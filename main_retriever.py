import torch
from retriever import UnifiedRetriever
from data_retriever import ZeshelDataset, transform_entities, load_data, \
    get_all_entity_hiddens, get_hard_negative, \
    Data
from util import Logger
import argparse
import numpy as np
import os
import random
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW, \
    get_linear_schedule_with_warmup, get_constant_schedule, RobertaTokenizer, \
    RobertaModel
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def configure_optimizer(args, model, num_train_examples):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
                      eps=args.adam_epsilon)

    num_train_steps = int(num_train_examples / args.B /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps)

    return optimizer, scheduler, num_train_steps, num_warmup_steps


def configure_optimizer_simple(args, model, num_train_examples):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_train_steps = int(num_train_examples / args.B /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = 0

    scheduler = get_constant_schedule(optimizer)

    return optimizer, scheduler, num_train_steps, num_warmup_steps


def evaluate(mention_loader, model, all_candidates_embeds, k, device,
             len_en_loader,
             too_large=False, en_hidden_path=None):
    model.eval()
    if not too_large:
        if hasattr(model, 'module'):
            model.module.evaluate_on = True
            model.module.candidates_embeds = all_candidates_embeds
        else:
            model.evaluate_on = True
            model.candidates_embeds = all_candidates_embeds
    nb_samples = 0
    r_k = 0
    acc = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(mention_loader)):
            if not too_large:
                scores = model(batch[0], batch[1], None, None)
            else:
                scores = []
                for j in range(len_en_loader):
                    file_path = os.path.join(en_hidden_path,
                                             'en_hiddens_%s.pt' % j)
                    en_embeds = torch.load(file_path)
                    if hasattr(model, 'module'):
                        model.module.evaluate_on = True
                        model.module.candidates_embeds = en_embeds
                    else:
                        model.evaluate_on = True
                        model.candidates_embeds = en_embeds
                    score = model(batch[0], batch[1], None,
                                  None).detach()
                    scores.append(score)
                scores = torch.cat(scores, dim=1)
            labels = batch[2].to(device)
            top_k = scores.topk(k, dim=1)[1]
            preds = top_k[:, 0]
            r_k += (top_k == labels.to(device)).sum().item()
            nb_samples += scores.size(0)
            acc += (preds == labels.squeeze(1).to(device)).sum().item()
    r_k /= nb_samples
    acc /= nb_samples
    if hasattr(model, 'module'):
        model.module.evaluate_on = False
        model.module.candidates_embeds = None
    else:
        model.evaluate_on = False
        model.candidates_embeds = None
    return r_k, acc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    set_seeds(args)
    # configure logger
    logger = Logger(args.model + '.log', True)
    logger.log(str(args))
    # load data and initialize model and dataset
    samples_train, samples_heldout_train_seen, \
    samples_heldout_train_unseen, samples_val, samples_test, \
    train_doc, heldout_train_doc, heldout_train_unseen_doc, \
    heldout_train_unseen_doc, val_doc, test_doc = load_data(
        args.data_dir)
    num_rands = int(args.num_cands * args.cands_ratio)
    num_hards = args.num_cands - num_rands
    # get model and tokenizer
    if args.pre_model == 'Bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoder = BertModel.from_pretrained('bert-base-uncased')
    elif args.pre_model == 'Roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        encoder = RobertaModel.from_pretrained('roberta-base')
    else:
        raise ValueError('wrong encoder type')
    # encoder=MLPEncoder(args.max_len)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.type_model == 'poly':
        attention_type = 'soft_attention'
    else:
        attention_type = 'hard_attention'
    if args.type_model == 'dual':
        num_mention_vecs = 1
        num_entity_vecs = 1
    elif args.type_model == 'multi_vector':
        num_mention_vecs = 1
        num_entity_vecs = args.num_entity_vecs
    else:
        num_mention_vecs = args.num_mention_vecs
        num_entity_vecs = args.num_entity_vecs
    model = UnifiedRetriever(encoder, device, num_mention_vecs, num_entity_vecs,
                             args.mention_use_codes, args.entity_use_codes,
                             attention_type, None, False)
    if args.resume_training:
        cpt = torch.load(args.model) if device.type == 'cuda' \
            else torch.load(args.model, map_location=torch.device('cpu'))
        model.load_state_dict(cpt['sd'])
    dp = torch.cuda.device_count() > 1
    if dp:
        logger.log('Data parallel across {:d} GPUs {:s}'
                   ''.format(len(args.gpus.split(',')), args.gpus))
        model = nn.DataParallel(model)
    model.to(device)
    logger.log('transform train entities')
    all_train_entity_token_ids, all_train_masks = transform_entities(train_doc,
                                                                     args.max_len,
                                                                     tokenizer)
    logger.log('transform valid and test entities')
    all_val_entity_token_ids, all_val_masks = transform_entities(val_doc,
                                                                 args.max_len,
                                                                 tokenizer)
    all_test_entity_token_ids, all_test_masks = transform_entities(test_doc,
                                                                   args.max_len,
                                                                   tokenizer)
    data = Data(train_doc, val_doc, test_doc, tokenizer,
                all_train_entity_token_ids, all_train_masks,
                all_val_entity_token_ids, all_val_masks,
                all_test_entity_token_ids, all_test_masks, args.max_len,
                samples_train, samples_val, samples_test)
    train_en_loader, val_en_loader, test_en_loader, train_men_loader, \
    val_men_loader, test_men_loader = data.get_loaders(args.mention_bsz,
                                                       args.entity_bsz)
    model.train()

    # configure optimizer
    num_train_samples = len(samples_train)
    if args.simpleoptim:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer_simple(args, model, num_train_samples)
    else:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer(args, model, num_train_samples)
    if args.resume_training:
        optimizer.load_state_dict(cpt['opt_sd'])
        scheduler.load_state_dict(cpt['scheduler_sd'])
    # train
    logger.log('***** train *****')
    logger.log('# train samples: {:d}'.format(num_train_samples))
    logger.log('# epochs: {:d}'.format(args.epochs))
    logger.log(' batch size: {:d}'.format(args.B))
    logger.log(' gradient accumulation steps {:d}'
               ''.format(args.gradient_accumulation_steps))
    logger.log(
        ' effective training batch size with accumulation: {:d}'
        ''.format(args.B * args.gradient_accumulation_steps))
    logger.log(' # training steps: {:d}'.format(num_train_steps))
    logger.log(' # warmup steps: {:d}'.format(num_warmup_steps))
    logger.log(' learning rate: {:g}'.format(args.lr))
    logger.log(' # parameters: {:d}'.format(count_parameters(model)))
    # start_time = datetime.now()
    step_num = 0
    tr_loss, logging_loss = 0.0, 0.0
    if args.resume_training:
        step_num = cpt['step_num']
        tr_loss, logging_loss = cpt['tr_loss'], cpt['logging_loss']
    model.zero_grad()
    best_val_perf = float('-inf')
    start_epoch = 1
    if args.resume_training:
        start_epoch = cpt['epoch'] + 1
    for epoch in range(start_epoch, args.epochs + 1):
        logger.log('\nEpoch {:d}'.format(epoch))
        if args.type_cands == 'hard_adjusted_negative':
            distribution_sampling = True
            adjust_logits = True
            num_cands = args.num_cands
        elif args.type_cands == 'hard_negative':
            distribution_sampling = True
            adjust_logits = False
            num_cands = args.num_cands
        elif args.type_cands == 'mixed_negative':
            distribution_sampling = False
            adjust_logits = False
            num_cands = num_hards
        else:
            raise ValueError('type candidates wrong')
        if args.type_cands == 'mixed_negative' and num_hards == 0:
            candidates = None
        else:
            all_train_cands_embeds = get_all_entity_hiddens(train_en_loader,
                                                            model,
                                                            args.store_en_hiddens,
                                                            args.en_hidden_path)
            candidates = get_hard_negative(train_men_loader, model, num_cands,
                                           len(train_en_loader), device,
                                           distribution_sampling,
                                           args.exclude_golds,
                                           args.store_en_hiddens,
                                           all_train_cands_embeds,
                                           args.en_hidden_path,
                                           adjust_logits, args.smoothing_value)
        train_set = ZeshelDataset(tokenizer, samples_train, train_doc,
                                  args.max_len,
                                  candidates, device, num_rands,
                                  args.type_cands,
                                  all_train_entity_token_ids,
                                  all_train_masks)

        train_loader = DataLoader(train_set, args.B, shuffle=True,
                                  drop_last=False)
        for step, batch in enumerate(train_loader):
            model.train()
            loss = model(*batch)[0]
            if len(args.gpus) > 1:
                loss = loss.mean()
            loss.backward()
            tr_loss += loss.item()
            # print('loss is %f' % loss.item())

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                step_num += 1

                if step_num % args.logging_steps == 0:
                    avg_loss = (tr_loss - logging_loss) / args.logging_steps
                    logger.log('Step {:10d}/{:d} | Epoch {:3d} | '
                               'Batch {:5d}/{:5d} | '
                               'Average Loss {:8.4f}'
                               ''.format(step_num, num_train_steps,
                                         epoch, step + 1,
                                         len(train_loader), avg_loss))
                    logging_loss = tr_loss

        #  eval_train_result = evaluate(train_loader, model, args.k,device)[0]
        all_val_cands_embeds = get_all_entity_hiddens(val_en_loader, model,
                                                      args.store_en_hiddens,
                                                      args.en_hidden_path)
        eval_result = evaluate(val_men_loader, model, all_val_cands_embeds,
                               args.k, device, len(val_en_loader),
                               args.store_en_hiddens, args.en_hidden_path)
        logger.log('Done with epoch {:3d} | train loss {:8.4f} | '
                   'validation recall {:8.4f}'
                   '|validation accuracy {:8.4f}'.format(
            epoch,
            tr_loss / step_num,
            eval_result[0],
            eval_result[1]
        ))
        save_model = (eval_result[0] >= best_val_perf) if args.eval_criterion \
                                                          == 'recall' else \
            (eval_result[1] >= best_val_perf)
        if save_model:
            logger.log('------- new best val perf: {:g} --> {:g} '
                       ''.format(best_val_perf, eval_result[0]))
            best_val_perf = eval_result[0]
            torch.save({'opt': args,
                        'sd': model.module.state_dict() if dp else model.state_dict(),
                        'perf': best_val_perf, 'epoch': epoch,
                        'opt_sd': optimizer.state_dict(),
                        'scheduler_sd': scheduler.state_dict(),
                        'tr_loss': tr_loss, 'step_num': step_num,
                        'logging_loss': logging_loss},
                       args.model)
        else:
            logger.log('')
        # update dataset and dataloader
        # torch.cuda.empty_cache()

        # torch.cuda.empty_cache()
    # test model on test dataset
    package = torch.load(args.model) if device.type == 'cuda' \
        else torch.load(args.model, map_location=torch.device('cpu'))
    new_state_dict = package['sd']
    # encoder=MLPEncoder(args.max_len)
    if args.pre_model == 'Bert':
        encoder = BertModel.from_pretrained('bert-base-uncased')
    elif args.pre_model == 'Roberta':
        encoder = RobertaModel.from_pretrained('roberta-base')
    else:
        raise ValueError('wrong encoder type')
    model = UnifiedRetriever(encoder, device, num_mention_vecs, num_entity_vecs,
                             args.mention_use_codes, args.entity_use_codes,
                             attention_type, None, False)
    model.load_state_dict(new_state_dict)
    if dp:
        logger.log('Data parallel across {:d} GPUs {:s}'
                   ''.format(len(args.gpus.split(',')), args.gpus))
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    all_test_cands_embeds = get_all_entity_hiddens(test_en_loader, model,
                                                   args.store_en_hiddens,
                                                   args.en_hidden_path)
    test_result = evaluate(test_men_loader, model, all_test_cands_embeds,
                           args.k, device, len(test_en_loader),
                           args.store_en_hiddens, args.en_hidden_path)
    logger.log(' test recall@{:d} : {:8.4f}'
               '| test accuracy : {:8.4f}'.format(args.k, test_result[0],
                                                  test_result[1]))


#    test_train_result = evaluate(train_loader, model,args)
#   logger.log('test train acc {:f}'.format(test_train_result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='model path')
    parser.add_argument('--resume_training', action='store_true',
                        help='resume training from checkpoint?')
    parser.add_argument('--max_len', type=int, default=128,
                        help='max length of the mention input '
                             'and the entity input')
    parser.add_argument('--num_hards', type=int, default=10,
                        help='the number of the nearest neighbors we use to '
                             'construct hard negatives')
    parser.add_argument('--type_cands', type=str,
                        default='mixed_negative',
                        choices=['mixed_negative',
                                 'hard_negative',
                                 'hard_adjusted_negative'],
                        help='the type of negative we use during training')
    parser.add_argument('--data_dir', type=str,
                        help='the  data directory')

    parser.add_argument('--B', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                        choices=[5e-6, 1e-5, 2e-5, 5e-5, 2e-4, 5e-4, 0.002,
                                 0.001],
                        help='the learning rate')
    parser.add_argument('--epochs', type=int, default=3,
                        help='the number of training epochs')
    parser.add_argument('--k', type=int, default=64,
                        help='recall@k when evaluate')
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='proportion of training steps to perform linear '
                             'learning rate warmup for [%(default)g]')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay [%(default)g]')
    parser.add_argument('--adam_epsilon', type=float, default=1e-6,
                        help='epsilon for Adam optimizer [%(default)g]')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='num gradient accumulation steps [%(default)d]')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers [%(default)d]')
    parser.add_argument('--simpleoptim', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--clip', type=float, default=1,
                        help='gradient clipping [%(default)g]')
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help='num logging steps [%(default)d]')
    parser.add_argument('--gpus', default='', type=str,
                        help='GPUs separated by comma [%(default)s]')
    parser.add_argument('--eval_criterion', type=str, default='recall',
                        choices=['recall', 'accuracy'],
                        help='the criterion for selecting model')
    parser.add_argument('--pre_model', default='Bert',
                        choices=['Bert', 'Roberta'],
                        type=str, help='the encoder for train')
    parser.add_argument('--cands_ratio', default=1.0, type=float,
                        help='the ratio between random candidates and hard '
                             'candidates')
    parser.add_argument('--num_cands', default=128, type=int,
                        help='the total number of candidates')
    parser.add_argument('--smoothing_value', default=1, type=float,
                        help=' smoothing factor when sampling negatives '
                             'according to model distribution')
    parser.add_argument('--eval_batchsize', type=int, default=512,
                        help='the batch size')
    parser.add_argument('--mention_bsz', type=int, default=512,
                        help='the batch size')
    parser.add_argument('--entity_bsz', type=int, default=512,
                        help='the batch size')
    parser.add_argument('--exclude_golds', action='store_true',
                        help='exclude golds when sampling?')
    parser.add_argument('--type_model', type=str,
                        default='dual',
                        choices=['dual',
                                 'sum_max',
                                 'multi_vector',
                                 'poly'],
                        help='the type of model')
    parser.add_argument('--num_mention_vecs', type=int, default=8,
                        help='the number of mention vectors ')
    parser.add_argument('--num_entity_vecs', type=int, default=8,
                        help='the number of entity vectors  ')
    parser.add_argument('--mention_use_codes', action='store_true',
                        help='use codes for mention embeddings?')
    parser.add_argument('--entity_use_codes', action='store_true',
                        help='use codes for entity embeddings?')
    parser.add_argument('--en_hidden_path', type=str,
                        help='all entity hidden states path')
    parser.add_argument('--store_en_hiddens', action='store_true',
                        help='store entity hiddens?')
    args = parser.parse_args()
    # Set environment variables before all else.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus  # Sets torch.cuda behavior
    main(args)
