import torch
from retriever import UnifiedRetriever
from data_retriever import MentionSet, EntitySet, transform_entities
import logging
import argparse
import numpy as np
from transformers import BertTokenizer, BertModel, RobertaModel
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
import torch.nn as nn


def get_all_entity_hiddens(en_loaders, model,
                           store_en_hiddens=False,
                           en_hidden_path=None):
    """
    get  all the entity embeddings
    :return:
    """
    model.eval()
    all_entity_embeds = []
    for i in range(len(en_loaders)):
        domain_entity_embeds = []
        en_loader = en_loaders[i]
        with torch.no_grad():
            for j, batch in enumerate(en_loader):
                if hasattr(model, 'module'):
                    en_embeds = (
                        model.module.encode(None, None, None, None, batch[0],
                                            batch[1])[2]).detach().cpu()
                else:
                    en_embeds = (
                        model.encode(None, None, None, None, batch[0],
                                     batch[1])[2]).detach().cpu()
                if store_en_hiddens:
                    file_path = os.path.join(en_hidden_path,
                                             'en_hiddens_{}_{}.pt'.format(i, j))
                    torch.save(en_embeds, file_path)
                else:
                    domain_entity_embeds.append(en_embeds)
            if not store_en_hiddens:
                domain_entity_embeds = torch.cat(domain_entity_embeds, dim=0)
                all_entity_embeds.append(domain_entity_embeds)
    if store_en_hiddens:
        return None
    else:
        return all_entity_embeds


def compute_cands_indices(men_loaders, model, en_loaders,
                          num_cands, too_large=False, en_hidden_path=None,
                          domain_entity_embeds=None):
    model.eval()
    all_cands_indices = []
    for i in range(len(men_loaders)):
        mention_loader = men_loaders[i]
        if not too_large:
            if hasattr(model, 'module'):
                model.module.evaluate_on = True
                model.module.candidates_embeds = domain_entity_embeds[i]
            else:
                model.evaluate_on = True
                model.candidates_embeds = domain_entity_embeds[i]
        len_entity_loader = len(en_loaders[i])
        cands_indices = []
        with torch.no_grad():
            for t, batch in tqdm(enumerate(mention_loader)):
                if not too_large:
                    scores = model(batch[0], batch[1], None,
                                   None).detach()
                else:
                    scores = []
                    for j in range(len_entity_loader):
                        file_path = os.path.join(en_hidden_path,
                                                 'en_hiddens_{}_{}.pt'.format(i,
                                                                              j))
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
                # golds
                cands = scores.topk(num_cands, dim=1)[1]
                cands_indices.append(cands)
        cands_indices = torch.cat(cands_indices, dim=0).cpu().numpy()
        all_cands_indices.append(cands_indices)
    if hasattr(model, 'module'):
        model.module.evaluate_on = False
        model.module.candidates_embeds = None
    else:
        model.evaluate_on = False
        model.candidates_embeds = None
    return all_cands_indices


def save_candidates(all_cands_indices, cands_dir, doc, all_mentions, part):
    cands = []
    for i in range(len(all_cands_indices)):
        cands_indices = all_cands_indices[i]
        entities = doc[i]
        mentions = all_mentions[i]
        all_entities = np.array(list(entities.keys()))
        assert cands_indices.shape[0] == len(mentions)
        for j in range(cands_indices.shape[0]):
            candidate = {}
            candidate['mention_id'] = mentions[j]['mention_id']
            candidate['candidates'] = all_entities[cands_indices[j]].tolist()
            cands.append(candidate)
    with open(os.path.join(cands_dir, 'candidates_%s.json' % part), 'w') as f:
        for item in cands:
            f.write('%s\n' % json.dumps(item))


def main(args):
    LOG_FORMAT = "%(levelname)s:  %(message)s"
    logging.basicConfig(

        level=logging.DEBUG,
        format=LOG_FORMAT)
    # load data and initialize model and dataset
    train_mentions, val_mentions, test_mentions, train_doc, val_doc, test_doc \
        = load_domain_data(args.data_dir)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # get model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    package = torch.load(args.model) if device.type == 'cuda' \
        else torch.load(args.model, map_location=torch.device('cpu'))
    new_state_dict = package['sd']
    # encoder=MLPEncoder(args.max_len)
    if args.pre_model == 'Bert':
        encoder = BertModel.from_pretrained('bert-base-uncased')
    if args.pre_model == 'Roberta':
        encoder = RobertaModel.from_pretrained('roberta-base')
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
    model.load_state_dict(new_state_dict)

    dp = torch.cuda.device_count() > 1
    if dp:
        print('Data parallel across {:d} GPUs {:s}'
              ''.format(len(args.gpus.split(',')), args.gpus))
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    train_loaders, val_loaders, test_loaders = [], [], []
    train_entity_loaders, val_entity_loaders, test_entity_loaders = [], [], []
    for i in range(len(train_mentions)):
        mentions = train_mentions[i]
        doc = train_doc[i]
        entity_token_ids, entity_masks = transform_entities(doc, args.max_len,
                                                            tokenizer)
        mention_set = MentionSet(tokenizer, mentions, doc, args.max_len)
        entity_set = EntitySet(tokenizer, entity_token_ids, entity_masks)
        train_loaders.append(DataLoader(mention_set, args.mention_bsz,
                                        shuffle=False))
        entity_loader = DataLoader(entity_set, args.entity_bsz,
                                   shuffle=False)
        train_entity_loaders.append(entity_loader)
    for i in range(len(val_mentions)):
        mentions = val_mentions[i]
        doc = val_doc[i]
        entity_token_ids, entity_masks = transform_entities(doc, args.max_len,
                                                            tokenizer)
        mention_set = MentionSet(tokenizer, mentions, doc, args.max_len)
        entity_set = EntitySet(tokenizer, entity_token_ids, entity_masks)
        val_loaders.append(DataLoader(mention_set, args.mention_bsz,
                                      shuffle=False))
        entity_loader = DataLoader(entity_set, args.entity_bsz,
                                   shuffle=False)
        val_entity_loaders.append(entity_loader)
    for i in range(len(test_mentions)):
        mentions = test_mentions[i]
        doc = test_doc[i]
        entity_token_ids, entity_masks = transform_entities(doc, args.max_len,
                                                            tokenizer)
        mention_set = MentionSet(tokenizer, mentions, doc, args.max_len)
        entity_set = EntitySet(tokenizer, entity_token_ids, entity_masks)
        test_loaders.append(DataLoader(mention_set, args.mention_bsz,
                                       shuffle=False))
        entity_loader = DataLoader(entity_set, args.entity_bsz,
                                   shuffle=False)
        test_entity_loaders.append(entity_loader)
    print('begin computing candidates indices')
    all_train_cands_embeds = get_all_entity_hiddens(train_entity_loaders,
                                                    model,
                                                    args.store_en_hiddens,
                                                    args.en_hidden_path)
    train_cands_indices = compute_cands_indices(train_loaders, model,
                                                train_entity_loaders,
                                                args.num_cands,
                                                args.store_en_hiddens,
                                                args.en_hidden_path,
                                                all_train_cands_embeds)
    all_val_cands_embeds = get_all_entity_hiddens(val_entity_loaders,
                                                  model,
                                                  args.store_en_hiddens,
                                                  args.en_hidden_path)
    val_cands_indices = compute_cands_indices(val_loaders, model,
                                              val_entity_loaders,
                                              args.num_cands,
                                              args.store_en_hiddens,
                                              args.en_hidden_path,
                                              all_val_cands_embeds)
    all_test_cands_embeds = get_all_entity_hiddens(test_entity_loaders,
                                                   model,
                                                   args.store_en_hiddens,
                                                   args.en_hidden_path)
    test_cands_indices = compute_cands_indices(test_loaders, model,
                                               test_entity_loaders,
                                               args.num_cands,
                                               args.store_en_hiddens,
                                               args.en_hidden_path,
                                               all_test_cands_embeds)

    print('begin saving train candidates')
    save_candidates(train_cands_indices, args.cands_dir, train_doc,
                    train_mentions, 'train')
    print('begin saving val candidates')
    save_candidates(val_cands_indices, args.cands_dir, val_doc,
                    val_mentions, 'val')
    print('begin saving test candidates')
    save_candidates(test_cands_indices, args.cands_dir, test_doc,
                    test_mentions, 'test')


def load_domain_data(data_dir):
    """

    :param data_dir: train_data_dir if args.train else eval_data_dir
    :return: mentions, entities,doc
    """
    print('begin loading data')
    men_path = os.path.join(data_dir, 'mentions')

    def get_domains(part):
        domains = set()
        with open(os.path.join(men_path, '%s.json' % part)) as f:
            for line in f:
                field = json.loads(line)
                domains.add(field['corpus'])
        return list(domains)

    def load_domain_mentions(domain, part):
        domain_mentions = []
        with open(os.path.join(men_path, '%s.json' % part)) as f:
            for line in f:
                field = json.loads(line)
                if field['corpus'] == domain:
                    domain_mentions.append(field)
        return domain_mentions

    def load_domain_entities(data_dir, domain):
        """

        :param domains: list of domains
        :return: all the entities in the domains
        """
        doc = {}
        doc_path = os.path.join(data_dir, 'documents')
        with open(os.path.join(doc_path, domain + '.json')) as f:
            for line in f:
                field = json.loads(line)
                page_id = field['document_id']
                doc[page_id] = field
        return doc

    train_domains = get_domains('train')
    val_domains = get_domains('val')
    test_domains = get_domains('test')
    train_mentions = []
    val_mentions = []
    test_mentions = []
    train_doc = []
    val_doc = []
    test_doc = []
    for train_domain in train_domains:
        domain_mentions = load_domain_mentions(train_domain, 'train')
        domain_entities = load_domain_entities(data_dir, train_domain)
        train_mentions.append(domain_mentions)
        train_doc.append(domain_entities)
    for val_domain in val_domains:
        domain_mentions = load_domain_mentions(val_domain, 'val')
        domain_entities = load_domain_entities(data_dir, val_domain)
        val_mentions.append(domain_mentions)
        val_doc.append(domain_entities)
    for test_domain in test_domains:
        domain_mentions = load_domain_mentions(test_domain, 'test')
        domain_entities = load_domain_entities(data_dir, test_domain)
        test_mentions.append(domain_mentions)
        test_doc.append(domain_entities)

    return train_mentions, val_mentions, test_mentions, train_doc, val_doc, \
           test_doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='model path')
    parser.add_argument('--max_len', type=int, default=128,
                        help='max length of the mention input '
                             'and the entity input')
    parser.add_argument('--data_dir', type=str,
                        help='the  data directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers [%(default)d]')
    parser.add_argument('--gpus', default='', type=str,
                        help='GPUs separated by comma [%(default)s]')
    parser.add_argument('--mention_bsz', default=128, type=int,
                        help='the mention batch size')
    parser.add_argument('--entity_bsz', default=512, type=int,
                        help='the entity batch size')
    parser.add_argument('--cands_dir', type=str,
                        help='the  data directory')
    parser.add_argument('--num_cands', default=64, type=int,
                        help='the total number of candidates')
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
    parser.add_argument('--pre_model', default='Bert',
                        choices=['Bert', 'Roberta'],
                        type=str, help='the encoder for train')
    parser.add_argument('--store_en_hiddens', action='store_true',
                        help='store entity hiddens?')
    args = parser.parse_args()
    # Set environment variables before all else.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus  # Sets torch.cuda behavior
    main(args)
