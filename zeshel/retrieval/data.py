import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import random
from tqdm import tqdm
import sys


class ZeshelDataset(Dataset):
    def __init__(self, tokenizer, mentions, doc, max_len,
                 candidates, device, num_rands, type_cands,
                 all_entity_token_ids, all_entity_masks):
        self.tokenizer = tokenizer
        self.mentions = mentions
        self.doc = doc
        self.max_len = max_len  # the max  length of input (mention or entity)
        self.Ms = '[unused0]'
        self.Me = '[unused1]'
        self.ENT = '[unused2]'
        self.device = device
        self.candidates = candidates
        self.num_rands = num_rands
        self.type_cands = type_cands
        self.all_entity_token_ids = all_entity_token_ids
        self.all_entity_masks = all_entity_masks
        self.all_entity_indices = {x: index for index, x in
                                   enumerate(list(self.doc.keys()))}
        random.seed(42)

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, index):
        """

        :param index: The index of mention
        :return: mention_token_ids,mention_masks,entity_token_ids,entity_masks : 1 X L
                entity_hard_token_ids, entity_hard_masks: k X L  (k<=10)
        """
        # process mention
        mention = self.mentions[index]
        mention_window = self.get_mention_window(mention)
        mention_encoded_dict = self.tokenizer.encode_plus(mention_window,
                                                          add_special_tokens=True,
                                                          max_length=self.max_len,
                                                          pad_to_max_length=True,
                                                          truncation=True)
        mention_token_ids = mention_encoded_dict['input_ids']
        mention_masks = mention_encoded_dict['attention_mask']
        # process entity
        target_page_id = mention['label_document_id']
        target_idx = self.all_entity_indices[target_page_id]
        entity_token_ids = self.all_entity_token_ids[target_idx]
        entity_masks = self.all_entity_masks[target_idx]
        candidate_token_ids = [entity_token_ids]
        candidate_masks = [entity_masks]
        cands_probs = None
        if self.type_cands == 'mixed_negative':
            random_cands_pool = list(range(len(self.all_entity_indices)))
            random_cands_pool.remove(target_idx)
            rand_cands = random.sample(random_cands_pool, self.num_rands)
            negatives_token_ids = self.all_entity_token_ids[rand_cands].tolist()
            candidate_token_ids += negatives_token_ids
            negatives_masks = self.all_entity_masks[rand_cands].tolist()
            candidate_masks += negatives_masks

            # process hard negatives
            if self.candidates is not None:
                hard_negs = self.candidates[index]
                hard_negative_token_ids = self.all_entity_token_ids[
                    hard_negs].tolist()
                hard_negative_masks = self.all_entity_masks[hard_negs].tolist()
                candidate_token_ids += hard_negative_token_ids
                candidate_masks += hard_negative_masks
        elif self.type_cands == 'hard_adjusted_negative':
            distributed_cands = self.candidates[0][index]
            cands_probs = torch.tensor(self.candidates[1][index]).float()
            negative_token_ids = self.all_entity_token_ids[
                distributed_cands].tolist()
            negative_masks = self.all_entity_masks[distributed_cands].tolist()
            candidate_token_ids += negative_token_ids
            candidate_masks += negative_masks
        elif self.type_cands == 'hard_negative':
            distributed_cands = self.candidates[index]
            negative_token_ids = self.all_entity_token_ids[
                distributed_cands].tolist()
            negative_masks = self.all_entity_masks[distributed_cands].tolist()
            candidate_token_ids += negative_token_ids
            candidate_masks += negative_masks
        else:
            raise ValueError('wrong type candidates')

        mention_token_ids = torch.tensor(mention_token_ids).long()
        mention_masks = torch.tensor(mention_masks).long()
        candidate_token_ids = torch.tensor(candidate_token_ids).long()
        candidate_masks = torch.tensor(candidate_masks).long()
        if self.type_cands == 'hard_adjusted_negative':
            return mention_token_ids, mention_masks, candidate_token_ids, \
                   candidate_masks, cands_probs
        else:  # mixed/hard negative
            return mention_token_ids, mention_masks, candidate_token_ids, \
                   candidate_masks

    def get_mention_window(self, mention):
        page_id = mention['context_document_id']
        tokens = self.doc[page_id]['text'].split()
        max_mention_len = self.max_len - 2  # cls and sep

        # assert men == context_tokens[start_index:end_index]
        ctx_l = tokens[max(0, mention['start_index'] - max_mention_len - 1):
                       mention['start_index']]
        ctx_r = tokens[mention['end_index'] + 1:
                       mention['end_index'] + max_mention_len + 2]
        mention_tokens = tokens[mention['start_index']:mention['end_index'] + 1]

        ctx_l = ' '.join(ctx_l)
        ctx_r = ' '.join(ctx_r)
        mention_tokens = ' '.join(mention_tokens)
        mention_tokens = self.tokenizer.tokenize(mention_tokens)
        ctx_l = self.tokenizer.tokenize(ctx_l)
        ctx_r = self.tokenizer.tokenize(ctx_r)
        return self.help_mention_window(ctx_l, mention_tokens, ctx_r,
                                        max_mention_len)

    def help_mention_window(self, ctx_l, mention, ctx_r, max_len):
        if len(mention) >= max_len:
            window = mention[:max_len]
            return window
        leftover = max_len - len(mention) - 2  # [Ms] token and [Me] token
        leftover_hf = leftover // 2
        if len(ctx_l) > leftover_hf:
            ctx_l_len = leftover_hf if len(
                ctx_r) > leftover_hf else leftover - len(ctx_r)
        else:
            ctx_l_len = len(ctx_l)
        window = ctx_l[-ctx_l_len:] + [self.Ms] + mention + [self.Me] + ctx_r
        window = window[:max_len]
        return window


def transform_entities(doc, len_max, tokenizer):  # get all entities token 
    # ids and token masks
    def get_entity_window(en_page_id):
        page_id = en_page_id
        en_title = doc[page_id]['title']
        en_text = doc[page_id]['text']
        max_len = len_max - 2  # cls , sep
        ENT = '[unused2]'
        en_title_tokens = tokenizer.tokenize(en_title)
        en_text_tokens = tokenizer.tokenize(en_text)
        window = (en_title_tokens + [ENT] + en_text_tokens)[:max_len]
        return window

    all_entity_token_ids = []
    all_entity_masks = []
    all_entities = list(doc.keys())
    for page_id in all_entities:
        entity_window = get_entity_window(page_id)
        entity_dict = tokenizer.encode_plus(entity_window,
                                            add_special_tokens=True,
                                            max_length=len_max,
                                            pad_to_max_length=True,
                                            truncation=True)
        all_entity_token_ids.append(entity_dict['input_ids'])
        all_entity_masks.append(entity_dict['attention_mask'])
    all_entity_token_ids = np.array(all_entity_token_ids)
    all_entity_masks = np.array(all_entity_masks)
    return all_entity_token_ids, all_entity_masks


def load_data(data_dir):
    """

    :param data_dir
    :return: mentions, entities,doc
    """
    print('begin loading data')
    mention_path = os.path.join(data_dir, 'mentions')

    def load_mentions(part):
        mentions = []
        domains = set()
        with open(os.path.join(mention_path, '%s.json' % part)) as f:
            for line in f:
                field = json.loads(line)
                mentions.append(field)
                domains.add(field['corpus'])
        return mentions, domains

    samples_train, train_domain = load_mentions('train')
    samples_heldout_train_seen, heldout_train_domain = load_mentions(
        'heldout_train_seen')
    samples_heldout_train_unseen, heldout_train_unseen_domain = load_mentions(
        'heldout_train_unseen')
    samples_val, val_domain = load_mentions('val')
    samples_test, test_domain = load_mentions('test')

    def load_entities(domains):
        """

        :param domains: list of domains
        :return: all the entities in the domains
        """
        doc = {}
        doc_path = os.path.join(data_dir, 'documents')
        for domain in domains:
            with open(os.path.join(doc_path, domain + '.json')) as f:
                for line in f:
                    field = json.loads(line)
                    page_id = field['document_id']
                    doc[page_id] = field
        return doc

    train_doc = load_entities(train_domain)
    heldout_train_doc = load_entities(heldout_train_domain)
    heldout_train_unseen_doc = load_entities(heldout_train_unseen_domain)
    val_doc = load_entities(val_domain)
    test_doc = load_entities(test_domain)

    return samples_train, samples_heldout_train_seen, \
           samples_heldout_train_unseen, samples_val, samples_test, \
           train_doc, heldout_train_doc, heldout_train_unseen_doc, \
           heldout_train_unseen_doc, val_doc, test_doc


def get_all_entity_hiddens(en_loader, model, store_en_hiddens=False,
                           en_hidden_path=None):
    """
    :param en_loader: entity data loader
    :param model: current model
    :param store_en_hiddens: store entity hiddens?
    :param en_hidden_path: entity hidden states store path
    :return: 
    """
    model.eval()
    all_en_embeds = []
    with torch.no_grad():
        for i, batch in enumerate(en_loader):
            if hasattr(model, 'module'):
                en_embeds = (
                    model.module.encode(None, None, None, None, batch[0],
                                        batch[1])[
                        2]).detach().cpu()
            else:
                en_embeds = (
                    model.encode(None, None, None, None, batch[0],
                                 batch[1])[
                        2]).detach().cpu()
            if store_en_hiddens:
                file_path = os.path.join(en_hidden_path, 'en_hiddens_%s.pt' % i)
                torch.save(en_embeds, file_path)
            else:
                all_en_embeds.append(en_embeds)
    if store_en_hiddens:
        return None
    else:
        all_en_embeds = torch.cat(all_en_embeds, dim=0)
        return all_en_embeds


def get_hard_negative(mention_loader, model,
                      num_hards, len_en_loader,
                      device, distribution_sampling=False,
                      exclude_golds=True,
                      too_large=False,
                      all_candidates_embeds=None,
                      en_hidden_path=None, adjust_logits=False,
                      smoothing_value=1.0):
    model.eval()
    if not too_large:
        if hasattr(model, 'module'):
            model.module.candidates_embeds = all_candidates_embeds
        else:
            model.candidates_embeds = all_candidates_embeds
    hard_indices = []
    all_candidates_probs = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(mention_loader)):
            if not too_large:  # entity embeddings are too large for memory
                scores = model(batch[0], batch[1], None, None)
            else:
                scores = []
                for j in range(len_en_loader):
                    file_path = os.path.join(en_hidden_path,
                                             'en_hiddens_%s.pt' % j)
                    en_embeds = torch.load(file_path)
                    if hasattr(model, 'module'):
                        model.module.candidates_embeds = en_embeds
                    else:
                        model.candidates_embeds = en_embeds
                    score = model(batch[0], batch[1], None,
                                  None).detach()
                    scores.append(score)
                scores = torch.cat(scores, dim=1)
            if distribution_sampling:
                scores = scores*smoothing_value
            if exclude_golds:
                label_cols = batch[2].view(-1).to(device)
                label_rows = torch.arange(scores.size(0)).to(device)
                scores[label_rows, label_cols] = float('-inf')  # exclude golds
            if distribution_sampling:
                probs = scores.softmax(dim=1)
                hard_cands = distribution_sample(probs, num_hards,
                                                 device)
                # keep q_i for adjustment
                if adjust_logits:
                    candidates_probs = probs.gather(1, hard_cands)
                    all_candidates_probs.append(candidates_probs)
            else:
                hard_cands = scores.topk(num_hards, dim=1)[1]
            hard_indices.append(hard_cands)
    hard_indices = torch.cat(hard_indices, dim=0).tolist()
    if hasattr(model, 'module'):
        model.module.candidates_embeds = None
    else:
        model.candidates_embeds = None
    if adjust_logits:
        all_candidates_probs = torch.cat(all_candidates_probs,
                                         dim=0).tolist()
        return hard_indices, all_candidates_probs
    else:
        return hard_indices


def distribution_sample(probs, num_cands, device):
    num_nonzero = (probs != 0).sum(1)
    less_indices = (num_nonzero < num_cands)
    min_num_nonzero = num_nonzero.min().item()
    if min_num_nonzero < num_cands:
        candidates = torch.zeros((probs.size(0), num_cands)).long().to(device)
        candidates[less_indices] = torch.multinomial(probs[less_indices],
                                                     num_cands,
                                                     replacement=True)
        if candidates[less_indices].size(0) != probs.size(0):
            candidates[~less_indices] = torch.multinomial(probs[~less_indices],
                                                          num_cands,
                                                          replacement=False)
    else:
        candidates = torch.multinomial(probs, num_cands, replacement=False)
    return candidates


class EntitySet(Dataset):
    def __init__(self, tokenizer, all_entity_token_ids, all_entity_masks):
        self.tokenizer = tokenizer
        self.all_entity_token_ids = all_entity_token_ids
        self.all_entity_masks = all_entity_masks

    def __len__(self):
        return len(self.all_entity_token_ids)

    def __getitem__(self, index):
        """

        :param index: The index of mention
        :return: mention_token_ids,mention_masks,entity_token_ids,entity_masks : 1 X L
                entity_hard_token_ids, entity_hard_masks: k X L  (k<=10)
        """
        # process entity(for in batch training)
        entity_token_ids = self.all_entity_token_ids[index]
        entity_masks = self.all_entity_masks[index]
        entity_token_ids = torch.tensor(entity_token_ids).long()
        entity_masks = torch.tensor(entity_masks).long()
        return entity_token_ids, entity_masks


class MentionSet(Dataset):
    def __init__(self, tokenizer, mentions, doc, max_len):
        self.tokenizer = tokenizer
        self.mentions = mentions
        self.doc = doc
        self.max_len = max_len  # the max  length of input (mention or entity)
        self.Ms = '[unused0]'
        self.Me = '[unused1]'
        self.all_entity_indices = {x: index for index, x in
                                   enumerate(list(self.doc.keys()))}

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, index):
        """

        :param index: The index of mention
        :return: mention_token_ids,mention_masks,entity_token_ids,entity_masks : 1 X L
                entity_hard_token_ids, entity_hard_masks: k X L  (k<=10)
        """
        # process mention
        mention = self.mentions[index]
        mention_window = self.get_mention_window(mention)
        mention_encoded_dict = self.tokenizer.encode_plus(mention_window,
                                                          add_special_tokens=True,
                                                          max_length=self.max_len,
                                                          pad_to_max_length=True,
                                                          truncation=True)
        mention_token_ids = mention_encoded_dict['input_ids']
        mention_masks = mention_encoded_dict['attention_mask']
        # process entity(for in batch training)
        target_page_id = mention['label_document_id']
        label = self.all_entity_indices[target_page_id]
        label = torch.tensor([label]).long()
        mention_token_ids = torch.tensor(mention_token_ids).long()
        mention_masks = torch.tensor(mention_masks).long()
        return mention_token_ids, mention_masks, label

    def get_mention_window(self, mention):
        page_id = mention['context_document_id']
        tokens = self.doc[page_id]['text'].split()
        max_mention_len = self.max_len - 2  # cls and sep

        # assert men == context_tokens[start_index:end_index]
        ctx_l = tokens[max(0, mention['start_index'] - max_mention_len - 1):
                       mention['start_index']]
        ctx_r = tokens[mention['end_index'] + 1:
                       mention['end_index'] + max_mention_len + 2]
        men = tokens[mention['start_index']:mention['end_index'] + 1]

        ctx_l = ' '.join(ctx_l)
        ctx_r = ' '.join(ctx_r)
        men = ' '.join(men)
        men = self.tokenizer.tokenize(men)
        ctx_l = self.tokenizer.tokenize(ctx_l)
        ctx_r = self.tokenizer.tokenize(ctx_r)
        return self.help_mention_window(ctx_l, men, ctx_r, max_mention_len)

    def help_mention_window(self, ctx_l, mention, ctx_r, max_len):
        if len(mention) >= max_len:
            window = mention[:max_len]
            return window
        leftover = max_len - len(mention) - 2  # [Ms] token and [Me] token
        leftover_hf = leftover // 2
        if len(ctx_l) > leftover_hf:
            ctx_l_len = leftover_hf if len(
                ctx_r) > leftover_hf else leftover - len(ctx_r)
        else:
            ctx_l_len = len(ctx_l)
        window = ctx_l[-ctx_l_len:] + [self.Ms] + mention + [self.Me] + ctx_r
        window = window[:max_len]
        return window


class Data:
    def __init__(self, train_doc, val_doc, test_doc, tokenizer,
                 all_train_en_token_ids, all_train_en_masks,
                 all_val_en_token_ids, all_val_en_masks, all_test_en_token_ids,
                 all_test_en_masks, max_len,
                 train_mention, val_mention, test_mention):
        self.train_doc = train_doc
        self.val_doc = val_doc
        self.test_doc = test_doc
        self.tokenizer = tokenizer
        self.all_train_en_token_ids = all_train_en_token_ids
        self.all_train_en_masks = all_train_en_masks
        self.all_val_en_token_ids = all_val_en_token_ids
        self.all_val_en_masks = all_val_en_masks
        self.all_test_en_token_ids = all_test_en_token_ids
        self.all_test_en_masks = all_test_en_masks
        self.train_men = train_mention
        self.val_men = val_mention
        self.test_men = test_mention
        self.max_len = max_len

    def get_loaders(self, mention_bsz, entity_bsz):
        train_en_set = EntitySet(self.tokenizer, self.all_train_en_token_ids,
                                 self.all_train_en_masks)
        val_en_set = EntitySet(self.tokenizer, self.all_val_en_token_ids,
                               self.all_val_en_masks)
        test_en_set = EntitySet(self.tokenizer, self.all_test_en_token_ids,
                                self.all_test_en_masks)
        train_men_set = MentionSet(self.tokenizer, self.train_men,
                                   self.train_doc,
                                   self.max_len)
        val_men_set = MentionSet(self.tokenizer, self.val_men, self.val_doc,
                                 self.max_len)
        test_men_set = MentionSet(self.tokenizer, self.test_men, self.test_doc,
                                  self.max_len)
        train_en_loader = DataLoader(train_en_set, entity_bsz, shuffle=False)
        val_en_loader = DataLoader(val_en_set, entity_bsz, shuffle=False)
        test_en_loader = DataLoader(test_en_set, entity_bsz, shuffle=False)
        train_men_loader = DataLoader(train_men_set, mention_bsz,
                                      shuffle=False)
        val_men_loader = DataLoader(val_men_set, mention_bsz, shuffle=False)
        test_men_loader = DataLoader(test_men_set, mention_bsz,
                                     shuffle=False)
        return train_en_loader, val_en_loader, test_en_loader, \
               train_men_loader, val_men_loader, test_men_loader


class Logger(object):

    def __init__(self, log_path, on=True):
        self.log_path = log_path
        self.on = on

        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += '+'

    def log(self, string, newline=True):
        if self.on:
            with open(self.log_path, 'a') as logf:
                logf.write(string)
                if newline: logf.write('\n')

            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()
