import json
import math
import os
import statistics as stat
import sys
import torch
import util

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from util import Logger as BasicLogger


class BasicDataset(Dataset):

    def __init__(self, documents, samples, tokenizer, max_len,
                 max_num_candidates,
                 is_training, indicate_mention_boundaries=False):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_len_mention = 0
        self.max_len_candidate = 0
        self.max_num_candidates = max_num_candidates
        self.is_training = is_training
        self.num_samples_original = len(samples[0])
        self.indicate_mention_boundaries = indicate_mention_boundaries
        self.MENTION_START = '[unused0]'
        self.MENTION_END = '[unused1]'
        self.samples = self.cull_samples(samples)

    def __len__(self):
        return len(self.samples[0])

    def __getitem__(self, index):
        raise NotImplementedError

    def prepare_candidates(self, mention, candidates):
        xs = candidates['candidates'][:self.max_num_candidates]
        y = mention['label_document_id']

        if self.is_training:
            # At training time we can include target if not already included.
            if not y in xs:
                xs.append(y)
        else:
            # At test time we assume candidates already include target.
            assert y in xs

        xs = [y] + [x for x in xs if x != y]  # Target index always 0
        return xs[:self.max_num_candidates]

    def cull_samples(self, samples):
        self.num_samples_original = len(samples[0])
        if self.is_training:
            return samples
        else:
            mentions = [mc for mc in samples[0] if mc['label_document_id'] in
                        samples[1][mc['mention_id']]['candidates'][
                        :self.max_num_candidates]]
            return mentions, samples[1]

    def get_mention_window(self, mention):
        # Get "enough" context from space-tokenized text.
        tokens = self.documents[mention['context_document_id']]['text'].split()
        prefix = tokens[max(0, mention['start_index'] - self.max_len_mention):
                        mention['start_index']]
        suffix = tokens[mention['end_index'] + 1:
                        mention['end_index'] + self.max_len_mention + 1]
        extracted_mention = tokens[mention['start_index']:
                                   mention['end_index'] + 1]
        prefix = ' '.join(prefix)
        suffix = ' '.join(suffix)
        extracted_mention = ' '.join(extracted_mention)
        assert extracted_mention == mention['text']

        # Get window under new tokenization.
        mention_tokens = self.tokenizer.tokenize(extracted_mention)
        if self.indicate_mention_boundaries:
            mention_tokens = [self.MENTION_START] + mention_tokens + \
                             [self.MENTION_END]
        return util.get_window(self.tokenizer.tokenize(prefix),
                               mention_tokens,
                               self.tokenizer.tokenize(suffix),
                               self.max_len_mention)

    def get_candidate_prefix(self, candidate_document_id):
        # Get "enough" context from space-tokenized text.
        tokens = self.documents[candidate_document_id]['text'].split()
        prefix = tokens[:self.max_len_candidate]
        prefix = ' '.join(prefix)

        # Get prefix under new tokenization.
        return self.tokenizer.tokenize(prefix)[:self.max_len_candidate]


class UnifiedDataset(BasicDataset):
    def __init__(self, documents, samples, tokenizer, max_len,
                 max_num_candidates,
                 is_training, indicate_mention_boundaries=False):
        super(UnifiedDataset, self).__init__(documents, samples, tokenizer,
                                             max_len, max_num_candidates,
                                             is_training,
                                             indicate_mention_boundaries)
        self.max_len = max_len // 2
        self.max_len_mention = self.max_len - 2  # cls and sep
        self.max_len_candidate = self.max_len - 2  # cls and sep

    def __getitem__(self, index):
        mention = self.samples[0][index]
        candidates = self.samples[1][mention['mention_id']]
        mention_window = self.get_mention_window(mention)[0]
        mention_encoded_dict = self.tokenizer.encode_plus(mention_window,
                                                          add_special_tokens=True,
                                                          max_length=self.max_len,
                                                          pad_to_max_length=True,
                                                          truncation=True)
        mention_token_ids = torch.tensor(mention_encoded_dict['input_ids'])
        mention_masks = torch.tensor(mention_encoded_dict['attention_mask'])
        candidates_token_ids = torch.zeros((self.max_num_candidates,
                                            self.max_len))
        candidates_masks = torch.zeros((self.max_num_candidates,
                                        self.max_len))

        candidate_document_ids = self.prepare_candidates(mention, candidates)

        for i, candidate_document_id in enumerate(candidate_document_ids):
            candidate_window = self.get_candidate_prefix(candidate_document_id)
            candidate_dict = self.tokenizer.encode_plus(candidate_window,
                                                        add_special_tokens=True,
                                                        max_length=self.max_len,
                                                        pad_to_max_length=True,
                                                        truncation=True)
            candidates_token_ids[i] = torch.tensor(candidate_dict['input_ids'])
            candidates_masks[i] = torch.tensor(candidate_dict['attention_mask'])
        return mention_token_ids, mention_masks, candidates_token_ids, \
               candidates_masks


class FullDataset(BasicDataset):
    def __init__(self, documents, samples, tokenizer, max_len,
                 max_num_candidates,
                 is_training, indicate_mention_boundaries=False):
        super(FullDataset, self).__init__(documents, samples, tokenizer,
                                          max_len, max_num_candidates,
                                          is_training,
                                          indicate_mention_boundaries)
        self.max_len_mention = (max_len - 3) // 2  # [CLS], [SEP], [SEP]
        self.max_len_candidate = (max_len - 3) - self.max_len_mention

    def __getitem__(self, index):
        mention = self.samples[0][index]
        candidates = self.samples[1][mention['mention_id']]
        mention_window, mention_start, mention_end \
            = self.get_mention_window(mention)
        mention_start += 1  # [CLS]
        mention_end += 1  # [CLS]

        assert self.tokenizer.pad_token_id == 0
        encoded_pairs = torch.zeros((self.max_num_candidates, self.max_len))
        type_marks = torch.zeros((self.max_num_candidates, self.max_len))
        input_lens = torch.zeros(self.max_num_candidates)

        candidate_document_ids = self.prepare_candidates(mention, candidates)

        for i, candidate_document_id in enumerate(candidate_document_ids):
            candidate_prefix = self.get_candidate_prefix(candidate_document_id)
            encoded_dict = self.tokenizer.encode_plus(
                mention_window, candidate_prefix, return_attention_mask=False,
                pad_to_max_length=True, max_length=self.max_len,
                truncation=True)
            encoded_pairs[i] = torch.tensor(encoded_dict['input_ids'])
            type_marks[i] = torch.tensor(encoded_dict['token_type_ids'])

            input_lens[i] = len(mention_window) + len(candidate_prefix) + 3

        return encoded_pairs, type_marks, input_lens


def get_loaders(data, tokenizer, max_len, max_num_candidates, batch_size,
                num_workers, indicate_mention_boundaries,
                max_num_cands_val,
                use_full_dataset=True, macro_eval=True):
    (documents, samples_train,
     samples_val, samples_test) = data
    print('get train loaders')
    if use_full_dataset:
        dataset_train = FullDataset(documents, samples_train, tokenizer,
                                    max_len, max_num_candidates, True,
                                    indicate_mention_boundaries)
    else:
        dataset_train = UnifiedDataset(documents, samples_train, tokenizer,
                                       max_len, max_num_candidates, True,
                                       indicate_mention_boundaries)

    loader_train = DataLoader(dataset_train, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    def help_loader(samples):
        num_samples = len(samples[0])
        if use_full_dataset:
            dataset = FullDataset(documents, samples, tokenizer,
                                  max_len, max_num_cands_val, False,
                                  indicate_mention_boundaries)
        else:
            dataset = UnifiedDataset(documents, samples, tokenizer,
                                     max_len, max_num_cands_val, False,
                                     indicate_mention_boundaries)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
        return loader, num_samples

    if macro_eval:
        loader_val = []
        loader_test = []
        val_num_samples = []
        test_num_samples = []
        for sample in samples_val:
            loader, num_samples = help_loader(sample)
            loader_val.append(loader)
            val_num_samples.append(num_samples)
        for sample in samples_test:
            loader, num_samples = help_loader(sample)
            loader_test.append(loader)
            test_num_samples.append(num_samples)
    else:
        loader_val, val_num_samples = help_loader(samples_val)
        loader_test, test_num_samples = help_loader(samples_test)

    return (loader_train, loader_val, loader_test,
            val_num_samples, test_num_samples)


def load_zeshel_data(data_dir, cands_dir, macro_eval=True):
    """

    :param data_dir: train_data_dir if args.train else eval_data_dir
    :return: mentions, entities,doc
    """
    print('begin loading data')
    men_path = os.path.join(data_dir, 'mentions')

    def load_documents():
        documents = {}
        path = os.path.join(data_dir, 'documents')
        for fname in os.listdir(path):
            with open(os.path.join(path, fname)) as f:
                for line in f:
                    fields = json.loads(line)
                    fields['corpus'] = os.path.splitext(fname)[0]
                    documents[fields['document_id']] = fields
        return documents

    documents = load_documents()

    def get_domains(part):
        domains = set()
        with open(os.path.join(men_path, '%s.json' % part)) as f:
            for line in f:
                field = json.loads(line)
                domains.add(field['corpus'])
        return list(domains)

    def load_mention_candidates_pairs(part, domain=None, in_domain=False):
        mentions = []
        with open(os.path.join(men_path, '%s.json' % part)) as f:
            for i, line in enumerate(f):
                field = json.loads(line)
                if in_domain:
                    if field['corpus'] == domain:
                        mentions.append(field)
                else:
                    mentions.append(field)
                    assert mentions[i]['context_document_id'] in documents
                    assert mentions[i]['label_document_id'] in documents
        candidates = {}
        with open(os.path.join(cands_dir,
                               'candidates_%s.json' % part)) as f:
            for line in f:
                field = json.loads(line)
                candidates[field['mention_id']] = field
        return mentions, candidates

    print('start getting train pairs')
    samples_train = load_mention_candidates_pairs('train')
    if macro_eval:
        print('compute the domains')
        val_domains = get_domains('val')
        test_domains = get_domains('test')
        samples_val = []
        samples_test = []
        print('start getting val pairs')
        for val_domain in val_domains:
            pair = load_mention_candidates_pairs('val', val_domain, True)
            samples_val.append(pair)
        print('start getting test pairs')
        for test_domain in test_domains:
            pair = load_mention_candidates_pairs('test', test_domain, True)
            samples_test.append(pair)
    else:
        samples_val = load_mention_candidates_pairs('val')
        samples_test = load_mention_candidates_pairs('test')

    print('load all done')

    return documents, samples_train, samples_val, samples_test


class Logger(BasicLogger):

    def __init__(self, log_path, on=True):
        super(Logger, self).__init__(log_path, on)

    def log_zeshel_data_stats(self, data):
        (documents, samples_train, samples_val, samples_test) = data

        def count_domains(dict_documents):
            domain_count = Counter()
            lens = []
            for doc in dict_documents.values():
                domain_count[doc['corpus']] += 1
                lens.append(len(doc['text'].split()))
            domain_count = sorted(domain_count.items(), key=lambda x: x[1],
                                  reverse=True)
            return domain_count, lens

        self.log('------------Zeshel data------------')
        domain_count_all, lens = count_domains(documents)
        self.log('%d documents in %d domains' % (len(documents),
                                                 len(domain_count_all)))
        for domain, count in domain_count_all:
            self.log('   {:20s}: {:5d}'.format(domain, count))
        self.log('Lengths (by space): mean %g, max %d, min %d, stdev %g' %
                 (stat.mean(lens), max(lens), min(lens), stat.stdev(lens)))

        def log_samples(samples, name, running_mention_ids,
                        train_label_document_ids=None):
            self.log('\n*** %s ***' % name)
            domain_count = Counter()
            mention_lens = []
            candidate_nums = []
            label_document_ids = Counter()
            label_id_seen_count = 0
            mention_id_overlap = False
            for mention in samples[0]:
                if mention['mention_id'] in running_mention_ids:
                    mention_id_overlap = True
                running_mention_ids[mention['mention_id']] = True
                domain_count[mention['corpus']] += 1
                mention_lens.append(mention['end_index'] -
                                    mention['start_index'] + 1)
                candidate_nums.append(len(samples[1][mention['mention_id']][
                                              'candidates']))
                # for doc_id in candidates['candidates']:
                #   assert mention['corpus'] == documents[doc_id]['corpus']
                label_document_ids[mention['label_document_id']] += 1

                if train_label_document_ids:
                    if mention['label_document_id'] in train_label_document_ids:
                        label_id_seen_count += 1

            domain_count = sorted(domain_count.items(), key=lambda x: x[1],
                                  reverse=True)
            self.log('%d mentions in %d domains' % (len(samples),
                                                    len(domain_count)))
            self.log('Mention ID overlap: %d' % mention_id_overlap)

            if train_label_document_ids:
                self.log('%d / %d linked to entities seen in train' %
                         (label_id_seen_count, len(samples)))

            if not name in ['heldout_train_seen', 'heldout_train_unseen']:
                for domain, count in domain_count:
                    self.log('   {:20s}: {:5d}'.format(domain, count))

            self.log('Mention lengths (by space): mean %g, max %d, min %d, '
                     'stdev %g' % (stat.mean(mention_lens), max(mention_lens),
                                   min(mention_lens), stat.stdev(mention_lens)))

            self.log('Candidate numbers: mean %g, max %d, min %d, stdev %g' %
                     (stat.mean(candidate_nums), max(candidate_nums),
                      min(candidate_nums), stat.stdev(candidate_nums)))

            return label_document_ids

        running_mention_ids = {}
        train_label_document_ids = log_samples(samples_train, 'train',
                                               running_mention_ids)
        log_samples(samples_val, 'val', running_mention_ids,
                    train_label_document_ids)
        log_samples(samples_test, 'test', running_mention_ids,
                    train_label_document_ids)

    def log_perfs(self, perfs, best_args):
        valid_perfs = [perf for perf in perfs if not np.isinf(perf)]
        best_perf = max(valid_perfs)
        self.log('%d perfs: %s' % (len(perfs), str(perfs)))
        self.log('best perf: %g' % best_perf)
        self.log('best args: %s' % str(best_args))
        self.log('')
        self.log('perf max: %g' % best_perf)
        self.log('perf min: %g' % min(valid_perfs))
        self.log('perf avg: %g' % stat.mean(valid_perfs))
        self.log('perf std: %g' % (stat.stdev(valid_perfs)
                                   if len(valid_perfs) > 1 else 0.0))
        self.log('(excluded %d out of %d runs that produced -inf)' %
                 (len(perfs) - len(valid_perfs), len(perfs)))
