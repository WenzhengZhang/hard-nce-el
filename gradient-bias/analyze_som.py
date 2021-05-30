import argparse
import json
import os
import random
import pickle

from collections import Counter


def get_context(mention_id, mentions, entities, context_size):
    mention = mentions[mention_id]
    context_document_id = mention['context_document_id']
    s = mention['start_index']
    t = mention['end_index']
    toks = entities[context_document_id]['text'].split()
    assert ' '.join(toks[s:t + 1]) == mention['text']
    half = context_size // 2
    left_toks = toks[max(0, s - half):s]
    right_toks = toks[t + 1:min(t + 1 + half, len(toks))]
    shortage = context_size - len(left_toks) - len(right_toks)

    if shortage > 0:
        # Get more context from the left.
        left_toks = toks[max(0, s - half - shortage):s]
        shortage = context_size - len(left_toks) - len(right_toks)
        if shortage > 0:
            # Get more context from the right.
            right_toks = toks[t + 1:min(t + 1 + half + shortage,
                                        len(toks))]
    shortage = context_size - len(left_toks) - len(right_toks)
    assert shortage == 0 or \
        len(toks) - len(mention['text'].split()) < context_size

    left = ' '.join(left_toks)
    right = ' '.join(right_toks)
    return '{:s} [[{:s}]] {:s}'.format(left, mention['text'], right)


def get_description(mention_id, mentions, entities, candidate,
                    context_size):
    mention = mentions[mention_id]
    toks = entities[candidate]['text'].split()
    return '{:s}'.format(' '.join(toks[:min(context_size, len(toks))]))


def verbalize(args, candidate_path):
    # Read mentions and their domains.
    mentions = {}
    domains = {}
    with open(os.path.join(args.zeshel_path, 'mentions/val.json')) as f:
        for line in f:
            attributes = json.loads(line)
            domains[attributes['corpus']] = True
            mentions[attributes['mention_id']] = attributes

    # Collect relevant entities.
    entities = {}
    for domain in domains:
        with open(os.path.join(args.zeshel_path,
                               'documents/{:s}.json'.format(domain))) as f:
            for line in f:
                attributes = json.loads(line)
                entities[attributes['document_id']] = attributes
                entities[attributes['document_id']]['domain'] = domain

    # Augment mentions with candidates.
    num_candidates = 0
    with open(candidate_path) as f:
        for line in f:
            attributes = json.loads(line)
            mention_id = attributes['mention_id']
            candidates = attributes['candidates']
            if num_candidates == 0:
                num_candidates = len(candidates)
            else:
                assert len(candidates) == num_candidates
            assert mention_id in mentions
            assert not 'candidates' in mentions[mention_id]
            mentions[mention_id]['candidates'] = candidates
            mentions[mention_id]['contains_gold'] \
                = mentions[mention_id]['label_document_id'] in candidates

    # Augment mentions with mention text and candidate text.
    final = {}
    for mention_id in mentions:
        context = get_context(mention_id, mentions, entities, args.context_size)
        final_candidates = []
        gold_rank = -1
        for i, candidate in enumerate(mentions[mention_id]['candidates']):
            description = get_description(mention_id, mentions, entities,
                                          candidate, args.context_size)
            is_gold = candidate == mentions[mention_id]['label_document_id']
            if is_gold:
                gold_rank = i
            final_candidates.append((description,
                                     entities[candidate]['domain'],
                                     is_gold))

        final[mention_id] = {
            'mention_string': context,
            'domain': mentions[mention_id]['corpus'],
            'category': mentions[mention_id]['category'],
            'gold_entity_title': entities[
                mentions[mention_id]['label_document_id']]['title'],
            'contains_gold': mentions[mention_id]['contains_gold'],
            'gold_rank': gold_rank,
            'candidates': final_candidates
        }
    return final


def main(args):
    if os.path.isfile('dual.p'):
        dual = pickle.load(open('dual.p', 'rb'))
    else:
        dual = verbalize(args, os.path.join(args.json_path,
                                            'candidates_dual_mixed.json'))
        pickle.dump(dual, open('dual.p', 'wb'))

    if os.path.isfile('som.p'):
        som = pickle.load(open('som.p', 'rb'))
    else:
        som = verbalize(args, os.path.join(args.json_path,
                                            'candidates_summax_mixed.json'))
        pickle.dump(som, open('som.p', 'wb'))

    assert len(dual) == len(som)
    recall_dual = sum([1 for mention_id in dual if
                       dual[mention_id]['contains_gold']]) / len(dual) * 100.
    recall_som = sum([1 for mention_id in som if
                       som[mention_id]['contains_gold']]) / len(som) * 100.

    improved = []
    pp = 0
    pn = 0
    np = 0
    nn = 0
    for mention_id in som:
        num_candidates = len(dual[mention_id]['candidates'])
        if som[mention_id]['contains_gold'] and \
           not dual[mention_id]['contains_gold']:
            improved.append(mention_id)
        if dual[mention_id]['contains_gold'] and som[mention_id]['contains_gold']:
            pp += 1
        if dual[mention_id]['contains_gold'] and not som[mention_id]['contains_gold']:
            pn += 1
        if (not dual[mention_id]['contains_gold']) and som[mention_id]['contains_gold']:
            np += 1
        if (not dual[mention_id]['contains_gold']) and not som[mention_id]['contains_gold']:
            nn += 1
    assert pp + pn + np + nn == len(dual)


    categories = Counter()
    for mention_id in dual:
        categories[dual[mention_id]['category']] += 1
    sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)


    wrong_categories_dual = Counter()
    wrong_categories_som = Counter()
    for mention_id in dual:
        if not dual[mention_id]['contains_gold']:
            wrong_categories_dual[dual[mention_id]['category']] += 1
        if not som[mention_id]['contains_gold']:
            wrong_categories_som[som[mention_id]['category']] += 1

    exf = open('dual_n_som_p.example', 'w')
    improved_categories = Counter()
    for mention_id in improved:
        cat = dual[mention_id]['category']
        improved_categories[cat] += 1

        if som[mention_id]['gold_rank'] > args.num_cands - 1:  # Too long
            continue
        exf.write('-'*200 +'\n')
        exf.write('Mention ({:s}, {:s}, {:s}): {:s}\n'.format(
            dual[mention_id]['domain'], cat,
            dual[mention_id]['gold_entity_title'],
            dual[mention_id]['mention_string']))
        exf.write('\nDual (mixed) candidates\n')
        for i, (description, domain, is_gold) in enumerate(
                dual[mention_id]['candidates'][:args.num_cands]):
            if is_gold:
                exf.write('**{:2d} ({:10s}). {:s}\n'.format(i + 1, domain,
                                                            description))
            else:
                exf.write('  {:2d} ({:10s}). {:s}\n'.format(i + 1, domain,
                                                            description))

        exf.write('\nSOM (mixed) candidates\n')
        for i, (description, domain, is_gold) in enumerate(
                som[mention_id]['candidates'][:args.num_cands]):
            if is_gold:
                exf.write('**{:2d} ({:10s}). {:s}\n'.format(i + 1, domain,
                                                            description))
            else:
                exf.write('  {:2d} ({:10s}). {:s}\n'.format(i + 1, domain,
                                                            description))

    sorted_wrong_cats_dual = sorted(wrong_categories_dual.items(),
                                    key=lambda x: x[1], reverse=True)
    sorted_wrong_cats_som = sorted(wrong_categories_som.items(),
                                    key=lambda x: x[1], reverse=True)

    sorted_improved_cats = sorted(improved_categories.items(),
                                  key=lambda x: x[1], reverse=True)

    print('\nNum val mentions: {:d}'.format(len(som)))
    print('Num candidates: {:d}'.format(num_candidates))
    print('Recall (dual, mixed): {:5.2f}'.format(recall_dual))
    print('Recall (som, mixed): {:5.2f}'.format(recall_som))
    print('\npp: {:10d}'.format(pp))
    print('pn: {:10d}'.format(pn))
    print('np: {:10d}'.format(np))
    print('nn: {:10d}'.format(nn))



    print('\nVal mention categories')
    for cat, num in sorted_cats:
        print('{:30s}: {:10d}'.format(cat, num))

    print('\nWrong mention categories - dual')
    for cat, num in sorted_wrong_cats_dual:
        print('{:30s}: {:10d}'.format(cat, num))

    print('\nWrong mention categories - som')
    for cat, num in sorted_wrong_cats_som:
        print('{:30s}: {:10d}'.format(cat, num))

    print('\nImproved categories')
    for cat, num in sorted_improved_cats:
        print('{:30s}: {:10d}'.format(cat, num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='/Users/stratos/work'
                        '/nce-gradient-bias/json/',
                        help='directory containing json files with '
                        'mention_id and candidates')
    parser.add_argument('--zeshel_path', type=str, default='/Users/stratos/work'
                        '/pytorch/knowledge/data/zeshel/zeshel_full/',
                        help='zeshel data directory')
    parser.add_argument('--context_size', type=int, default=100,
                        help='[%(default)d]')
    parser.add_argument('--num_cands', type=int, default=10,
                        help='[%(default)d]')
    parser.add_argument('--seed', type=int, default=42,
                        help='[%(default)d]')
    args = parser.parse_args()
    main(args)
