import argparse
import os
import random
import numpy as np
import pickle
import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, num_labels, dim=-1):
        super().__init__()
        self.num_labels = num_labels
        if dim < 0:
            self.emb = nn.Embedding(1, num_labels)
            self.ff = None
        else:
            self.emb = nn.Embedding(1, dim)
            #self.ff = nn.Sequential(nn.ReLU(), nn.Linear(dim, num_labels))
            self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                                    nn.Linear(dim, num_labels))
        self.loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, batch_size, labels=None):
        logits = self.emb.weight.expand(batch_size, -1)
        if self.ff is not None:
            logits = self.ff(logits)

        loss = None if labels is None else self.loss(logits, labels)
        return logits, loss

    def get_grad(self, pop, subsample=False, negative_distribution='rand',
                 num_negatives=-1, replacement=False, monte_carlo=100):
        logits = self.forward(1)[0].squeeze(0)

        if subsample:
            assert num_negatives > 0
            if not replacement:
                assert num_negatives < self.num_labels

            if negative_distribution == 'rand':
                q = torch.full((self.num_labels,), 1. / self.num_labels)
            elif negative_distribution == 'pop':
                q = pop
            elif negative_distribution == 'model':
                q = logits.detach().softmax(dim=0)
            else:
                raise NotImplementedError(
                    'Unknown value "{:s}"'.format(negative_distribution))

            estimate = self.sampled_loss_replacement if replacement else \
                       self.sampled_loss
            loss = estimate(pop, q, num_negatives, monte_carlo)
        else:
            loss = -(pop * logits.log_softmax(dim=0)).sum()

        loss.backward()
        grad = torch.cat([p.grad.detach().view(-1) for p in self.parameters()])
        self.zero_grad()
        return grad, loss

    def sampled_loss(self, pop, q, num_negatives, monte_carlo):
        weighted_nats = []
        for y in range(self.num_labels):
            q0 = q.clone()
            q0[y] = 0.  # Exclude gold.
            q0 = q0 / q0.sum()

            # (monte_carlo, num_negatives): Can't parallelize sampling because
            # each MC sample needs to sample without replacement.
            negative_samples = [torch.multinomial(q0, num_negatives,
                                                  replacement=False)
                                for _ in range(monte_carlo)]
            negative_samples = torch.stack(negative_samples, dim=0)

            # (monte_carlo, num_negatives + 1)
            candidates = torch.cat([torch.full((monte_carlo, 1), float(y)),
                                    negative_samples], dim=1).long()

            logits = self.forward(1)[0].squeeze(0)[candidates]
            num_nats = -logits.log_softmax(dim=1)[:, 0].mean()
            weighted_nats.append(pop[y] * num_nats)

        loss = torch.stack(weighted_nats).sum()
        return loss

    def sampled_loss_replacement(self, pop, q, num_negatives, monte_carlo):
        # (num_labels * monte_carlo, num_negatives)
        negative_samples = torch.multinomial(q,
                                             num_negatives * monte_carlo *
                                             self.num_labels,
                                             replacement=True).\
                                             view(-1, num_negatives)
        # (num_labels * monte_carlo, 1)
        positive_samples = torch.arange(self.num_labels).unsqueeze(1).\
                           repeat(1, monte_carlo).view(-1, 1)

        # (num_labels * monte_carlo, num_negatives + 1)
        candidates = torch.cat([positive_samples, negative_samples],
                                dim=1).long()
        logits = self.forward(1)[0].squeeze(0)[candidates]

        # Mask golds in negative samples.
        for i in range(0, logits.size(0), monte_carlo):
            partial = candidates[i:i + monte_carlo, 1:]
            logits[i:i + monte_carlo, 1:][partial == candidates[i, 0]] \
                = float('-inf')

        # (num_labels * monte_carlo)
        num_nats_all = -logits.log_softmax(dim=1)[:, 0]

        # (num_labels, num_lables * monte_carlo)
        diag = torch.diag(torch.full((self.num_labels,), 1 / monte_carlo)).\
               view(-1, 1).repeat(1, monte_carlo).view(self.num_labels, -1)

        # (num_labels)
        num_nats = torch.matmul(diag, num_nats_all).view(-1)

        loss = (pop * num_nats).sum()
        return loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_xent(p, q):
    inds_nonzero = (q > 0.).nonzero(as_tuple=True)[0]
    xent = - (p[inds_nonzero] * q[inds_nonzero].log()).sum().item()
    assert xent > -1E-42
    return max(0., xent)


def train(pop, model, steps, batch_size, lr, num_negatives, replacement,
          monte_carlo):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    records = []
    for step in range(steps):
        labels = torch.multinomial(pop, batch_size, replacement=True)
        logits, loss = model(batch_size, labels)
        loss_avg = loss / batch_size
        loss_avg.backward()
        optimizer.step()
        optimizer.zero_grad()

        grad_xent, xent = model.get_grad(pop)
        grad_rand, rand = model.get_grad(pop, True, 'rand', num_negatives,
                                         replacement, monte_carlo)
        grad_hard, hard = model.get_grad(pop, True, 'model', num_negatives,
                                         replacement, monte_carlo)

        diff_rand = torch.norm(grad_xent - grad_rand)
        diff_hard = torch.norm(grad_xent - grad_hard)
        record = {'loss_avg': loss_avg.item(), 'xent': xent.item(),
                  'rand': rand.item(), 'hard': hard.item(),
                  'diff_rand': diff_rand.item(),
                  'diff_hard': diff_hard.item()}
        records.append(record)
        print('Step {:3d} | '.format(step + 1),
              ' | '.join(['{:s} {:2.2f}'.format(k, v)
                          for k, v in record.items()]))
    return records


def main(args):
    print(args)
    set_seed(args.seed)
    pop = torch.rand(args.num_labels) / args.temp
    if args.support > 0:
        pop[args.support:].fill_(float('-inf'))
    pop = pop.softmax(dim=0)
    entropy_pop = get_xent(pop, pop)
    print('pop (temp={:g}): {:d} labels, support {:d}, entropy {:.2f}'.format(
        args.temp, args.num_labels, (pop > 0).sum(), entropy_pop))
    print('Example iid samples: {:s}'.format(
        str(torch.multinomial(pop, 30, replacement=True).tolist())))

    model = Model(args.num_labels, dim=args.dim)
    print('Model: {:d} params'.format(count_parameters(model)))

    records = train(pop, model, args.steps, args.batch_size, args.lr,
                    args.num_negatives, args.replacement, args.monte_carlo)

    pickle.dump((entropy_pop, records), open(args.pickle, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_labels', type=int, default=1000,
                        help='Num label types [%(default)d]')
    parser.add_argument('--support', type=int, default=-1,
                        help='Support size [%(default)d]')
    parser.add_argument('--temp', type=float, default=0.001,
                        help='Temperature for softmax [%(default)g]')
    parser.add_argument('--num_negatives', type=int, default=4,
                        help='Num negative samples [%(default)d]')
    parser.add_argument('--replacement', action='store_true',
                        help='Sample with replacement?')
    parser.add_argument('--monte_carlo', type=int, default=10,
                        help='Num Monte Carlo samples [%(default)d]')
    parser.add_argument('--dim', type=int, default=300,
                        help='Feedforward dimension [%(default)d]')
    parser.add_argument('--steps', type=int, default=30,
                        help='Num training steps [%(default)d]')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size [%(default)d]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate [%(default)g]')
    parser.add_argument('--pickle', type=str, default='cukes.p',
                        help='output pickle file path [%(default)s]')
    parser.add_argument('--seed', type=int, default=42,
                        help='[%(default)d]')
    args = parser.parse_args()
    main(args)
