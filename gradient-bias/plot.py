import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle


def main(args):

    entropy_pop, records = pickle.load(open(args.pickle, 'rb'))

    losses_xent = [record['xent'] for record in records]
    losses_hard = [record['hard'] for record in records]
    losses_rand = [record['rand'] for record in records]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(5,5))
    x = range(1, len(records) + 1)
    plt.plot(x, losses_hard, "-r", label='Hard NCE',
             linewidth=5, alpha=1.0)
    plt.plot(x, losses_rand, "-b", label='Random NCE',
             linewidth=5, alpha=1.0)
    plt.plot(x, losses_xent, "-g", label='Cross Entropy',
             linewidth=5, alpha=1.0, linestyle='dotted')
    plt.plot(x, [entropy_pop for _ in range(len(records))], '-k',
                 label='Entropy', linewidth=5, linestyle='dotted')

    #plt.legend(loc="center right", bbox_to_anchor=(0.5, 1))
    plt.legend(prop={'size': 15})
    plt.ylim(0, max([max(losses_xent) - 1,
                     max(losses_hard),
                     max(losses_rand)]) + 0.5)
    plt.xlim(1, len(records))
    plt.xticks([5, 10, 15, 20, 25, 30])
    plt.savefig(args.loss, bbox_inches='tight')

    plt.clf()
    biases_hard = [record['diff_hard'] for record in records]
    biases_rand = [record['diff_rand'] for record in records]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(5,5))
    x = range(1, len(records) + 1)
    plt.plot(x, biases_hard, "-r", label='Hard NCE',
             linewidth=5, alpha=1.0)
    plt.plot(x, biases_rand, "-b", label='Random NCE',
             linewidth=5, alpha=1.0)

    plt.legend(prop={'size': 15})
    plt.ylim(0, max([max(biases_hard),
                     max(biases_rand)]) + 0.5)
    plt.xlim(1, len(records))
    plt.xticks([5, 10, 15, 20, 25, 30])
    plt.savefig(args.bias, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('pickle', type=str,
                        help='path to saved pickled file from main.py')
    parser.add_argument('--loss', type=str, default='loss.pdf',
                        help='loss figure file path [%(default)s]')
    parser.add_argument('--bias', type=str, default='bias.pdf',
                        help='bias figure file path [%(default)s]')
    args = parser.parse_args()
    main(args)
