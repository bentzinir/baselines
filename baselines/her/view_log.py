import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def parse_log(logfile):
    with open(logfile, "r") as fid:
        lines = fid.read().splitlines()
        success_rate = [float(line.split('|')[-2]) for line in lines if "train/success_rate" in line]
        epochs = [float(line.split('|')[-2]) for line in lines if "epoch" in line]
    return epochs, success_rate


def main(args):

    epochs1, val1 = parse_log(args.log1)
    epochs2, val2 = parse_log(args.log2)

    fig, ax = plt.subplots(1, figsize=(6, 5))
    if args.semilog:
        ax.semilogx(epochs1, val1, lw=2, label='log1', color='blue')
        ax.semilogx(epochs2, val2, lw=2, label='log2', color='red')
    else:
        ax.plot(epochs1, val1, lw=2, label='log1', color='blue')
        ax.plot(epochs2, val2, lw=2, label='log2', color='red')
    ax.legend(loc='upper left')
    ax.set_xlabel('Epochs', fontsize=18)
    ax.grid()
    plt.legend(loc=2, prop={'size': 18})
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.show(block=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log1', help='Logdir path', default=None)
    parser.add_argument('--log2', help='Logdir path', default=None)
    parser.add_argument('--semilog', help='Semilog', action='store_true')
    args = parser.parse_args()
    main(args)
