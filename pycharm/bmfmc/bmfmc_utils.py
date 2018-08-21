import matplotlib.pyplot as plt
import numpy as np


def plot_1d_data(x, y, marker='None', markersize=5, linestyle='-', linewidth=3, color='C0', label='', num=1,
                 xlabel='$x$', ylabel='$y$', title='', xlim=[], ylim=[]):
    if len(str(num)) >= 3:
        plt.subplot(num)
    else:
        plt.figure(num)

    if len(xlim) == 0:
        xlim = [np.min(x), np.max(x)]
    if len(ylim) == 0:
        ylim = [np.min(y), np.max(y)]

    plt.plot(x, y, marker=marker, markersize=markersize, linestyle=linestyle, linewidth=linewidth,
             color=color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid()
    plt.legend(loc='upper left')
    plt.title(title)
    plt.gcf().set_figheight(6)
    plt.gcf().set_figwidth(6)


def plot_1d_kde(qkde, xmin=0.0, xmax=1.0, linestyle='-', linewidth=3, color='C0', num=1,
                xlabel='$x$', ylabel='$p(x)$', label='', title=''):

    if len(str(num)) >= 3:
        plt.subplot(num)
    else:
        plt.figure(num)

    qplot = np.linspace(xmin, xmax, num=200)
    plt.plot(qplot, qkde(qplot), linestyle=linestyle, linewidth=linewidth, color=color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend(loc='upper left')
    plt.title(title)
    plt.gcf().set_figheight(6)
    plt.gcf().set_figwidth(6)


def plot_1d_hist(samples, num=1, title=''):

    if len(str(num)) >= 3:
        plt.subplot(num)
    else:
        plt.figure(num)

    plt.hist(samples, 20, facecolor='C0', alpha=0.5, label=r'Histogram', density=1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$p \left( x \right)$')
    plt.grid()
    plt.legend(loc='upper left')
    plt.title(title)
    plt.gcf().set_figheight(6)
    plt.gcf().set_figwidth(6)


def plot_1d_conf(x_pred, y_pred, sigma, num=1, title=''):

    if len(str(num)) >= 3:
        plt.subplot(num)
    else:
        plt.figure(num)

    ysd = 3.0*sigma
    plt.plot(x_pred, y_pred, 'o', markersize=3, label=r'Prediction')
    plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
             np.concatenate([(y_pred - ysd),
                             (y_pred + ysd)[::-1]]),
             alpha=.4, ec='None', label='99% Confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.grid()
    plt.legend(loc='upper left')
    plt.title(title)
    plt.gcf().set_figheight(6)
    plt.gcf().set_figwidth(6)


def plot_2d_scatter(x, y, z, marker='o', num=1, title='', xlim=[], ylim=[], xlabel="$x$", ylabel="$y$"):

    if len(str(num)) >= 3:
        plt.subplot(num)
    else:
        plt.figure(num)

    if len(z) == 0:
        plt.scatter(x, y, s=50, edgecolor='', marker=marker)
    else:
        plt.scatter(x, y, c=z, s=50, edgecolor='', marker=marker)

    if len(xlim) == 0:
        xlim = [np.min(x), np.max(x)]
    if len(ylim) == 0:
        ylim = [np.min(y), np.max(y)]

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.gcf().set_figheight(6)
    plt.gcf().set_figwidth(6)


def print_bmfmc_stats(samples_hf_mc, samples_hf, samples_lf):
    print('')
    print('################### BMFMC statistics:')
    print('MC mean: ' + str(np.mean(samples_hf_mc)))
    print('MC std: ' + str(np.std(samples_hf_mc)))
    print('BMFMC mean: ' + str(np.mean(samples_hf)))
    print('BMFMC std: ' + str(np.std(samples_hf)))
    print('Low-fidelity mean: ' + str(np.mean(samples_lf)))
    print('Low-fidelity std: ' + str(np.std(samples_lf)))
    print('')