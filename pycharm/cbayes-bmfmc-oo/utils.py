import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as gkde
import numpy as np
import seaborn as sns
import pandas as pd


# Plot arbitrary 1D data i.e. vector x over vector y
def plot_1d_data(x, y, marker='None', markersize=5, linestyle='-', linewidth=3, color='C0', label='', num=1,
                 xlabel='$x$', ylabel='$y$', title='', xlim=None, ylim=None):
    if len(str(num)) >= 3:
        plt.subplot(num)
    else:
        plt.figure(num)

    if xlim is None:
        xlim = [np.min(x), np.max(x)]
    if ylim is None:
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


# Plot a 1D kernel density estimation
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


# Plot a 2D kernel density estimation
def plot_2d_kde(samples, xlabel='$x_1$', ylabel='$x_2$', num=1, cmap='Blues'):

    if len(str(num)) >= 3:
        plt.subplot(num)
    else:
        plt.figure(num)

    df = pd.DataFrame(samples, columns=[xlabel, ylabel])
    g = sns.jointplot(x=xlabel, y=ylabel, data=df, kind='kde', color='C0', shade=True, shade_lowest=True, cmap=cmap)
    g.plot_joint(plt.scatter, c='k', alpha=0.3, s=1, linewidth=0.0, marker='o')
    g.ax_joint.collections[0].set_alpha(0)
    g.set_axis_labels(xlabel, ylabel)


# Plot a 1D histogram given samples
def plot_1d_hist(samples, num=1, title='', xlabel='$x$', ylabel='$p(x)$'):

    if len(str(num)) >= 3:
        plt.subplot(num)
    else:
        plt.figure(num)

    plt.hist(samples, 20, facecolor='C0', alpha=0.5, label=r'Histogram', density=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend(loc='upper left')
    plt.title(title)
    # plt.gcf().set_figheight(6)
    # plt.gcf().set_figwidth(6)


# Plot a 1D regression model with 99% confidence intervals
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


# Scatter two sets of data samples x and y
def plot_2d_scatter(samples_x, samples_y, marker='o', num=1, title='', xlim=None, ylim=None, xlabel="$x$", ylabel="$y$"):

    if len(str(num)) >= 3:
        plt.subplot(num)
    else:
        plt.figure(num)

    x = samples_x
    y = samples_y
    xy = np.vstack([x, y])
    z = gkde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x, y, c=z, s=50, edgecolor='', marker=marker)

    if xlim is None:
        xlim = [np.min(x), np.max(x)]
    if ylim is None:
        ylim = [np.min(y), np.max(y)]

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)


# Create a 2D contour plot using KDE given two sets of samples x and y
def plot_2d_contour(samples_x, samples_y, num=1, title='', xlim=None, ylim=None, xlabel="$x$", ylabel="$y$"):

    if len(str(num)) >= 3:
        plt.subplot(num)
    else:
        plt.figure(num)

    if xlim is None:
        xlim = [np.min(samples_x), np.max(samples_x)]

    if ylim is None:
        ylim = [np.min(samples_y), np.max(samples_y)]

    xy_kde = gkde(np.vstack([samples_x, samples_y]))
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 80), np.linspace(ylim[0], ylim[1], 80))
    zz = np.reshape(xy_kde(np.vstack([xx.ravel(), yy.ravel()])).T, xx.shape)
    ax = plt.gca()
    cfset = ax.contourf(xx, yy, zz, cmap='Blues', alpha=1.0)
    cset = ax.contour(xx, yy, zz, colors='k', alpha=1.0, linewidths=0.5)
    ax.clabel(cset, fontsize=4)
    ax.set_xlabel('$q$')
    ax.set_ylabel('$Q$')
    plt.colorbar(cfset)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
