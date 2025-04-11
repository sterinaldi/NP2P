import numpy as np
import warnings

from pathlib import Path
from corner import corner
from inspect import signature

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import style
from matplotlib.pyplot import hist
from distutils.spawn import find_executable

from np2p._utils import implemented_processes

style.use('default')

# Settings
if find_executable('latex'):
    rcParams["text.usetex"] = True
rcParams["xtick.labelsize"] = 14
rcParams["ytick.labelsize"] = 14
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["legend.fontsize"] = 12
rcParams["legend.frameon"]  = False
rcParams["legend.loc"]      = "best"
rcParams["axes.labelsize"]  = 16
rcParams["axes.grid"]       = True
rcParams["grid.alpha"]      = 0.6
rcParams["grid.linestyle"]  = "dotted"
rcParams["lines.linewidth"] = 0.7
rcParams["hist.bins"]       = "sqrt"
rcParams["savefig.bbox"]    = "tight"

def nicer_hist(func):
    def decorated_func(*args, **kwargs):
        if not 'density' in kwargs.keys():
            kwargs['density'] = True
        if not 'histtype' in kwargs.keys():
            kwargs['histtype'] = 'step'
        return func(*args, **kwargs)
    return decorated_func

plt.hist = nicer_hist(plt.hist)

def plot_posterior(samples, labels = None, truths = None, save = True, model_name = None, out_folder = '.'):
    """
    Corner plot of the posterior samples
    
    Arguments:
        np.ndarray samples: the posterior samples
        str process:
        list-of-str labels: LaTeX-style labels
        np.ndarray truths: true values of parameters (if known)
        bool save: whether to save the plot
        str or Path out_folder: output folder
    
    Returns:
        plt.figure: figure
    """
    if len(samples.shape) > 1:
        n_pars = samples.shape[-1]
    else:
        n_pars = 1
        samples = np.atleast_2d(samples).T
    if labels is not None:
        if not (len(labels) == n_pars or len(labels) == n_pars-1):
            raise Exception('Please provide all the parameter names')
        dirichlet = False
        if len(labels) == n_pars-1:
            dirichlet = True
            labels  = list(labels) + ['\\beta_\\mathrm{DP}']
        labels = ['${}$'.format(lab) for lab in labels]
    if truths is not None:
        if not ((len(truths) == n_pars-1) or (len(truths) == n_pars)):
            raise Exception('Please provide all the true values for the parameters')
        if len(truths) == n_pars-1:
            dirichlet = True
            truths = list(truths) + [None]
    if dirichlet:
        # Prune plot from very large values of beta (orders of magnitude)
        exp_beta = np.median(samples[:,-1])
        unc_beta = np.diff(np.percentile(samples[:,-1], [5,95]))
        samples  = samples[np.abs(samples[:,-1]-exp_beta) < 3*unc_beta]
    # Corner plot
    if samples.shape[-1] > 1 and truths is not None:
        fig = corner(samples,
                     labels          = labels,
                     truths          = truths,
                     quantiles       = [0.16, 0.5, 0.84],
                     show_titles     = True,
                     hist_kwargs     = {'density': True, 'linewidth':0.7},
                     hist_bin_factor = int(np.sqrt(len(samples)))/20,
                     quiet           = True,
                     )
    else:
        # Bug in corner for 1d dist with true val
        fig = corner(samples,
             labels          = labels,
             truths          = None,
             quantiles       = [0.16, 0.5, 0.84],
             show_titles     = True,
             hist_kwargs     = {'density': True, 'linewidth':0.7},
             hist_bin_factor = int(np.sqrt(len(samples)))/20,
             quiet           = True,
             )
        fig.axes[0].axvline(truths[0])
    if save:
        if model_name is None:
            fig.savefig(Path(out_folder, 'joint_posterior.pdf'), bbox_inches = 'tight')
        else:
            fig.savefig(Path(out_folder, 'joint_posterior_{}.pdf'.format(model_name)), bbox_inches = 'tight')
        plt.close()
    else:
        return fig

def plot_comparison_1d(bins, draws, model, samples, label = 'x', unit = None, out_folder = '.', model_name = 'model', colors_data = ['steelblue','darkturquoise','mediumturquoise'], colors_inferred = ['orangered', 'darksalmon', 'lightsalmon'], label_data = '\\mathrm{Data}', label_model = '\\mathrm{Model}', save = True):
    """
    Plot a 1D distribution of both the reconstructed distribution and the inferred parametric distribution

    Arguments:
        iterable bins: bin values
        iterable draws: reconstructed distribution (either callable or evaluated on the bins)
        callable model: model pdf (callable)
        np.ndarray samples: parameter samples
        str label: LaTeX-style label for x axis
        str unit: LaTeX-style unit for x axis
        str or Path out_folder: output folder
        str model_name: model name
        list-of-str colors_data: list of colors for median, 68% and 90% credible regions of the reconstructed distribution
        list-of-str colors_inferred: list of colors for median, 68% and 90% credible regions of the inferred parametric distribution
        str label_name: LaTeX-style label for the reconstructed distribution
        str label_model: LaTeX-style label for the inferred parametric distribution
        bool save: whether to save the figure
    
    Returns:
        plt.figure: figure
    """
    fig, ax = plt.subplots()
    percentiles = [50, 5, 16, 84, 95]
    
    # Evaluate non-parametric draws
    if hasattr(draws[0], 'pdf') or callable(draws[0]):
        q = np.atleast_2d([d.pdf(bins) for d in draws])
    elif hasattr(draws[0], 'logpdf'):
        q = np.atleast_2d([np.exp(d.logpdf(bins)) for d in draws])
    elif isinstance(draws[0], (list, np.ndarray, float)):
        q = np.atleast_2d(draws)
        if not len(q[0]) == len(bins):
            raise Exception('The number of bins and the number of evaluated points in draws does not match')
    # Percentiles
    p = {}
    for perc in percentiles:
        p[perc] = np.percentile(q, perc, axis = 0)
    norm = np.sum(p[50]*(bins[1]-bins[0]))
    for perc in percentiles:
        p[perc] /= norm
    color_med, color_68, color_90 = colors_data
    # CR
    ax.fill_between(bins, p[95], p[5], color = color_90, alpha = 0.25)
    ax.fill_between(bins, p[84], p[16], color = color_68, alpha = 0.25)
    if label_data is not None:
        label_data = '$'+label_data+'$'
    ax.plot(bins, p[50], lw = 0.7, color = color_med, label = label_data)
    
    # Evaluate parametric draws
    if len(signature(model).parameters) == samples.shape[-1]:
        samples = samples[:,:-1]
    q_par = np.array([model(bins, *s) for s in samples])
    # Percentiles
    p_par = {}
    for perc in percentiles:
        p_par[perc] = np.percentile(q_par, perc, axis = 0)
    norm = np.sum(p_par[50]*(bins[1]-bins[0]))
    for perc in percentiles:
        p_par[perc] /= norm
    color_med, color_68, color_90 = colors_inferred
    # CR
    ax.fill_between(bins, p_par[95], p_par[5], color = color_90, alpha = 0.25)
    ax.fill_between(bins, p_par[84], p_par[16], color = color_68, alpha = 0.25)
    if label_model is not None:
        label_model = '$'+label_model+'$'
    ax.plot(bins, p_par[50], lw = 0.7, color = color_med, label = label_model)

    # Maquillage
    if unit is None or unit == '':
        ax.set_xlabel('${0}$'.format(label))
    else:
        ax.set_xlabel('${0}\ [{1}]$'.format(label, unit))
    ax.set_ylabel('$p({0})$'.format(label))
    ax.set_ylim(bottom = np.max([np.min(p[5]), np.min(p_par[5])])*0.9, top = np.max([np.max(p[95]), np.max(p_par[95])])*1.1)
    ax.legend(loc = 0)
    fig.align_labels()
    if save:
        fig.savefig(Path(out_folder, 'plot_{0}.pdf'.format(model_name)), bbox_inches = 'tight')
        ax.set_yscale('log')
        fig.savefig(Path(out_folder, 'plot_log_{0}.pdf'.format(model_name)), bbox_inches = 'tight')
        plt.close()
    else:
        return fig

def plot_model_selection(models, folder, names = None, out_folder = '.', name = None, save = True):
    """
    Plot log(beta) credible intervals for different models.
    
    Arguments:
        list-of-str models:     list of model labels
        Path or str folder:     folder containing the posterior samples
        list-of-str names:      list of model names (LaTeX is accepted)
        Path or str out_folder: folder where to save the plot
        str name:               name to give to the plot
        bool save:              whether to save the plot
    
    Return:
        plt.figure: the figure
    """
    step = 0.45
    marker    = 'o'
    color     = 'steelblue'
    facecolor = 'w'
    edgecolor = color
    fig, ax = plt.subplots(figsize = (6.4, len(models)*step))
    
    if names is None:
        names = models
    
    betas  = [np.log(np.genfromtxt(Path(folder, 'posterior_samples_{}.txt'.format(model)), names = True)['beta']) for model in models]
    # Ascending betas
    medians  = np.median(betas, axis = 1)
    names    = np.array(names)[np.argsort(medians)]
    betas    = np.array(betas)[np.argsort(medians)]
    
    # Axis limits
    bmin     = np.min([np.percentile(beta, 5) for beta in betas])
    bmax     = np.max([np.percentile(beta, 95) for beta in betas])
    midpoint = (bmin+bmax)/2
    range    = (bmax-bmin)
    # Set right limits
    ax.scatter([bmax], [-step], marker = '', c = 'none')
    ax.scatter([bmin], [-step], marker = '', c = 'none')
    
    for i, (beta, n) in enumerate(zip(betas, names)):
        ll, l, m, u, uu = np.percentile(beta, [5, 16, 50, 84, 95])
        if m < bmin + (0.1*range):
            textpos = bmin
            ha = 'left'
        elif m > bmin + (0.9*range):
            textpos = bmax
            ha = 'right'
        else:
            textpos = m
            ha = 'center'
        ax.errorbar([m], [i*step], xerr = [[m-ll],[uu-m]], marker = '', color = color, alpha = 0.5, lw = 1, capsize = 2)
        ax.errorbar([m], [i*step], xerr = [[m-l],[u-m]], marker = '', color = color, alpha = 0.75, lw = 1, capsize = 2)
        ax.scatter([m], [i*step], marker = marker, edgecolors = edgecolor, facecolors = facecolor, zorder = 3*len(betas)+10, lw = 1)
        ax.text(x = textpos, y = i*step+0.185, s = '$\\mathrm{'+'{}'.format(n)+'}$', ha = ha, va = 'center')
        
    ax.set_ylim(-0.6, (len(betas))*step)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.set_xlabel('$\\log\\beta_\\mathrm{DP}$')
    if save:
        if name is not None:
            fig.savefig(Path(out_folder, 'model_selection_{}.pdf'.format(name)), bbox_inches = 'tight')
        else:
            fig.savefig(Path(out_folder, 'model_selection.pdf'), bbox_inches = 'tight')
        plt.close(fig)
    else:
        return fig
