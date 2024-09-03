import numpy as np
import warnings

from pathlib import Path
from corner import corner

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import style
from matplotlib.pyplot import hist
from distutils.spawn import find_executable

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
histdefaults = list(hist.__defaults__)
histdefaults[2] = True   # density
histdefaults[6] = 'step' # histtype
hist.__defaults__ = tuple(histdefaults)

def plot_posterior(samples, labels = None, truths = None, save = True, model_name = None, out_folder = '.'):
    """
    Corner plot of the posterior samples
    
    Arguments:
        np.ndarray samples: the posterior samples
        list-of-str labels: LaTeX-style labels
        np.ndarray truths: true values of parameters (if known)
        bool save: whether to save the plot
        str or Path out_folder: output folder
    
    Returns:
        plt.figure: figure
    """
    n_pars = samples.shape[-1]
    if labels is not None:
        if not (len(labels) == n_pars or len(labels) == n_pars-1):
            print(labels)
            raise Exception('Please provide all the parameter names')
        if len(labels) == n_pars-1:
            labels = list(labels) + ['\\beta']
        labels = ['${}$'.format(lab) for lab in labels]
    if truths is not None:
        if not len(truths) == n_pars-1:
            raise Exception('Please provide all the true values for the parameters')
        truths = list(truths) + [None]
    # Prune plot from very large values of beta (orders of magnitude)
    exp_beta = np.median(samples[:,-1])
    unc_beta = np.diff(np.percentile(samples[:,-1], [5,95]))
    samples  = samples[np.abs(samples[:,-1]-exp_beta) < 3*unc_beta]
    # Corner plot
    fig = corner(samples,
                 labels          = labels,
                 truths          = truths,
                 quantiles       = [0.16, 0.5, 0.84],
                 show_titles     = True,
                 hist_kwargs     = {'density': True, 'linewidth':0.7},
                 hist_bin_factor = int(np.sqrt(len(samples)))/20,
                 quiet           = True,
                 )
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
    color_med, color_68, color_90 = colors_data
    # CR
    ax.fill_between(bins, p[95], p[5], color = color_90, alpha = 0.25)
    ax.fill_between(bins, p[84], p[16], color = color_68, alpha = 0.25)
    if label_data is not None:
        label_data = '$'+label_data+'$'
    ax.plot(bins, p[50], lw = 0.7, color = color_med, label = label_data)
    
    # Evaluate parametric draws
    q_par = np.array([model(bins, *s) for s in samples])
    # Percentiles
    p_par = {}
    for perc in percentiles:
        p_par[perc] = np.percentile(q_par, perc, axis = 0)
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
    ax.set_ylim(bottom = 1e-5, top = np.max([np.max(p[95]), np.max(p_par[95])])*1.1)
    ax.legend(loc = 0)
    fig.align_labels()
    if save:
        fig.savefig(Path(out_folder, '{0}.pdf'.format(model_name)), bbox_inches = 'tight')
        plt.close()
    else:
        return fig
