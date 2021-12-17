import numpy as np
import matplotlib.pyplot as plt
from parest.ParEst import DirichletProcess
import json
import pickle
import cpnest
import corner
import os
from scipy.interpolate import interp1d
from scipy.special import logsumexp
from parest.loglikelihood import truncated, broken_pl, pl_peak, multi_peak, broken_pl_peak, tapered_plpeak

# OPTIONS
#------------------------
# Select dataset
dataset = 'O3a' # 'O3', 'O3a', 'hdp'
# Select a model:
model = 'tapered_plpeak' # 'truncated', 'broken_pl', 'pl_peak', 'multi_peak', 'broken_pl_peak', 'tapered_plpeak'
# Postprocessing
postprocessing = False
# Data folder
folder = '/Users/stefanorinaldi/Documents/parametric/population/' # CHANGEME
# Mass boundaries
x_min = 4
x_max = 90
# Concentration parameter
max_alpha = 10000
#------------------------

folder = folder + dataset + '/'
out_folder = folder + model + '/'

# Files
if dataset == 'O3':
    draws_file = folder + 'mf_samples.json'
    rec_file   = folder + 'log_m_astro.txt'
if dataset == 'O3a':
    draws_file = folder + 'combined_posteriors.pkl'
    rec_file   = folder + 'log_combined_prob_mf.txt'
if dataset == 'hdp':
    draws_file = folder + 'astro_samples.json'
    rec_file   = folder + 'rec_prob.txt'

# Comparison with DPGMM outcome
rec = np.genfromtxt(rec_file, names = True)

# Load data
if dataset == 'O3':
    openfile = open(draws_file, 'r')
    json_dict = json.load(openfile)
    x = np.ascontiguousarray([xi for xi in json_dict[0] if x_min < xi < x_max])
    logdx = np.log(x[1] - x[0])
    draws = []
    for i, p in enumerate(json_dict[1:]):
        draws.append(p)
    samples = np.array([d[np.where([xi for xi in x if x_min < xi < x_max])] + logdx for d in np.array(draws).T])
    samples = np.array([s - logsumexp(s) for s in samples])
    openfile.close()

elif dataset == 'hdp':
    openfile = open(draws_file, 'r')
    json_dict = json.load(openfile)
    x = np.array([float(xi) for xi in json_dict.keys() if x_min < float(xi) < x_max])
    logdx = np.log(x[1] - x[0])
    draws = []
    for i, p in enumerate(json_dict.values()):
        draws.append(p)
    draws = np.array(draws).T
    samples = np.array([d[np.where([xi for xi in x if x_min < xi < x_max])] + logdx for d in np.array(draws).T])
    samples = np.array([s - logsumexp(s) for s in samples])
    openfile.close()
    
elif dataset == 'O3a':
    openfile = open(draws_file, 'rb')
    samps = np.array(pickle.load(openfile)).T
    openfile.close()
    x = np.array([xi for xi in rec['m'] if x_min < xi < x_max])
    logdx = np.log(x[1]-x[0])
    samples = []
    for d in samps:
        samples.append(d[np.where([x_min < xi < x_max for xi in x])] + logdx)
    samples = np.array([s - logsumexp(s) for s in samples])

else:
    print('Unsupported dataset')
    exit()

N_bins = len(x)
print('{0} bins between {1:.1f} and {2:.1f}'.format(N_bins, x_min, x_max))


if model == 'truncated':
    names = ['b', 'mmin', 'mmax', 'd']
    bounds = [[1,12], [2,10], [30,100],[0,10]]
    labels = ['\\beta', 'm_{min}', 'm_{max}','\\delta_m']
    label_selected_model = 7 # Truncated
    true_vals = None
    model = truncated
    model_label = 'Truncated'
    
if model == 'broken_pl':
    names = ['a1','a2','mmin','mmax','b','d']
    bounds = [[1,12], [1,12], [2,10], [30,100], [0,1], [0,10]]
    labels = ['\\beta_1', '\\beta_2', 'm_{min}', 'm_{max}', 'b', '\\delta_m']
    label_selected_model = 8
    true_vals = [1.58,5.59,3.96,87.14,0.43,4.83]
    model = broken_pl
    model_label = 'Broken\ PowerLaw'

if model == 'pl_peak':
    names = ['l','b','mmin','d_min','mmax','d_max', 'mu','s']
    bounds = [[0,1], [1,12], [2,10], [0,10], [30, 100], [0,10], [20, 50], [1,10]]
    labels = ['\\lambda_{peak}', '\\beta', 'm_{min}', '\\delta_{min}', 'm_{max}', '\\delta_{max}', '\\mu_m', '\\sigma_m']
    label_selected_model = 14
    true_vals = [0.1,2.63,4.59,4.82,86.22,10,33.07,5.69]
    model = pl_peak_smoothed
    model_label = 'PowerLaw\ +\ Peak'

if model == 'multi_peak':
    names = ['l', 'lg', 'b', 'mmin', 'd', 'mmax', 'mu1', 's1', 'mu2', 's2']
    bounds = [[0,1],[0,1],[1,12],[2,10],[0,10],[30,100],[20,50],[1,10],[50,100],[1,10]]
    labels = ['\\lambda','\\lambda_1','\\beta','m_{min}','\\delta_m','m_{max}','\\mu_{m,1}','\\sigma_{m,1}','\\mu_{m,2}','\\sigma_{m,2}']
    label_selected_model = 10
    true_vals = None
    model = multi_peak
    model_label = 'Multi\ Peak'

if model == 'broken_pl_peak':
    names = ['a1','a2','mmin','mmax','b','d', 'mu', 's', 'l']
    bounds = [[1,12], [1,12], [2,10], [30,100], [0,1], [0,10],[20,50],[1,10], [0,1]]
    labels = ['\\beta_1', '\\beta_2', 'm_{min}', 'm_{max}', 'b', '\\delta_m', '\\mu_m', '\\sigma_m', '\\lambda_{peak}']
    label_selected_model = 11
    true_vals = None
    model = broken_pl_peak
    model_label = 'Broken\ PowerLaw\ +\ Peak'

if model == 'tapered_plpeak':
    names = ['b', 'mmin', 'mmax', 'lmin', 'lmax', 'mu', 's', 'w']
    bounds = [[0,5], [5,40], [50,120],[1,30], [1,30], [40,70], [1,10], [0,1]]
    labels = ['\\beta', 'm_{min}', 'm_{max}','\\lambda_{min}', '\\lambda_{max}', '\\mu', '\\sigma', 'w']
    label_selected_model = 12 # Tapered PowerLaw
    true_vals = None
    model = tapered_plpeak
    model_label = 'Tapered\ PowerLaw\ +\ Peak'

PE = DirichletProcess(
    label_selected_model,
    names,
    bounds,
    samples,
    x = x,
    max_a = max_alpha,
    out_folder = out_folder
    )
    
if not postprocessing:
    work = cpnest.CPNest(PE,
                        verbose = 2,
                        nlive = 1000,
                        maxmcmc = 5000,
                        nensemble = 4,
                        output  = out_folder
                        )
    work.run()
    post = work.posterior_samples.ravel()
else:
    post = np.genfromtxt(os.path.join(out_folder,'posterior.dat'),names=True)

# Postprocessing

labels = labels + ['\\alpha','\\alpha/N']
par_names = names
names = names + ['a']
if true_vals is not None:
    true_vals = true_vals + [1, 1]
samps = np.column_stack([post[lab] for lab in names] + [post['a']/N_bins])

# Plots

fig = corner.corner(samps,
       labels= [r'${0}$'.format(lab) for lab in labels],
       quantiles=[0.05, 0.16, 0.5, 0.84, 0.95],
       show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
       use_math_text=True, truths = true_vals,
       filename=os.path.join(out_folder,'joint_posterior.pdf'))
fig.savefig(os.path.join(out_folder,'joint_posterior.pdf'), bbox_inches='tight')

# Comparison: (H)DPGMM vs model
fig, ax = plt.subplots(figsize = (10,6))
ax.fill_between(rec['m'], np.exp(rec['95']), np.exp(rec['5']), color = 'mediumturquoise', alpha = 0.5)
ax.plot(rec['m'], np.exp(rec['50']), color = 'steelblue', label = '$Non-parametric$')
pdf = []
dx = x[1]-x[0]
for i,si in enumerate(post):
    s = np.array([si[lab] for lab in par_names])
    f = model(x, *s)
    pdf.append(f)
low,med,high = np.percentile(pdf,[5,50,95],axis=0)
ax.fill_between(x, high, low, color = 'lightsalmon', alpha = 0.5)
ax.plot(x, med, color = 'r', lw = 0.5, label = '${0}$'.format(model_label))
ax.set_xlim(x_min, x_max)
ax.set_xlabel('$M\ [M_\\odot]$')
ax.set_ylabel('$p(M)$')
ax.grid(True,dashes=(1,3))
ax.legend(loc=0,frameon=False,fontsize=10)
fig.savefig(os.path.join(out_folder,'compare_50.pdf'), bbox_inches='tight')
ax.set_yscale('log')
ax.set_ylim(1e-5, 0.25)
fig.savefig(os.path.join(out_folder,'compare_50_log.pdf'), bbox_inches='tight')
