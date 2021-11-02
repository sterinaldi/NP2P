import numpy as np
import matplotlib.pyplot as plt
from ParEst import DirichletProcess
import pickle
import cpnest
import corner
import os
from scipy.interpolate import interp1d
from scipy.special import logsumexp
from loglikelihood import tapered_plpeak

# OPTIONS
#------------------------
# Postprocessing
postprocessing = True
# Data folder
folder = '/Users/stefanorinaldi/Documents/parametric/plpeak/' # CHANGEME
# Mass boundaries
x_min = 10
x_max = 129
# Concentration parameter
max_alpha = 10000
# Model parameters
names = ['b', 'mmin', 'mmax', 'lmin', 'lmax', 'mu', 's', 'w']
bounds = [[0,5], [5,20], [60,100],[2,20], [2,20],[40,70],[1,10],[0,1]]
labels = ['\\beta', 'm_{min}', 'm_{max}','\\lambda_{min}', '\\lambda_{max}', '\\mu_m', '\\sigma_m', 'w']
label_selected_model = 12 # Tapered PowerLaw + Peak
true_vals = [0.5,15, 90, 5, 10, 55, 6, 0.9]
model = tapered_plpeak
model_label = 'Tapered\ PowerLaw\ +\ Peak'
#------------------------

out_folder = folder + 'inference/'

# Files
draws_file = folder + 'samples.pkl'
rec_file   = folder + 'reconstruction.txt'

# Comparison with DPGMM outcome
rec = np.genfromtxt(rec_file, names = True)

# Load data
openfile = open(draws_file, 'rb')
samps = np.array(pickle.load(openfile)).T
openfile.close()
x = np.ascontiguousarray([xi for xi in samps[0].x if x_min < xi < x_max])
logdx = np.log(x[1]-x[0])
samples = []
for d in samps:
    samples.append(d.y[np.where([x_min < xi < x_max for xi in d.x])] + logdx)
samples = np.array([s - logsumexp(s) for s in samples])
# MEDIAN
#x = np.array(m[np.where([x_min < mi < x_max for mi in rec['m']])])
#logdx = np.log(x[1]-x[0])
#s = rec['50'][np.where([x_min < mi < x_max for mi in rec['m']])] - logdx
#samples = np.array([s - logsumexp(s)])

N_bins = len(x)
print('{0} bins between {1:.1f} and {2:.1f}'.format(N_bins, x_min, x_max))

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
dx = x[1]-x[0]
fig, ax = plt.subplots(figsize = (10,6))
#ax.fill_between(rec['m'], np.exp(rec['95']), np.exp(rec['5']), color = 'mediumturquoise', alpha = 0.5)
#ax.plot(rec['m'], np.exp(rec['50']), color = 'steelblue', label = '$Non-parametric$')
pr = []
for d in samples:
    p = np.exp(d)/dx
    pr.append(p)
    ax.plot(x, p, lw = 0.1, alpha=0.5)
ax.plot(x,np.array(pr).mean(axis = 0))
pdf = []
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
