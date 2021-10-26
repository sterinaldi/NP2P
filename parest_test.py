import numpy as np
import matplotlib.pyplot as plt
from ParEst import DirichletProcess
import json
import cpnest
import corner
from scipy.special import erf, logsumexp
from scipy.stats import norm, dirichlet
from scipy.interpolate import interp1d
import os
from parametric_models import normal, power_law
from numba import jit

def log_add(x, y): return x+np.log(1.0+np.exp(y-x)) if x >= y else y+np.log(1.0+np.exp(x-y))
def log_norm(x, x0, s): return -((x-x0)**2)/(2*s*s) - np.log(np.sqrt(2*np.pi)) - np.log(s)

def PL(m, alpha, m_max, m_min, l_max, l_min):
    f = ((1-alpha)/(m_max**(1-alpha) - m_min**(1-alpha)))*m**(-alpha)*(1+erf((m-m_min)/(l_min)))*(1+erf((m_max-m)/l_max))/4.
    return f

@jit
def gauss(x, x0, s):
    return np.exp(-((x-x0)**2/(2*s**2)))/(np.sqrt(2*np.pi)*s)

def bimodal(x,x0,s0,x1,s1):
    return (np.exp(-((x-x0)**2/(2*s0**2)))/(np.sqrt(2*np.pi)*s0) + np.exp(-((x-x1)**2/(2*s1**2)))/(np.sqrt(2*np.pi)*s1))/2.

def untapered(m, b, mmin, l):
    m = np.asarray(m)
    pre_PL = (b-1)/(mmin**(1-b))
    N = 1 + pre_PL*mmin**(-b)*l*np.sqrt(2*np.pi)/2.
    return np.where(m > mmin, pre_PL*m**(-b) / N, np.exp(-(m-mmin)**2/(2*l**2)) *pre_PL*mmin**(-b) / N)

def PLpeak(m, b, mmin, l, x0, s, w):
    m = np.asarray(m)
    pre_PL = (b-1)/(mmin**(1-b))
    N = 1 + pre_PL*mmin**(-b)*l*np.sqrt(2*np.pi)/2.
    return w*np.where(m > mmin, pre_PL*m**(-b) / N, np.exp(-(m-mmin)**2/(2*l**2)) *pre_PL*mmin**(-b) / N) + (1-w)*gauss(m, x0, s)

def logPrior(*args):
    return 0

# OPTIONS
#------------------------

samp_file = 'path/to/file' # CHANGEME
out_folder  = '/Users/stefanorinaldi/Documents/parametric/gaussian/' # CHANGEME
rec_file = 'path/to/file' # CHANGEME

names = ['mean', 'sigma']
nargs = len(names)
bounds = [[20,60], [1,6]]
labels = ['\\mu', '\\sigma']
label_selected_model = 0 #Gaussian
true_vals = [40, 5]
x_min = 20
x_max = 60
N_bins = 30
max_alpha

# change also the model in comparison plot (see below)
#-----------------------

openfile = open(file, 'r')
json_dict = json.load(samp_file)
for d in np.array(json_dict.values()).T:
    samples.append(d)
openfile.close()
m = np.array([float(m) for m in json_dict.keys()])
samples = np.array([interp1d(m, p) for p in samples_set])

# Comparison with DPGMM outcome
rec = np.genfromtxt(rec_file, names = True)

PE = DirichletProcess(
    label_selected_model,
    names,
    bounds,
    samples,
    x_min = x_min
    x_max = x_max,
    prior_pars = logPrior,
    max_a = max_alpha,
    N_bins = N_bins,
    out_folder = out_folder
    )

if 1:
    work = cpnest.CPNest(PE,
                        verbose = 2,
                        nlive = 100,
                        maxmcmc = 100,
                        nensemble = 4,
                        output  = out_folder
                        )
    work.run()
    post = work.posterior_samples.ravel()
else:
    post = np.genfromtxt(os.path.join(out_folder,'posterior.dat'),names=True)

labels = labels + ['\\alpha']
par_names = names
names = names + ['a']
true_vals = true_vals + [1]
samps = np.column_stack([post[lab] for lab in names])
#print('log Evidence: {0}'.format(work.logZ))

# Posteriors

fig = corner.corner(samps,
       labels= [r'${0}$'.format(lab) for lab in labels],
       quantiles=[0.05, 0.16, 0.5, 0.84, 0.95],
       show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
       use_math_text=True, truths = true_vals + [1],
       filename=os.path.join(out_folder,'joint_posterior.pdf'))
fig.savefig(os.path.join(out_folder,'joint_posterior.pdf'), bbox_inches='tight')

# Comparison: (H)DPGMM vs model

x = np.linspace(PE.x_min,PE.x_max,100)
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.fill_between(rec['m'], np.exp(rec['95']), np.exp(rec['5']), color = 'magenta', alpha = 0.5)
ax.plot(rec['m'], np.exp(rec['50']), color = 'r')
pdf = []
for i,si in enumerate(post):
    # Smarter way to update?
    f = np.array([gauss(xi, si['mean'], si['sigma']) for xi in x])
    pdf.append(f)
    if i%10 == 0:
        ax.plot(x, f, color='turquoise', linewidth = 0.1)
l,m,h = np.percentile(pdf,[5,50,95],axis=0)
ax.fill_between(x, l, h, color = 'turquoise', alpha = 0.5)
ax.plot(x, m, color = 'k')

fig.savefig(os.path.join(out_folder,'compare_50.pdf'), bbox_inches='tight')

