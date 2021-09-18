import numpy as np
import matplotlib.pyplot as plt
from ParEst import DirichletProcess
import pickle
import cpnest
import corner
from scipy.special import erf, logsumexp
from scipy.stats import norm, dirichlet
from scipy.interpolate import interp1d
import os


def log_add(x, y): return x+np.log(1.0+np.exp(y-x)) if x >= y else y+np.log(1.0+np.exp(x-y))
def log_norm(x, x0, s): return -((x-x0)**2)/(2*s*s) - np.log(np.sqrt(2*np.pi)) - np.log(s)

def PL(m, alpha, m_max, m_min, l_max, l_min):
    f = ((1-alpha)/(m_max**(1-alpha) - m_min**(1-alpha)))*m**(-alpha)*(1+erf((m-m_min)/(l_min)))*(1+erf((m_max-m)/l_max))/4.
    return f

def gauss(x, x0, s):
    return np.exp(-((x-x0)**2/(2*s**2)))/(np.sqrt(2*np.pi)*s)

def bimodal(x,x0,s0,x1,s1):
    return (np.exp(-((x-x0)**2/(2*s0**2)))/(np.sqrt(2*np.pi)*s0) + np.exp(-((x-x1)**2/(2*s1**2)))/(np.sqrt(2*np.pi)*s1))/2.

def logPrior(*args):
    return 0

# Simulated values
mu = 50
sigma = 4
a = 20
g = 20
true_vals = [mu, sigma, a, 1]

# Samples
samp_file = '/Users/stefanorinaldi/Documents/mass_inference/DPGMM/reconstructed_events/posteriors/posterior_functions_gaussian.pkl' # CHANGEME
openfile  = open(samp_file, 'rb')
samples   = pickle.load(openfile)[-10:] # subset
openfile.close()

# Comparison with DPGMM outcome
rec_file = '/Users/stefanorinaldi/Documents/mass_inference/DPGMM/reconstructed_events/rec_prob/log_rec_prob_gaussian.txt' # CHANGEME
rec = np.genfromtxt(rec_file, names = True)

out_folder  = '/Users/stefanorinaldi/Documents/parametric/DP' # CHANGEME

names = ['mu1', 'sigma1']
bounds = [[45,60], [2,10]]
labels = ['\mu_1', '\sigma_1']
selected_model = gauss

PE = DirichletProcess(
    selected_model,
    names,
    bounds,
    samples,
    x_min = 40,
    x_max = 60,
    prior_pars = logPrior,
    max_a = 10000,
    max_g = 2,
    max_N = 7,
    out_folder = out_folder,
    load_preprocessed = False
    )

work = cpnest.CPNest(PE,
                    verbose = 2,
                    nlive = 1000,
                    maxmcmc = 1000,
                    nthreads = 4,
                    output  = out_folder
                    )
work.run()
print('log Evidence: {0}'.format(work.NS.logZ))

# Posteriors
labels = labels + ['\\alpha', '\\gamma', 'N']
par_names = names
names = names + ['a','g', 'N']
x = work.posterior_samples.ravel()
samps = np.column_stack([x[lab] for lab in names])
fig = corner.corner(samps,
       labels= [r'${0}$'.format(lab) for lab in labels],
       quantiles=[0.05, 0.16, 0.5, 0.84, 0.95],
       show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
       use_math_text=True,
       filename=os.path.join(out_folder,'joint_posterior.pdf'))
fig.savefig(os.path.join(out_folder,'joint_posterior.pdf'), bbox_inches='tight')

# Comparison: HDPGMM vs model (median of all inferred parameters)
p = np.percentile(np.column_stack([x[lab] for lab in par_names]), 50, axis = 0)
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.fill_between(rec['m'], np.exp(rec['95']), np.exp(rec['5']), color = 'lightgreen', alpha = 0.5)
ax.fill_between(rec['m'], np.exp(rec['84']), np.exp(rec['16']), color = 'aqua', alpha = 0.5)
ax.plot(rec['m'], np.exp(rec['50']), color = 'r')
print(*p)
ax.plot(rec['m'], selected_model(rec['m'], *p), color = 'k')

fig.savefig(os.path.join(out_folder,'compare_50.pdf'), bbox_inches='tight')

