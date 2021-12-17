import numpy as np
import matplotlib.pyplot as plt
from ParEst import DirichletProcess
import json
import cpnest
import corner
import os
from scipy.interpolate import interp1d
from scipy.special import logsumexp, gammaln
from scipy.stats import dirichlet
from loglikelihood import normal, log_likelihood

# OPTIONS
#------------------------
# Data folder
folder = '/Users/stefanorinaldi/Documents/parametric/gaussian_test/' # CHANGEME
# Concentration parameter
max_alpha = 10000
# Select a model:
model = 'normal' # 'normal', 'uniform', 'exponential', 'cauchy', 'gen_normal'
# Postprocessing
postprocessing = False
#------------------------

# Files
samps_file = folder + 'gaussian.txt'
draws_file = folder + 'posterior_functions_gaussian.json'
rec_file   = folder + 'log_rec_prob_gaussian.txt'

out_folder = folder + model + '/'

# Comparison with DPGMM outcome
rec = np.genfromtxt(rec_file, names = True)

# Load samples
ss = np.genfromtxt(samps_file)

# Boundaries and number of bins
x_min = np.min(ss)
x_max = np.max(ss)

# Load data
openfile = open(draws_file, 'r')
json_dict = json.load(openfile)
draws = []
samps = []
for i, p in enumerate(json_dict.values()):
    draws.append(p)
draws = np.array(draws).T
for p in draws:
    samps.append(p)
openfile.close()
m = np.array([float(m) for m in json_dict.keys()])
x = np.array(m[np.where([x_min < mi < x_max for mi in m])])
dx = x[1]-x[0]
logdx = np.log(x[1]-x[0])
samples = []
for d in samps[1:]:
    samples.append(d[np.where([x_min < mi < x_max for mi in m])] + logdx)
    
#samples = np.array([rec['50'][np.where([x_min < mi < x_max for mi in m])]+logdx])
#samples = np.array([np.log(normal(x,40,5))])
samples = np.array([s - logsumexp(s) for s in samples])

logP = np.mean(samples, axis = 0)

lenMu    = 1000
lenSigma = 1000

mu = np.linspace(39,41, lenMu)
sigma = np.linspace(4.8,5.2,lenSigma)
dm = mu[1] - mu[0]
ds = sigma[1] - sigma[0]
M, S = np.meshgrid(mu,sigma)

N_bins = len(x)
print('{0} bins between {1:.1f} and {2:.1f}, {3} draws. dmu = {4}, ds = {5}'.format(N_bins, x_min, x_max, len(samples), dm, ds))



concentration = 1e5

likelihood = np.zeros(shape = (len(mu), len(sigma)))

for i in range(lenMu):
    for j in range(lenSigma):
        print('\r{0}/{1}'.format(i*lenSigma+j+1, lenMu*lenSigma), end = '')
        model = normal(x, mu[i], sigma[j])*dx
        model = concentration*model#/np.sum(model)
        lnB = gammaln(np.sum(model)) - np.sum(gammaln(model))
        likelihood[i,j] = lnB + np.sum((model-1)*logP) #log_likelihood(model, samples, concentration)

i_mu, j_sigma = np.where([li == likelihood.max() for li in likelihood])

print('\nmu = {0}, sigma = {1}, logL_max = {2}, logL_true = {3}'.format(mu[i_mu][0], sigma[j_sigma][0], likelihood.max(), log_likelihood(normal(x, 40, 5)*dx, samples, concentration)))
post = normal(x, mu[i_mu], sigma[j_sigma])*dx

N = logsumexp(likelihood + np.log(dm) + np.log(ds))

plt.figure(1,figsize=(10,7))
plt.contourf(sigma, mu, np.exp(likelihood - N), cmap='PuBuGn', levels = 20)    #2D plot of the posterior
cbar = plt.colorbar(orientation='horizontal', pad=0.07, shrink=1, label = '$log(L)$')

plt.xlabel('$\\sigma$')
plt.ylabel('$\\mu$')

plt.tight_layout()
#plt.grid()
plt.savefig(out_folder + '/plot_logL.pdf', bbox_inches = 'tight')

plt.figure()
for s in samples:
    plt.plot(x, np.exp(s)/np.sum(np.exp(s)), lw = 0.1)
#plt.plot(x, normal(x, 40, 5)*dx, lw = 0.6, color = 'r')
plt.plot(x, post, lw = 0.6, color = 'g')
    
plt.savefig(out_folder + '/all_draws.pdf', bbox_inches = 'tight')
