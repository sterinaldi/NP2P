import numpy as np
import matplotlib.pyplot as plt
from ParEst import DirichletProcess
import json
import cpnest
import corner
import os
from scipy.interpolate import interp1d
from loglikelihood import normal, uniform, exponential, cauchy, generalized_normal

# OPTIONS
#------------------------
# Postprocessing
postprocessing = False

# Files
samps_file = '/Users/stefanorinaldi/Documents/parametric/gaussian/gaussian.txt'
draws_file = '/Users/stefanorinaldi/Documents/parametric/gaussian/posterior_functions_gaussian.json' # CHANGEME
rec_file   = '/Users/stefanorinaldi/Documents/parametric/gaussian/log_rec_prob_gaussian.txt' # CHANGEME

# Output folder
out_folder = '/Users/stefanorinaldi/Documents/parametric/' # CHANGEME

# Select a model:
model = 'normal' # 'normal', 'uniform', 'exponential', 'cauchy', 'gen_normal'
out_folder = out_folder + model + '/'

if model == 'normal':
    names = ['mean', 'sigma']
    nargs = len(names)
    bounds = [[20,60], [1,10]]
    labels = ['\\mu', '\\sigma']
    label_selected_model = 0 # Normal
    true_vals = [40, 5]
    model = normal
    model_label = 'Normal'

if model == 'exponential':
    names = ['x0', 'l']
    nargs = len(names)
    bounds = [[20,60], [1,10]]
    labels = ['x_0', '\\lambda']
    label_selected_model = 4 # Exponential
    true_vals = None
    model = exponential
    model_label = 'Exponential'
    
if model == 'uniform':
    names = ['x_min', 'x_max']
    nargs = len(names)
    bounds = [[10, 30], [50,70]]
    labels = ['x_{min}', 'x_{max}']
    label_selected_model = 3 # Uniform
    true_vals = None
    model = uniform
    model_label = 'Uniform'

if model == 'cauchy':
    names = ['x0', 'g']
    nargs = len(names)
    bounds = [[20, 60], [1,10]]
    labels = ['x_0', '\\gamma']
    label_selected_model = 5 # Cauchy
    true_vals = None
    model = cauchy
    model_label = 'Cauchy'

if model == 'gen_normal':
    names = ['x0','s','b']
    nargs = len(names)
    bounds = [[20, 60], [1,10], [1,4]]
    labels = ['\\mu', '\\sigma', '\\beta']
    label_selected_model = 6 # Generalized Normal
    true_vals = None
    model = generalized_normal
    model_label = 'Generalized\ Normal'


# Load data
openfile = open(draws_file, 'r')
json_dict = json.load(openfile)
draws = []
samps = []
for i, p in enumerate(json_dict.values()):
    draws.append(p)
draws = np.array(draws).T
for p in draws[1:]:
    samps.append(p)
openfile.close()
m = np.array([float(m) for m in json_dict.keys()])
samples = []
for d in samps:
    samples.append(interp1d(m, d))

# Load samples
ss = np.genfromtxt(samps_file)

# Boundaries, c_par and number of bins
max_alpha = 10000
x_min = np.min(ss)
x_max = np.max(ss)
N_bins = len(np.where([x_min <= mi <= x_max for mi in m])[0])
print('{0} bins between {1:.1f} and {2:.1f}'.format(N_bins, x_min, x_max))

# Comparison with DPGMM outcome
rec = np.genfromtxt(rec_file, names = True)

PE = DirichletProcess(
    label_selected_model,
    names,
    bounds,
    samples,
    x_min = x_min,
    x_max = x_max,
    max_a = max_alpha*N_bins,
    N_bins = N_bins,
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

labels = labels + ['\\alpha','\\alpha/N']
par_names = names
names = names + ['a']
if true_vals is not None:
    true_vals = true_vals + [1, 1]
samps = np.column_stack([post[lab] for lab in names] + [post['a']/N_bins)])

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
for i,si in enumerate(post):
    s = np.array([si[lab] for lab in par_names])
    f = model(m, *s)
    pdf.append(f)

low,med,high = np.percentile(pdf,[5,50,95],axis=0)
ax.fill_between(m, high, low, color = 'lightsalmon', alpha = 0.5)
ax.plot(m, med, color = 'r', lw = 0.5, label = '${0}$'.format(model_label))
ax.set_xlim(x_min, x_max)
ax.set_xlabel('$x$')
ax.set_ylabel('$p(x)$')
ax.grid(True,dashes=(1,3))
ax.legend(loc=0,frameon=False,fontsize=10)
fig.savefig(os.path.join(out_folder,'compare_50.pdf'), bbox_inches='tight')

