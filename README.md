# NP2P
Parameter estimation from nonparametric inference.

## Installation
You can install this package from the repository:
```
git clone git@github.com:sterinaldi/NP2P.git
cd NP2P
pip install .
```

## Usage
This analysis a set of draws from a non-parametric scheme (represented as list of objects with a `logpdf` method).
Basic usage:
```python
from np2p.ParEst import DirichletProcess as DP
# Import (or define) your parametric model
from your_module import parametric_model

# Interval where the samples are defined
domain_bounds = [xmin, xmax]
# Load non-parametric reconstruction
draws = load_np_draws(np_file)
# Parameters of the model
names  = ['par1', 'par2']
# Parameter bounds
bounds = [[0,1], [0,1]]
# Model name
model_name = 'mymodel'
# Desired number of bins
n_bins = N

sampler = DP(model         = parametric_model, 
             names         = names, 
             bounds        = bounds,
             draws         = draws,
             domain_bounds = domain_bounds,
             model_name    = model_name,
             n_bins        = n_bins,
             )
sampler.run()
samples = sampler.samples
```
