# parametric
Parameter estimation from nonparametric inference.

## Installation
You can install this package from the repository:
```
git clone git@github.com:sterinaldi/parametric.git
cd parametric
pip install .
```

## Usage
This analysis a set of draws from a non-parametric scheme (represented as list of objects with a `logpdf` method).
Basic usage:
```python
import numpy as np
from parest.ParEst import DirichletProcess as DP
# Import (or define) your parametric model
from your_module import parametric_model

# Interval where the samples are defined
domain_bounds = [xmin, xmax]
# Load non-parametric reconstruction
draws = load_np_draws(np_file)
# Parameters of the model
pars  = ['par1', 'par2']
# Parameter bounds
bounds = [[0,1], [0,1]]
# Number of observations used for the non-parametric analysis
n_data = N
# number of samples to draw
n_samples = K

sampler = DP(model         = parametric_model, 
             pars          = pars, 
             bounds        = bounds,
             draws         = draws,
             domain_bounds = domain_bounds,
             n_data        = n_data,
             )
sampler.run(size = n_samples)
samples = sampler.samples
```
