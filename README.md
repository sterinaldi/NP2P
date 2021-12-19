# parametric
Parameter estimation from nonparametric inference

## Usage
Requires a set of draws from a non-parametric scheme (logarithmic).
Basic usage:
```python
from parest.ParEst import DirichletProcess as DP
import cpnest
import numpy as np

x = np.loadtxt('x_values.txt')      # Interval where the samples are defined
samples = np.loadtxt('samples.txt') # Non-parametric samples (rows)
model = 1                           # Model number
pars  = ['par1', 'par2']            # Parameters of the model
bounds = [[0,1], [0,1]]             # Parameter bounds

sampler = DP(model   = model, 
             pars    = pars, 
             bounds  = bounds,
             samples = samples,
             x       = x,
             )

work = cpnest.CPNest(DP)
work.run()
```
