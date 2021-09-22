import numpy as np
import numpy.random as rd

def read_samples(samples, m_min, m_max, N):
    m  = np.linspace(x_min, x_max, N)
    dm = m[1] - m[0]
    probs = []
    for samp in samples:
        p = np.ones(N) * -np.inf
        p = samp(m)
        probs.append(p+np.log(dm))
    probs.append(np.zeros(N))
    probs = np.array([p - logsumexp(p) for p in probs])
    return probs.T

def random_walk(probs, N_draws):
    draws = np.array([recursive_draw() for _ in range(N_draws)])
    return draws
        
