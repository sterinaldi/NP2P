import numpy as np
import numpy.random as rd
import pickle
from scipy.special import logsumexp
import ray
import matplotlib.pyplot as plt

def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if array_equal(elem, myarr)), False)

def read_samples(samples, x_min, x_max, N):
    m  = np.linspace(x_min, x_max, N)
    dm = m[1] - m[0]
    probs = []
    for samp in samples:
        p = samp(m)
        probs.append(np.exp(p)*dm)
    probs = np.array([p/p.sum() for p in probs])
    return probs.T

def random_walk(probs, N_draws):
    ray.init(ignore_reinit_error = True)
    draws = []
    draws.append(ray.get([recursive_draw.remote(probs) for _ in range(N_draws)]))
    return np.array(draws)

@ray.remote
def recursive_draw(probs):
    while True:
        finish_flag = True
        ps = []
        partial_p = 0.0
        for i in range(probs.shape[0]):
            a = rd.choice(probs[i,:])
            partial_p += a
            if partial_p > 1.0:
                finish_flag = False
                break
            if partial_p < 1.0:
                ps.append(a)
        if np.isclose(partial_p,1.0) and finish_flag:
            break
    return np.array(ps)

def random_paths(samples, N_draws):
    p = random_walk(samples, N_draws)
    set_p = []
    for pi in p:
        if not arreq_in_list(pi, set_p):
            set_p.append(pi)
    set_p = np.array(set_p[0])
    return np.log(set_p)
    
if __name__=="__main__":

    ray.init()
    samp_file ='/Users/stefanorinaldi/Documents/mass_inference/DPGMM/reconstructed_events/posteriors/posterior_functions_gaussian.pkl' # CHANGEME
    openfile  = open(samp_file, 'rb')
    samples   = pickle.load(openfile)
    openfile.close()
    probs = read_samples(samples, 40, 60, 100)
    print(len(samples))
    print(probs.shape)
    s = np.array([p for p in random_walk(probs, 1000)])
#    print(s)
    set_p = []
    for si in s:
        if not arreq_in_list(si, set_p):
            set_p.append(si)
    set_p = np.array(set_p)
    print(s.shape, set_p.shape)
    ray.shutdown()
