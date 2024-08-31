import numpy as np

def recursive_grid(bounds, n_pts):
    """
    Recursively generates the n-dimensional grid points (extremes are excluded).
    
    Arguments:
        list-of-lists bounds: extremes for each dimension (excluded)
        int n_pts:            number of points for each dimension
        
    Returns:
        np.ndarray: grid
    """
    bounds = np.atleast_2d(bounds)
    n_pts  = np.atleast_1d(n_pts)
    if len(bounds) == 1:
        d = np.linspace(*bounds[0], n_pts[0])
        return np.atleast_2d(d).T
    else:
        grid_nm1 = recursive_grid(np.array(bounds)[1:], n_pts[1:])
        d        = np.linspace(*bounds[0], n_pts[0])
        grid     = []
        for di in d:
            for gi in grid_nm1:
                grid.append([di,*gi])
        return np.array(grid)
