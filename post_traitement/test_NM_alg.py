from Nelder_Mead_parameters_estimation import write_parameters, save_best_parameters, bhattacharyya_distance
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import subprocess


## Testing the algo by putting silly parameters
theta = np.array([
    
    0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8,
    0.9, 1.0, 1.1
])

write_parameters(theta)

save_best_parameters(theta, 0.123)

p = np.array([0.1, 0.2, 0.7])
q = np.array([0.9, 0.9, 0.1])

d = bhattacharyya_distance(p, q)
