

import numpy as np
import matplotlib.pyplot as plt
import active_utils as autils
from plot_flower_queries import get_query_data 


#object centers
centers = np.array([[0.25, 0.75],[0.75, 0.75], [0.25, 0.25], [0.75, 0.25]])


#get leanred weights for MAP Reward function
filename = "./data/flowers_seed0_randomFalse_demo10.txt"
_, _, obj_weights, abs_weights, _, _  = get_query_data(filename)


map_reward = autils.RbfComplexReward(centers, obj_weights, abs_weights)
pi_map, _ = map_reward.estimate_best_placement()
print(pi_map)
