# -*- coding: utf-8 -*-
"""
Created on Thu May 10 21:11:08 2018

@author: daniel
"""

import birl
import active_utils
import numpy as np
import matplotlib.pyplot as plt


def generate_random_configuration(num_centers):
    return np.random.rand(num_centers,2)

def generate_one_tweak_configuration(num_centers, birl):
    #get latest placement demo from birl and tweak it a bit for one item
    centers = birl.demonstrations[-1][0].copy()
    num_centers = centers.shape[0]
    tweak_indx = np.random.randint(num_centers)
    centers[tweak_indx] = np.random.rand(1,2)

    return centers

#calculate the policy loss between the hypothesis return and the map return
def calculate_policy_loss(config, hyp_params, map_params):
    #calculate reward for optimal placement under hyp_reward
    hyp_obj_weights, hyp_abs_weights = hyp_params
    hyp_reward_fn = active_utils.RbfComplexReward(config, hyp_obj_weights, hyp_abs_weights)
    #get optimal placement under the hypothesis reward function and new configuration
    hyp_placement, hyp_return = hyp_reward_fn.estimate_best_placement()

    #calculate reward for map placement under hyp_reward
    map_obj_weights, map_abs_weights = map_params
    map_reward_fn = active_utils.RbfComplexReward(config, map_obj_weights, map_abs_weights)
    #get optimal placement under map reward function and new configuration
    map_placement, _ = map_reward_fn.estimate_best_placement()
    map_return = hyp_reward_fn.get_reward(map_placement)

    return hyp_return - map_return

def calculate_placement_loss(config, hyp_params, map_params):
    #calculate reward for optimal placement under hyp_reward
    hyp_obj_weights, hyp_abs_weights = hyp_params
    hyp_reward_fn = active_utils.RbfComplexReward(config, hyp_obj_weights, hyp_abs_weights)
    #active_utils.visualize_reward(hyp_reward_fn, "hypothesis reward")
    #get optimal placement under the hypothesis reward function and new configuration
    hyp_placement, _ = hyp_reward_fn.estimate_best_placement()

    #calculate reward for map placement under hyp_reward
    map_obj_weights, map_abs_weights = map_params
    map_reward_fn = active_utils.RbfComplexReward(config, map_obj_weights, map_abs_weights)
    #active_utils.visualize_reward(map_reward_fn, "map reward")
    #get optimal placement under map reward function and new configuration
    map_placement, _ = map_reward_fn.estimate_best_placement()
    #print "placement loss", np.linalg.norm(hyp_placement - map_placement)
    #plt.show()
    return np.linalg.norm(hyp_placement - map_placement)


def calculate_var(alpha, config, birl):
    param_chain = birl.get_mcmc_chain()
    map_params = birl.get_map_params()
    plosses = []
    hyp_cnt = 0
    for i in range(birl.burn, len(param_chain), birl.skip):
        hyp_params = param_chain[i]
        #print hyp_cnt
        ploss = calculate_policy_loss(config, hyp_params, map_params)
        plosses.append(ploss)
        hyp_cnt += 1
    #sort and take alpha quantile
    plosses.sort()
    alpha_index = int(alpha*len(plosses))
    return plosses[alpha_index]

def calculate_placement_var(alpha, config, birl):
    param_chain = birl.get_mcmc_chain()
    map_params = birl.get_map_params()
    plosses = []
    hyp_cnt = 0
    for i in range(birl.burn, len(param_chain), birl.skip):
        #print "config", config
        hyp_params = param_chain[i]
        #print "hyp_params", hyp_params
        #print hyp_cnt
        ploss = calculate_placement_loss(config, hyp_params, map_params)
        plosses.append((ploss, hyp_params))
        hyp_cnt += 1
    #sort and take alpha quantile
    #print(plosses)
    plosses.sort()
    #print(plosses)
    ploss = [p for p,x in plosses]
    hyp_params = [x for p,x in plosses]
    #print(ploss)
    plosses = ploss
    #print(hyp_params)
    #input("Enter")
    alpha_index = int(alpha*len(plosses))
    #print "placement var", plosses[alpha_index]
    return plosses[alpha_index], hyp_params[alpha_index]

def find_max_var_config(num_objs, birl, num_configs=10, alpha = 0.95, random_queries=False):
    max_var = 0
    query_config = None
    if random_queries:
        query_config = generate_random_configuration(num_objs)
    else:
        #generate random config and evaluate VaR
        for i in range(num_configs):
            #print "trying config", i
            new_config = generate_random_configuration(num_objs)
            var = calculate_var(alpha, new_config, birl)
            if var > max_var:
                max_var  = var
                query_config = new_config
    return query_config, max_var



def find_max_placement_var_config(num_objs, birl, num_configs=10, alpha = 0.95, random_queries=False):
    max_var = 0
    query_config = None
    query_rbf = None
    if random_queries:
        #query_config = generate_random_configuration(num_objs)
        query_config = generate_one_tweak_configuration(num_objs,birl)
    else:
        #generate random config and evaluate VaR
        for i in range(num_configs):
            print "-------trying config", i
            #new_config = generate_random_configuration(num_objs)
            new_config = generate_one_tweak_configuration(num_objs, birl)
            config_rbf = active_utils.RbfComplexReward(new_config, birl.get_map_params()[0], birl.get_map_params()[1])
            #active_utils.visualize_reward(config_rbf, "config with pi")
            #plt.show()
            #print "new config", new_config
            var, var_rbf = calculate_placement_var(alpha, new_config, birl)
            #print "var", var
            if var > max_var:
                #print "best so far"
                max_var  = var
                query_config = new_config
                query_rbf = var_rbf
    return query_config, max_var, query_rbf
