# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:08:25 2019

@author: user
"""

import numpy as np
import random

class GA(object):
    
    def __init__(self,num_of_generations=1000,
                    num_of_parents=8,
                    num_of_offsprings=64,
                    mutation_percent=0.05,
                    cross_point=1/2):
        
        self.num_of_generations=num_of_generations
        self.num_of_parents=num_of_parents
        self.num_of_offsprings=num_of_offsprings
        self.mutation_percent=mutation_percent
        self.cross_point=cross_point
    
    def mat_to_vector(self,mat_pop_weights):
        pop_weights_vector = []
        for sol_idx in range(mat_pop_weights.shape[0]):
            curr_vector = []
            for layer_idx in range(mat_pop_weights.shape[1]):
                vector_weights = np.reshape(mat_pop_weights[sol_idx, layer_idx],\
                                            newshape=(mat_pop_weights[sol_idx, layer_idx].size))
                curr_vector.extend(vector_weights)
            pop_weights_vector.append(curr_vector)
        return np.array(pop_weights_vector)

    def vector_to_mat(self,vector_pop_weights, mat_pop_weights):
        mat_weights = []
        for sol_idx in range(mat_pop_weights.shape[0]):
            start = 0
            end = 0
            for layer_idx in range(mat_pop_weights.shape[1]):
                end = end + mat_pop_weights[sol_idx, layer_idx].size
                curr_vector = vector_pop_weights[sol_idx, start:end]
                mat_layer_weights = np.reshape(curr_vector,\
                                               newshape=(mat_pop_weights[sol_idx, layer_idx].shape))
                mat_weights.append(mat_layer_weights)
                start = end
        return np.reshape(mat_weights, newshape=mat_pop_weights.shape)

    def select_mating_pool(self,pop_w_vc_vector, pop_w_cv_vector,fitness):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents_of_w_vc=[];parents_of_w_cv=[];
        for parent_num in range(self.num_of_parents):
            min_fitness_idx = np.where(fitness == np.min(fitness))
            min_fitness_idx = min_fitness_idx[0][0]
            parents_of_w_vc.append(pop_w_vc_vector[min_fitness_idx, :]) # dims:[num_of_parents,(m*n*8)]
            parents_of_w_cv.append(pop_w_cv_vector[min_fitness_idx, :]) # dims:[num_of_parents,(m*n*8)]
            fitness[min_fitness_idx] = 9999999
        return np.array(parents_of_w_vc),np.array(parents_of_w_cv)


    def crossover(self,parents_of_w_vc,parents_of_w_cv):
        offspring_of_w_vc=[];offspring_of_w_cv=[]
        offs_w_vc=np.empty([parents_of_w_vc.shape[1]])
        offs_w_cv=np.empty([parents_of_w_cv.shape[1]])
        crossover_point = np.uint8(parents_of_w_vc.shape[1]*self.cross_point);
        for k in range(self.num_of_parents):
            parent1_idx=k;
            for j in range(self.num_of_parents):                    
                parent2_idx=j;
                # The new offspring will have its first half of its genes taken from the first parent.
                offs_w_vc[0:crossover_point] = parents_of_w_vc[parent1_idx, 0:crossover_point]
                # The new offspring will have its second half of its genes taken from the second parent.
                offs_w_vc[crossover_point:] = parents_of_w_vc[parent2_idx, crossover_point:]
                offs_w_cv[0:crossover_point] = parents_of_w_cv[parent1_idx, 0:crossover_point]
                offs_w_cv[crossover_point:] = parents_of_w_cv[parent2_idx, crossover_point:]
                offspring_of_w_vc.append(offs_w_vc);
                offspring_of_w_cv.append(offs_w_cv);
        return np.array(offspring_of_w_vc), np.array(offspring_of_w_cv ) 
    
    def mutation(self,offspring_of_w_vc, offspring_of_w_cv):#,mutation_degree):
        num_mutations = np.uint8(self.mutation_percent*offspring_of_w_cv.shape[1])
        if num_mutations==0:
            num_mutations=1;

        nonzero_index_cv=np.nonzero(offspring_of_w_cv[0])[0]
        nonzero_index_vc=np.nonzero(offspring_of_w_vc[0])[0]
        mutation_indices_cv = np.array(random.sample(nonzero_index_cv.tolist(), num_mutations))
        mutation_indices_vc = np.array(random.sample(nonzero_index_vc.tolist(), num_mutations))   

        for idx in range(offspring_of_w_vc.shape[0]):
            #random_value = np.random.uniform(-0.1,0.1, num_mutations)
            #offspring_of_w_vc[idx,mutation_indices_vc] += random_value;
            #offspring_of_w_cv[idx,mutation_indices_cv] += np.random.uniform(0.5, 1, num_mutations);
            offspring_of_w_vc[idx,mutation_indices_vc] = np.random.uniform(0.1, 2, num_mutations);
            offspring_of_w_cv[idx,mutation_indices_cv] = np.random.uniform(0.1, 2, num_mutations);
        return offspring_of_w_vc, offspring_of_w_cv;
    
    
    