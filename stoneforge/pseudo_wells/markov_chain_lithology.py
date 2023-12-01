"""
Created on Mon Oct 02 2023
@authors: Mario Martins, Jhefferson Oliveira 
"""

import numpy as np

import scipy

def simple(markov_chain, sampling, lithology_code = False, initial_state = 0, seed_value = 42):

    """ 
    Ex.: lithology_code  = [49,57,2]

    Ex.: markov_chain = np.array(
                        [[0.90, 0.07, 0.03], # Arenito
                         [0.03, 0.95, 0.02], # Folhelho
                         [0.02, 0.10, 0.88]] # Carbonato
                         )
    """

    initial_state = initial_state

    np.random.seed(seed_value)

    if lithology_code:
        litho_index = np.arange(len(lithology_code))
        value_map = dict(zip(litho_index,lithology_code))
    else:
        litho_index = np.arange(np.shape(markov_chain)[0])
    

    sorted_values = []

    current_state = initial_state

    for _ in range(sampling):
        # Use the Markov matrix to transition to the next state
        next_state = np.random.choice(litho_index, p=markov_chain[current_state])
        sorted_values.append(next_state)
        current_state = next_state

    if lithology_code:
        new_array = [value_map[val] for val in np.array(sorted_values)]
    else:
        new_array = np.array(sorted_values)
    return new_array

def extended(markov_chain, sampling, lithology_code = False, initial_state = 0, single_lithology = True, seed_value = 42):

    """ 
    Ex.: lithology_code  = [49,57,2]
    
    Ex.: markov_chain = np.array(
                        [[0.90, 0.07, 0.03], # Arenito
                         [0.03, 0.95, 0.02], # Folhelho
                         [0.02, 0.10, 0.88]] # Carbonato
                         )
    """

    np.random.seed(seed_value)

    if lithology_code:
        litho_index = np.arange(len(lithology_code))
        value_map = dict(zip(litho_index,lithology_code))
    else:
        litho_index = np.arange(np.shape(markov_chain)[0])

    num = 256
    facies = np.zeros((num, sampling), dtype=int)
    facies[0, :] = initial_state

    for j in range(sampling):
        for i in range(1, num):
            line = facies[i-1, j]
            probabilities = P[line]
            facies[i, j] = np.random.choice(litho_index, p=probabilities)
        facies[:, j] = facies[::-1, j]

    if lithology_code:
        new_array = np.vectorize(value_map.get)(facies)
    else:
        new_array = facies.copy()

    if single_lithology:
        ii = np.random.randint(0,255)
        return new_array[ii]
    else:
        return new_array
    