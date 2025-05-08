import numpy as np

def simple(markov_chain, sampling, lithology_code = False, initial_state = 0, seed_value = 42):
    """ 
    Ex.: lithology_code  = [14,7,21,36]

    Ex.: markov_chain = np.array(
                        [[0.93, 0.07, 0.00, 0.00], # Carbonato Fechado
                        [0.02, 0.97, 0.01, 0.00], # Carbonato Poroso
                        [0.05, 0.10, 0.85, 0.00] # Carbonato Rico em Argila
                        [0.00, 0.00, 0.00, 0.00]] # Carbonato c/ sílica
                         )
    """
    initial_state = initial_state

    np.random.seed(seed_value)

    if lithology_code:
        litho_index = np.arange(len(lithology_code))
        value_map = dict(zip(litho_index,lithology_code))
        current_state = list(lithology_code).index(initial_state)
    else:
        litho_index = np.arange(np.shape(markov_chain)[0])
        current_state = initial_state

    sorted_values = []

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

def markov_chain(lito):
        """Generate markov chain matrix from lithology data
        """
        lito = np.array(lito)
        _lito = lito[~np.isnan(lito)]
        states = list(set(_lito))

        matrix_size = len(states)
        transition_matrix = np.zeros((matrix_size, matrix_size))

        state_to_index = {state: i for i, state in enumerate(states)}

        for i in range(len(_lito) - 1):
            current_state = _lito[i]
            next_state = _lito[i + 1]

            transition_matrix[state_to_index[current_state], state_to_index[next_state]] += 1

        row_sums = transition_matrix.sum(axis=1)
        MC = transition_matrix / row_sums[:, np.newaxis]
        states = states

        return (MC,states)

def extended(markov_chain, sampling, lithology_code = False, initial_state = 0, single_lithology = True, seed_value = 42):
    """ 
    Ex.: lithology_code  = [14,7,21,36]
    
    Ex.: markov_chain = np.array(
                        [[0.93, 0.07, 0.00, 0.00], # Carbonato Fechado
                        [0.02, 0.97, 0.01, 0.00], # Carbonato Poroso
                        [0.05, 0.10, 0.85, 0.00] # Carbonato Rico em Argila
                        [0.00, 0.00, 0.00, 0.00]] # Carbonato c/ sílica
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
            probabilities = markov_chain[line]
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