import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


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


def generate_lithology_section(markov_chain, sampling, lithology_code=None, initial_state=0, seed_value=42):
    
    litho_sequence = simple(markov_chain, sampling, lithology_code, initial_state, seed_value)
    
    fig, ax = plt.subplots(figsize=(6, 8))  
    
    if lithology_code:
        color_map = {code: np.random.rand(3,) for code in lithology_code}
    else:
        color_map = {i: np.random.rand(3,) for i in set(litho_sequence)}

    depth = np.linspace(3000, 7000, len(litho_sequence) + 1)  
    
    for i, lithology in enumerate(litho_sequence):
        if i < len(litho_sequence) - 1:  
            ax.fill_betweenx([depth[i], depth[i + 1]], 0, 1, color=color_map[lithology])

    ax.invert_yaxis()

    ax.set_xlabel('Litologia')
    ax.set_ylabel('Profundidade (m)')

    ax.set_title('Seção Litológica')
    ax.set_xlim(0, 1)  
   
    plt.tight_layout()
    plt.show()  

markov_chain = np.array(
    [[0.33, 0.07, 0.30, 0.30],
     [0.02, 0.87, 0.01, 0.10],
     [0.05, 0.10, 0.85, 0.00],
     [0.30, 0.15, 0.25, 0.30]]
)
lithology_code = [14, 7, 21, 36]

generate_lithology_section(markov_chain, 300, lithology_code, seed_value=42)
