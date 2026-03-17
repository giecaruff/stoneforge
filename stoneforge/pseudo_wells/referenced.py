import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import pickle
from typing import Annotated

def anadrill_siliciclastic(
    structure: Annotated[dict, "Data structure"],
    step: Annotated[float, "Depth step"]=1.0,
    top: Annotated[float, "Top depth"]=None,
    bottom: Annotated[float, "Bottom depth"]=None,
    random_state: Annotated[bool, "Random state"]=False,
    noise: Annotated[float, "Noise level"]=0.005):

    """Generate synthetic well log data based on the SLB/Anadrill siliciclastic facies model adapted from :footcite:t:`slb1972,freire2020youtube`
    
    Parameters
    ----------
    structure : dict
        Dictionary in format {facies_code: n_samples}
    step : float
        Depth step for synthetic log
    top : float or None
        Top depth of the synthetic log (if None, starts at 0)
    bottom : float or None
        Bottom depth of the synthetic log (if None, determined by total samples and step)
    random_state : int or False
        Random seed for reproducibility (if False, no noise is added)
    noise : float
        Proportional noise level to add to the curves (e.g., 0.005 for 0.5%)
    Returns
    -------
    dict
        Dictionary containing synthetic well log data, with keys for each curve and facies.
    """
    return (_generate(
        structure,
        data_path='anadrill_siliciclastic.ggf',
        step=step,
        top=top,
        bottom=bottom,
        random_state=random_state,
        noise=noise
    ),{'DEPTH':'m','GR':'API','RES':'ohm.m','NPHI':'m3/m3','DEN':'g/cm3','DT':'us/ft','INDEX':'','CODE':'','ROCK':'','FLUID':''})

def _generate(
    structure,
    data_path='anadrill_siliciclastic.ggf',
    step=1.0,
    top=None,
    bottom=None,
    random_state=False,
    noise=0.005
):
    """
    Generate synthetic well log data based on facies structure.

    Parameters
    ----------
    structure : dict
        Dictionary in format {facies_code: n_samples}
    data_path : str
        Path to GGF file containing statistical data (standard is 'anadrill_siliciclastic.ggf')
    step : float
        Depth step for synthetic log
    top : float or None
        Top depth of the synthetic log (if None, starts at 0)
    bottom : float or None
        Bottom depth of the synthetic log (if None, determined by total samples and step)
    random_state : int or False
        Random seed for reproducibility (if False, no noise is added)
    noise : float
        Proportional noise level to add to the curves (e.g., 0.005 for 0.5%)

    Returns
    -------
    dict
        Dictionary containing synthetic well log data, with keys for each curve and facies.
    """
    
    module_dir = Path(__file__).parent
    file_path = module_dir / data_path

    with open(file_path, 'rb') as handle:
        example = pickle.load(handle)

    # RNG only if noise is enabled
    rng = None
    if random_state is not False:
        rng = np.random.default_rng(random_state)

    # Detect numeric vs categorical fields
    header_f = []
    header_s = []

    sample_facies = next(iter(structure))

    for k in example:
        print(k)
        if isinstance(example[k][sample_facies], float):
            header_f.append(k)
        else:
            header_s.append(k)

    n_total = sum(structure.values())

    curves = np.zeros((n_total, len(header_f)))
    classes = {h: [] for h in header_s}

    idx = 0

    for facies, n_samples in structure.items():

        values_f = np.array([example[h][facies] for h in header_f])
        values_s = [example[h][facies] for h in header_s]

        block = np.tile(values_f, (n_samples, 1))

        # Add proportional Gaussian noise
        if rng is not None:
            eps = rng.normal(0, noise, size=block.shape)
            block = block + block * eps

        curves[idx:idx+n_samples, :] = block

        for h, v in zip(header_s, values_s):
            classes[h].extend([v]*n_samples)

        idx += n_samples
        
    if top is not None:
        top = float(top)
    else:
        top = 0.0
        
    if bottom is not None:
        bottom = float(bottom)
    else:
        bottom = top + n_total * step

    final_data = {}
    final_data['DEPTH'] = np.linspace(top, bottom, num=n_total)

    for i, h in enumerate(header_f):
        final_data[h] = curves[:, i]

    for h in header_s:
        final_data[h] = classes[h]

    return final_data