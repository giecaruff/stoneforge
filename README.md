<p align="center">
<img src="https://raw.githubusercontent.com/giecaruff/logos/main/APPY/stoneforge.png" width="400"/>

<h2 align="center">Algorithms, methods and equations for geophysics in Python</h2>

<p align="center">
Part of the <strong>Appy</strong> project
</p>



<p align="center">
<a href="https://giecaruff.github.io/stoneforge/"><strong>Documentation</strong> (latest)</a> |
<a href="http://gcr.sites.uff.br/"><strong>Institutional</strong> (GIECAR website)</a> 
</p>


<p align="center">
<a href="https://github.com/giecaruff/stoneforge/actions"><img src="https://github.com/giecaruff/stoneforge/actions/workflows/CI.yml/badge.svg" alt="Latest version on PyPI"/></a>
<a href="https://badge.fury.io/py/stoneforge"><img src="https://badge.fury.io/py/stoneforge.svg" alt="PyPI version" height="20"></a>
</p>
  
<!-- 
[![https://github.com/giecaruff/stoneforge/actions](https://github.com/giecaruff/stoneforge/actions/workflows/CI.yml/badge.svg)](https://github.com/giecaruff/stoneforge/actions)
-->

  
## About

The Stoneforge library is linked to the GIECAR laboratory at the Universidade Federal Fluminense (UFF) related to the Geology and Geophysics Departmnet <a href="http://geologiaegeofisica.sites.uff.br/"> (GGO)</a>, and its purpose is to teach and develop routines in Python to solve geological and geophysical problems.


## Installation and first steps

To install stoneforge use the following syntax in your command interpreter environment:

```
$pip install stoneforge
```

and them verify the installation using the following command in Python:

```
>>> import stoneforge
```

## Testing

### Petrophysical Computation and Validation Framework

This library implements a comprehensive suite of petrophysical models for porosity, shale volume, water saturation, and permeability, following well-established formulations from the geoscience and reservoir engineering literature (e.g., Archie, Simandoux, Indonesia, Larionov, Clavier, Timur, Coates, Tixier).

Beyond numerical implementation, the library adopts a multi-layered scientific validation strategy designed to ensure correctness, numerical robustness, and physical consistency across scalar and log-scale data. The testing framework is structured into four complementary levels:

#### 1. Analytical (Equation-Based) Validation

Each petrophysical model is validated against its analytical formulation using deterministic unit tests. For fixed, physically meaningful inputs, the numerical output is compared directly to the closed-form equation, ensuring exact correspondence with the theoretical model. This guarantees that the implementation faithfully reproduces the published equations and assumptions.

#### 2. Vectorization and Numerical Behavior

Because petrophysical data are inherently log-based and vectorized, all models are tested using NumPy arrays to verify correct broadcasting, array-wise computation, and performance on realistic datasets. These tests explicitly check:

- Vectorized execution (arrays in → arrays out)

- Correct propagation of NaN values

- Numerical stability across valid physical ranges

- Enforcement of physical bounds (e.g., 0 ≤ 𝜙, 𝑉𝑠ℎ, 𝑆𝑤 ≤ 1; 𝑘 ≥ 0)

This level ensures the models behave reliably when applied to full well logs rather than isolated scalar values.

#### 3. Physical Invariants and Limiting Behavior (Property-Based Testing)

Advanced property-based testing is employed to validate physical laws and monotonic trends, independent of any specific numerical example. Using randomized but physically constrained inputs, the models are verified against invariant properties such as:

- Increasing resistivity leads to decreasing water saturation

- Increasing porosity leads to increasing permeability

- Increasing shale volume increases shale-corrected water saturation

- Limiting cases (e.g., 𝑉𝑠ℎ→0, 𝑅𝑡→∞, 𝑆𝑤→1) converge to physically meaningful results

This approach provides strong guarantees that the models remain physically consistent even outside hand-picked test cases and is particularly effective at identifying subtle numerical or logical errors.

#### 4. Cross-Model Consistency

Where applicable, cross-model relationships are validated (e.g., shaly-sand models converging to Archie behavior in clean formations). This ensures internal coherence across the petrophysical workflow and reinforces interpretational reliability.

  
## Dataset

The stoneforge dataset comprises the following data: 

Four .las data from wildcat wells in the National Petroleum Reserve in Alaska. Those where drilled in the National Petroleum Reserve in Alaska (NPRA). </br>
[USGS Well Index](https://pubs.usgs.gov/of/1999/ofr-99-0015/Wells/WellIdx.htm)

Two .dlis data from the Mississippi Fan (Gulf of Mexico) from the DSDP (Deep Sea Drilling Project); DSDP Leg 96 - Hole 616; Processed and Original data. </br>
[DSDP Leg 96 - Hole 616](https://mlp.ldeo.columbia.edu/data/dsdp/leg96/616/)

Zihlman, F. N creator ; Oliver, H. L ; Geological Survey (U.S.)
Reston, Va. : Denver, Colo : U.S. Dept. of the Interior, U.S. Geological Survey1999
