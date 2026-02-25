=========================
Porosity
=========================

The library provides a set of conventional porosity estimation techniques derived from density, neutron, and sonic well logs, as well as hybrid formulations designed to mitigate lithology and shale effects. These methods reflect standard petrophysical models and enable users to compute both total and effective porosity under varying geological and logging conditions.

- Density-based porosity calculations rely on the bulk density response of the formation. The Density Porosity model estimates total porosity from the contrast between the measured bulk density (RHOB) and assumed matrix (ρ_ma) and fluid (ρ_f) densities. This formulation is grounded in the volumetric mixing law and is particularly effective in clean formations where matrix properties are reasonably constrained.

- Neutron-derived porosity exploits the sensitivity of neutron tools to hydrogen concentration. The Neutron Porosity estimator corrects the raw neutron response for shale effects by incorporating shale volume (Vsh) and an apparent shale porosity term. This correction is essential because bound water in clays can significantly inflate neutron readings, leading to overestimated porosity if untreated.

- Sonic-based porosity is computed using the Wyllie Time-Average Equation, which models acoustic transit time (Δt) as a linear combination of matrix and fluid contributions. Sonic porosity is often useful in consolidated formations but may require caution in poorly compacted or complex lithologies where the time-average assumption breaks down.

To improve robustness and reduce tool-specific biases, the library includes combined estimators:

- Neutron–Density Porosity – Computes effective porosity from the joint response of density and neutron logs. Depending on configuration, the estimator uses either the arithmetic mean or a root-mean-square formulation, helping stabilize results where individual logs are affected by lithology or fluid variations.

- Gaymard–Poupon Method – A crossplot-inspired correction that integrates neutron and density porosities to better resolve shale-free porosity. This method is widely applied in shaly sand interpretations to compensate for systematic tool deviations.

Effective porosity calculations are also explicitly supported:

- Effective Porosity – Derives shale-corrected porosity from total porosity and shale volume, accounting for the non-reservoir pore space associated with clays. This adjustment is critical for volumetric evaluations and saturation models.

For workflow simplification, the library exposes a façade function that unifies all porosity models under a single interface. This design allows users to select estimation methods based on available logs, formation characteristics, and interpretation strategy while maintaining methodological consistency across analyses.

Porosity
----------------

.. automodule:: stoneforge.petrophysics.porosity
    :members:
    :undoc-members:
    :show-inheritance:

References
----------------

.. footbibliography::