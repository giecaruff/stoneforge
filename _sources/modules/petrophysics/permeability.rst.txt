=========================
Permeability
=========================

The library incorporates several classical empirical permeability (k) estimators derived from porosity, water saturation, and resistivity measurements. These models reflect historically validated correlations between rock storage capacity, fluid distribution, and flow properties, providing practical permeability proxies in the absence of core data. While empirical in nature, these methods remain widely used for quick-look evaluations and comparative reservoir analysis.

Porosity–saturation–based formulations represent the primary class of estimators:

- **Timur (1968) Model** – A foundational empirical relationship linking permeability to porosity (φ) and irreducible water saturation (Sw). The model captures the intuitive dependence of permeability on pore volume and fluid occupancy, typically predicting higher permeability for well-connected pore systems exhibiting low water saturation.

- **Coates (1981) Model** – A refinement of the Timur-type approach, incorporating similar dependencies but calibrated to better reflect movable fluid fractions and pore geometry effects. The formulation is frequently applied in conjunction with NMR-derived parameters, although it can be used with conventional porosity and saturation logs.

Resistivity-integrated estimators extend permeability prediction by exploiting electrical responses:

- **Coates–Dumanoir (1974) Model** – Combines deep formation resistivity (Rt or ResD) with porosity to estimate permeability. This model implicitly links pore structure and fluid distribution to electrical behavior, offering an alternative pathway when saturation estimates are uncertain or unavailable. Proper calibration is essential, as unit consistency and formation-specific constants strongly influence results.

- **Tixier (1949) Model** – Utilizes the contrast between deep and shallow resistivity measurements to infer permeability through invasion-profile behavior. The method is based on the premise that permeability governs mud filtrate invasion dynamics; therefore, resistivity differentials can serve as indirect indicators of pore connectivity and transmissibility.

These empirical methods do not replace laboratory-derived permeability but provide operationally useful approximations, particularly during early-stage interpretation, interval ranking, or uncertainty screening. Their reliability depends on lithology, pore type, fluid system, and calibration against representative core or test data.

Permeability
----------------

.. automodule:: stoneforge.petrophysics.permeability
    :members:
    :undoc-members:
    :show-inheritance:

References
----------------

.. footbibliography::