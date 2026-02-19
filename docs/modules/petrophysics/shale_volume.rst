=========================
Shale Volume
=========================

The library implements a set of established shale volume (Vshale) estimation techniques widely used in petrophysical interpretation. These methods derive shale content from different logging measurements and reflect distinct theoretical assumptions about rock response, mineralogy, and compaction effects. Collectively, they provide flexible alternatives suitable for a variety of geological settings and data availability scenarios.

Gamma ray–based approaches constitute the primary group of models. The Gamma Ray Index (IGR) serves as the fundamental normalization step, scaling the measured gamma ray log between clean (GRmin) and shale (GRmax) reference values. From this index, several transformation models are provided:

- Linear Model – Assumes a direct proportional relationship between gamma ray response and shale fraction. It is simple, robust, and commonly used as a baseline estimator.

- Larionov Model (Young Rocks) – Introduces a nonlinear correction to account for the typical gamma ray behavior of Tertiary or weakly compacted formations, where the linear assumption tends to overestimate shale volume.

- Larionov Model (Old Rocks) – Adapts the nonlinear response for older, more compacted lithologies, reflecting reduced gamma ray sensitivity with increasing diagenesis.

- Clavier Model – Applies an empirical nonlinear transformation designed to moderate shale volume estimates, particularly in formations exhibiting intermediate gamma ray responses.

- Stieber Model – Incorporates a correction that limits shale volume inflation at higher gamma ray index values, often producing more conservative estimates in laminated or dispersed shale systems.

Beyond gamma ray methods, the library includes multi-log and specialized estimators:

- Neutron–Density Method (Neutron–Density Crossplot) – Computes shale volume using neutron porosity (NPHI) and bulk density (RHOB) measurements via a three-point mixing model. This approach is particularly useful where gamma ray logs are unreliable or where lithology effects must be explicitly handled.

- NMR-Based Method – Estimates shale volume from the relationship between total and effective porosity derived from nuclear magnetic resonance (NMR) data, leveraging differences in fluid and bound water responses.

By offering multiple formulations, the library enables users to select methods consistent with formation age, mineralogical complexity, logging conditions, and calibration strategy. This multi-model design supports comparative analysis and uncertainty assessment in shale volume interpretation workflows.

Shale Volume
----------------

.. automodule:: stoneforge.petrophysics.shale_volume
    :members:
    :undoc-members:
    :show-inheritance:

References
----------------

.. footbibliography::