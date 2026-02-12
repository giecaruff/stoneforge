=========================
Water Saturation
=========================

The library implements a suite of widely adopted water saturation (Sw) models derived from resistivity and porosity measurements. These formulations cover both clean and shaly reservoir scenarios, reflecting distinct assumptions about conductive pathways, clay effects, and rock electrical behavior. The available methods enable consistent saturation evaluation across a broad range of lithological and petrophysical conditions.

- The Archie Equation constitutes the fundamental clean-formation model. It relates water saturation to true formation resistivity (Rt), porosity (φ), and water resistivity (Rw), governed by the tortuosity factor (a), cementation exponent (m), and saturation exponent (n). Archie’s model assumes that the rock matrix is non-conductive and that electrical conduction occurs exclusively through the formation water. As such, it is most reliable in clean, clay-free reservoirs where surface conductivity is negligible.

For shaly formations, the library provides models that explicitly incorporate clay conductivity effects:

- Simandoux Model – Extends Archie’s formulation by accounting for the parallel conductive contribution of shale. The model integrates shale volume (Vsh) and shale resistivity (Rsh), making it suitable for dispersed clay systems and laminated shaly sands. It remains one of the most commonly applied shaly-sand saturation equations.

- Indonesia (Poupon–Leveaux) Model – Introduces a nonlinear coupling between matrix and shale conductivity terms. This formulation often produces more stable results in formations with moderate-to-high shale content and is particularly valued where Simandoux may overestimate Sw.

- Fertl Model – A simplified shaly-formation model designed to reduce sensitivity to shale resistivity uncertainties. Instead of requiring explicit Rsh input, the method uses an empirical correction term controlled by the alpha parameter (α). This approach is operationally convenient when shale resistivity is poorly constrained or unavailable.

All shale-corrected models require effective porosity rather than total porosity, reflecting the need to isolate the interconnected pore space contributing to fluid flow and electrical conduction.

To streamline interpretation workflows, the library exposes a façade function that unifies all saturation models under a single interface. This design allows users to select the appropriate equation based on reservoir type, shale content, and data quality while preserving parameter consistency.

Water Saturation
----------------

.. automodule:: stoneforge.petrophysics.water_saturation
    :members:
    :undoc-members:
    :show-inheritance:

References
----------------

.. footbibliography::