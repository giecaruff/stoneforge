=========================
Petrophysics
=========================

**About:** This section contains documentation for the Petrophysics module. We indicate the reference (Pt-Br) of :footcite:t:`freire2020youtube` for the methods used in this module.

This module implements a comprehensive suite of petrophysical models for porosity, shale volume, water saturation, and permeability, following well-established formulations from the geoscience and reservoir engineering literature (e.g., Archie, Simandoux, Indonesia, Larionov, Clavier, Timur, Coates, Tixier).

Beyond numerical implementation, the library adopts a multi-layered scientific validation strategy designed to ensure correctness, numerical robustness, and physical consistency across scalar and log-scale data. The testing framework is structured into four complementary levels:

1. Analytical (Equation-Based) Validation

Each petrophysical model is validated against its analytical formulation using deterministic unit tests. For fixed, physically meaningful inputs, the numerical output is compared directly to the closed-form equation, ensuring exact correspondence with the theoretical model. This guarantees that the implementation faithfully reproduces the published equations and assumptions.

2. Vectorization and Numerical Behavior

Because petrophysical data are inherently log-based and vectorized, all models are tested using NumPy arrays to verify correct broadcasting, array-wise computation, and performance on realistic datasets. These tests explicitly check:

- Vectorized execution (arrays in → arrays out)

- Correct propagation of NaN values

- Numerical stability across valid physical ranges

- Enforcement of physical bounds (e.g., 0 ≤ 𝜙, 𝑉𝑠ℎ, 𝑆𝑤 ≤ 1; 𝑘 ≥ 0)

This level ensures the models behave reliably when applied to full well logs rather than isolated scalar values.

3. Physical Invariants and Limiting Behavior (Property-Based Testing)

Advanced property-based testing is employed to validate physical laws and monotonic trends, independent of any specific numerical example. Using randomized but physically constrained inputs, the models are verified against invariant properties such as:

- Increasing resistivity leads to decreasing water saturation

- Increasing porosity leads to increasing permeability

- Increasing shale volume increases shale-corrected water saturation

- Limiting cases (e.g., 𝑉𝑠ℎ→0, 𝑅𝑡→∞, 𝑆𝑤→1) converge to physically meaningful results

This approach provides strong guarantees that the models remain physically consistent even outside hand-picked test cases and is particularly effective at identifying subtle numerical or logical errors.

4. Cross-Model Consistency

Where applicable, cross-model relationships are validated (e.g., shaly-sand models converging to Archie behavior in clean formations). This ensures internal coherence across the petrophysical workflow and reinforces interpretational reliability.

.

.. toctree::
   :maxdepth: 1

   shale_volume
   porosity
   water_saturation
   permeability
   net_pay

References
----------------

.. footbibliography::