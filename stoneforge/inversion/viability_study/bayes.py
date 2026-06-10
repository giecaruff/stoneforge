def get_posterior_lithologies_prob(lithologies_distributions, lithologies_priori_probabilities):
    """Calculate posterior lithologies probabilities using Bayes' Theorem for seismic inversion viability study.

    Bayes' Theorem Formula:
    P(Lithology | Data) = P(Data | Lithology) * P(Lithology) / P(Data)

    Parameters
    ----------
    lithologies_distributions : list
        List of lithologies probability density functions (PDFs).
        These represent P(Data | Lithology), the likelihood of the observed data given each lithology.

    lithologies_priori_probabilities : list
        List of prior probabilities for each lithology.
        These represent P(Lithology), the probability of each lithology before observing the data.

    Returns:
    -------
    list
        List of posterior lithologies probabilities (P(Lithology | Data)).


    Examples:
    --------
    >>> lith_distributions = [pdf1, pdf2, pdf3]
    >>> lith_priors = [prob1, prob2, prob3]
    >>> get_posterior_lithologies_prob(lith_distributions, lith_priors)
    [post_prob1, post_prob2, post_prob3]
    """
    # Validate inputs: lengths must match
    if len(lithologies_distributions) != len(lithologies_priori_probabilities):
        raise ValueError("lithologies_distributions and lithologies_priori_probabilities must have the same length")

    # Calculate numerators: P(Data | Lithology) * P(Lithology)
    numerators = [p_conditional * p_lithology for p_conditional, p_lithology in zip(lithologies_distributions, lithologies_priori_probabilities)]

    # Calculate the denominator: P(Data)
    p_data = sum(numerators)

    # If the evidence is numerically zero (possible underflow), fall back to normalized priors
    if p_data == 0.0:
        total_prior = sum(lithologies_priori_probabilities)
        if total_prior > 0.0:
            return [p / total_prior for p in lithologies_priori_probabilities]
        # If priors also sum to zero, return uniform probabilities
        n = max(1, len(lithologies_priori_probabilities))
        return [1.0 / n] * n

    # Calculate posterior probabilities: P(Lithology | Data) = (P(Data | Lithology) * P(Lithology)) / P(Data)
    posterior_probabilities = [numerator / p_data for numerator in numerators]

    return posterior_probabilities