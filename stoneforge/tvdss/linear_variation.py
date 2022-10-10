import numpy as np


def md_to_tvd(MD, md, incl, tvd):
    """
    Determines the True Vertical Depth (TVD) in well resolution using the linear variation method.

    Parameters
    ----------
    MD : array_like
        Default Masured Depth from well.

    Returns
    -------
    md : array_like
        Measured Depth from survey file.

    inlc : array_like
        Incliation relative to vertical of wellpath from survey file.

    tvd : array_like
        True vertical depth from survey file.

    References
    ----------
    .. [1] Mason, C.M. and Taylor, H.L.: "A Systematic Approach to WellSurveying Calculations," SPEJ (June 1971). doi:10.2118/3362-PA

    """
    TVD = np.empty(len(MD))

    for i in range(len(md)-1):
        w = (MD >= md[i])*(MD <= md[i+1])
        if incl[i] == incl[i+1]:
            theta = np.ones(np.sum(w), dtype=np.float64)*incl[i]
            TVD[w] = tvd[i] + (MD[w] - md[i])*np.cos(theta)
        else:
            theta = (incl[i]*(md[i+1] - MD[w]) + incl[i+1]*(MD[w] - md[i]))/(md[i+1] - md[i])
            TVD[w] = tvd[i] + (md[i+1] - md[i])*(np.sin(theta) - np.sin(incl[i]))/(incl[i+1] - incl[i])

    return TVD