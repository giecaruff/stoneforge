from typing import Tuple

import numpy as np
from numpy.core.multiarray import normalize_axis_index
from numpy.typing import NDArray


def sobel(arr: NDArray, axes: Tuple[int, int] = (-2, -1)) -> NDArray:
    if arr.ndim != 2:
        raise NotImplementedError
    if any(
        normalize_axis_index(ax, arr.ndim) != i
        for i, ax in zip(range(2), axes)
    ):
        raise NotImplementedError

    # Define our filter kernels. Notice they inherit the input type, so
    # that a float32 input never has to be cast to float64 for computation.
    # But can you see where using another dtype for Gx and Gy might make
    # sense for some input dtypes?
    Gx = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=arr.dtype,
    )
    Gy = np.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=arr.dtype,
    )

    # Create the output array and fill with zeroes
    arr = np.pad(arr, (1,))  # After padding, it is shaped (nx + 2, ny + 2)
    s = np.zeros_like(arr)
    s = np.zeros_like(arr)
    for ix in range(1, arr.shape[0] - 1):
        for iy in range(1, arr.shape[1] - 1):
            # Pointwise multiplication followed by sum, aka convolution
            s1 = (Gx * arr[ix - 1 : ix + 2, iy - 1 : iy + 2]).sum()
            s2 = (Gy * arr[ix - 1 : ix + 2, iy - 1 : iy + 2]).sum()
            s[ix, iy] = np.hypot(s1, s2)  # np.sqrt(s1**2 + s2**2)
    return s