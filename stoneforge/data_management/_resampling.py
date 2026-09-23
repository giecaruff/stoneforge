import pandas as pd
import numpy as np
from typing import Annotated

def resampling(
        dataframe: Annotated[pd.DataFrame, "Dataframe of well data"],
        depth: Annotated[str, "Depth mnemonic"],
        step: Annotated[float, "Data sampling step"] = 1.0,
        top: Annotated[float, "Top depth"] = None,
        bottom: Annotated[float, "Bottom depth"] = None,
        mode: Annotated[str, "Resampling mode"] = "nearest"):
    
    """
    Resample well log data to a regular depth grid.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing well log data, with a depth column.
    depth : str
        Name of the depth column in the DataFrame.
    step : float
        Desired depth step for resampling (e.g., 0.5 for 0.5 m).
    top : float or None
        Top depth for resampling. If None, uses minimum depth in data.
    bottom : float or None
        Bottom depth for resampling. If None, uses maximum depth in data.
    mode : str
        Resampling mode. Options:
        - "nearest": Nearest neighbor (no interpolation, just pick closest sample)
        - "mean": Mean of samples within depth bin
        - "weighted_mean": Weighted mean of samples within depth bin (weights = inverse distance to center)
        - "least_squares": Fit a line to samples within depth bin and evaluate at center depth

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with regular depth intervals.
    """

    df = dataframe.copy()

    # --- Apply depth window ---
    if top is not None or bottom is not None:
        if top is None:
            top = df[depth].min()
        if bottom is None:
            bottom = df[depth].max()

        if top > bottom:
            top, bottom = bottom, top

        df = df[(df[depth] >= top) & (df[depth] <= bottom)]

        if df.empty:
            raise ValueError("No data in the specified depth range.")

    # --- Sort ---
    df = df.sort_values(by=depth).reset_index(drop=True)

    # --- New depth grid ---
    #dmin, dmax = df[depth].min(), df[depth].max()
    new_depth = np.arange(top, bottom + step, step)

    old_depth = df[depth].values

    # --- Helper for categorical columns ---
    def most_common(series):
        return series.mode().iloc[0] if not series.mode().empty else np.nan

    # --- NEAREST (unchanged) ---
    if mode == "nearest":
        idx = np.searchsorted(old_depth, new_depth)
        idx[idx == len(old_depth)] = len(old_depth) - 1
        prev_idx = np.maximum(idx - 1, 0)

        choose_prev = np.abs(new_depth - old_depth[prev_idx]) < np.abs(new_depth - old_depth[idx])
        final_idx = np.where(choose_prev, prev_idx, idx)

        resampled_df = df.iloc[final_idx].copy()
        resampled_df[depth] = new_depth
        return resampled_df.reset_index(drop=True)

    # --- BIN-BASED MODES ---
    results = []

    for d in new_depth:
        lower = d - step / 2
        upper = d + step / 2

        window = df[(df[depth] >= lower) & (df[depth] <= upper)]

        # fallback if empty → nearest
        if window.empty:
            idx = np.abs(old_depth - d).argmin()
            row = df.iloc[idx].copy()
            row[depth] = d
            results.append(row)
            continue

        new_row = {}

        for col in df.columns:
            if col == depth:
                new_row[col] = d
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                values = window[col].values
                depths = window[depth].values

                if mode == "mean":
                    new_row[col] = np.mean(values)

                elif mode == "weighted_mean":
                    # weight = inverse distance to center
                    dist = np.abs(depths - d)
                    weights = 1 / (dist + 1e-6)
                    new_row[col] = np.sum(values * weights) / np.sum(weights)

                elif mode == "least_squares":
                    if len(values) == 1:
                        new_row[col] = values[0]
                    else:
                        # linear fit (degree 1)
                        coeffs = np.polyfit(depths, values, 1)
                        new_row[col] = np.polyval(coeffs, d)

                else:
                    raise NotImplementedError(f"Mode '{mode}' not implemented.")

            else:
                # categorical/string → most frequent
                new_row[col] = most_common(window[col])

        results.append(new_row)

    return pd.DataFrame(results)