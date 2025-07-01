=========================
Quickstart
=========================

.. code-block:: bash

   pip install stoneforge

   (...)

.. code-block:: python

   from stoneforge.preprocessing import las_import
   from stoneforge.petrophysics import shale_volume, porosity, water_saturation

   # Import a LAS file
   # This will return a DataFrame with the data and a dictionary with the units
   data,units = las_import("..//stoneforge//datasets/DP1.las")

   # Remove non sampling rows
   data_c = data[~data.isin([-999.0]).any(axis=1)]

   # Calculate shale volume
   GR = np.array(data_c["GR"])
   GR_min = np.percentile(GR, 10)
   GR_max = np.percentile(GR, 90)

   VSH = shale_volume.vshale_larionov(GR, GR_min, GR_max, "larionov")