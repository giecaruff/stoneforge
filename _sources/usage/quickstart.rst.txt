=========================
Quickstart
=========================

.. code-block:: bash

   pip install stoneforge

.. code-block:: python

   from stoneforge.data_management.preprocessing import DataLoader
   from stoneforge.petrophysics.shale_volume import vshale_larionov_old

   # Import a LAS file
   # This will return a DataFrame with the data and a dictionary with the units
   data,units = las_import("..//stoneforge//datasets/DP1.las")
   las2 = DataLoader(r"DP1.las", filetype='las2')

   print('header itens:',las2.data_obj.header.keys())
   las2.data_obj.header['well']

   data_las2, units_las2 = las2.dataframe(las2.data_obj.data)
   data_las2

   # Remove non sampling rows
   data_c = data[~data.isin([-999.0]).any(axis=1)]

   # Calculate shale volume
   GR = np.array(data_c["GR"])
   VSH = vshale_larionov_old(gr=GR,grmin=40,grmax=120)