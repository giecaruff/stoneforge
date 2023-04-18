import numpy as np
import platform
import pickle
import json
import os
import pandas

from . import las2

class project():
    
    def __init__(self,data_path):
        
        self.project = {}
        self.data_path = data_path
        self.outpath = '.'
        self.well_names_paths = {}
        self.well_data = {}
        self.well_names_las = []
        
    # ============================================ #

    def import_folder(self,ext = '.las'):

        # ------------------------------------ #
        # all paths 
        files = []
        # r=root, d=directories, f = files
        for r, _, f in os.walk(self.data_path):
            for file in f:
                if ext in file:
                    files.append(os.path.join(r, file))

        c_resumo = self.data_path+'\\'


        for i in files:
            n1 = i.replace(c_resumo, '')
            self.well_names_paths[n1.replace(ext,'')] = i

    # ============================================ #

    def import_well(self,name):
        
        # ------------------------------------ #
        
        path = self.well_names_paths[name]
        self.well_names_las.append(name)

        read_data = las2.read(path)

        mnemonic = [a['mnemonic'] for a in read_data['curve']]
        unit = [a['unit'] for a in read_data['curve']]
        self.well_data[name] = {}
 
        for i in range(len(mnemonic)):
            self.well_data[name][mnemonic[i]] = {}
            self.well_data[name][mnemonic[i]]['data'] = read_data['data'][i]
            self.well_data[name][mnemonic[i]]['unit'] = unit[i]

    # ============================================ #

    def import_several_wells(self):

        for name in self.well_names_paths:
            self.import_well(name)

    # ============================================ #

    def data_replacement(self,ref,forced = True):

        mnemonics_list = list(ref.keys())

        new_well_data = {}
        for i in self.well_data:
            new_well_data[i] = {}
            local = {}

            for j in self.well_data[i]:
                new_mnemonic = self._find_mnemonic(j,ref)
                if new_mnemonic:
                    local[new_mnemonic[0]] = self.well_data[i][j]
                else:
                    pass
            new_well_data[i] = local

        self.well_data = new_well_data

    def _find_mnemonic(self,value,ref):

        for i in ref:
            for j in ref[i]:
                if value == j:
                    return i,value

    # ============================================ #

    def convert_into_matrix(self,reference_mnemonics=False):
        """
        converts an manly dictionary database into an
        matrix database with tree values: 
        mnemonics, units and data.
        """

        wells = {}
        for i in self.well_data:
            data = []
            units = []
            mnemonics = []
            well = {}
            if reference_mnemonics:
                well_data = reference_mnemonics
            else:
                well_data = self.well_data[i]

            for j in well_data:
                data.append(self.well_data[i][j]['data'])
                units.append(self.well_data[i][j]['unit'])
                mnemonics.append(j)

            well['mnemonics'] = mnemonics
            well['units'] = units
            well['data'] = np.array(data)
            wells[i] = well

        self.well_data = wells

    # ============================================ #

    def class_counts(self,class_value,seed = 99):

        np.random.seed(seed)

        n_class = list(set(class_value))
        class_count = {}
        for c in n_class:
            class_count[c] = {}
            class_count[c]['name'] = c
            class_count[c]['color'] = str(tuple(np.random.choice(range(256), size=3)))
            counts = 0
            for i in class_value:
                if i == c:
                    counts += 1
            class_count[c]['value'] = counts

        return class_count

    def shape_check(self,ref):
        """
         If an well has less mnemonics than the others,
         than this function removes this well.
        """

        value = len(ref.keys())

        well_data = {}

        for i in self.well_data:
            if np.shape(self.well_data[i]['data'])[0] == value:
                well_data[i] = self.well_data[i]
            else:
                print("well: '{}'".format(i),"because it has less logs")

        self.well_data = well_data

    # ============================================ #
        


