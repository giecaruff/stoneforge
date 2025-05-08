import os

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