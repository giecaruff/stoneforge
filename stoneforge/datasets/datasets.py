from ..preprocessing.data_management import project

def alaska_usgs(a = 'a'):
    """
    Import well data from the Alaska USGS dataset.
    """
    proj = project('alaska_usgs')
    proj.import_folder()
    proj.import_several_wells()
    return proj.well_data