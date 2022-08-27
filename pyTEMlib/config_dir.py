# -*- coding: utf-8 -*-
# part of pyTEMlib
# 2022/08 Changed this to be hopefully compatible with conda-forge
#
#
"""
config_dir: setup of directory ~/.pyTEMlib for custom sources and database
"""
import os
import numpy as np

# import wget
if os.name == 'posix':
    config_path = os.path.join(os.path.expanduser('~'), '.pyTEMlib')
    os_name = 'posix'
elif os.name in ['nt', 'dos']:
    config_path = os.path.join(os.path.expanduser('~'), '.pyTEMlib')
    os_name = 'windows'
else:
    config_path = '.'

if os.path.isdir(config_path) is False:
    # messages.information("Creating config directory: %s" % config_path)
    os.mkdir(config_path)

lines = ['Microscope,E0,alpha,beta,pppc,correlation_factor,VOA_conv,EELS_b1,EELS_b2,EELS_b100,MADF_offset,MADF_slope,'
         'HADF_offset,HADF_slope,BF_offset,BF_slope',
         'Libra 200,2.00E+05,10,15,1,1,6241509.647,0.0634,0.0634,0.0634,0,0,0,0,0,0',
         'UltraSTEM 60,6.00E+04,30,50,1,1,1.79E+07,0.2,0.45,0.9,0.001383,4.04E-06,0,0,0,0',
         'UltraSTEM 100,1.00E+05,30,50,1,1,6.24E+06,0.45,1,2,0,0,0,0,0,0',
         'UltraSTEM 200,2.00E+05,30,50,1,1,6.24E+06,0.45,1,2,0,0,0,0,0,0']

config_file = os.path.join(config_path, 'microscopes.csv')
if os.path.isfile(config_file) is False:  # Do not overwrite users microscopy files
    with open(config_file, 'w') as f:
        f.write('\n'.join(lines))

""" 
import pickle
from pkg_resources import resource_filename

data_path = resource_filename(__name__, 'data')
pkl_file = open(data_path + '/old/edges_db.pkl', 'rb')
x_sections = pickle.load(pkl_file)
pkl_file.close()
for key in range(80, 83):
    print(f'\'{key}\': ', x_sections[str(key)], ',')

# config_file = os.path.join(config_path, 'edges_db.pkl')
ref = importlib_resources.files('pyTEMlib') / 'data/edges_db.pkl'
with importlib_resources.as_file(ref) as templates_file:
    if os.path.isfile(config_file) is False:
        try:
            shutil.copy(templates_file, config_file)
        except FileNotFoundError:
            pass
            # wget('https://github.com/pycroscopy/pyTEMlib/tree/main/pyTEMlib/data/'+file)
"""
