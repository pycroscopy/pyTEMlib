# -*- coding: utf-8 -*-
# part of pyTEMlib
# 2022/08 Changed this to be hopefully compatible with conda-forge
#
#
"""
config_dir: setup of directory ~/.pyTEMlib for custom sources and database
"""
import os
import importlib
import shutil

if os.name in ['nt', 'dos', 'posix']:
    config_path = os.path.join(os.path.expanduser('~'), '.pyTEMlib')
else:
    config_path = '.'

origin_path = os.path.join(importlib.resources.files('pyTEMlib'), 'data')

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
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

for filename in os.listdir(origin_path):
    source_file = os.path.join(origin_path, filename)
    target_file = os.path.join(config_path, filename)
    if os.path.isfile(source_file)is False:
        continue
    if os.path.isfile(target_file) is False:  # Do not overwrite users files
        shutil.copy(source_file, target_file)
        print(f"copied from {origin_path}: {filename}")
