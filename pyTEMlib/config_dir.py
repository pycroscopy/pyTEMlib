# -*- coding: utf-8 -*-
# Gleaned from
# Copyright © 2007 Francisco Javier de la Peña
# file of EELSLab.
#
"""
config_dir: setup of directory ~/.pyTEMlib for custom sources and database
"""
import os
import shutil

config_files = ['microscopes.csv', 'edges_db.csv', 'edges_db.pkl', 'fparam.txt']
data_path = os.sep.join([os.path.dirname(__file__), 'data'])

if os.name == 'posix':
    config_path = os.path.join(os.path.expanduser('~'), '.TEMlib')
    os_name = 'posix'
elif os.name in ['nt', 'dos']:
    config_path = os.path.join(os.path.expanduser('~'), '.TEMlib')
    os_name = 'windows'
else:
    config_path = '.'
if os.path.isdir(config_path) is False:
    # messages.information("Creating config directory: %s" % config_path)
    os.mkdir(config_path)

for file in config_files:
    templates_file = os.path.join(data_path, file)
    config_file = os.path.join(config_path, file)
    if os.path.isfile(config_file) is False:
        shutil.copy(templates_file, config_file)
