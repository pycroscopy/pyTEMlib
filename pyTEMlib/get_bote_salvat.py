""" Convert Bote and Salvat cross sections to json format
    Data from:  Bote and Salvat (1998)  Atomic Data and Nuclear Data Tables 71, 1-15
"""


import json
import os
import sys

import numpy as np

sys.path.insert(0, './')
import pyTEMlib
print(pyTEMlib.__version__)

def write_bote_salvat_json():
    """ Convert Bote and Salvat cross sections to json format"""
    line = 'not empty'
    x_sec = {}
    with open('.//data//Bote_Salvat.txt', 'r', encoding='utf-8') as f:
        while line:
            line = f.readline()
            if 'BoteSalvatElementDatum' in line:
                ele = line.split('[')
                z = int(ele[0].split('(')[1][:-2])
                be = []
                for value in ele[1].split(']')[0].split(','):
                    be.append(float(value))
                x_sec[z]= {'Be': be}
                current_x = x_sec[z]
            else:
                if 'Anlj' not in current_x.keys():
                    alij = []
                    for value in line.split('[')[1].split(']')[0].split(','):
                        alij.append(float(value))
                    current_x['Anlj'] = alij
                elif 'G' not in current_x.keys():
                    g = []
                    gg = []
                    g_lines =line.split('[')[1].split(']')[0].split(';')
                    for g_line in g_lines:
                        g = []
                        for value in g_line.split(' '):
                            if value != '':
                                g.append(float(value))
                        gg.append(g)
                    current_x['G'] = gg
                elif 'edge' not in current_x.keys():
                    edge = []
                    for value in line.split('[')[1].split(']')[0].split(','):
                        edge.append(float(value))
                    current_x['edge'] = edge
                elif 'A' not in current_x.keys():
                    a = []
                    aa = []
                    a_lines =line.split('[')[1].split(']')[0].split(';')
                    for a_line in a_lines:
                        a = []
                        for value in a_line.split(' '):
                            if value != '':
                                a.append(float(value))
                        aa.append(a)
                    current_x['A'] = aa
        for key, value in x_sec[3].items():
            print(key, np.array(value))


    file_name_out = os.path.join(pyTEMlib.config_path, 'Bote_Salvat.json')
    json.dump(x_sec, open(file_name_out, 'w', encoding='utf-8'), indent=4)
