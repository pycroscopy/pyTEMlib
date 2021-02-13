""" dft simulations tools

Part of pyTEMlib
by Gerd Duscher
created 10/29/2020

Supports the conversion of DFT data to simulated EELS spectra

- exciting_get_spectra:  importing dielectric function from the exciting program
- final_state_broadening: apply final state broadening to loss-spectra
"""

import numpy as np
from lxml import etree


def exciting_get_spectra(file):
    """get EELS spectra from exciting calculation"""

    tags = {'data': {}}

    tree = etree.ElementTree(file=file)
    root = tree.getroot()

    data = tags['data']

    if root.tag in ['loss', 'dielectric']:
        print(' reading ', root.tag, ' function from file ', file)
        # print(root[0].tag, root[0].text)
        map_def = root[0]
        i = 0
        v = {}
        for child_of_root in map_def:
            data[child_of_root.tag] = child_of_root.attrib
            v[child_of_root.tag] = []
            i += 1

        for elem in tree.iter(tag='map'):
            m_dict = elem.attrib
            for key in m_dict:
                v[key].append(float(m_dict[key]))

        for key in data:
            data[key]['data'] = np.array(v[key])
        data['type'] = root.tag+' function'
        return tags


def final_state_broadening(x, y, start, instrument):
    """Final state smearing of ELNES edges

    Parameters
    ----------
    x: numpy array
        x or energy loss axis of density of states
    y: numpy array
        y or intensity axis of density of states
    start: float
        start energy of edge
    instrument: float
        instrument broadening

    Return
    ------
    out_data: numpy array
        smeared intensity according to final state and instrument broadening
    """

    # Getting the smearing
    a_i = 107.25*5
    b_i = 0.04688*2.
    x = np.array(x)-start
    zero = int(-x[0]/(x[1]-x[0]))+1
    smear_i = x*0.0
    smear_i[zero:-1] = (a_i/x[zero:-1]**2)+b_i*np.sqrt(x[zero:-1])
    h_bar = 6.58e-16  # h/2pi
    pre = 1.0
    m = 6.58e-31
    smear = x*0.0
    smear[zero:-1] = pre*(h_bar/(smear_i[zero:-1]*0.000000001))*np.sqrt((2*x[zero:-1]*1.6E-19)/m)

    def lorentzian(xx, pp):
        yy = ((0.5 * pp[1]/3.14)/((xx-pp[0])**2 + ((pp[1]/2)**2)))
        return yy/sum(yy)

    p = [0, instrument]
    in_data = y.copy()
    out_data = np.array(y)*0.0
    for i in range(zero+5, len(x)):
        p[0] = x[i]
        p[1] = smear[i]/1.0
        lor = lorentzian(x+1e-9, p)
        out_data[i] = sum(in_data*lor)
        if np.isnan(out_data[i]):
            out_data[i] = 0.0

    p[1] = instrument
    in_data = out_data.copy()
    for i in range(zero-5, len(x)):
        p[0] = x[i]
        lor = lorentzian(x+1e-9, p)
        out_data[i] = sum(in_data*lor)
        # print(out_data[i],in_data[i], lor[i],in_data[i-1], lor[i-1], )
    return out_data
