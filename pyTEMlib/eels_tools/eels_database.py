"""
eels_tools
Model based quantification of electron energy-loss data
Copyright by Gerd Duscher

"""
import numpy as np
import requests



##########################
# EELS Database
##########################


def read_msa(msa_string):
    """read msa formated file"""
    parameters = {}
    y = []
    x = []
    # Read the keywords
    data_section = False
    msa_lines = msa_string.split('\n')

    for line in msa_lines:
        if data_section is False:
            if len(line) > 0:
                if line[0] == "#":
                    try:
                        key, value = line.split(': ')
                        value = value.strip()
                    except ValueError:
                        key = line
                        value = None
                    key = key.strip('#').strip()

                    if key != 'SPECTRUM':
                        parameters[key] = value
                    else:
                        data_section = True
        else:
            # Read the data
            if len(line) > 0 and line[0] != "#" and line.strip():
                if parameters['DATATYPE'] == 'XY':
                    xy = line.replace(',', ' ').strip().split()
                    y.append(float(xy[1]))
                    x.append(float(xy[0]))
                elif parameters['DATATYPE'] == 'Y':
                    print('y')
                    data = [
                        float(i) for i in line.replace(',', ' ').strip().split()]
                    y.extend(data)
    parameters['data'] = np.array(y)
    if 'XPERCHAN' in parameters:
        parameters['XPERCHAN'] = str(parameters['XPERCHAN']).split(' ', maxsplit=1)[0]
        parameters['OFFSET'] = str(parameters['OFFSET']).split(' ', maxsplit=1)[0]
        dispersion = float(parameters['XPERCHAN'])  # eV per channel
        offset = float(parameters['OFFSET'])  # eV offset
        parameters['energy_scale'] = np.arange(len(y)) * dispersion + offset
    return parameters


def get_spectrum_eels_db(formula=None, edge=None, title=None, element=None):
    """
    get spectra from EELS database
    chemical formula and edge is accepted.
    Could expose more of the search parameters
    """
    valid_edges = ['K', 'L1', 'L2,3', 'M2,3', 'M4,5', 'N2,3', 'N4,5', 'O2,3', 'O4,5']
    if edge is not None and edge not in valid_edges:
        print('edge should be a in ', valid_edges)

    params = {"type": None,
              "title": title,
              "author": None,
              "edge": edge,
              "element": element,
              "min_energy": None,
              "max_energy": None,
              "resolution": None,
              "resolution_compare": "lt",
              "monochromated": None,
              "formula": formula,
              "min_energy_compare": "gt",
              "max_energy_compare": "lt",
              "per_page": -1,
              "order": None,
              "order_direction": "ASC"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}
    request = requests.get("https://api.eelsdb.eu/spectra", params=params, verify=True,
                           headers=headers, timeout=10)
    jsons = request.json()
    if "message" in jsons:
        # Invalid query, EELSdb raises error.
        raise IOError(f"Error: {jsons['message']}")
    reference_spectra = {}
    for json_spectrum in jsons:
        download_link = json_spectrum['download_link']
        msa_string = requests.get(download_link, verify=True, headers=headers, timeout=10).text
        parameters = read_msa(msa_string)
        if 'XPERCHAN' in parameters:
            reference_spectra[parameters['TITLE']] = parameters
            print(parameters['TITLE'])
    print(f'found {len(reference_spectra.keys())} spectra in EELS database)')

    return reference_spectra
