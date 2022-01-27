"""
crystal_tools

part of pyTEMlib

Author: Gerd Duscher

Provides convenient functions to make most regular crystal structures

Contains also a dictionary of crystal structures and atomic form factors

Units:
    everything is in SI units, except length is given in nm.
    angles are assumed to be in degree but will be internally converted to rad

Usage:
    See the notebooks for examples of these routines

"""

import numpy as np
import itertools
import ase
import ase.spacegroup
import ase.build
import ase.data.colors

import matplotlib.pylab as plt  # basic plotting

_spglib_present = True
try:
    import spglib
except ModuleNotFoundError:
    _spglib_present = False

if _spglib_present:
    print('Symmetry functions of spglib enabled')
else:
    print('spglib not installed; Symmetry functions of spglib disabled')


# from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
# from matplotlib.patches import Circle  # , Ellipse, Rectangle
# from matplotlib.collections import PatchCollection


def get_dictionary(atoms):
    """
    structure dictionary from ase.Atoms object
    """
    tags = {'unit_cell': atoms.cell.array,
            'elements': atoms.get_chemical_formula(),
            'base': atoms.get_scaled_positions(),
            'metadata': atoms.info}

    return tags


def atoms_from_dictionary(tags):
    atoms = ase.Atoms(cell=tags['unit_cell'],
                      symbols=tags['elements'],
                      scaled_positions=tags['base'])
    if 'metadata' in tags:
        atoms.info = tags['metadata']
    return atoms


def get_symmetry(atoms, verbose=True):
    """
    Symmetry analysis with spglib

    spglib must be installed

    Parameters
    ----------
    atoms: ase.Atoms object
        crystal structure
    verbose: bool

    Returns
    -------

    """
    if _spglib_present:
        if verbose:
            print('#####################')
            print('# Symmetry Analysis #')
            print('#####################')

        base = atoms.get_scaled_positions()
        for i, atom in enumerate(atoms):
            if verbose:
                print(f'{i + 1}: {atom.number} = {2} : [{base[i][0]:.2f}, {base[i][1]:.2f}, {base[i][2]:.2f}]')

        lattice = (atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
        spgroup = spglib.get_spacegroup(lattice, symprec=1e-2)
        sym = spglib.get_symmetry(lattice)

        if verbose:
            print("  Spacegroup  is %s." % spgroup)
            print('  Crystal has {0} symmetry operation'.format(sym['rotations'].shape[0]))

        p_lattice, p_positions, p_numbers = spglib.find_primitive(lattice, symprec=1e-5)
        print("\n########################\n #Basis vectors of primitive Cell:")
        for i in range(3):
            print('[{0:.4f}, {1:.4f}, {2:.4f}]'.format(p_lattice[i][0], p_lattice[i][1], p_lattice[i][2]))

        print('There {0} atoms and {1} species in primitive unit cell:'.format(len(p_positions), p_numbers))
    else:
        print('spglib is not installed')

    return True


def set_bond_radii(atoms):
    bond_radii = np.ones(len(atoms))
    for i in range(len(atoms)):
        bond_radii[i] = electronFF[atoms.symbols[i]]['bond_length'][1]
    atoms.info['bond_radii'] = bond_radii


def jmol_viewer(atoms, size=2):
    """
    jmol viewer of ase .Atoms object
    requires jupyter-jsmol to be installed (available through conda or pip)

    Parameter
    ---------
    atoms: ase.Atoms
        structure info
    size: int, list, or np.array of size 3; default 1
        size of unit_cell; maximum = 8

    Returns
    -------
    view: JsmolView object

    Example
    -------
    from jupyter_jsmol import JsmolView
    import ase
    import ase.build
    import itertools
    import numpy as np
    atoms = ase.build.bulk('Cu', 'fcc', a=5.76911, cubic=True)
    for pos in list(itertools.product([0.25, .75], repeat=3)):
        atoms += ase.Atom('Al', al2cu.cell.lengths()*pos)

    view = plot_ase(atoms, size = 8)
    display(view)
    """
    try:
        from jupyter_jsmol import JsmolView
        from IPython.display import display
    except ImportError:
        print('this function is based on jupyter-jsmol, please install with: \n '
              'conda install -c conda-forge jupyter-jsmol')
        return

    if isinstance(size, int):
        size = [size] * 3

    [a, b, c] = atoms.cell.lengths()
    [alpha, beta, gamma] = atoms.cell.angles()

    view = JsmolView.from_ase(atoms, f"{{{size[0]} {size[1]} {size[2]}}}"
                                     f" unitcell {{{a:.3f} {b:.3f} {c:.3f} {alpha:.3f} {beta:.3f} {gamma:.3f}}}")

    display(view)

    return view


def plot_super_cell(super_cell, shift_x=0.):
    """ make a super_cell to plot with extra atoms at periodic boundaries"""

    if not isinstance(super_cell, ase.Atoms):
        raise TypeError('Need an ase Atoms object')

    super_cell2plot = super_cell * (2, 2, 2)
    super_cell2plot.positions[:, 0] = super_cell2plot.positions[:, 0] - super_cell2plot.cell[0, 0] * shift_x

    del super_cell2plot[super_cell2plot.positions[:, 2] > super_cell.cell[2, 2] + 0.1]
    del super_cell2plot[super_cell2plot.positions[:, 1] > super_cell.cell[1, 1] + 0.1]
    del super_cell2plot[super_cell2plot.positions[:, 0] > super_cell.cell[0, 0] + 0.1]
    del super_cell2plot[super_cell2plot.positions[:, 0] < -0.1]
    super_cell2plot.cell = super_cell.cell * (1, 1, 1)

    return super_cell2plot


def ball_and_stick(atoms, extend=1, max_bond_length=0.):
    """Calculates the data to plot a ball and stick model

    Parameters
    ----------
    atoms: ase.Atoms object
        object containing the structural information like 'cell', 'positions', and 'symbols' .

    extend: integer or list f 3 integers
        The *extend* argument scales the effective cell in which atoms
        will be included. It must either be a list of three integers or a single
        integer scaling all 3 directions.  By setting this value to one,
        all  corner and edge atoms will be included in the returned cell.
        This will of cause make the returned cell non-repeatable, but this is
        very useful for visualisation.

    max_bond_length: 1 float
        The max_bond_length argument defines the distance for which a bond will be shown.
        If max_bond_length is zero, the tabulated atom radii will be used.

    Returns
    -------
    super_cell: ase.Atoms object
        structure with additional information in info dictionary
    """

    if not isinstance(atoms, ase.Atoms):
        raise TypeError('Need an ase Atoms object')

    from ase import neighborlist
    from scipy import sparse
    from scipy.sparse import dok_matrix

    super_cell = plot_super_cell(atoms*extend)
    cell = super_cell.cell.array
    # Corners and Outline of unit cell
    h = (0, 1)
    corner_vectors = np.dot(np.array(list(itertools.product(h, h, h))), cell)
    corner_matrix = dok_matrix((8, 8), dtype=bool)
    trace = [[0, 1], [1, 3], [2, 3], [0, 2], [0, 4], [4, 5], [5, 7], [6, 7], [4, 6], [1, 5], [2, 6], [3, 7]]
    for s, e in trace:
        corner_matrix[s, e] = True

    # List of bond lengths taken from electronFF database below
    bond_lengths = []
    for atom in super_cell:
        bond_lengths.append(electronFF[atom.symbol]['bond_length'][1])

    super_cell.set_cell(cell*2, scale_atoms=False)   # otherwise, corner atoms have distance 0
    neighbor_list = neighborlist.NeighborList(bond_lengths, self_interaction=False, bothways=False)
    neighbor_list.update(super_cell)
    bond_matrix = neighbor_list.get_connectivity_matrix()

    del_double = []
    for (k, s) in bond_matrix.keys():
        if k > s:
            del_double.append((k, s))
    for key in del_double:
        bond_matrix.pop(key)

    if super_cell.info is None:
        super_cell.info = {}
    super_cell.info['plot_cell'] = {'bond_matrix': bond_matrix, 'corner_vectors': corner_vectors,
                                    'bond_length': bond_lengths, 'corner_matrix': corner_matrix}
    super_cell.set_cell(cell/2, scale_atoms=False)
    return super_cell


def plot_unit_cell(atoms, extend=1, max_bond_length=1.0):
    """
    Simple plot of unit cell
    """

    super_cell = ball_and_stick(atoms, extend=extend, max_bond_length=max_bond_length)

    corners = super_cell.info['plot_cell']['corner_vectors']
    positions = super_cell.positions - super_cell.cell.lengths()/2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # draw unit_cell

    for line in super_cell.info['plot_cell']['corner_matrix'].keys():
        ax.plot3D(corners[line, 0], corners[line, 1], corners[line, 2], color="blue")

    # draw bonds
    bond_matrix = super_cell.info['plot_cell']['bond_matrix']
    for bond in super_cell.info['plot_cell']['bond_matrix'].keys():
        ax.plot3D(positions[bond, 0], positions[bond, 1], positions[bond, 2], color="black", linewidth=4)
        # , tube_radius=0.02)

    # draw atoms
    ax.scatter(super_cell.positions[:, 0], super_cell.positions[:, 1], super_cell.positions[:, 2],
               color=tuple(jmol_colors[super_cell.get_atomic_numbers()]), alpha=1.0, s=50)
    maximum_position = super_cell.positions.max()*1.05
    ax.set_proj_type('ortho')

    ax.set_zlim(-maximum_position/2, maximum_position/2)
    ax.set_ylim(-maximum_position/2, maximum_position/2)
    ax.set_xlim(-maximum_position/2, maximum_position/2)

    if 'name' in super_cell.info:
        ax.set_title(super_cell.info['name'])

    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_zlabel('z [Å]')
    return fig


# Jmol colors.  See: http://jmol.sourceforge.net/jscolors/#color_U
jmol_colors = ase.data.colors.jmol_colors


def structure_by_name(crystal_name):
    """
        Provides crystal structure in ase.Atoms format.
        Additional information is stored in the info attribute as a dictionary


        Parameter
        ---------
        crystal_name: str
            Please note that the chemical expressions are not case-sensitive.

        Returns
        -------
        atoms: ase.Atoms
            structure

        Example
        -------
        >> #  for a list of pre-defined crystal structures
        >> import pyTEMlib.crystal_tools
        >> print(pyTEMlib.crystal_tools.crystal_data_base.keys())
        >>
        >> atoms = pyTEMlib.crystal_tools.structure_by_name('Silicon')
        >> print(atoms)
        >> print(atoms.info)

    """

    # Check whether name is in the crystal_data_base
    import ase
    import ase.build

    if crystal_name.lower() in cdb:
        tags = cdb[crystal_name.lower()].copy()
    else:
        print(f'Crystal name {crystal_name.lower()} not defined')
        return

    if 'symmetry' in tags:
        if tags['symmetry'].lower() == 'fcc':
            atoms = ase.build.bulk(tags['elements'], 'fcc', a=tags['a'], cubic=True)

        elif tags['symmetry'].lower() == 'bcc':
            atoms = ase.build.bulk(tags['elements'], 'bcc', a=tags['a'], cubic=True)

        elif tags['symmetry'].lower() == 'diamond':
            import ase.lattice.cubic
            atoms = ase.lattice.cubic.Diamond(tags['elements'], latticeconstant=tags['a'])

        elif 'rocksalt' in tags['symmetry']:  # B1
            import ase.lattice.compounds
            atoms = ase.lattice.compounds.Rocksalt(tags['elements'], latticeconstant=tags['a'])

        elif 'zincblende' in tags['symmetry']:
            import ase.lattice.compounds
            atoms = ase.lattice.compounds.B3(tags['elements'], latticeconstant=tags['a'])

        elif 'B2' in tags['symmetry']:
            import ase.lattice.compounds
            atoms = ase.lattice.compounds.B2(tags['elements'], latticeconstant=tags['a'])

        elif 'graphite' in tags['symmetry']:
            base = [(0, 0, 0), (0, 0, 1/2), (2/3, 1/3, 0), (1/3, 2/3, 1/2)]
            structure_matrix = np.array([[tags['a'], 0., 0.],
                                         [np.cos(np.pi/3*2)*tags['a'], np.sin(np.pi/3*2)*tags['a'], 0.],
                                         [0., 0., tags['c']]])

            atoms = ase.Atoms(tags['elements'], cell=structure_matrix, scaled_positions=base)

        elif 'perovskite' in tags['symmetry']:
            import ase.spacegroup
            atom_positions = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, 0.5, 0.0)]
            atoms = ase.spacegroup.crystal(tags['elements'], atom_positions, spacegroup=221, cellpar=tags['a'])

        elif 'wurzite' in tags['symmetry']:
            import ase.spacegroup
            atom_positions = [(1/3, 2/3, 0.0), (1/3, 2/3, tags['u'])]
            atoms = ase.spacegroup.crystal(tags['elements'], atom_positions, spacegroup=186,
                                           cellpar=[tags['a'], tags['a'], tags['c'], 90, 90, 120])

        elif 'rutile' in tags['symmetry']:
            import ase.spacegroup
            atoms = ase.spacegroup.crystal(tags['elements'], basis=[(0, 0, 0), (0.3, 0.3, 0.0)],
                                           spacegroup=136, cellpar=[tags['a'], tags['a'], tags['c'], 90, 90, 90])
        elif 'dichalcogenide' in tags['symmetry']:
            import ase.spacegroup

            u = tags['u']
            base = [(1 / 3., 2 / 3., 1 / 4.), (2 / 3., 1 / 3., 3 / 4.),
                    (2 / 3., 1 / 3., 1 / 4. + u), (2 / 3., 1 / 3., 1 / 4. - u),
                    (1 / 3., 2 / 3., 3 / 4. + u), (1 / 3., 2 / 3., 3 / 4. - u)]
            atoms = ase.spacegroup.crystal(tags['elements'][0] * 2 + tags['elements'][1] * 4, base, spacegroup=194,
                                           cellpar=[tags['a'], tags['a'], tags['c'], 90, 90, 120])

        elif tags['symmetry'].lower() in ['primitive', 'hexagonal']:
            atoms = ase.Atoms(tags['elements'], cell=tags['unit_cell'], scaled_positions=tags['base'])

        else:
            print(' symmetry of structure is wrong')

    atoms.info = {'structure': {'reference': tags['reference'], 'link': tags['link']},
                  'title': tags['crystal_name']}
    return atoms


# crystal data base cbd
cdb = {'aluminum': {'crystal_name': 'aluminum',
                    'symmetry': 'FCC',
                    'elements': 'Al',
                    'a': 4.05,  # Angstrom
                    'reference': 'W. Witt, Z. Naturforsch. A, 1967, 22A, 92',
                    'link': 'http://doi.org/10.1515/zna-1967-0115'}}
cdb['al'] = cdb['aluminium'] = cdb['aluminum']

cdb['gold'] = {'crystal_name': 'gold',
               'symmetry': 'FCC',
               'elements': 'Au',
               'a': 4.0782,  # Angstrom
               'reference': '',
               'link': ''}
cdb['au'] = cdb['gold']

cdb['silver'] = {'crystal_name': 'silver',
                 'symmetry': 'FCC',
                 'elements': 'Ag',
                 'a': 4.0853,  # Angstrom
                 'reference': '', 'link': ''}
cdb['ag'] = cdb['silver']

cdb['copper'] = {'crystal_name': 'copper',
                 'symmetry': 'FCC',
                 'elements': 'Cu',
                 'a': 4.0853,  # Angstrom
                 'reference': '', 'link': ''}
cdb['cu'] = cdb['copper']

cdb['diamond'] = {'crystal_name': 'diamond',
                  'symmetry': 'diamond',
                  'elements': 'C',
                  'a': 3.5668,  # Angstrom
                  'reference': '', 'link': ''}

cdb['germanium'] = {'crystal_name': 'germanium',
                    'symmetry': 'diamond',
                    'elements': 'Ge',
                    'a': 5.66806348,  # Angstrom for 300K
                    'reference': 'H. P. Singh, Acta Crystallogr., 1968, 24A, 469',
                    'link': 'https://doi.org/10.1107/S056773946800094X'}
cdb['ge'] = cdb['germanium']

cdb['silicon'] = {'crystal_name': 'silicon',
                  'symmetry': 'diamond',
                  'elements': 'Si',
                  'a': 5.430880,  # Angstrom for 300K
                  'reference': 'C. R. Hubbard, H. E. Swanson, and F. A. Mauer, J. Appl. Crystallogr., 1975, 8, 45',
                  'link': 'https://doi.org/10.1107/S0021889875009508'}
cdb['si'] = cdb['silicon']

cdb['gaas'] = {'crystal_name': 'GaAs',
               'symmetry': 'zincblende(B3)',
               'elements': ['Ga', 'As'],
               'a': 5.65325,  # Angstrom for 300K
               'reference': 'J.F.C. Baker, M. Hart, M.A.G. Halliwell, R. Heckingbottom, Solid-State Electronics, 19, '
                            '1976, 331-334,',
               'link': 'https://doi.org/10.1016/0038-1101(76)90031-9'}

cdb['fcc fe'] = {'crystal_name': 'FCC Fe',
                 'symmetry': 'FCC',
                 'elements': 'Fe',
                 'a': 3.3571,  # Angstrom
                 'reference': 'R. Kohlhaas, P. Donner, and N. Schmitz-Pranghe, Z. Angew. Phys., 1967, 23, 245',
                 'link': ''}

cdb['iron'] = {'crystal_name': 'BCC Fe',
               'symmetry': 'BCC',
               'elements': 'Fe',
               'a': 2.866,  # Angstrom
               'reference': 'Z. S. Basinski, W. Hume-Rothery and A. L. Sutton, Proceedings of the Royal Society of '
                            'London. Series A, Mathematical and Physical Sciences Vol. 229, No. 1179 '
                            '(May 24, 1955), pp. 459-467',
               'link': 'http://www.jstor.org/stable/99693'}
cdb['bcc fe'] = cdb['alpha iron'] = cdb['iron']

cdb['srtio3'] = {'crystal_name': 'SrTiO3',
                 'symmetry': 'perovskite',
                 'elements': ['Sr', 'Ti', 'O'],
                 'a': 3.905268,  # Angstrom
                 'reference': 'M. Schmidbauer, A. Kwasniewski and J. Schwarzkopf, Acta Cryst. (2012). B68, 8-14',
                 'link': 'http://doi.org/10.1107/S0108768111046738'}
cdb['strontium titanate'] = cdb['srtio3']

cdb['graphite'] = {'crystal_name': 'graphite',
                   'symmetry': 'graphite hexagonal',
                   'elements': 'C4',
                   'a': 2.46772414,
                   'c': 6.711,
                   'reference': 'P. Trucano and R. Chen, Nature, 1975, 258, 136',
                   'link': 'https://doi.org/10.1038/258136a0'}

cdb['cscl'] = {'crystal_name': 'CsCl',
               'symmetry': 'CsCl (B2)',
               'a': 4.209,  # Angstrom
               'elements': ['Cs', 'Cl'],
               'reference': '', 'link': ''}
cdb['cesium chlorid'] = cdb['cscl']

cdb['mgo'] = {'crystal_name': 'MgO',
              'symmetry': 'rocksalt (B1)',
              'elements': ['Mg', 'O'],
              'a': 4.256483,  # Angstrom
              'reference': '', 'link': ''}

cdb['titanium nitride'] = {'crystal_name': 'TiN',
                           'symmetry': 'rocksalt (B1)',
                           'elements': ['Ti', 'N'],
                           'a': 4.25353445,  # Angstrom
                           'reference': '', 'link': '',
                           'space_group': 225,
                           'symmetry_name': 'Fm-3m'}

cdb['zno wurzite'] = {'crystal_name': 'ZnO Wurzite',
                      'symmetry': 'wurzite',
                      'elements': ['Zn', 'O'],
                      'a': 3.278,  # Angstrom
                      'c': 5.292,  # Angstrom
                      'u': 0.382,
                      'reference': '', 'link': ''}
cdb['zno'] = cdb['wzno'] = cdb['zno wurzite']

cdb['gan'] = {'crystal_name': 'GaN Wurzite',
              'symmetry': 'wurzite',
              'elements': ['Ga', 'N'],
              'a': 3.186,  # Angstrom
              'c': 5.186,  # Angstrom
              'u': 0.376393,
              'reference': '', 'link': ''}
cdb['gan wurzite'] = cdb['wgan'] = cdb['gallium nitride'] = cdb['gan']


cdb['tio2'] = {'crystal_name': 'TiO2 rutile',
               'symmetry': 'rutile',
               'elements': ['Ti', 'O'],
               'a': 4.6,  # Angstrom
               'c': 2.95,  # Angstrom
               'reference': '', 'link': ''}

cdb['mos2'] = {'crystal_name': 'MoS2',
               'symmetry': 'dichalcogenide',
               'elements': ['Mo', 'S'],
               'a': 3.19031573,  # Angstrom
               'c': 14.87900430,  # Angstrom
               'u': 0.105174,
               'reference': '', 'link': ''}

cdb['ws2'] = {'crystal_name': 'WS2',
              'symmetry': 'dichalcogenide',
              'elements': ['W', 'S'],
              'a': 3.19073051,  # Angstrom
              'c': 14.20240204,  # Angstrom
              'u': 0.110759,
              'reference': '', 'link': ''}

cdb['wse2'] = {'crystal_name': 'WSe2',
               'symmetry': 'dichalcogenide',
               'elements': ['W', 'Se'],
               'a': 3.32706918,  # Angstrom
               'c': 15.06895072,  # Angstrom
               'u': 0.111569,
               'reference': '', 'link': ''}

cdb['mose2'] = {'crystal_name': 'MoSe2',
                'symmetry': 'dichalcogenide',
                'elements': ['Mo', 'Se'],
                'a': 3.32694913,  # Angstrom
                'c': 15.45142322,  # Angstrom
                'u': 0.108249,
                'reference': '', 'link': ''}
a_l = 0.3336
c_l = 0.4754
base_l = [(2. / 3., 1. / 3., .5), (1. / 3., 2. / 3., 0.), (2. / 3., 1. / 3., 0.), (1. / 3., 2. / 3., .5)]

cdb['zno hexagonal'] = {'crystal_name': 'ZnO hexagonal',
                        'symmetry': 'hexagonal',
                        'a': a_l,  # nm
                        'c': c_l,  # not np.sqrt(8/3)*1
                        'elements': ['Zn', 'Zn', 'O', 'O'],
                        'unit_cell': [[a_l, 0., 0.],
                                      [np.cos(120 / 180 * np.pi) * a_l, np.sin(120 / 180 * np.pi) * a_l, 0.],
                                      [0., 0., c_l]],
                        'base': np.array(base_l),
                        'reference': '', 'link': ''}

cdb['pdse2'] = {'crystal_name': 'PdSe2',
                'symmetry': 'primitive',
                'unit_cell': (np.identity(3) * (.579441832, 0.594542204, 0.858506072)),
                'elements': ['Pd'] * 4 + ['Se'] * 8,
                'base': np.array([[.5, .0, .0], [.0, 0.5, 0.0],
                                  [.5, 0.5, 0.5], [.0, 0.5, 0.5],
                                  [0.611300, 0.119356, 0.585891],
                                  [0.111300, 0.380644, 0.414109],
                                  [0.388700, 0.619356, 0.914109],
                                  [0.888700, 0.880644, 0.085891],
                                  [0.111300, 0.119356, 0.914109],
                                  [0.611300, 0.380644, 0.085891],
                                  [0.888700, 0.619356, 0.585891],
                                  [0.388700, 0.880644, 0.414109]]),
                'reference': '', 'link': ''}

crystal_data_base = cdb

# From Appendix C of Kirkland, "Advanced Computing in Electron Microscopy", 2nd ed.
electronFF = {
    # form factor coefficients
    # Z= 6, chisq= 0.143335
    # a1 b1 a2 b2
    # a3 b3 c1 d1
    # c2 d2 c3 d3

    # name of the file:  feKirkland.txt
    # converted with program sortFF.py
    # form factor parametrized in 1/Angstrom
    # bond_length as a list of atom Sizes, bond radii, angle radii, H-bond radii

    'H': {'Z': 1, 'chisq': 0.170190,
          'bond_length': [0.98, 0.78, 1.20, 0],
          'fa': [4.20298324e-003, 6.27762505e-002, 3.00907347e-002],
          'fb': [2.25350888e-001, 2.25366950e-001, 2.25331756e-001],
          'fc': [6.77756695e-002, 3.56609237e-003, 2.76135815e-002],
          'fd': [4.38854001e+000, 4.03884823e-001, 1.44490166e+000]},
    'He': {'Z':  2,   'chisq': 0.396634,
           'bond_length': [1.45, 1.25, 1.40, 0],
           'fa': [1.87543704e-005, 4.10595800e-004, 1.96300059e-001],
           'fb': [2.12427997e-001, 3.32212279e-001, 5.17325152e-001],
           'fc': [8.36015738e-003, 2.95102022e-002, 4.65928982e-007],
           'fd': [3.66668239e-001, 1.37171827e+000, 3.75768025e+004]},
    'Li': {'Z':  3,   'chisq': 0.286232,
           'bond_length': [1.76,  1.56, 1.82, 0],
           'fa': [7.45843816e-002, 7.15382250e-002, 1.45315229e-001],
           'fb': [8.81151424e-001, 4.59142904e-002, 8.81301714e-001],
           'fc': [1.12125769e+000, 2.51736525e-003, 3.58434971e-001],
           'fd': [1.88483665e+001, 1.59189995e-001, 6.12371000e+000]},
    'Be': {'Z':  4,   'chisq': 0.195442,
           'bond_length': [1.33,  1.13, 1.70, 0],
           'fa': [6.11642897e-002, 1.25755034e-001, 2.00831548e-001],
           'fb': [9.90182132e-002, 9.90272412e-002, 1.87392509e+000],
           'fc': [7.87242876e-001, 1.58847850e-003, 2.73962031e-001],
           'fd': [9.32794929e+000, 8.91900236e-002, 3.20687658e+000]},
    'B': {'Z':  5,   'chisq': 0.146989,
          'bond_length': [1.18,  0.98, 2.08, 0],
          'fa': [1.25716066e-001, 1.73314452e-001, 1.84774811e-001],
          'fb': [1.48258830e-001, 1.48257216e-001, 3.34227311e+000],
          'fc': [1.95250221e-001, 5.29642075e-001, 1.08230500e-003],
          'fd': [1.97339463e+000, 5.70035553e+000, 5.64857237e-002]},
    'C': {'Z':  6,   'chisq': 0.102440,
          'bond_length': [1.12,  0.92, 1.95, 0],
          'fa': [2.12080767e-001, 1.99811865e-001, 1.68254385e-001],
          'fb': [2.08605417e-001, 2.08610186e-001, 5.57870773e+000],
          'fc': [1.42048360e-001, 3.63830672e-001, 8.35012044e-004],
          'fd': [1.33311887e+000, 3.80800263e+000, 4.03982620e-002]},
    'N': {'Z':  7,   'chisq': 0.060249,
          'bond_length': [1.08,  0.88, 1.85, 1.30],
          'fa': [5.33015554e-001, 5.29008883e-002, 9.24159648e-002],
          'fb': [2.90952515e-001, 1.03547896e+001, 1.03540028e+001],
          'fc': [2.61799101e-001, 8.80262108e-004, 1.10166555e-001],
          'fd': [2.76252723e+000, 3.47681236e-002, 9.93421736e-001]},
    'O': {'Z':  8,   'chisq': 0.039944,
          'bond_length': [1.09,  0.89, 1.70, 1.40],
          'fa': [3.39969204e-001, 3.07570172e-001, 1.30369072e-001],
          'fb': [3.81570280e-001, 3.81571436e-001, 1.91919745e+001],
          'fc': [8.83326058e-002, 1.96586700e-001, 9.96220028e-004],
          'fd': [7.60635525e-001, 2.07401094e+000, 3.03266869e-002]},
    'F': {'Z':  9,   'chisq': 0.027866,
          'bond_length': [1.30,  1.10, 1.73, 0],
          'fa': [2.30560593e-001, 5.26889648e-001, 1.24346755e-001],
          'fb': [4.80754213e-001, 4.80763895e-001, 3.95306720e+001],
          'fc': [1.24616894e-003, 7.20452555e-002, 1.53075777e-001],
          'fd': [2.62181803e-002, 5.92495593e-001, 1.59127671e+000]},
    'Ne': {'Z':  10,   'chisq': 0.021836,
           'bond_length': [1.50,  1.30, 1.54, 0],
           'fa': [4.08371771e-001, 4.54418858e-001, 1.44564923e-001],
           'fb': [5.88228627e-001, 5.88288655e-001, 1.21246013e+002],
           'fc': [5.91531395e-002, 1.24003718e-001, 1.64986037e-003],
           'fd': [4.63963540e-001, 1.23413025e+000, 2.05869217e-002]},
    'Na': {'Z':  11,   'chisq': 0.064136,
           'bond_length': [2.10,  1.91, 2.27, 0],
           'fa': [1.36471662e-001, 7.70677865e-001, 1.56862014e-001],
           'fb': [4.99965301e-002, 8.81899664e-001, 1.61768579e+001],
           'fc': [9.96821513e-001, 3.80304670e-002, 1.27685089e-001],
           'fd': [2.00132610e+001, 2.60516254e-001, 6.99559329e-001]},
    'Mg': {'Z':  12,   'chisq': 0.051303,
           'bond_length': [1.80,  1.60, 1.73, 0],
           'fa': [3.04384121e-001, 7.56270563e-001, 1.01164809e-001],
           'fb': [8.42014377e-002, 1.64065598e+000, 2.97142975e+001],
           'fc': [3.45203403e-002, 9.71751327e-001, 1.20593012e-001],
           'fd': [2.16596094e-001, 1.21236852e+001, 5.60865838e-001]},
    'Al': {'Z':  13,   'chisq': 0.049529,
           'bond_length': [1.60,  1.43, 2.05, 0],
           'fa': [7.77419424e-001, 5.78312036e-002, 4.26386499e-001],
           'fb': [2.71058227e+000, 7.17532098e+001, 9.13331555e-002],
           'fc': [1.13407220e-001, 7.90114035e-001, 3.23293496e-002],
           'fd': [4.48867451e-001, 8.66366718e+000, 1.78503463e-001]},
    'Si': {'Z':  14,   'chisq': 0.071667,
           'bond_length': [1.52,  1.32, 2.10, 0],
           'fa': [1.06543892e+000, 1.20143691e-001, 1.80915263e-001],
           'fb': [1.04118455e+000, 6.87113368e+001, 8.87533926e-002],
           'fc': [1.12065620e+000, 3.05452816e-002, 1.59963502e+000],
           'fd': [3.70062619e+000, 2.14097897e-001, 9.99096638e+000]},
    'P': {'Z':  15,   'chisq': 0.047673,
          'bond_length': [1.48,  1.28, 2.08, 0],
          'fa': [1.05284447e+000, 2.99440284e-001, 1.17460748e-001],
          'fb': [1.31962590e+000, 1.28460520e-001, 1.02190163e+002],
          'fc': [9.60643452e-001, 2.63555748e-002, 1.38059330e+000],
          'fd': [2.87477555e+000, 1.82076844e-001, 7.49165526e+000]},
    'S': {'Z':  16,   'chisq': 0.033482,
          'bond_length': [1.47,  1.27, 2.00, 0],
          'fa': [1.01646916e+000, 4.41766748e-001, 1.21503863e-001],
          'fb': [1.69181965e+000, 1.74180288e-001, 1.67011091e+002],
          'fc': [8.27966670e-001, 2.33022533e-002, 1.18302846e+000],
          'fd': [2.30342810e+000, 1.56954150e-001, 5.85782891e+000]},
    'Cl': {'Z':  17,   'chisq': 0.206186,
           'bond_length': [1.70,  1.50, 1.97, 0],
           'fa': [9.44221116e-001, 4.37322049e-001, 2.54547926e-001],
           'fb': [2.40052374e-001, 9.30510439e+000, 9.30486346e+000],
           'fc': [5.47763323e-002, 8.00087488e-001, 1.07488641e-002],
           'fd': [1.68655688e-001, 2.97849774e+000, 6.84240646e-002]},
    'Ar': {'Z':  18,   'chisq': 0.263904,
           'bond_length': [2.00,  1.80, 1.88, 0],
           'fa': [1.06983288e+000, 4.24631786e-001, 2.43897949e-001],
           'fb': [2.87791022e-001, 1.24156957e+001, 1.24158868e+001],
           'fc': [4.79446296e-002, 7.64958952e-001, 8.23128431e-003],
           'fd': [1.36979796e-001, 2.43940729e+000, 5.27258749e-002]},
    'K': {'Z':  19,   'chisq': 0.161900,
          'bond_length': [2.58,  2.38, 2.75, 0],
          'fa': [6.92717865e-001, 9.65161085e-001, 1.48466588e-001],
          'fb': [7.10849990e+000, 3.57532901e-001, 3.93763275e-002],
          'fc': [2.64645027e-002, 1.80883768e+000, 5.43900018e-001],
          'fd': [1.03591321e-001, 3.22845199e+001, 1.67791374e+000]},
    'Ca': {'Z':  20,   'chisq': 0.085209,
           'bond_length': [2.17,  1.97, 1.97, 0],
           'fa': [3.66902871e-001, 8.66378999e-001, 6.67203300e-001],
           'fb': [6.14274129e-002, 5.70881727e-001, 7.82965639e+000],
           'fc': [4.87743636e-001, 1.82406314e+000, 2.20248453e-002],
           'fd': [1.32531318e+000, 2.10056032e+001, 9.11853450e-002]},
    'Sc': {'Z':  21,   'chisq': 0.052352,
           'bond_length': [1.84,  1.64, 1.70, 0],
           'fa': [3.78871777e-001, 9.00022505e-001, 7.15288914e-001],
           'fb': [6.98910162e-002, 5.21061541e-001, 7.87707920e+000],
           'fc': [1.88640973e-002, 4.07945949e-001, 1.61786540e+000],
           'fd': [8.17512708e-002, 1.11141388e+000, 1.80840759e+001]},
    'Ti': {'Z':  22,   'chisq': 0.035298,
           'bond_length': [1.66,  1.46, 1.70, 0],
           'fa': [3.62383267e-001, 9.84232966e-001, 7.41715642e-001],
           'fb': [7.54707114e-002, 4.97757309e-001, 8.17659391e+000],
           'fc': [3.62555269e-001, 1.49159390e+000, 1.61659509e-002],
           'fd': [9.55524906e-001, 1.62221677e+001, 7.33140839e-002]},
    'V': {'Z':  23,   'chisq': 0.030745,
          'bond_length': [1.55,  1.35, 1.70, 0],
          'fa': [3.52961378e-001, 7.46791014e-001, 1.08364068e+000],
          'fb': [8.19204103e-002, 8.81189511e+000, 5.10646075e-001],
          'fc': [1.39013610e+000, 3.31273356e-001, 1.40422612e-002],
          'fd': [1.48901841e+001, 8.38543079e-001, 6.57432678e-002]},
    'Cr': {'Z':  24,   'chisq': 0.015287,
           'bond_length': [1.56,  1.36, 1.70, 0],
           'fa': [1.34348379e+000, 5.07040328e-001, 4.26358955e-001],
           'fb': [1.25814353e+000, 1.15042811e+001, 8.53660389e-002],
           'fc': [1.17241826e-002, 5.11966516e-001, 3.38285828e-001],
           'fd': [6.00177061e-002, 1.53772451e+000, 6.62418319e-001]},
    'Mn': {'Z':  25,   'chisq': 0.031274,
           'bond_length': [1.54,  1.30, 1.70, 0],
           'fa': [3.26697613e-001, 7.17297000e-001, 1.33212464e+000],
           'fb': [8.88813083e-002, 1.11300198e+001, 5.82141104e-001],
           'fc': [2.80801702e-001, 1.15499241e+000, 1.11984488e-002],
           'fd': [6.71583145e-001, 1.26825395e+001, 5.32334467e-002]},
    'Fe': {'Z':  26,   'chisq': 0.031315,
           'bond_length': [1.47,  1.27, 1.70, 0],
           'fa': [3.13454847e-001, 6.89290016e-001, 1.47141531e+000],
           'fb': [8.99325756e-002, 1.30366038e+001, 6.33345291e-001],
           'fc': [1.03298688e+000, 2.58280285e-001, 1.03460690e-002],
           'fd': [1.16783425e+001, 6.09116446e-001, 4.81610627e-002]},
    'Co': {'Z':  27,   'chisq': 0.031643,
           'bond_length': [1.45,  1.25, 1.70, 0],
           'fa': [3.15878278e-001, 1.60139005e+000, 6.56394338e-001],
           'fb': [9.46683246e-002, 6.99436449e-001, 1.56954403e+001],
           'fc': [9.36746624e-001, 9.77562646e-003, 2.38378578e-001],
           'fd': [1.09392410e+001, 4.37446816e-002, 5.56286483e-001]},
    'Ni': {'Z':  28,   'chisq': 0.032245,
           'bond_length': [1.45,  1.25, 1.63, 0],
           'fa': [1.72254630e+000, 3.29543044e-001, 6.23007200e-001],
           'fb': [7.76606908e-001, 1.02262360e-001, 1.94156207e+001],
           'fc': [9.43496513e-003, 8.54063515e-001, 2.21073515e-001],
           'fd': [3.98684596e-002, 1.04078166e+001, 5.10869330e-001]},
    'Cu': {'Z':  29,   'chisq': 0.010467,
           'bond_length': [1.48,  1.28, 1.40, 0],
           'fa': [3.58774531e-001, 1.76181348e+000, 6.36905053e-001],
           'fb': [1.06153463e-001, 1.01640995e+000, 1.53659093e+001],
           'fc': [7.44930667e-003, 1.89002347e-001, 2.29619589e-001],
           'fd': [3.85345989e-002, 3.98427790e-001, 9.01419843e-001]},
    'Zn': {'Z':  30,   'chisq': 0.026698,
           'bond_length': [1.59,  1.39, 1.39, 0],
           'fa': [5.70893973e-001, 1.98908856e+000, 3.06060585e-001],
           'fb': [1.26534614e-001, 2.17781965e+000, 3.78619003e+001],
           'fc': [2.35600223e-001, 3.97061102e-001, 6.85657228e-003],
           'fd': [3.67019041e-001, 8.66419596e-001, 3.35778823e-002]},
    'Ga': {'Z':  31,   'chisq': 0.008110,
           'bond_length': [1.61,  1.41, 1.87, 0],
           'fa': [6.25528464e-001, 2.05302901e+000, 2.89608120e-001],
           'fb': [1.10005650e-001, 2.41095786e+000, 4.78685736e+001],
           'fc': [2.07910594e-001, 3.45079617e-001, 6.55634298e-003],
           'fd': [3.27807224e-001, 7.43139061e-001, 3.09411369e-002]},
    'Ge': {'Z':  32,   'chisq': 0.032198,
           'bond_length': [1.57,  1.37, 1.70, 0],
           'fa': [5.90952690e-001, 5.39980660e-001, 2.00626188e+000],
           'fb': [1.18375976e-001, 7.18937433e+001, 1.39304889e+000],
           'fc': [7.49705041e-001, 1.83581347e-001, 9.52190743e-003],
           'fd': [6.89943350e+000, 3.64667232e-001, 2.69888650e-002]},
    'As': {'Z':  33,   'chisq': 0.034014,
           'bond_length': [1.59,  1.39, 1.85, 0],
           'fa': [7.77875218e-001, 5.93848150e-001, 1.95918751e+000],
           'fb': [1.50733157e-001, 1.42882209e+002, 1.74750339e+000],
           'fc': [1.79880226e-001, 8.63267222e-001, 9.59053427e-003],
           'fd': [3.31800852e-001, 5.85490274e+000, 2.33777569e-002]},
    'Se': {'Z':  34,   'chisq': 0.035703,
           'bond_length': [1.60,  1.40, 1.90, 0],
           'fa': [9.58390681e-001, 6.03851342e-001, 1.90828931e+000],
           'fb': [1.83775557e-001, 1.96819224e+002, 2.15082053e+000],
           'fc': [1.73885956e-001, 9.35265145e-001, 8.62254658e-003],
           'fd': [3.00006024e-001, 4.92471215e+000, 2.12308108e-002]},
    'Br': {'Z':  35,   'chisq': 0.039250,
           'bond_length': [1.80,  1.60, 2.10, 0],
           'fa': [1.14136170e+000, 5.18118737e-001, 1.85731975e+000],
           'fb': [2.18708710e-001, 1.93916682e+002, 2.65755396e+000],
           'fc': [1.68217399e-001, 9.75705606e-001, 7.24187871e-003],
           'fd': [2.71719918e-001, 4.19482500e+000, 1.99325718e-002]},
    'Kr': {'Z':  36,   'chisq': 0.045421,
           'bond_length': [2.10,  1.90, 2.02, 0],
           'fa': [3.24386970e-001, 1.31732163e+000, 1.79912614e+000],
           'fb': [6.31317973e+001, 2.54706036e-001, 3.23668394e+000],
           'fc': [4.29961425e-003, 1.00429433e+000, 1.62188197e-001],
           'fd': [1.98965610e-002, 3.61094513e+000, 2.45583672e-001]},
    'Rb': {'Z':  37,   'chisq': 0.130044,
           'bond_length': [2.75,  2.55, 1.70, 0],
           'fa': [2.90445351e-001, 2.44201329e+000, 7.69435449e-001],
           'fb': [3.68420227e-002, 1.16013332e+000, 1.69591472e+001],
           'fc': [1.58687000e+000, 2.81617593e-003, 1.28663830e-001],
           'fd': [2.53082574e+000, 1.88577417e-002, 2.10753969e-001]},
    'Sr': {'Z':  38,   'chisq': 0.188055,
           'bond_length': [2.35,  2.15, 1.70, 0],
           'fa': [1.37373086e-002, 1.97548672e+000, 1.59261029e+000],
           'fb': [1.87469061e-002, 6.36079230e+000, 2.21992482e-001],
           'fc': [1.73263882e-001, 4.66280378e+000, 1.61265063e-003],
           'fd': [2.01624958e-001, 2.53027803e+001, 1.53610568e-002]},
    'Y': {'Z':  39,   'chisq': 0.174927,
          'bond_length': [2.00,  1.80, 1.70, 0],
          'fa': [6.75302747e-001, 4.70286720e-001, 2.63497677e+000],
          'fb': [6.54331847e-002, 1.06108709e+002, 2.06643540e+000],
          'fc': [1.09621746e-001, 9.60348773e-001, 5.28921555e-003],
          'fd': [1.93131925e-001, 1.63310938e+000, 1.66083821e-002]},
    'Zr': {'Z':  40,   'chisq': 0.072078,
           'bond_length': [1.80,  1.60, 1.70, 0],
           'fa': [2.64365505e+000, 5.54225147e-001, 7.61376625e-001],
           'fb': [2.20202699e+000, 1.78260107e+002, 7.67218745e-002],
           'fc': [6.02946891e-003, 9.91630530e-002, 9.56782020e-001],
           'fd': [1.55143296e-002, 1.76175995e-001, 1.54330682e+000]},
    'Nb': {'Z':  41,   'chisq': 0.011800,
           'bond_length': [1.67,  1.47, 1.70, 0],
           'fa': [6.59532875e-001, 1.84545854e+000, 1.25584405e+000],
           'fb': [8.66145490e-002, 5.94774398e+000, 6.40851475e-001],
           'fc': [1.22253422e-001, 7.06638328e-001, 2.62381591e-003],
           'fd': [1.66646050e-001, 1.62853268e+000, 8.26257859e-003]},
    'Mo': {'Z':  42,   'chisq': 0.008976,
           'bond_length': [1.60,  1.40, 1.70, 0],
           'fa': [6.10160120e-001, 1.26544000e+000, 1.97428762e+000],
           'fb': [9.11628054e-002, 5.06776025e-001, 5.89590381e+000],
           'fc': [6.48028962e-001, 2.60380817e-003, 1.13887493e-001],
           'fd': [1.46634108e+000, 7.84336311e-003, 1.55114340e-001]},
    'Tc': {'Z':  43,   'chisq': 0.023771,
           'bond_length': [1.56,  1.36, 1.70, 0],
           'fa': [8.55189183e-001, 1.66219641e+000, 1.45575475e+000],
           'fb': [1.02962151e-001, 7.64907000e+000, 1.01639987e+000],
           'fc': [1.05445664e-001, 7.71657112e-001, 2.20992635e-003],
           'fd': [1.42303338e-001, 1.34659349e+000, 7.90358976e-003]},
    'Ru': {'Z':  44,   'chisq': 0.010613,
           'bond_length': [1.54,  1.34, 1.70, 0],
           'fa': [4.70847093e-001, 1.58180781e+000, 2.02419818e+000],
           'fb': [9.33029874e-002, 4.52831347e-001, 7.11489023e+000],
           'fc': [1.97036257e-003, 6.26912639e-001, 1.02641320e-001],
           'fd': [7.56181595e-003, 1.25399858e+000, 1.33786087e-001]},
    'Rh': {'Z':  45,   'chisq': 0.012895,
           'bond_length': [1.54,  1.34, 1.70, 0],
           'fa': [4.20051553e-001, 1.76266507e+000, 2.02735641e+000],
           'fb': [9.38882628e-002, 4.64441687e-001, 8.19346046e+000],
           'fc': [1.45487176e-003, 6.22809600e-001, 9.91529915e-002],
           'fd': [7.82704517e-003, 1.17194153e+000, 1.24532839e-001]},
    'Pd': {'Z':  46,   'chisq': 0.009172,
           'bond_length': [1.58,  1.38, 1.63, 0],
           'fa': [2.10475155e+000, 2.03884487e+000, 1.82067264e-001],
           'fb': [8.68606470e+000, 3.78924449e-001, 1.42921634e-001],
           'fc': [9.52040948e-002, 5.91445248e-001, 1.13328676e-003],
           'fd': [1.17125900e-001, 1.07843808e+000, 7.80252092e-003]},
    'Ag': {'Z':  47,   'chisq': 0.006648,
           'bond_length': [1.64,  1.44, 1.72, 0],
           'fa': [2.07981390e+000, 4.43170726e-001, 1.96515215e+000],
           'fb': [9.92540297e+000, 1.04920104e-001, 6.40103839e-001],
           'fc': [5.96130591e-001, 4.78016333e-001, 9.46458470e-002],
           'fd': [8.89594790e-001, 1.98509407e+000, 1.12744464e-001]},
    'Cd': {'Z':  48,   'chisq': 0.005588,
           'bond_length': [1.77,  1.57, 1.58, 0],
           'fa': [1.63657549e+000, 2.17927989e+000, 7.71300690e-001],
           'fb': [1.24540381e+001, 1.45134660e+000, 1.26695757e-001],
           'fc': [6.64193880e-001, 7.64563285e-001, 8.61126689e-002],
           'fd': [7.77659202e-001, 1.66075210e+000, 1.05728357e-001]},
    'In': {'Z':  49,   'chisq': 0.002569,
           'bond_length': [1.86,  1.66, 1.93, 0],
           'fa': [2.24820632e+000, 1.64706864e+000, 7.88679265e-001],
           'fb': [1.51913507e+000, 1.30113424e+001, 1.06128184e-001],
           'fc': [8.12579069e-002, 6.68280346e-001, 6.38467475e-001],
           'fd': [9.94045620e-002, 1.49742063e+000, 7.18422635e-001]},
    'Sn': {'Z':  50,   'chisq': 0.005051,
           'bond_length': [1.82,  1.62, 2.17, 0],
           'fa': [2.16644620e+000, 6.88691021e-001, 1.92431751e+000],
           'fb': [1.13174909e+001, 1.10131285e-001, 6.74464853e-001],
           'fc': [5.65359888e-001, 9.18683861e-001, 7.80542213e-002],
           'fd': [7.33564610e-001, 1.02310312e+001, 9.31104308e-002]},
    'Sb': {'Z':  51,   'chisq': 0.004383,
           'bond_length': [1.79,  1.59, 2.20, 0],
           'fa': [1.73662114e+000, 9.99871380e-001, 2.13972409e+000],
           'fb': [8.84334719e-001, 1.38462121e-001, 1.19666432e+001],
           'fc': [5.60566526e-001, 9.93772747e-001, 7.37374982e-002],
           'fd': [6.72672880e-001, 8.72330411e+000, 8.78577715e-002]},
    'Te': {'Z':  52,   'chisq': 0.004105,
           'bond_length': [1.80,  1.60, 2.06, 0],
           'fa': [2.09383882e+000, 1.56940519e+000, 1.30941993e+000],
           'fb': [1.26856869e+001, 1.21236537e+000, 1.66633292e-001],
           'fc': [6.98067804e-002, 1.04969537e+000, 5.55594354e-001],
           'fd': [8.30817576e-002, 7.43147857e+000, 6.17487676e-001]},
    'I': {'Z':  53,   'chisq': 0.004068,
          'bond_length': [1.90,  1.70, 2.15, 0],
          'fa': [1.60186925e+000, 1.98510264e+000, 1.48226200e+000],
          'fb': [1.95031538e-001, 1.36976183e+001, 1.80304795e+000],
          'fc': [5.53807199e-001, 1.11728722e+000, 6.60720847e-002],
          'fd': [5.67912340e-001, 6.40879878e+000, 7.86615429e-002]},
    'Xe': {'Z':  54,   'chisq': 0.004381,
           'bond_length': [2.30,  2.10, 2.16, 0],
           'fa': [1.60015487e+000, 1.71644581e+000, 1.84968351e+000],
           'fb': [2.92913354e+000, 1.55882990e+001, 2.22525983e-001],
           'fc': [6.23813648e-002, 1.21387555e+000, 5.54051946e-001],
           'fd': [7.45581223e-002, 5.56013271e+000, 5.21994521e-001]},
    'Cs': {'Z':  55,   'chisq': 0.042676,
           'bond_length': [2.93,  2.73, 1.70, 0],
           'fa': [2.95236854e+000, 4.28105721e-001, 1.89599233e+000],
           'fb': [6.01461952e+000, 4.64151246e+001, 1.80109756e-001],
           'fc': [5.48012938e-002, 4.70838600e+000, 5.90356719e-001],
           'fd': [7.12799633e-002, 4.56702799e+001, 4.70236310e-001]},
    'Ba': {'Z':  56,   'chisq': 0.043267,
           'bond_length': [2.44,  2.24, 1.70, 0],
           'fa': [3.19434243e+000, 1.98289586e+000, 1.55121052e-001],
           'fb': [9.27352241e+000, 2.28741632e-001, 3.82000231e-002],
           'fc': [6.73222354e-002, 4.48474211e+000, 5.42674414e-001],
           'fd': [7.30961745e-002, 2.95703565e+001, 4.08647015e-001]},
    'La': {'Z':  57,   'chisq': 0.033249,
           'bond_length': [2.08,  1.88, 1.70, 0],
           'fa': [2.05036425e+000, 1.42114311e-001, 3.23538151e+000],
           'fb': [2.20348417e-001, 3.96438056e-002, 9.56979169e+000],
           'fc': [6.34683429e-002, 3.97960586e+000, 5.20116711e-001],
           'fd': [6.92443091e-002, 2.53178406e+001, 3.83614098e-001]},
    'Ce': {'Z':  58,   'chisq': 0.029355,
           'bond_length': [2.02,  1.82, 1.70, 0],
           'fa': [3.22990759e+000, 1.57618307e-001, 2.13477838e+000],
           'fb': [9.94660135e+000, 4.15378676e-002, 2.40480572e-001],
           'fc': [5.01907609e-001, 3.80889010e+000, 5.96625028e-002],
           'fd': [3.66252019e-001, 2.43275968e+001, 6.59653503e-002]},
    'Pr': {'Z':  59,   'chisq': 0.029725,
           'bond_length': [2.03,  1.83, 1.70, 0],
           'fa': [1.58189324e-001, 3.18141995e+000, 2.27622140e+000],
           'fb': [3.91309056e-002, 1.04139545e+001, 2.81671757e-001],
           'fc': [3.97705472e+000, 5.58448277e-002, 4.85207954e-001],
           'fd': [2.61872978e+001, 6.30921695e-002, 3.54234369e-001]},
    'Nd': {'Z':  60,   'chisq': 0.027597,
           'bond_length': [2.02,  1.82, 1.70, 0],
           'fa': [1.81379417e-001, 3.17616396e+000, 2.35221519e+000],
           'fb': [4.37324793e-002, 1.07842572e+001, 3.05571833e-001],
           'fc': [3.83125763e+000, 5.25889976e-002, 4.70090742e-001],
           'fd': [2.54745408e+001, 6.02676073e-002, 3.39017003e-001]},
    'Pm': {'Z':  61,   'chisq': 0.025208,
           'bond_length': [2.01,  1.81, 1.70, 0],
           'fa': [1.92986811e-001, 2.43756023e+000, 3.17248504e+000],
           'fb': [4.37785970e-002, 3.29336996e-001, 1.11259996e+001],
           'fc': [3.58105414e+000, 4.56529394e-001, 4.94812177e-002],
           'fd': [2.46709586e+001, 3.24990282e-001, 5.76553100e-002]},
    'Sm': {'Z':  62,   'chisq': 0.023540,
           'bond_length': [2.00,  1.80, 1.70, 0],
           'fa': [2.12002595e-001, 3.16891754e+000, 2.51503494e+000],
           'fb': [4.57703608e-002, 1.14536599e+001, 3.55561054e-001],
           'fc': [4.44080845e-001, 3.36742101e+000, 4.65652543e-002],
           'fd': [3.11953363e-001, 2.40291435e+001, 5.52266819e-002]},
    'Eu': {'Z':  63,   'chisq': 0.022204,
           'bond_length': [2.24,  2.04, 1.70, 0],
           'fa': [2.59355002e+000, 3.16557522e+000, 2.29402652e-001],
           'fb': [3.82452612e-001, 1.17675155e+001, 4.76642249e-002],
           'fc': [4.32257780e-001, 3.17261920e+000, 4.37958317e-002],
           'fd': [2.99719833e-001, 2.34462738e+001, 5.29440680e-002]},
    'Gd': {'Z':  64,   'chisq': 0.017492,
           'bond_length': [2.00,  1.80, 1.70, 0],
           'fa': [3.19144939e+000, 2.55766431e+000, 3.32681934e-001],
           'fb': [1.20224655e+001, 4.08338876e-001, 5.85819814e-002],
           'fc': [4.14243130e-002, 2.61036728e+000, 4.20526863e-001],
           'fd': [5.06771477e-002, 1.99344244e+001, 2.85686240e-001]},
    'Tb': {'Z':  65,   'chisq': 0.020036,
           'bond_length': [1.98,  1.78, 1.70, 0],
           'fa': [2.59407462e-001, 3.16177855e+000, 2.75095751e+000],
           'fb': [5.04689354e-002, 1.23140183e+001, 4.38337626e-001],
           'fc': [2.79247686e+000, 3.85931001e-002, 4.10881708e-001],
           'fd': [2.23797309e+001, 4.87920992e-002, 2.77622892e-001]},
    'Dy': {'Z':  66,   'chisq': 0.019351,
           'bond_length': [1.97,  1.77, 1.70, 0],
           'fa': [3.16055396e+000, 2.82751709e+000, 2.75140255e-001],
           'fb': [1.25470414e+001, 4.67899094e-001, 5.23226982e-002],
           'fc': [4.00967160e-001, 2.63110834e+000, 3.61333817e-002],
           'fd': [2.67614884e-001, 2.19498166e+001, 4.68871497e-002]},
    'Ho': {'Z':  67,   'chisq': 0.018720,
           'bond_length': [1.98,  1.78, 1.70, 0],
           'fa': [2.88642467e-001, 2.90567296e+000, 3.15960159e+000],
           'fb': [5.40507687e-002, 4.97581077e-001, 1.27599505e+001],
           'fc': [3.91280259e-001, 2.48596038e+000, 3.37664478e-002],
           'fd': [2.58151831e-001, 2.15400972e+001, 4.50664323e-002]},
    'Er': {'Z':  68,   'chisq': 0.018677,
           'bond_length': [1.96,  1.76, 1.70, 0],
           'fa': [3.15573213e+000, 3.11519560e-001, 2.97722406e+000],
           'fb': [1.29729009e+001, 5.81399387e-002, 5.31213394e-001],
           'fc': [3.81563854e-001, 2.40247532e+000, 3.15224214e-002],
           'fd': [2.49195776e-001, 2.13627616e+001, 4.33253257e-002]},
    'Tm': {'Z':  69,   'chisq': 0.018176,
           'bond_length': [1.95,  1.75, 1.70, 0],
           'fa': [3.15591970e+000, 3.22544710e-001, 3.05569053e+000],
           'fb': [1.31232407e+001, 5.97223323e-002, 5.61876773e-001],
           'fc': [2.92845100e-002, 3.72487205e-001, 2.27833695e+000],
           'fd': [4.16534255e-002, 2.40821967e-001, 2.10034185e+001]},
    'Yb': {'Z':  70,   'chisq': 0.018460,
           'bond_length': [2.10,  1.90, 1.70, 0],
           'fa': [3.10794704e+000, 3.14091221e+000, 3.75660454e-001],
           'fb': [6.06347847e-001, 1.33705269e+001, 7.29814740e-002],
           'fc': [3.61901097e-001, 2.45409082e+000, 2.72383990e-002],
           'fd': [2.32652051e-001, 2.12695209e+001, 3.99969597e-002]},
    'Lu': {'Z':  71,   'chisq': 0.015021,
           'bond_length': [1.93,  1.73, 1.70, 0],
           'fa': [3.11446863e+000, 5.39634353e-001, 3.06460915e+000],
           'fb': [1.38968881e+001, 8.91708508e-002, 6.79919563e-001],
           'fc': [2.58563745e-002, 2.13983556e+000, 3.47788231e-001],
           'fd': [3.82808522e-002, 1.80078788e+001, 2.22706591e-001]},
    'Hf': {'Z':  72,   'chisq': 0.012070,
           'bond_length': [1.78,  1.58, 1.70, 0],
           'fa': [3.01166899e+000, 3.16284788e+000, 6.33421771e-001],
           'fb': [7.10401889e-001, 1.38262192e+001, 9.48486572e-002],
           'fc': [3.41417198e-001, 1.53566013e+000, 2.40723773e-002],
           'fd': [2.14129678e-001, 1.55298698e+001, 3.67833690e-002]},
    'Ta': {'Z':  73,   'chisq': 0.010775,
           'bond_length': [1.67,  1.47, 1.70, 0],
           'fa': [3.20236821e+000, 8.30098413e-001, 2.86552297e+000],
           'fb': [1.38446369e+001, 1.18381581e-001, 7.66369118e-001],
           'fc': [2.24813887e-002, 1.40165263e+000, 3.33740596e-001],
           'fd': [3.52934622e-002, 1.46148877e+001, 2.05704486e-001]},
    'W': {'Z':  74,   'chisq': 0.009479,
          'bond_length': [1.61,  1.41, 1.70, 0],
          'fa': [9.24906855e-001, 2.75554557e+000, 3.30440060e+000],
          'fb': [1.28663377e-001, 7.65826479e-001, 1.34471170e+001],
          'fc': [3.29973862e-001, 1.09916444e+000, 2.06498883e-002],
          'fd': [1.98218895e-001, 1.35087534e+001, 3.38918459e-002]},
    'Re': {'Z':  75,   'chisq': 0.004620,
           'bond_length': [1.58,  1.38, 1.70, 0],
           'fa': [1.96952105e+000, 1.21726619e+000, 4.10391685e+000],
           'fb': [4.98830620e+001, 1.33243809e-001, 1.84396916e+000],
           'fc': [2.90791978e-002, 2.30696669e-001, 6.08840299e-001],
           'fd': [2.84192813e-002, 1.90968784e-001, 1.37090356e+000]},
    'Os': {'Z':  76,   'chisq': 0.003085,
           'bond_length': [1.55,  1.35, 1.70, 0],
           'fa': [2.06385867e+000, 1.29603406e+000, 3.96920673e+000],
           'fb': [4.05671697e+001, 1.46559047e-001, 1.82561596e+000],
           'fc': [2.69835487e-002, 2.31083999e-001, 6.30466774e-001],
           'fd': [2.84172045e-002, 1.79765184e-001, 1.38911543e+000]},
    'Ir': {'Z':  77,   'chisq': 0.003924,
           'bond_length': [1.56,  1.36, 1.70, 0],
           'fa': [2.21522726e+000, 1.37573155e+000, 3.78244405e+000],
           'fb': [3.24464090e+001, 1.60920048e-001, 1.78756553e+000],
           'fc': [2.44643240e-002, 2.36932016e-001, 6.48471412e-001],
           'fd': [2.82909938e-002, 1.70692368e-001, 1.37928390e+000]},
    'Pt': {'Z':  78,   'chisq': 0.003817,
           'bond_length': [1.59,  1.39, 1.72, 0],
           'fa': [9.84697940e-001, 2.73987079e+000, 3.61696715e+000],
           'fb': [1.60910839e-001, 7.18971667e-001, 1.29281016e+001],
           'fc': [3.02885602e-001, 2.78370726e-001, 1.52124129e-002],
           'fd': [1.70134854e-001, 1.49862703e+000, 2.83510822e-002]},
    'Au': {'Z':  79,   'chisq': 0.003143,
           'bond_length': [1.64,  1.44, 1.66, 0],
           'fa': [9.61263398e-001, 3.69581030e+000, 2.77567491e+000],
           'fb': [1.70932277e-001, 1.29335319e+001, 6.89997070e-001],
           'fc': [2.95414176e-001, 3.11475743e-001, 1.43237267e-002],
           'fd': [1.63525510e-001, 1.39200901e+000, 2.71265337e-002]},
    'Hg': {'Z':  80,   'chisq': 0.002717,
           'bond_length': [1.77,  1.57, 1.55, 0],
           'fa': [1.29200491e+000, 2.75161478e+000, 3.49387949e+000],
           'fb': [1.83432865e-001, 9.42368371e-001, 1.46235654e+001],
           'fc': [2.77304636e-001, 4.30232810e-001, 1.48294351e-002],
           'fd': [1.55110144e-001, 1.28871670e+000, 2.61903834e-002]},
    'Tl': {'Z':  81,   'chisq': 0.003492,
           'bond_length': [1.92,  1.72, 1.96, 0],
           'fa': [3.75964730e+000, 3.21195904e+000, 6.47767825e-001],
           'fb': [1.35041513e+001, 6.66330993e-001, 9.22518234e-002],
           'fc': [2.76123274e-001, 3.18838810e-001, 1.31668419e-002],
           'fd': [1.50312897e-001, 1.12565588e+000, 2.48879842e-002]},
    'Pb': {'Z':  82,   'chisq': 0.001158,
           'bond_length': [1.95,  1.75, 2.02, 0],
           'fa': [1.00795975e+000, 3.09796153e+000, 3.61296864e+000],
           'fb': [1.17268427e-001, 8.80453235e-001, 1.47325812e+001],
           'fc': [2.62401476e-001, 4.05621995e-001, 1.31812509e-002],
           'fd': [1.43491014e-001, 1.04103506e+000, 2.39575415e-002]},
    'Bi': {'Z':  83,   'chisq': 0.026436,
           'bond_length': [1.90,  1.70, 1.70, 0],
           'fa': [1.59826875e+000, 4.38233925e+000, 2.06074719e+000],
           'fb': [1.56897471e-001, 2.47094692e+000, 5.72438972e+001],
           'fc': [1.94426023e-001, 8.22704978e-001, 2.33226953e-002],
           'fd': [1.32979109e-001, 9.56532528e-001, 2.23038435e-002]},
    'Po': {'Z':  84,   'chisq': 0.008962,
           'bond_length': [1.96,  1.76, 1.70, 0],
           'fa': [1.71463223e+000, 2.14115960e+000, 4.37512413e+000],
           'fb': [9.79262841e+001, 2.10193717e-001, 3.66948812e+000],
           'fc': [2.16216680e-002, 1.97843837e-001, 6.52047920e-001],
           'fd': [1.98456144e-002, 1.33758807e-001, 7.80432104e-001]},
    'At': {'Z':  85,   'chisq': 0.033776,
           'bond_length': [2.00,  1.80, 1.70, 0],
           'fa': [1.48047794e+000, 2.09174630e+000, 4.75246033e+000],
           'fb': [1.25943919e+002, 1.83803008e-001, 4.19890596e+000],
           'fc': [1.85643958e-002, 2.05859375e-001, 7.13540948e-001],
           'fd': [1.81383503e-002, 1.33035404e-001, 7.03031938e-001]},
    'Rn': {'Z':  86,   'chisq': 0.050132,
           'bond_length': [2.40,  2.20, 1.70, 0],
           'fa': [6.30022295e-001, 3.80962881e+000, 3.89756067e+000],
           'fb': [1.40909762e-001, 3.08515540e+001, 6.51559763e-001],
           'fc': [2.40755100e-001, 2.62868577e+000, 3.14285931e-002],
           'fd': [1.08899672e-001, 6.42383261e+000, 2.42346699e-002]},
    'Fr': {'Z':  87,   'chisq': 0.056720,
           'bond_length': [3.00,  2.80, 1.70, 0],
           'fa': [5.23288135e+000, 2.48604205e+000, 3.23431354e-001],
           'fb': [8.60599536e+000, 3.04543982e-001, 3.87759096e-002],
           'fc': [2.55403596e-001, 5.53607228e-001, 5.75278889e-003],
           'fd': [1.28717724e-001, 5.36977452e-001, 1.29417790e-002]},
    'Ra': {'Z':  88,   'chisq': 0.081498,
           'bond_length': [2.46,  2.26, 1.70, 0],
           'fa': [1.44192685e+000, 3.55291725e+000, 3.91259586e+000],
           'fb': [1.18740873e-001, 1.01739750e+000, 6.31814783e+001],
           'fc': [2.16173519e-001, 3.94191605e+000, 4.60422605e-002],
           'fd': [9.55806441e-002, 3.50602732e+001, 2.20850385e-002]},
    'Ac': {'Z':  89,   'chisq': 0.077643,
           'bond_length': [2.09,  1.88, 1.70, 0],
           'fa': [1.45864127e+000, 4.18945405e+000, 3.65866182e+000],
           'fb': [1.07760494e-001, 8.89090649e+001, 1.05088931e+000],
           'fc': [2.08479229e-001, 3.16528117e+000, 5.23892556e-002],
           'fd': [9.09335557e-002, 3.13297788e+001, 2.08807697e-002]},
    'Th': {'Z':  90,   'chisq': 0.048096,
           'bond_length': [2.00,  1.80, 1.70, 0],
           'fa': [1.19014064e+000, 2.55380607e+000, 4.68110181e+000],
           'fb': [7.73468729e-002, 6.59693681e-001, 1.28013896e+001],
           'fc': [2.26121303e-001, 3.58250545e-001, 7.82263950e-003],
           'fd': [1.08632194e-001, 4.56765664e-001, 1.62623474e-002]},
    'Pa': {'Z':  91,   'chisq': 0.070186,
           'bond_length': [1.83,  1.63, 1.70, 0],
           'fa': [4.68537504e+000, 2.98413708e+000, 8.91988061e-001],
           'fb': [1.44503632e+001, 5.56438592e-001, 6.69512914e-002],
           'fc': [2.24825384e-001, 3.04444846e-001, 9.48162708e-003],
           'fd': [1.03235396e-001, 4.27255647e-001, 1.77730611e-002]},
    'U': {'Z':  92,   'chisq': 0.072478,
          'bond_length': [1.76,  1.56, 1.86, 0],
          'fa': [4.63343606e+000, 3.18157056e+000, 8.76455075e-001],
          'fb': [1.63377267e+001, 5.69517868e-001, 6.88860012e-002],
          'fc': [2.21685477e-001, 2.72917100e-001, 1.11737298e-002],
          'fd': [9.84254550e-002, 4.09470917e-001, 1.86215410e-002]},
    'Np': {'Z':  93,   'chisq': 0.074792,
           'bond_length': [1.80,  1.60, 1.70, 0],
           'fa': [4.56773888e+000, 3.40325179e+000, 8.61841923e-001],
           'fb': [1.90992795e+001, 5.90099634e-001, 7.03204851e-002],
           'fc': [2.19728870e-001, 2.38176903e-001, 1.38306499e-002],
           'fd': [9.36334280e-002, 3.93554882e-001, 1.94437286e-002]},
    'Pu': {'Z':  94,   'chisq': 0.071877,
           'bond_length': [1.84,  1.64, 1.70, 0],
           'fa': [5.45671123e+000, 1.11687906e-001, 3.30260343e+000],
           'fb': [1.01892720e+001, 3.98131313e-002, 3.14622212e-001],
           'fc': [1.84568319e-001, 4.93644263e-001, 3.57484743e+000],
           'fd': [1.04220860e-001, 4.63080540e-001, 2.19369542e+001]},
    'Am': {'Z':  95,   'chisq': 0.062156,
           'bond_length': [2.01,  1.81, 1.70, 0],
           'fa': [5.38321999e+000, 1.23343236e-001, 3.46469090e+000],
           'fb': [1.07289857e+001, 4.15137806e-002, 3.39326208e-001],
           'fc': [1.75437132e-001, 3.39800073e+000, 4.69459519e-001],
           'fd': [9.98932346e-002, 2.11601535e+001, 4.51996970e-001]},
    'Cm': {'Z':  96,   'chisq': 0.050111,
           'bond_length': [2.20,  2.00, 1.70, 0],
           'fa': [5.38402377e+000, 3.49861264e+000, 1.88039547e-001],
           'fb': [1.11211419e+001, 3.56750210e-001, 5.39853583e-002],
           'fc': [1.69143137e-001, 3.19595016e+000, 4.64393059e-001],
           'fd': [9.60082633e-002, 1.80694389e+001, 4.36318197e-001]},
    'Bk': {'Z':  97,   'chisq': 0.044081,
           'bond_length': [2.20,  2.00, 1.70, 0],
           'fa': [3.66090688e+000, 2.03054678e-001, 5.30697515e+000],
           'fb': [3.84420906e-001, 5.48547131e-002, 1.17150262e+001],
           'fc': [1.60934046e-001, 3.04808401e+000, 4.43610295e-001],
           'fd': [9.21020329e-002, 1.73525367e+001, 4.27132359e-001]},
    'Cf': {'Z':  98,   'chisq': 0.041053,
           'bond_length': [2.20,  2.00, 1.70, 0],
           'fa': [3.94150390e+000, 5.16915345e+000, 1.61941074e-001],
           'fb': [4.18246722e-001, 1.25201788e+001, 4.81540117e-002],
           'fc': [4.15299561e-001, 2.91761325e+000, 1.51474927e-001],
           'fd': [4.24913856e-001, 1.90899693e+001, 8.81568925e-002]}
    }
