'''
KinsCat
Kinematic Scattering Theory
Copyright by Gerd Duscher

The Univerity of Tennessee, Knoxville
Department of Materials Science & Engineering

Sources:
           Scattering Theory:
           Zuo and Spence, "Advanced TEM", 2017

           Spence and Zuo, Electron Microdiffraction, Plenum 1992
  
           Atomic Form Factor:
           Kirkland: Advanced Computing in Electron Microsocpy 2nd edition
           Appendix C

Units: everything is in SI units, except length is given in nm.

Usage:
All the input and ouptut is done through a dictionary
(See Zuo_fig_3_18 for full example input)

Available Subroutines:

    Kinematic_Scattering(tags, verbose=False ) # Main Routine

    CsCl()
    Silicon()
    Zuo_fig_3_18()
    
    feq(element, q) # atomic form factor at reciprocal distance q   
    get_rotation_matrix(zone, verbose = False):
    zone_mistilt (zone, angles):
    vector_norm(g)  # Length of vextor
    get_waveLength(E0) # Wavelength in 1/nm

    plotSAED(tags, grey=False)
    holzLZ(tags, grey=False)
    plotKikuchi(tags, grey=False)


'''
# numerical packages used
import numpy as np
import  itertools 
import scipy.constants as const

import h5py
import pyUSID as usid


_version_ = 0.5

print('Using KinsCat library version ',_version_,' by G.Duscher')
_spglib_present = True
try:
    import spglib
except:
     _spglib_present = False

    
if _spglib_present:
    print('Symmetry functions of spglib enabled')
else:
    print('spglib not installed; Symmetry functions of spglib disabled')

# plotting package used
import matplotlib.pylab as plt # basic plotting
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D # 3D plotting



### Some Structure Determination Routines




def example():
    return(Zuo_fig_3_18())
def Zuo_fig_3_18():
    
    # Create Silicon structure (Could be produced with Silicon routine)
    # INPUT
    print('Sample Input for Figure 3.18 in Zuo and Specnce \"Adavanced TEM\", 2017')
    tags = {}
    tags['crystal_name'] = 'Silicon'
    print('tags[\'crystal\'] = ',tags['crystal_name'])
    a  = 0.514 #nm
    tags['lattice_parameter_nm'] = a
    print('tags[\'lattice_parameter_nm\'] =',tags['lattice_parameter_nm'])
    tags['unit_cell'] = [ [a,0,0],[0,a,0], [0,0,a] ]
    print('tags[\'unit_cell\'] =',tags['unit_cell'])
    tags['elements'] =  list(itertools.repeat('Si',8) )
    print('tags[\'elements\'] =',tags['elements'])
    base = [(0.,0.,0.),(0.5,0.0,0.5),(0.5,0.5,0.),(0.,0.5,0.5)]
    tags['base'] = np.array(base + (np.array(base) + (.25,.25,.25)).tolist())
    print('tags[\'base\'] =',tags['base'])

    # Define Experimental Conditions
    tags['convergence_angle_mrad'] = 7
 
    tags['acceleration_voltage_V'] = 101.6*1000.0 #V
    print('tags[\'acceleration_voltage_V\'] =',tags['acceleration_voltage_V'])
    
    tags['convergence_angle_mrad'] = 7.1 #mrad;  0 is parallel illumination
    print('tags[\'convergence_angle_mrad\'] =',tags['convergence_angle_mrad'])
    
    tags['zone_hkl'] = np.array([-2,2,1])  # incident neares zone axis: defines Laue Zones!!!!
    print('tags[\'zone_hkl\'] =',tags['zone_hkl'])
    tags['mistilt']  = np.array([0,0,0])  # mistilt in degrees
    print('tags[\'mistilt\'] =',tags['mistilt'])
    
    # Define Simulation Parameters 

    tags['Sg_max'] = .2 # 1/nm  maximum allowed excitation error
    print('tags[\'Sg_max\'] =',tags['Sg_max'])
    
    tags['hkl_max'] = 9   # Highest evaluated Miller indices
    print('tags[\'hkl_max\'] =',tags['hkl_max'])

    print('##################')
    print('# Output Options #')
    print('##################')


    # Output options

    tags['background'] = 'black' # 'white'  'grey'
    print('tags[\'background\'] =',tags['background'], '# \'white\',  \'grey\' ')
    tags['color map'] = 'plasma'
    print('tags[\'color map\'] =',tags['color map'],'#,\'cubehelix\',\'Greys\',\'jet\' ')

    tags['plot HOLZ'] = 1
    print('tags[\'plot HOLZ\'] =',tags['plot HOLZ'])
    tags['plot HOLZ excess'] = 1
    print('tags[\'plot HOLZ excess\'] =',tags['plot HOLZ excess'])
    tags['plot Kikuchi'] = 1
    print('tags[\'plot Kikuchi\'] =',tags['plot Kikuchi'])
    tags['plot reflections'] = 1
    print('tags[\'plot reflections\'] =',tags['plot reflections'])
    
    tags['label HOLZ'] = 0
    print('tags[\'label HOLZ\'] =',tags['label HOLZ'])
    tags['label Kikuchi'] = 0
    print('tags[\'label Kikuchi\'] =',tags['label Kikuchi'])
    tags['label reflections'] = 0
    print('tags[\'label reflections\'] =',tags['label reflections'])
    

    tags['label color'] = 'black'
    print('tags[\'label color\'] =',tags['label color'])
    tags['label size'] = 10
    print('tags[\'label size\'] =',tags['label size'])
    
    tags['color Laue Zones'] = ['red',  'blue', 'green', 'blue', 'green']#, 'green', 'red'] #for OLZ give a sequence
    print('tags[\'color Laue Zones\'] =',tags['color Laue Zones'],' #[\'red\', \'blue\', \'lightblue\']')

    tags['color Kikuchi'] = 'green'
    print('tags[\'color Kikuchi\'] =',tags['color Kikuchi'])
    tags['linewidth HOLZ'] = -1 # -1: linewidth according to intensity (structure factor F^2
    print('tags[\'linewidth HOLZ\'] =',tags['linewidth HOLZ'],'# -1: linewidth according to intensity (structure factor F^2')
    tags['linewidth Kikuchi'] = -1 # -1: linewidth according to intensity (structure factor F^2
    print('tags[\'linewidth Kikuchi\'] =',tags['linewidth Kikuchi'],'# -1: linewidth according to intensity (structure factor F^2')

    
    tags['color reflections'] = 'intensity', #'Laue Zone'
    print('tags[\'color reflections\'] =',tags['color reflections'],'#\'Laue Zone\' ')
    tags['color zero'] = 'white'#'None', 'white', 'blue'
    print('tags[\'color zero\'] =',tags['color zero'],'#\'None\', \'white\', \'blue\' ')
    tags['color ring zero'] = 'None'# 'Red' #'white' #, 'None'
    print('tags[\'color ring zero\'] =',tags['color ring zero'],'#\'None\', \'white\', \'Red\' ')

    print('########################')
    print('# End of Example Input #')
    print('########################\n\n')
    
    return(tags)



def h5_add_crystal_structure(h5_file, crystal_tags):
    """
    Adds the basic crystal information to a pyUSID style hdf5 file.

    example usage:
    >> crystal_tags = ks.structure_by_name(h5_file, 'Graphite')
    >> ks.add_crystal_structure(h5_file, crystal_tags)
    """
    structure_group = usid.io.hdf_utils.create_indexed_group(h5_file,'Structure')
    
    structure_group['unit_cell'] = crystal_tags['unit_cell']
    structure_group['relative_positions'] = crystal_tags['base']
    structure_group['title'] = str(crystal_tags['crystal_name'])
    structure_group['_'+crystal_tags['crystal_name']] = str(crystal_tags['crystal_name'])
    structure_group['elements'] = np.array(crystal_tags['elements'],dtype='S')
    if 'zone_axis' in structure_group:
        structure_group['zone_axis'] = np.array(crystal_tags['zone_axis'], dtype=float)
    else:
        structure_group['zone_axis'] = np.array([1.,1.,1.], dtype=float)

    h5_file.flush()
    
    return structure_group

def h5_add2Structure(structure_group, crystal_tags):
    for key in crystal_tags:
        if key in structure_group.keys():
            print(key, ' not written; use new name')

        else:
            structure_group[key] = crystal_tags[key]

def cubic(a):
    """ Cubic lattice of dimensions a x a x a. """
    return np.identity(3)*a


def from_parameters(a, b, c, alpha, beta, gamma):
    """ Create a Lattice using unit cell lengths and angles (in degrees)."""

    alpha_r = np.radians(alpha)
    beta_r = np.radians(beta)
    gamma_r = np.radians(gamma)
    val = (np.cos(alpha_r) * np.cos(beta_r) - np.cos(gamma_r))\
        / (np.sin(alpha_r) * np.sin(beta_r))
    
    # Sometimes rounding errors result in values slightly > 1.
    if val > 1.:  val = 1.
    if val <-1.:  val = -1.

    gamma_star = np.arccos(val)
    vector_a = [a * np.sin(beta_r), 0.0, a * np.cos(beta_r)]
    vector_b = [-b * np.sin(alpha_r) * np.cos(gamma_star),
                b * np.sin(alpha_r) * np.sin(gamma_star),
                b * np.cos(alpha_r)]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])

def tetragonal(a, c):
    """Tetragonal lattice of dimensions a x a x c."""
    return from_parameters(a, a, c, 90, 90, 90)

def BCC(lattice_parameter, elements):
    unit_cell = cubic(lattice_parameter)
    base = np.array([(0.,0.,0.),(0.5,0.5,0.5)])
    if len(elements) >1:
        atoms = [elements[0], elements[1]]
    else:
        atoms = [elements[0], elements[0]]
    return unit_cell, base, atoms
        
   
def FCC(lattice_parameter, elements):
    unit_cell = cubic(lattice_parameter)
    base = np.array([(0.,0.,0.),(0.5,0.0,0.5),(0.5,0.5,0.),(0.,0.5,0.5)])
    if len(elements) >3:
        atoms = [elements[0], elements[1], elements[2], elements[3]]
    elif len(elements) >1:
        atoms = [elements[0], elements[1], elements[1], elements[1]]
    else:
        atoms = [elements[0]]*4
    return unit_cell, base, atoms

def dichalcogenide(a,c,u, elements):
    unit_cell= from_parameters(a, a, c, 90., 90., 120.)
    base = [(1/3., 2/3. , 1/4.),(2/3., 1/3. , 3/4.),
            (2/3., 1/3. , 1/4.+u),(2/3., 1/3. , 1/4.-u),
            (1/3., 2/3. , 3/4.+u), (1/3., 2/3. , 3/4.-u)]
    if len(elements) >1:
        atoms = [elements[0]]*2 + [elements[1]]*4
    else:
        atoms = elements[0]*6
    return unit_cell, base, atoms

def wurzite(a,c,u, elements):
    unit_cell= from_parameters(a, a, c, 90., 90., 120.)
    base = [(2./3.,1./3.,.500),(1./3.,2./3.,0.000),(2./3.,1./3.,0.5+u ),(1./3.,2./3.,u )]
    if len(elements) >1:
        atoms = [elements[0]]*2 + [elements[1]]*2
    else:
        atoms = [elements[0]]*4
    return unit_cell, base, atoms

def rocksalt(lattice_parameter, elements):
    unit_cell = np.identity(3) * lattice_parameter
    base = [(0.,0.,0.),(0.5,0.0,0.5),(0.5,0.5,0.),(0.,0.5,0.5)]
    base2 = np.array(base)+(0.5,0.5,0.5)
    base2 = np.where(base2==1. ,0, base2)
    base = base+ base2.tolist()
    if len(elements) >1:
        atoms = [elements[0]]*4 + [elements[1]]*4
    else:
        atoms = [elements[0]]*8
    return unit_cell, base, atoms


def zinc_blende(lattice_parameter, elements):
    unit_cell = np.identity(3) * lattice_parameter
    base = [(0.,0.,0.),(0.5,0.0,0.5),(0.5,0.5,0.),(0.,0.5,0.5)]
    base = base + (np.array(base) + (.25,.25,.25)).tolist()
    if len(elements) >3:
        atoms = [elements[0], elements[0], elements[1], elements[1], elements[2], elements[2], elements[3], elements[3]]
        #Z = [electronFF[elements[0]][Z],
    elif len(elements) >1:
        atoms = [elements[0]]*4 + [elements[1]]*4
    else:
        atoms = [elements[0]]*8
    return unit_cell, base, atoms

def Perovskite(lattice_parameter, elements):
    unit_cell = cubic(lattice_parameter)
    base = np.array([(0.,0.,0.),(0.5,0.5,0.5),(0.5,0.0,0.5),(0.5,0.5,0.),(0.,0.5,0.5)])
    if len(elements) >2:
        atoms = [elements[0], elements[1]]+ [elements[2]]*3
    elif len(elements) >1:
        atoms = [elements[0]]+ [elements[1]]*4
    else:
        atoms = [elements[0]]*5
    return unit_cell, base, atoms

def zone_mistilt (zone, angles):
    alpha, beta, gamma = np.radians(angles)

    # first we rotate alpha about x axis
    c, s = np.cos(alpha), np.sin(alpha)
    rotx = np.array([[1,0,0],[0, c, -s],[0, s,c]])
    
    
    # second we rotate beta about y axis
    c, s = np.cos(beta), np.sin(beta)
    roty = np.array([[c, 0 ,s],[0,1,0],[-s,0,c]])
    
    #third we rotate gamma about z-axis
    c, s = np.cos(gamma), np.sin(gamma)
    rotz = np.array([[c, -s , 0],[s,c,0],[0,0,1]])

    return np.dot(np.dot(np.dot(zone,rotx),roty),rotz)


def structure_by_name(crystal):
    """
        Provides unit cell as a structure matrix, the list of elements and the atom base
        
        Input:
        ------
            crystal name as string:
            type "print(ks.crystal_data_base.keys())" for a list of pre-defined crystal structures

            Please note that the chemical expressions are case sensitive.

        Ouput:
        ------
            new dictionary with the following keys:
            
            ['unit_cell']: the structure matrix
            ['base']:      relative coordinates of atoms
            ['elements']:     name of elements in same order as base

            an empty dictionary will be returned if the name is not recognized.
    """

    ## Check whether name is in the crystal_data_base
    if crystal in cdb:
        tags  = cdb[crystal]
    else:
        print('Crystal name not defined')
        return {}   

    ## Make crystal structure dictionary based on symmetry 
    if 'symmetry' in tags:
        if tags['symmetry'] == 'BCC':
            unit_cell, base, atoms = BCC(tags['a'], tags['elements'])
         
        elif tags['symmetry'] == 'FCC':
            unit_cell, base, atoms = FCC(tags['a'], tags['elements'])

        elif tags['symmetry'] == 'perovskite':
            unit_cell, base, atoms = Perovskite(tags['a'], tags['elements'])

        elif tags['symmetry'] == 'zinc_blende':
            unit_cell, base, atoms = zinc_blende(tags['a'], tags['elements'])
            
        elif tags['symmetry'] == 'wurzite':
            unit_cell, base, atoms = wurzite(tags['a'],tags['c'],tags['u'], tags['elements'])

        elif tags['symmetry'] == 'rocksalt':
            unit_cell, base, atoms = rocksalt(tags['a'], tags['elements'])

        elif tags['symmetry'] == 'dichalcogenide':
            unit_cell, base, atoms = dichalcogenide(tags['a'],tags['c'],tags['u'], tags['elements'])

        tags['unit_cell'] = unit_cell
        tags['elements'] =  atoms
        tags['base'] =base
    
    return tags

def get_Symmetry(unit_cell,base, atoms):
    if _spglib_present:

        print('#####################')
        print('# Symmetry Analysis #')
        print('#####################')
        
        Z = []
        for i in range(len(atoms)):
            a = atoms[i]
            b = base[i]
            Z.append(electronFF[a]['Z'])
            print('{0}: {1} = {2} : [{3:.2f}, {4:.2f}, {5:.2f}]'.format( i+1, a, Z[i],base[i][0],base[i][1],base[i][2] ) )

        lattice = (unit_cell,base,Z)
        spgroup = spglib.get_spacegroup(lattice)
        sym = spglib.get_symmetry(lattice)

        print("  Spacegroup  is %s." % spgroup)
        print('  Crystal has {0} symmetry operation'.format(sym['rotations'].shape[0]))

        Plattice, Ppositions, Pnumbers = spglib.find_primitive(lattice, symprec=1e-5)
        print("\n########################\n #Basis vectors of primitive Cell:")
        for i in range(3):
            print('[{0:.4f}, {1:.4f}, {2:.4f}]'.format( Plattice[i][0],Plattice[i][1],Plattice[i][2] ) )


        print('There {0} atoms and {1} species in primitive unit cell:'.format(len(Ppositions),Pnumbers))
    else:
        print('spglib is not installed')

import itertools
from scipy import spatial

def ball_and_stick(tags,extend=1, max_bond_length = 0.):
    
    """
    calculates the data to plor a ball and stick model
    
    Input:
    ------
    tags: dictionary containing the 'unit_cell', 'base' and 'elements' tags.
    
    extend: 1 or 3 integers
        The *extend* argument scales the effective cell in which atoms
        will be included. It must either be a list of three integers or a single
        integer scaling all 3 directions.  By setting this value to one, 
        all  corner and edge atoms will be included in the returned cell.  
        This will of cause make the returned cell non-repeatable, but this is 
        very useful for visualisation.
    
    max_bond_length: 1 float
        The max_bond_length argument defines the distance for which a bond will be shown.
        If max_bond_length is zero, the tabulated atom radii will be used.
        
        
    Output:
    -------
    corners,balls, Z, bonds: lists
        These lists can be used to plot the 
        unit cell:
            for x, y, z in corners:
                l=mlab.plot3d( x,y,z, tube_radius=0.002)
        bonds:
            for x, y, z in bonds:
                mlab.plot3d( x,y,z, tube_radius=0.02)
        and atoms:
            for i,atom in enumerate(balls):
                mlab.points3d(atom[0],atom[1],atom[2],
                              scale_factor=0.1,##ks.vdw_radii[Z[i]]/5,
                              resolution=20,
                              color=tuple(ks.jmol_colors [Z[i]]),
                              scale_mode='none')
                              
        Please note that you'll need the *Z* list for coloring, or for radii that depend on elements
    
    """
    ## Check in which form extend is given
    if isinstance(extend, int):
        extend = [extend]*3
        
    extend = np.array(extend, dtype=int)
    
    ## Make the x,y, and z multiplicators
    if len(extend) == 3:
        x  = np.linspace(0,extend[0],extend[0]+1)  
        y  = np.linspace(0,extend[1],extend[1]+1)  
        z  = np.linspace(0,extend[2],extend[2]+1)  
    else:
        print('wrong extend paramter')
        return   
    
    
    ## Check whether this is the right kind of dictionary                  
    if 'unit_cell' not in tags:
        return
    cell = tags['unit_cell']
    
    ### Corners and Outline of unit cell
    h = (0,1)
    corner_vectors = np.dot(np.array(list(itertools.product(h,h,h))) , cell)
    trace = [[0,1],[1,3],[3,2],[2,0], [0,4], [4,5], [5,7], [6,7], [6,4], [1,5], [2,6], [3,7]]
    corners = []
    for s, e in trace:
        corners.append([*zip(corner_vectors[s], corner_vectors[e])])

        
    ## ball position and elements in supercell
    super_cell  = np.array(list(itertools.product(x,y,z) )) # all evaluated Miller indices

    pos = np.add(super_cell,np.array(tags['base'])[:,np.newaxis])
    
    Z = []
    for i in range(len(tags['elements'])):
        Z.append(electronFF[tags['elements'][i]]['Z'])

    ## List of bond lengths taken from electronFF database below
    bond_lengths = []
    for atom in tags['elements']:
        bond_lengths.append(electronFF[atom]['bond_length'][0])
        
    # extend list of atomic numbers
    Zpp = []
    for z in Z:
        Zpp.append([z]*pos.shape[1]) 
    Zpp = np.array(Zpp).flatten()

    # reshape supercell atom positions
    pos = pos.reshape((pos.shape[0]*pos.shape[1],pos.shape[2]))
    
    ## Make a mask that excludes all atoms outside of super cell
    maskz = pos[:,2] <= extend[2]
    masky = pos[:,1] <= extend[1]
    maskx = pos[:,0] <= extend[0]
    mask = np.logical_and(maskx,np.logical_and(masky,maskz))#, axis=1)
    
    ## Only use balls and elements inside super cell
    balls =  np.dot(pos[mask],cell)
    Z= Zpp[mask]

    ## Get maximum bond length from list of bond -lengths taken from electronFF database 
    if max_bond_length == 0:
        max_bond_length = np.median(bond_lengths)/5
    ## Check nearest neighbours within max_bond_length
    tree = spatial.KDTree(balls)
    nearest_neighbours = np.array(tree.query(balls,k=8,distance_upper_bound = max_bond_length))


    ## Calculate bonds
    bonds = []
    bond_indices = []
    for i in range(nearest_neighbours.shape[1]):
        for j in range(nearest_neighbours.shape[2]):
            if nearest_neighbours[0,i,j]<max_bond_length:
                if nearest_neighbours[0,i,j]>0:
                    #print(atoms[i],atoms[int(bonds[1,i,j])],bonds[:,i,j])
                    bonds.append([*zip(balls[i], balls[int(nearest_neighbours[1,i,j])])])
                    bond_indices.append([i,int(nearest_neighbours[1,i,j])])
    
        
    return corners, balls, Z, bonds

def plot_unitcell_pyqtgraph(tags):
    
    #from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg
    
    if 'max_bond_length' not in tags:
        max_bond_length = 0.
    else:
        max_bond_length = tags['max_bond_length']

    if 'extend' not in tags:
        extend = 1
    else:
        extend = tags['extend']


    corners,balls, Z, bonds = ball_and_stick(tags,extend=extend, max_bond_length = max_bond_length)

    print('Now plotting')
    #app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 20000
    w.opts['fov'] = .01

    w.show()
    w.setWindowTitle('pyqtgraph example: GLImageItem')


    ax = gl.GLAxisItem()
    w.addItem(ax)

    def AtomSphere(radius,color):
        md = gl.MeshData.sphere(rows=30, cols=30, radius=radius, offset=True)
        colors = np.ones((md.faceCount(), 4), dtype=float)
        colors[:,0:3] = color[0:3]
        md.setFaceColors(colors)
        s = gl.GLMeshItem(meshdata=md, smooth=True,shader='shaded')    
        return s  

    size =.05

    for i,corner in enumerate(corners):
        line = gl.GLLinePlotItem(pos = np.array(list(zip(*corner))),width = 2 , color = (0,1,0,1))
        w.addItem(line)
    for i,bond in enumerate(bonds):
        line = gl.GLLinePlotItem(pos = np.array(list(zip(*bond))),width = 10 , color = (1,1,1,1))
        w.addItem(line)
    for i,atom in enumerate(balls):    
        sphere2 =AtomSphere(size,(jmol_colors [Z[i]]))
        sphere2.translate(atom[0],atom[1],atom[2])
        w.addItem(sphere2)

def plot_unitcell_nglview():
    try:
        pass
    except:
        pass
    
def plot_unitcell_mayavi(tags):
    """
        Makes a 3D plot of crystal structure

        Input:
        ------
            Dictionary with tags: 'unit_cell, 'elements', 'base'

        Output:
        -------
            3D plot

        Dependencies:
        -------------
            ball_and_stick function of KinsCat
            mlab of mayavi 
    """
    from mayavi import mlab

    # Make sure "max_bond_length" and "extend" variables are initialized
    if 'max_bond_length' not in tags:
        max_bond_length = 0.
    else:
        max_bond_length = tags['max_bond_length']

    if 'extend' not in tags:
        extend = 1
    else:
        extend = tags['extend']

    ## get balls, sticks and atomic numbers for colors and sizes
    corners,balls, Z, bonds = ball_and_stick(tags,extend=extend, max_bond_length = max_bond_length)

    print('Now plotting')
    fig = mlab.figure(1, bgcolor=(0, 0, 0), size=(350, 350))

    mlab.clf() # clear figure

    ## parallel projection
    mlab.gcf().scene.parallel_projection = True
    mlab.gcf().scene.camera.parallel_scale = 5

    # plot unit cell 
    for x, y, z in corners:
        l=mlab.plot3d( x,y,z, tube_radius=0.002)

    # plot bonds as sticks
    for x, y, z in bonds:
        mlab.plot3d( x,y,z, tube_radius=0.02)

    # plot atoms
    for i,atom in enumerate(balls):
        mlab.points3d(atom[0],atom[1],atom[2],
                      scale_factor=0.1,##ks.vdw_radii[Z[i]]/50,
                      resolution=20,
                      color=tuple(jmol_colors [Z[i]]),
                      scale_mode='none')


    ## parallel projection    
    mlab.gcf().scene.parallel_projection = True
    mlab.gcf().scene.camera.parallel_scale = 5
    #show plot
    mlab.show()
    

def plot_unitcell(tags):

    if 'max_bond_length' not in tags:
        max_bond_length = 0.
    else:
        max_bond_length = tags['max_bond_length']

    if 'extend' not in tags:
        extend = 1
    else:
        extend = tags['extend']


    corners,balls, Z, bonds = ball_and_stick(tags,extend=extend, max_bond_length = max_bond_length)
    
    fig = plt.figure();ax = fig.add_subplot(111, projection='3d')

    # draw unit_cell
    for x, y, z in corners:
        ax.plot3D( x,y,z, color="blue")

    # draw bonds
    for x, y, z in bonds:
        ax.plot3D( x,y,z, color="black", linewidth = 4)#, tube_radius=0.02)

    # draw atoms
    for i,atom in enumerate(balls):
        ax.scatter(atom[0],atom[1],atom[2],
                  color=tuple(jmol_colors [Z[i]]),
                  alpha = 1.0, s=200)

    ax.axis('equal')
    ax.set_aspect("equal")
    plt.show()

    
# The metric tensor of the lattice.
def metric_tensor( matrix):
    #Use like: metric_tensor(unit_cell)
    metric_tensor = np.dot(matrix, matrix.T)
    return metric_tensor

# Length of vextor
def vector_norm(g):
    return np.sqrt(g[:,0]**2+g[:,1]**2+g[:,2]**2)

def make_pretty_labels(hkls, hex_label = False):
    """
    ###########################
    # Make pretty labels
    ###########################
    Input:
    ------
        hkls: a numpy array with all the Miller indices to be labeled
    Optional:
    ---------
        hex_label: if True this will make for Miller indices.

    Ouput:
    ------
        list of labels in Latex format
    """
    hkl_label = []
    for i in range (len(hkls)):
        h,k,l = np.array(hkls)[i]
        
        if h < 0:
            h_string = r'[$\bar {'+str(int(-h))+'},'
        else:
            h_string = r'[$\bar {'+str(int(h))+'},'
        if k < 0:
            k_string = r'\bar {'+str(int(-k))+'},'
        else:
            k_string = str(int(k))+','
        if hex_label:
            ii = -(h+k)
            if ii<0:
                k_string = kstring + r'\bar {'+str(int(-ii))+'},'
            else:
                k_string = kstring + str(int(ii))+','
        if l < 0:
            l_string = r'\bar {'+str(int(-l))+'} $]'
        else:
            l_string = str(int(l))+'} $]'
        label = h_string+k_string+l_string
        hkl_label.append(label)
    return hkl_label

# Wavelength in 1/nm
def get_waveLength(E0):
    """
    Calculates the relativistic corrected de Broglie wave length of an electron

    Input:
    ------
        acceleration voltage in volt
    Output:
    -------
        wave length in nm
    """

    eV = const.e * E0 
    return const.h/np.sqrt(2*const.m_e*eV*(1+eV/(2*const.m_e*const.c**2)))*10**9
    

# Rotation of zone axis by mistilt
def zone_mistilt (zone, angles):
    alpha, beta, gamma = np.radians(angles)

    # first we rotate alpha about x axis
    c, s = np.cos(alpha), np.sin(alpha)
    rotx = np.array([[1,0,0],[0, c, -s],[0, s,c]])
    
    
    # second we rotate beta about y axis
    c, s = np.cos(beta), np.sin(beta)
    roty = np.array([[c, 0 ,s],[0,1,0],[-s,0,c]])
    
    #third we rotate gamma about z-axis
    c, s = np.cos(gamma), np.sin(gamma)
    rotz = np.array([[c, -s , 0],[s,c,0],[0,0,1]])

    return np.dot(np.dot(np.dot(zone,rotx),roty),rotz)

# Determine rotation matrix to tilt zone axis onto z-axis
# We determine spherical coordinates to do that

def get_rotation_matrix(zone, verbose = False):
    #spherical coordinates of zone
    r = np.sqrt((zone*zone).sum())
    theta = np.arccos(zone[2]/r)
    if zone[0] < 0:
        theta = -theta
    if zone[0] == 0:
        phi= np.pi/2
    else: 
        phi = (np.arctan(zone[1]/zone[0]))

    if verbose:
        print('Rotation theta ',np.degrees(theta),' phi ',np.degrees(phi))
    #unit = np.array([[1, 0 , 0],[0,1,0],[0,0,1]])

    #first we rotate phi about z-axis
    c, s = np.cos(phi), np.sin(phi)
    rotz = np.array([[c, -s , 0],[s,c,0],[0,0,1]])

    # second we rotate theta about y axis
    c, s = np.cos(theta), np.sin(theta)
    roty = np.array([[c, 0 ,s],[0,1,0],[-s,0,c]])
    
    # the rotation now makes z-axis coincide with plane normal
    return np.dot(rotz,roty), np.degrees(theta), np.degrees(phi)

def check_sanity(tags):
    ############################################
    #Check sanity
    ############################################
    stop = False
    if 'unit_cell' not in tags:
        print(' No unit_cell defined')
        stop = True
    if 'base' not in tags:
        print(' No base defined')
        stop = True
    if 'elements' not in tags:
        print(' No atoms defined')
        stop = True
    if 'acceleration_voltage_V' not in tags:
        print(' No acceleration_voltage_V defined')
        stop = True
    if 'zone_hkl' not in tags:
        print(' No zone_hkl defined')
        stop = True
    if 'Sg_max' not in tags:
        print(' No Sg_max defined')
        stop = True
    if 'hkl_max' not in tags:
        print(' No hkl_max defined')
        stop = True
    
    if 'crystal_name' not in tags:
        tags['crystal_name'] = 'undefined'
        print('tags[\'crystal\'] = \'undefined\'')
    if 'mistilt' not in tags:
        tags['mistilt'] = [0.,0.,0.]
        print ('tags[\'mistilt\'] = [0.,0.,0.]')
    if 'convergence_angle_mrad' not in tags:
        tags['convergence_angle_mrad'] = 0. 
        print ('tags[\'convergence_angle_mrad\'] = 0')
    
    return stop
def Ring_Pattern_Calculation(tags, verbose = False):
    #Reciprocal Lattice 
    # We use the linear algebra package of numpy to invert the unit_cell "matrix"
    reciprocal_unit_cell = np.linalg.inv(tags['unit_cell']).T # transposed of inverted unit_cell
    
    #INPUT
    hkl_max = 7#  maximum allowed Miller index
    
    acceleration_voltage = 200.0 *1000.0 #V
    wave_length  = get_waveLength(acceleration_voltage)
    
    
    
    h  = np.linspace(-hkl_max,hkl_max,2*hkl_max+1)   # all to be evaluated single Miller Index
    hkl  = np.array(list(itertools.product(h,h,h) )) # all to be evaluated Miller indices
    g_hkl = np.dot(hkl,reciprocal_unit_cell)  
    
    # Calculate Structure Factors
    
    structure_factors = []
    
    base = np.dot(tags['base'],tags['unit_cell'])  # transformation from relative to Carthesian coordinates
    for j  in range(len(g_hkl)):
        F = 0
        for b in range(len(base)):
            f = feq(tags['elements'][b],np.linalg.norm(g_hkl[j])) # Atomic form factor for element and momentum change (g vector)
            F += f * np.exp(-2*np.pi*1j*(g_hkl[j]*base[b]).sum())        
        structure_factors.append(F)
    F = structure_factors = np.array(structure_factors)
    
    # Allowed reflections have a non zero structure factor F (with a  bit of numerical error)
    allowed = np.absolute(structure_factors) > 0.001
    
    distances = np.linalg.norm(g_hkl, axis = 1)
    
    if verbose:
        print(f' Of the evaluated {hkl.shape[0]} Miller indices {allowed.sum()} are allowed. ')
    # We select now all the 
    zero = distances == 0.
    allowed = np.logical_and(allowed,np.logical_not(zero))
    
    F = F[allowed]
    g_hkl = g_hkl[allowed]
    hkl = hkl[allowed]
    distances = distances[allowed]
    
    sorted_allowed = np.argsort(distances)
    
    distances = distances[sorted_allowed]
    hkl = hkl[sorted_allowed]
    F = F[sorted_allowed]
    
    # How many have unique distances and what is their muliplicity
    
    unique, indices  = np.unique(distances, return_index=True)
    
    if verbose:
        print(f' Of the {allowed.sum()} allowed Bragg reflections there are {len(unique)} families of reflections.')
    
    intensity = np.absolute(F[indices]**2*(np.roll(indices,-1)-indices))
    if verbose:
        print('\n index \t  hkl \t      1/d [1/nm]       d [pm]     F     multip.  intensity' )
    family = []
    out_tags={}
    for j in range(len(unique)-1):
        i = indices[j]    
        i2 = indices[j+1]   
        family.append(hkl[i+np.argmax(hkl[i:i2].sum(axis=1))])
        index = '{'+f'{family[j][0]:.0f} {family[j][1]:.0f} {family[j][2]:.0f}'+'}'
        if verbose:
            print(f'{i:3g}\t {index} \t  {distances[i]:.2f}  \t {1/distances[i]*1000:.0f} \t {np.absolute(F[i]):.2f}, \t  {indices[j+1]-indices[j]:3g} \t {intensity[j]:.2f}') 
        out_tags[index]={}
        out_tags[index]['reciprocal_distance']=distances[i]
        out_tags[index]['real_distance']=1/distances[i]
        out_tags[index]['F']=np.absolute(F[i])
        out_tags[index]['multiplicity']=indices[j+1]-indices[j]
        out_tags[index]['intensity']=intensity[j]
    return out_tags  
        
        
    
# All the kinematic scattering
# Calculates SAED, Kikuchi, and HOLZ pattern
def Kinematic_Scattering( tags, verbose=False ):
    """
        All kinematic scattering calculation
        Calculates Bragg spots, Kikuchi lines, excess, and deficient HOLZ lines

        Input:
        ------
            a dictionary with crystal structure:
                'unit_cell', 'base' 'elements'
            and with experimental parameters:
                'acceleration_voltage_V', 'zone_hkl', 'Sg_max', 'hkl_max'
            Optional parameters are:
                'mistilt', convergence_angle_mrad', and 'crystal_name'

            vebose = True will give extended output of the calculation

        Output to dictionary:
        ---------------------

        There are three sub_dictinaries: ['allowed'], ['forbidden'], and ['HOLZ']
        ['allowed'] and ['forbidden'] dictionaries contain:
            ['Sg'], ['hkl'], ['g'], ['structure factor'], ['intensities'],
            ['ZOLZ'], ['FOLZ'], ['SOLZ'], ['HOLZ'], ['HHOLZ'], ['label'], and ['Laue_zone']
        the ['HOLZ'] dictionary contains:
            ['slope'], ['distance'], ['theta'], ['g deficient'], ['g excess'], ['hkl'], ['intensities'],
            ['ZOLZ'], ['FOLZ'], ['SOLZ'], ['HOLZ'], and  ['HHOLZ']
        Please note that the Kikuchi lines are the HOLZ lines of ZOLZ

        There are also a few parameters stored in the main dictionary:
            ['wave_length_nm'], ['reciprocal_unit_cell'], ['inner_potential_V'], ['incident_wave_vector'],
            ['volume'], ['theta'], ['phi'], and ['incident_wave_vector_vacuum']
    """

    if check_sanity(tags):
        print('Input is not complete, stopping')
        print('Try \'example()\' for example input')
        return

    tags['wave_length_nm'] = get_waveLength(tags['acceleration_voltage_V'])
    
    ############################################
    #reciprocal_unit_cell 
    ############################################
    unit_cell = np.array(tags['unit_cell'])
    # We use the linear algebra package of numpy to invert the unit_cell "matrix"
    reciprocal_unit_cell = np.linalg.inv(unit_cell).T # transposed of inverted unit_cell
    tags['reciprocal_unit_cell'] = reciprocal_unit_cell
    
    if verbose:
        print('reciprocal_unit_cell')
        print(np.round(reciprocal_unit_cell,3))
        
    ############################################
    #Incident wave vector K0 in vacuum and material 
    ############################################

    #Incident wave vector K0 in vacuum and material 
    U0 = 0
    for i in range(len(tags['elements'])):
        a = tags['elements'][i]
        U0 += feq(a,0)*0.023933754

    tags['volume'] = np.linalg.det(tags['unit_cell'])
    volume = tags['volume']*1000. # Needs to be in Angstrom for form factors
    AngstromConversion = 1.0e10 # So [1A (in m)] * AngstromConversion = 1
    NanometerConversion = 1.0e9 

    ScattFacToVolts=(const.h**2)*(AngstromConversion**2)/(2*np.pi*const.m_e*const.e)*volume
    U0=U0*ScattFacToVolts
    tags['inner_potential_A'] = U0   
    tags['inner_potential_V'] = U0*ScattFacToVolts
    if verbose:
        print('The inner potential is {0:.3f}kV'.format(tags['inner_potential_V']/1000))

    # Calculating incindent wave vector magnitude 'K0' in material
    wl = tags['wave_length_nm']
    tags['incident_wave_vector_vacuum'] = 1/wl
    
    K0 = tags['incident_wave_vector'] = np.sqrt(1/wl**2 -(U0/volume*100.)) #1/nm

    tags['convergence_angle_nm-1']  = K0*np.sin(tags['convergence_angle_mrad']/1000.)

    if verbose:
        print('Magnitude of incident wave vector in material {0:.1f} 1/nm and vacuum {1:.1f} 1/nm'.format(K0, 1/wl))
        print('The convergence angle of {0}mrad = {1:.2f} 1/nm'.format(tags['convergence_angle_mrad'], tags['convergence_angle_nm-1']))

    
    #############
    # Rotate
    #############
    
    # first we take care of mistilt: zone axis is then in fractional Miller indices
    zone = tags['zone'] = zone_mistilt(tags['zone_hkl'], tags['mistilt'])

    # zone axis in global coordinate system
    zone_vector = np.dot(zone,reciprocal_unit_cell)

    rotation_matrix, theta, phi = get_rotation_matrix(zone_vector, verbose=False)
    
    if verbose:
        print('Rotation angles are {0:.1f} deg and {1:.1f} deg'.format(theta, phi))
    tags['theta'] = theta
    tags['phi'] = phi
    
    # rotate incident wave vector 
    w_vector = np.dot(zone_vector, rotation_matrix)
    K0_unit_vector = w_vector/ np.linalg.norm(w_vector) # incident unit wave vector 
    K0_vector =K0_unit_vector*K0                        # incident  wave vector 
       
    if verbose:
        print('Center of Ewald sphere ' ,K0_vector)

    ########################
    #Find all Miller indices whose reciprocal point lays near the Ewald sphere with radius K0 
    #within a maximum excitation error Sg
    ########################
    
    hkl_max = tags['hkl_max']
    Sg_max  = tags['Sg_max'] # 1/nm  maximum allowed excitation error
    
    h  = np.linspace(-hkl_max,hkl_max,2*hkl_max+1)  # all evaluated single Miller Indices
    hkl  = np.array(list(itertools.product(h,h,h) )) # all evaluated Miller indices
    g = np.dot(hkl,reciprocal_unit_cell) # all evaluated reciprocal_unit_cell points
    g_norm = np.linalg.norm(g, axis=1)   # length of all vectors
    not_zero =  g_norm > 0
    g = g[not_zero]  # zero reflection will make problems further on, so we exclude it.
    g_norm = g_norm[not_zero]
    hkl = hkl[not_zero]
    g_non_rot = g
    g = np.dot(g, rotation_matrix)
    
    ########################
    # Calculate exitation errors for all reciprocal_unit_cell points
    ########################
    
    ## Zuo and Spence, 'Adv TEM', 2017 -- Eq 3:14
    #S=(K0**2-np.linalg.norm(g - K0_vector, axis=1)**2)/(2*K0)
    gMz = g - K0_vector

    in_sqrt = gMz[:,2]**2  + np.linalg.norm(gMz,axis=1)**2 - K0**2
    in_sqrt[in_sqrt <0] = 0.
    S = -gMz[:,2] - np.sqrt(in_sqrt)
    
    ########################
    # Determine reciprocal_unit_cell points with excitation error less than the maximum allowed one: Sg_max
    ########################
    
    reflections = abs(S)< Sg_max   #This is now a boolean array with True for all possible reflections
    
    Sg = S[reflections]
    g_hkl =g[reflections]
    g_hkl_non_rot = g_non_rot[reflections]
    hkl = hkl[reflections]
    g_norm = g_norm[reflections]

    if verbose:
        print ('Of the {0} tested reciprocal_unit_cell points, {1} have an excitation error less than {2:.2f} 1/nm'.format( len(g) , len(g_hkl), Sg_max))
        
    ##################################
    # Calculate Structure Factors
    #################################
    
    structure_factors = []
    """for j  in range(len(g_hkl)):
        F = 0
        for b in range(len(tags['base'])):
            f = feq(tags['elements'][b],np.linalg.norm(g_hkl[j]))
            #F += f * np.exp(-2*np.pi*1j*(hkl*tags['base'][b]).sum()) # may only work for cubic Gerd
            F += f * np.exp(-2*np.pi*1j*(g_hkl_non_rot*np.dot(tags['base'][b],unit_cell)).sum())
            

        structure_factors.append(F)
    
    F = structure_factors = np.array(structure_factors)
    """
    base = np.dot(tags['base'],tags['unit_cell'])  # transformation from relative to Carthesian coordinates
    for j  in range(len(g_hkl)):
        F = 0
        for b in range(len(base)):
            f = feq(tags['elements'][b],g_norm[j]) # Atomic form factor for element and momentum change (g vector)
            F += f * np.exp(-2*np.pi*1j*(g_hkl_non_rot[j]*base[b]).sum())        
        structure_factors.append(F)
    F = structure_factors = np.array(structure_factors)
    
    ############################################
    # Sort reflection in allowed and forbidden #
    ############################################
    
    allowed = np.absolute(F) > 0.000001    # allowed within numerical error 

    if verbose:
        print('Of the {0} possible reflection {1} are allowed.'.format(hkl.shape[0], allowed.sum() ) )
        
    # information of allowed reflections
    Sg_allowed = Sg[allowed]
    hkl_allowed = hkl[allowed][:]
    g_allowed = g_hkl[allowed,:]
    F_allowed = F[allowed]
    g_norm_allowed = g_norm[allowed]
    
    tags['allowed'] = {}
    tags['allowed']['Sg'] = Sg_allowed
    tags['allowed']['hkl'] = hkl_allowed
    tags['allowed']['g'] = g_allowed
    tags['allowed']['structure factor'] = F_allowed
    
    
    
    # information of forbidden reflections
    forbidden = np.logical_not(allowed)
    Sg_forbidden = Sg[forbidden]
    hkl_forbidden = hkl[forbidden]
    g_forbidden = g_hkl[forbidden]
    
    tags['forbidden'] = {}
    tags['forbidden']['Sg'] = Sg_forbidden
    tags['forbidden']['hkl'] = hkl_forbidden
    tags['forbidden']['g'] = g_forbidden

    
        ###########################
    # Make pretty labels
    ###########################
    hkl_label = make_pretty_labels(hkl_allowed)
    tags['allowed']['label'] = hkl_label
    
    #hkl_label = make_pretty_labels(hkl_forbidden)
    #tags['forbidden']['label'] = hkl_label
    
    ############################
    # Calculate Intensities (of allowed reflections)
    ############################
    
    intensities = np.absolute(F_allowed)**2

    
    tags['allowed']['intensities'] = intensities
    
    ############################
    # Calculate Laue Zones (of allowed reflections)
    ############################
    ## Below is the expresion given in most books.
    ## However, that would only work for orthogonal crystal systems
    #Laue_Zone = abs(np.dot(hkl_allowed,tags['zone_hkl']))  # works only for orthogonal systems

    # This expression works for all crystal systems
    # Remember we have already tilted, and so the dot product is trivial and gives only the z-component.
    length_zone_axis = np.linalg.norm(np.dot(tags['zone_hkl'], tags['unit_cell']))
    Laue_Zone = abs(np.floor(g_allowed[:,2]*length_zone_axis+0.5))
    
    tags['allowed']['Laue_Zone'] = Laue_Zone

    ZOLZ_forbidden = abs(np.floor(g_forbidden[:,2]*length_zone_axis+0.5))==0
    
    tags['forbidden']['Laue_Zone'] = ZOLZ_forbidden
    ZOLZ = Laue_Zone == 0
    FOLZ = Laue_Zone == 1
    SOLZ = Laue_Zone == 2
    HOLZ = Laue_Zone > 0
    HOLZp = Laue_Zone > 2

    tags['allowed']['ZOLZ'] = ZOLZ
    tags['allowed']['FOLZ'] = FOLZ
    tags['allowed']['SOLZ'] = SOLZ
    tags['allowed']['HOLZ'] = HOLZ
    tags['allowed']['HOLZ_plus'] = tags['allowed']['HHOLZ'] = HOLZp
    
    if verbose:
        print(' There are {0} allowed reflections in the zero order Laue Zone'.format(ZOLZ.sum() ))
        print(' There are {0} allowed reflections in the first order Laue Zone'.format((Laue_Zone ==1).sum() ))
        print(' There are {0} allowed reflections in the second order Laue Zone'.format((Laue_Zone ==2).sum() ))
        print(' There are {0} allowed reflections in the other higher order Laue Zones'.format((Laue_Zone>2).sum() ))
        
    if verbose==2:
        print(' hkl  \t Laue zone \t Intensity (*1 and \t log) \t length \n')
        for i in range(len(hkl_allowed)):
            print(' {0} \t {1} \t {2:.3f} \t  {3:.3f} \t  {4:.3f}   '.format(hkl_allowed[i], g_allowed[i], intensities[i], np.log(intensities[i]+1) , g_norm_allowed[i]) )  
   
    ###########################
    # Dynamically Activated forbidden reflections
    ###########################
   

    
    double_diffraction=(np.sum(np.array(list(itertools.combinations(hkl_allowed[ZOLZ],2))),axis=1))
    
    dynamical_allowed = []
    still_forbidden = []
    for i, hkl in enumerate(hkl_forbidden):
        if ZOLZ_forbidden[i]:
            if hkl.tolist() in double_diffraction.tolist():
                dynamical_allowed.append(i)
            else:
                still_forbidden.append(i)
    tags['forbidden']['dynamically_activated']= dynamical_allowed
    tags['forbidden']['forbidden']= dynamical_allowed
    if verbose:
        print('Length of zone axis vector in real space {0} nm'.format(np.round(length_zone_axis,3)))
        print(f'There are {len(dynamical_allowed)} forbidden but dynamical activated diffraction spots:')
        #print(tags['forbidden']['hkl'][dynamical_allowed])



    ####################################
    # Calculate HOLZ and Kikuchi Lines #
    ####################################

    ## Dynamic Correction

    #Equation Spence+Zuo 3.86a
    gamma_1 = - 1./(2.*K0) * (intensities/ (2.*K0*Sg_allowed)).sum()
    #print('gamma_1',gamma_1)

    #Equation Spence+Zuo 3.84
    Kg = K0 - K0*gamma_1/(g_allowed[:,2]+1e-15)
    Kg[ZOLZ] = K0
    
    #print(Kg, Kg.shape)

    # Calculate angle between K0 and deficient cone vector
    # For dynamic calculations K0 is replaced by Kg
    Kg[:] = K0
    dtheta = np.arcsin(g_norm_allowed/Kg/2.)-np.arcsin(np.abs(g_allowed[:,2])/g_norm_allowed)

    # calculate length of distance of deficient cone to K0 in ZOLZ plane
    gd_length =2*np.sin((dtheta)/2 )*K0
    
    # Calculate nearest point of HOLZ and Kikuchi lines
    g_closest = g_allowed.copy()
    g_closest =  g_closest*(gd_length/np.linalg.norm(g_closest, axis =1))[:,np.newaxis]
    
    g_closest[:,2] = 0.
    
    ### calculate and save line in Hough space coordinates (distance and theta)
    slope = g_closest[:,0]/(g_closest[:,1]+1e-10)
    distance = gd_length
    theta = np.arctan2(g_allowed[:,0],g_allowed[:,1])
    
                        
    tags['HOLZ'] = {}
    tags['HOLZ']['slope'] = slope
    # a line is now given by
    
    tags['HOLZ']['distance'] = distance
    tags['HOLZ']['theta'] = theta


    
    tags['HOLZ']['g deficient'] = g_closest
    tags['HOLZ']['g excess'] = g_closest+g_allowed
    
    tags['HOLZ']['ZOLZ'] = ZOLZ
    tags['HOLZ']['HOLZ'] = HOLZ
    tags['HOLZ']['FOLZ'] = FOLZ
    tags['HOLZ']['SOLZ'] = SOLZ
    tags['HOLZ']['HHOLZ']= HOLZp # even higher HOLZ
    
    tags['HOLZ']['hkl'] = tags['allowed']['hkl']
    tags['HOLZ']['intensities']=intensities
    
    
    if verbose:
        print('KinsCat\'s  \"Kinematic_Scattering\" finished')

#####################
# Plot SAED Pattern #
#####################
def plotSAED(tags, gray=False):

    saed = tags.copy()
    saed['convergence_angle_nm-1'] = 0
    
    saed['background'] = 'white'   # 'white'  'grey'
    saed['color map'] = 'plasma'#,'cubehelix'#'Greys'#'plasma'
    saed['color reflections'] = 'ZOLZ'
    
    if gray:
        saed['color map'] = 'gray'
        saed['background'] = '#303030'#'darkgray'
        saed['color reflections'] = 'intensity'
    saed['plot HOLZ'] = 0
    saed['plot HOLZ excess'] = 0
    saed['plot Kikuchi'] = 0
    saed['plot reflections'] = 1

    saed['label HOLZ'] = 0
    saed['label Kikuchi'] = 0
    saed['label reflections'] = 0

    saed['label color'] = 'white'
    saed['label size'] = 10

    saed['color Laue Zones'] = ['red',  'blue', 'green', 'blue', 'green']#, 'green', 'red'] #for OLZ give a sequence
    saed['color zero'] = 'red'#'None' #'white'
    saed['color ring zero'] = 'None'# 'Red' #'white' #, 'None'
    saed['width ring zero'] = 2
 
    
    plotDiffPattern(saed,True)
    
########################
# Plot Kikuchi Pattern #
########################
def plotKikuchi(tags, grey=False):
    Kikuchi = tags.copy()
    
    Kikuchi['background'] = 'black'   # 'white'  'grey'
    Kikuchi['color map'] = 'plasma'#,'cubehelix'#'Greys'#'plasma'
    Kikuchi['color reflections'] = 'intensity'

    Kikuchi['plot HOLZ'] = 0
    Kikuchi['plot HOLZ excess'] = 0
    Kikuchi['plot Kikuchi'] = 1
    Kikuchi['plot reflections'] = 1

    Kikuchi['label HOLZ'] = 0
    Kikuchi['label Kikuchi'] = 0
    Kikuchi['label reflections'] = 0

    Kikuchi['label color'] = 'white'
    Kikuchi['label size'] = 10

    Kikuchi['color Kikuchi'] = 'green'
    Kikuchi['linewidth HOLZ'] = -1 # -1: linewidth according to intensity (structure factor F^2
    Kikuchi['linewidth Kikuchi'] = -1 # -1: linewidth according to intensity (structure factor F^2
    

    Kikuchi['color Laue Zones'] = ['red',  'blue', 'green', 'blue', 'green']#, 'green', 'red'] #for OLZ give a sequence
    Kikuchi['color zero'] = 'white'#'None' #'white'
    Kikuchi['color ring zero'] = 'None'# 'Red' #'white' #, 'None'
    Kikuchi['width ring zero'] = 2
    
    
    plotDiffPattern(Kikuchi,True)
########################
# Plot HOLZ Pattern #
########################

def plotHOLZ(tags, grey=False):
    holz = tags.copy()
    
    holz['background'] = 'black'   # 'white'  'grey'
    holz['color map'] = 'plasma'#,'cubehelix'#'Greys'#'plasma'
    holz['color reflections'] = 'intensity'

    holz['plot HOLZ'] = 1
    holz['plot HOLZ excess'] = 0
    holz['plot Kikuchi'] = 1
    holz['plot reflections'] = 0

    holz['label HOLZ'] = 0
    holz['label Kikuchi'] = 0
    holz['label reflections'] = 0

    holz['label color'] = 'white'
    holz['label size'] = 10

    holz['color Kikuchi'] = 'green'
    holz['linewidth HOLZ'] = -1 # -1: linewidth according to intensity (structure factor F^2
    holz['linewidth Kikuchi'] = -1 # -1: linewidth according to intensity (structure factor F^2
    

    holz['color Laue Zones'] = ['red',  'blue', 'green']#, 'green', 'red'] #for OLZ give a sequence
    holz['color zero'] = 'white'#'None' #'white'
    holz['color ring zero'] = 'Red'# 'Red' #'white' #, 'None'
    holz['width ring zero'] = 2
    
    plotDiffPattern(holz,True)


########################
# Plot CBED Pattern #
########################

def plotCBED(tags, grey=False):
    cbed = tags.copy()
    
    cbed['background'] = 'black'   # 'white'  'grey'
    cbed['color map'] = 'plasma'#,'cubehelix'#'Greys'#'plasma'
    cbed['color reflections'] = 'intensity'

    cbed['plot HOLZ'] = 1
    cbed['plot HOLZ excess'] = 1
    cbed['plot Kikuchi'] = 1
    cbed['plot reflections'] = 1

    cbed['label HOLZ'] = 0
    cbed['label Kikuchi'] = 0
    cbed['label reflections'] = 0

    cbed['label color'] = 'white'
    cbed['label size'] = 10

    cbed['color Kikuchi'] = 'green'
    cbed['linewidth HOLZ'] = -1 # -1: linewidth according to intensity (structure factor F^2
    cbed['linewidth Kikuchi'] = -1 # -1: linewidth according to intensity (structure factor F^2

    cbed['color reflections'] = 'intensity'

    cbed['color Laue Zones'] = ['red',  'blue', 'green']#, 'green', 'red'] #for OLZ give a sequence
    cbed['color zero'] = 'white'#'None' #'white'
    cbed['color ring zero'] = 'Red'# 'Red' #'white' #, 'None'
    cbed['width ring zero'] = 2
    
    plotDiffPattern(cbed,True)


    
########################
# Plot HOLZ Pattern #
########################

from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.collections import PatchCollection

def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter plot of circles. 
    Similar to plt.scatter, but the size of circles are in data scale.
    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, ) 
        Radius of circles.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence 
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)  
        `c` can be a 2-D array in which the rows are RGB or RGBA, however. 
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls), 
        norm, cmap, transform, etc.
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
    Examples
    --------
    a = np.arange(11)
    circles(a, a, s=a*0.2, c=a, alpha=0.5, ec='none')
    plt.colorbar()
    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    zipped = np.broadcast(x, y, s)
    patches = [Circle((x_, y_), s_)
               for x_, y_, s_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection


def plotDiffPattern(tags, grey=False):


    if 'plot_rotation' not in tags:
        tags['plot_rotation'] = 0.

    if 'new_plot' not in tags:
        tags['new_plot'] = True

    if tags['new_plot']:
        fig = plt.figure()
        ax = fig.add_subplot(111, facecolor=tags['background'])
        
        DiffPattern(tags, grey)
            
        plt.axis('equal')
        if 'plot FOV' in tags:
            x = tags['plot FOV'] # in 1/nm
            plt.xlim(-x,x)
            plt.ylim(-x,x)
        plt.title( tags['crystal_name'])
        plt.show()
    else:
        DiffPattern(tags, grey)



def DiffPattern(tags, grey=False):
    
    # Get information from dictionary
    HOLZ = tags['allowed']['HOLZ']
    ZOLZ = tags['allowed']['ZOLZ']
    
    Laue_Zone = tags['allowed']['Laue_Zone']

    if 'label' in tags['allowed']:
        label = tags['allowed']['label']


    angle = np.radians(tags['plot_rotation']) #mrad
    c = np.cos(angle)
    s = np.sin(angle)
    r_mat = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    
    # HOLZ and Kikuchi lines coordinates in Hough space
    gd = np.dot(tags['HOLZ']['g deficient'],r_mat)
    ge = np.dot(tags['HOLZ']['g excess'],r_mat)

    points =  np.dot(tags['allowed']['g'],r_mat)
    
    theta = tags['HOLZ']['theta']+angle
    
    intensity = tags['allowed']['intensities']
    radius = tags['convergence_angle_nm-1']
    tags['Bragg'] = {}
    tags['Bragg']['points'] = points
    tags['Bragg']['intensity'] = intensity
    tags['Bragg']['radius'] = radius
    
    if radius < 0.1:
        radiusI = 2
    else:
        radiusI = radius
    # Beginning and ends of HOLZ lines
    maxlength = radiusI *1.3
    Hxp = gd[:,0] + maxlength * np.cos(np.pi-theta);
    Hyp = gd[:,1] + maxlength * np.sin(np.pi-theta);
    Hxm = gd[:,0] - maxlength * np.cos(np.pi-theta);
    Hym = gd[:,1] - maxlength * np.sin(np.pi-theta);
    tags['HOLZ lines'] = {}
    tags['HOLZ lines']['Hxp'] = Hxp
    tags['HOLZ lines']['Hyp'] = Hyp
    tags['HOLZ lines']['Hxm'] = Hxm
    tags['HOLZ lines']['Hym'] = Hym

    # Beginning and ends of excess HOLZ lines
    maxlength = radiusI *.8
    Exp = ge[:,0] + maxlength * np.cos(np.pi-theta);
    Eyp = ge[:,1] + maxlength * np.sin(np.pi-theta);
    Exm = ge[:,0] - maxlength * np.cos(np.pi-theta);
    Eym = ge[:,1] - maxlength * np.sin(np.pi-theta);
    tags['excess HOLZ lines'] = {}
    tags['excess HOLZ lines']['Exp'] = Exp
    tags['excess HOLZ lines']['Eyp'] = Eyp
    tags['excess HOLZ lines']['Exm'] = Exm
    tags['excess HOLZ lines']['Eym'] = Eym

    # Beginning and ends of HOLZ lines
    maxlength = 20
    Kxp = gd[:,0] + maxlength * np.cos(np.pi-theta);
    Kyp = gd[:,1] + maxlength * np.sin(np.pi-theta);
    Kxm = gd[:,0] - maxlength * np.cos(np.pi-theta);
    Kym = gd[:,1] - maxlength * np.sin(np.pi-theta);
    tags['Kikuchi lines'] = {}
    tags['Kikuchi lines']['Kxp'] = Kxp
    tags['Kikuchi lines']['Kyp'] = Kyp
    tags['Kikuchi lines']['Kxm'] = Kxm
    tags['Kikuchi lines']['Kym'] = Kym
    
    intensity_Kikuchi = intensity*4./intensity[ZOLZ].max()
    if len(intensity[tags['HOLZ']['HOLZ']])>1:
        intensity_HOLZ = intensity*4./intensity[tags['HOLZ']['HOLZ']].max()*.75
        tags['HOLZ lines']['intensity_HOLZ'] = intensity_HOLZ
    tags['Kikuchi lines']['intensity_Kikuchi'] = intensity_Kikuchi
    ########
    # Plot #
    ########
    cms = mpl.cm
    #cm = cms.plasma#jet#, cms.gray, cms.autumn]
    cm = plt.get_cmap(tags['color map']) 

    fig = plt.gcf()
    
    ax = plt.gca()

    
    if 'plot image' in tags:
        l = -tags['plot image FOV']/2+tags['plot shift x']
        r = tags['plot image FOV']/2+tags['plot shift x']
        t = -tags['plot image FOV']/2+tags['plot shift y']
        b = tags['plot image FOV']/2+tags['plot shift y']
        ax.imshow(tags['plot image'], extent=(l,r,t,b))
        print('image') 
    
    ix = np.argsort( (points**2).sum(axis=1))
    #print(tags['allowed']['hkl'][ix])
    p = points[ix]
    inten = intensity[ix]
    tags['Bragg']['points'] = p
    tags['Bragg']['intensity'] = inten

    
    Lauecolor =[]
    for i in range(int(Laue_Zone.max())+1):
        if i < len(tags['color Laue Zones']):
            Lauecolor.append(tags['color Laue Zones'][i])
        else:
            Lauecolor.append(tags['color Laue Zones'][-1])
    
    if tags['plot reflections']:
        if radius < 0.1:
            if tags['color reflections'] == 'intensity':
                ax.scatter(points[:,0], points[:,1], c=np.log(intensity),cmap=cm, s=20)
            else:
                for i in range(len(Laue_Zone)):
                    color = Lauecolor[int(Laue_Zone[i])]
                    ax.scatter(points[i,0], points[i,1], c=color,cmap=cm, s=20)
                
            ax.scatter(0,0, c=tags['color zero'], s=100)
            radius = 2
        else:
            ix = np.argsort( (points**2).sum(axis=1))
            p = points[ix]
            inten = intensity[ix]
            if tags['color reflections'] == 'intensity':
                circles(p[:,0], p[:,1], s=radius, c=np.log(inten+1),cmap=cm, alpha = 0.9, edgecolor = '') #
            else:
                for i in range(len(Laue_Zone)):
                    color = Lauecolor[int(Laue_Zone[i])]
                    circles(p[i,0], p[i,1], s=radius, c=color,cmap=cm, alpha = 0.9, edgecolor = '') #
            
    
    if not tags['color zero'] == 'None':
        circle = plt.Circle((0, 0), radius, color=tags['color zero'])
        ax.add_artist(circle)
    

    
    for i in range (len(Hxp)):
        if tags['HOLZ']['HOLZ'][i]:
            color = Lauecolor[int(Laue_Zone[i])]
            if tags['plot HOLZ']:
                # plot HOLZ lines
                ax.plot((Hxp[i],Hxm[i]), (Hyp[i],Hym[i]), c = color, linewidth = intensity_HOLZ[i])
            if tags['plot HOLZ excess']:
                ax.plot((Exp[i],Exm[i]), (Eyp[i],Eym[i]), c = color, linewidth = intensity_HOLZ[i])
            
                if tags['label HOLZ']: ## Add indices
                    ax.text(Hxp[i],Hyp[i],label[i],fontsize=10) 
                    ax.text(Exp[i],Eyp[i],label[i],fontsize=10) 
                    
        else:
            # Plot Kikuchi lines
            if tags['plot Kikuchi']:
                ax.plot((Kxp[i],Kxm[i]),(Kyp[i],Kym[i]),c = tags['color Kikuchi'], linewidth = intensity_Kikuchi[i]) 
                if tags['label Kikuchi']: ## Add indices
                    
                    ax.text(Kxp[i],Kyp[i],label[i],fontsize=tags['label size'], color = tags['label color']) 
                tags['Kikuchi lines']  
                
                

    if not (tags['color ring zero']  == 'None'):
        ring = plt.Circle((0, 0), radius, color=tags['color ring zero'], fill=False,linewidth = 2)
        ax.add_artist(ring)
        print(ring)

    return tags    
    
    
    
electronFF = {
#form factor coefficients
    #Z= 6, chisq= 0.143335
    #a1 b1 a2 b2
    #a3 b3 c1 d1
    #c2 d2 c3 d3

#name of the file:  feKirkland.txt
# converted with program sortFF.py
# form factor parametrized in 1/Angstrom
# bond_length as a list of atom Sizes, bond radii, angle radii, H-bond radii

    'H' : {'Z':  1,   'chisq' : 0.170190 ,
           'bond_length': [0.98,  0.78, 1.20, 0],
           'fa':[ 4.20298324e-003 , 6.27762505e-002 , 3.00907347e-002 ],
           'fb':[ 2.25350888e-001 , 2.25366950e-001 , 2.25331756e-001 ],
           'fc':[ 6.77756695e-002 , 3.56609237e-003 , 2.76135815e-002 ],
           'fd':[ 4.38854001e+000 , 4.03884823e-001 , 1.44490166e+000 ]},
    'He': {'Z':  2,   'chisq' : 0.396634 ,
           'bond_length': [1.45,  1.25, 1.40, 0],
           'fa':[ 1.87543704e-005 , 4.10595800e-004 , 1.96300059e-001 ],
           'fb':[ 2.12427997e-001 , 3.32212279e-001 , 5.17325152e-001 ],
           'fc':[ 8.36015738e-003 , 2.95102022e-002 , 4.65928982e-007 ],
           'fd':[ 3.66668239e-001 , 1.37171827e+000 , 3.75768025e+004 ]},
    'Li': {'Z':  3,   'chisq' : 0.286232 ,
           'bond_length': [1.76,  1.56, 1.82, 0],
           'fa':[ 7.45843816e-002 , 7.15382250e-002 , 1.45315229e-001 ],
           'fb':[ 8.81151424e-001 , 4.59142904e-002 , 8.81301714e-001 ],
           'fc':[ 1.12125769e+000 , 2.51736525e-003 , 3.58434971e-001 ],
           'fd':[ 1.88483665e+001 , 1.59189995e-001 , 6.12371000e+000 ]},
    'Be': {'Z':  4,   'chisq' : 0.195442 ,
           'bond_length': [1.33,  1.13, 1.70, 0],
           'fa':[ 6.11642897e-002 , 1.25755034e-001 , 2.00831548e-001 ],
           'fb':[ 9.90182132e-002 , 9.90272412e-002 , 1.87392509e+000 ],
           'fc':[ 7.87242876e-001 , 1.58847850e-003 , 2.73962031e-001 ],
           'fd':[ 9.32794929e+000 , 8.91900236e-002 , 3.20687658e+000 ]},
    'B' : {'Z':  5,   'chisq' : 0.146989 ,
           'bond_length': [1.18,  0.98, 2.08, 0],
           'fa':[ 1.25716066e-001 , 1.73314452e-001 , 1.84774811e-001 ],
           'fb':[ 1.48258830e-001 , 1.48257216e-001 , 3.34227311e+000 ],
           'fc':[ 1.95250221e-001 , 5.29642075e-001 , 1.08230500e-003 ],
           'fd':[ 1.97339463e+000 , 5.70035553e+000 , 5.64857237e-002 ]},
    'C' : {'Z':  6,   'chisq' : 0.102440 ,
           'bond_length': [1.12,  0.92, 1.95, 0],
           'fa':[ 2.12080767e-001 , 1.99811865e-001 , 1.68254385e-001 ],
           'fb':[ 2.08605417e-001 , 2.08610186e-001 , 5.57870773e+000 ],
           'fc':[ 1.42048360e-001 , 3.63830672e-001 , 8.35012044e-004 ],
           'fd':[ 1.33311887e+000 , 3.80800263e+000 , 4.03982620e-002 ]},
    'N' : {'Z':  7,   'chisq' : 0.060249 ,
           'bond_length': [1.08,  0.88, 1.85, 1.30],
           'fa':[ 5.33015554e-001 , 5.29008883e-002 , 9.24159648e-002 ],
           'fb':[ 2.90952515e-001 , 1.03547896e+001 , 1.03540028e+001 ],
           'fc':[ 2.61799101e-001 , 8.80262108e-004 , 1.10166555e-001 ],
           'fd':[ 2.76252723e+000 , 3.47681236e-002 , 9.93421736e-001 ]},
    'O' : {'Z':  8,   'chisq' : 0.039944 ,
           'bond_length': [1.09,  0.89, 1.70, 1.40],
           'fa':[ 3.39969204e-001 , 3.07570172e-001 , 1.30369072e-001 ],
           'fb':[ 3.81570280e-001 , 3.81571436e-001 , 1.91919745e+001 ],
           'fc':[ 8.83326058e-002 , 1.96586700e-001 , 9.96220028e-004 ],
           'fd':[ 7.60635525e-001 , 2.07401094e+000 , 3.03266869e-002 ]},
    'F' : {'Z':  9,   'chisq' : 0.027866 ,
           'bond_length': [1.30,  1.10, 1.73, 0],
           'fa':[ 2.30560593e-001 , 5.26889648e-001 , 1.24346755e-001 ],
           'fb':[ 4.80754213e-001 , 4.80763895e-001 , 3.95306720e+001 ],
           'fc':[ 1.24616894e-003 , 7.20452555e-002 , 1.53075777e-001 ],
           'fd':[ 2.62181803e-002 , 5.92495593e-001 , 1.59127671e+000 ]},
    'Ne': {'Z':  10,   'chisq' : 0.021836 ,
           'bond_length': [1.50,  1.30, 1.54, 0],
           'fa':[ 4.08371771e-001 , 4.54418858e-001 , 1.44564923e-001 ],
           'fb':[ 5.88228627e-001 , 5.88288655e-001 , 1.21246013e+002 ],
           'fc':[ 5.91531395e-002 , 1.24003718e-001 , 1.64986037e-003 ],
           'fd':[ 4.63963540e-001 , 1.23413025e+000 , 2.05869217e-002 ]},
    'Na': {'Z':  11,   'chisq' : 0.064136 ,
           'bond_length': [2.10,  1.91, 2.27, 0],
           'fa':[ 1.36471662e-001 , 7.70677865e-001 , 1.56862014e-001 ],
           'fb':[ 4.99965301e-002 , 8.81899664e-001 , 1.61768579e+001 ],
           'fc':[ 9.96821513e-001 , 3.80304670e-002 , 1.27685089e-001 ],
           'fd':[ 2.00132610e+001 , 2.60516254e-001 , 6.99559329e-001 ]},
    'Mg': {'Z':  12,   'chisq' : 0.051303 ,
           'bond_length': [1.80,  1.60, 1.73, 0], 
           'fa':[ 3.04384121e-001 , 7.56270563e-001 , 1.01164809e-001 ],
           'fb':[ 8.42014377e-002 , 1.64065598e+000 , 2.97142975e+001 ],
           'fc':[ 3.45203403e-002 , 9.71751327e-001 , 1.20593012e-001 ],
           'fd':[ 2.16596094e-001 , 1.21236852e+001 , 5.60865838e-001 ]},
    'Al': {'Z':  13,   'chisq' : 0.049529 ,
           'bond_length': [1.60,  1.43, 2.05, 0],
           'fa':[ 7.77419424e-001 , 5.78312036e-002 , 4.26386499e-001 ],
           'fb':[ 2.71058227e+000 , 7.17532098e+001 , 9.13331555e-002 ],
           'fc':[ 1.13407220e-001 , 7.90114035e-001 , 3.23293496e-002 ],
           'fd':[ 4.48867451e-001 , 8.66366718e+000 , 1.78503463e-001 ]},
    'Si': {'Z':  14,   'chisq' : 0.071667 ,
           'bond_length': [1.52,  1.32, 2.10, 0],
           'fa':[ 1.06543892e+000 , 1.20143691e-001 , 1.80915263e-001 ],
           'fb':[ 1.04118455e+000 , 6.87113368e+001 , 8.87533926e-002 ],
           'fc':[ 1.12065620e+000 , 3.05452816e-002 , 1.59963502e+000 ],
           'fd':[ 3.70062619e+000 , 2.14097897e-001 , 9.99096638e+000 ]},
    'P' : {'Z':  15,   'chisq' : 0.047673 ,
           'bond_length': [1.48,  1.28, 2.08, 0],
           'fa':[ 1.05284447e+000 , 2.99440284e-001 , 1.17460748e-001 ],
           'fb':[ 1.31962590e+000 , 1.28460520e-001 , 1.02190163e+002 ],
           'fc':[ 9.60643452e-001 , 2.63555748e-002 , 1.38059330e+000 ],
           'fd':[ 2.87477555e+000 , 1.82076844e-001 , 7.49165526e+000 ]},
    'S' : {'Z':  16,   'chisq' : 0.033482 ,
           'bond_length': [1.47,  1.27, 2.00, 0],
           'fa':[ 1.01646916e+000 , 4.41766748e-001 , 1.21503863e-001 ],
           'fb':[ 1.69181965e+000 , 1.74180288e-001 , 1.67011091e+002 ],
           'fc':[ 8.27966670e-001 , 2.33022533e-002 , 1.18302846e+000 ],
           'fd':[ 2.30342810e+000 , 1.56954150e-001 , 5.85782891e+000 ]},
    'Cl': {'Z':  17,   'chisq' : 0.206186 ,
           'bond_length': [1.70,  1.50, 1.97, 0],
           'fa':[ 9.44221116e-001 , 4.37322049e-001 , 2.54547926e-001 ],
           'fb':[ 2.40052374e-001 , 9.30510439e+000 , 9.30486346e+000 ],
           'fc':[ 5.47763323e-002 , 8.00087488e-001 , 1.07488641e-002 ],
           'fd':[ 1.68655688e-001 , 2.97849774e+000 , 6.84240646e-002 ]},
    'Ar': {'Z':  18,   'chisq' : 0.263904 ,
           'bond_length': [2.00,  1.80, 1.88, 0],
           'fa':[ 1.06983288e+000 , 4.24631786e-001 , 2.43897949e-001 ],
           'fb':[ 2.87791022e-001 , 1.24156957e+001 , 1.24158868e+001 ],
           'fc':[ 4.79446296e-002 , 7.64958952e-001 , 8.23128431e-003 ],
           'fd':[ 1.36979796e-001 , 2.43940729e+000 , 5.27258749e-002 ]},
    'K' : {'Z':  19,   'chisq' : 0.161900 ,
           'bond_length': [2.58,  2.38, 2.75, 0],
           'fa':[ 6.92717865e-001 , 9.65161085e-001 , 1.48466588e-001 ],
           'fb':[ 7.10849990e+000 , 3.57532901e-001 , 3.93763275e-002 ],
           'fc':[ 2.64645027e-002 , 1.80883768e+000 , 5.43900018e-001 ],
           'fd':[ 1.03591321e-001 , 3.22845199e+001 , 1.67791374e+000 ]},
    'Ca': {'Z':  20,   'chisq' : 0.085209 ,
           'bond_length': [2.17,  1.97, 1.97, 0],
           'fa':[ 3.66902871e-001 , 8.66378999e-001 , 6.67203300e-001 ],
           'fb':[ 6.14274129e-002 , 5.70881727e-001 , 7.82965639e+000 ],
           'fc':[ 4.87743636e-001 , 1.82406314e+000 , 2.20248453e-002 ],
           'fd':[ 1.32531318e+000 , 2.10056032e+001 , 9.11853450e-002 ]},
    'Sc': {'Z':  21,   'chisq' : 0.052352 ,
           'bond_length': [1.84,  1.64, 1.70, 0],
           'fa':[ 3.78871777e-001 , 9.00022505e-001 , 7.15288914e-001 ],
           'fb':[ 6.98910162e-002 , 5.21061541e-001 , 7.87707920e+000 ],
           'fc':[ 1.88640973e-002 , 4.07945949e-001 , 1.61786540e+000 ],
           'fd':[ 8.17512708e-002 , 1.11141388e+000 , 1.80840759e+001 ]},
    'Ti': {'Z':  22,   'chisq' : 0.035298 ,
           'bond_length': [1.66,  1.46, 1.70, 0],
           'fa':[ 3.62383267e-001 , 9.84232966e-001 , 7.41715642e-001 ],
           'fb':[ 7.54707114e-002 , 4.97757309e-001 , 8.17659391e+000 ],
           'fc':[ 3.62555269e-001 , 1.49159390e+000 , 1.61659509e-002 ],
           'fd':[ 9.55524906e-001 , 1.62221677e+001 , 7.33140839e-002 ]},
    'V' : {'Z':  23,   'chisq' : 0.030745 ,
           'bond_length': [1.55,  1.35, 1.70, 0],
           'fa':[ 3.52961378e-001 , 7.46791014e-001 , 1.08364068e+000 ],
           'fb':[ 8.19204103e-002 , 8.81189511e+000 , 5.10646075e-001 ],
           'fc':[ 1.39013610e+000 , 3.31273356e-001 , 1.40422612e-002 ],
           'fd':[ 1.48901841e+001 , 8.38543079e-001 , 6.57432678e-002 ]},
    'Cr': {'Z':  24,   'chisq' : 0.015287 ,
           'bond_length': [1.56,  1.36, 1.70, 0],
           'fa':[ 1.34348379e+000 , 5.07040328e-001 , 4.26358955e-001 ],
           'fb':[ 1.25814353e+000 , 1.15042811e+001 , 8.53660389e-002 ],
           'fc':[ 1.17241826e-002 , 5.11966516e-001 , 3.38285828e-001 ],
           'fd':[ 6.00177061e-002 , 1.53772451e+000 , 6.62418319e-001 ]},
    'Mn': {'Z':  25,   'chisq' : 0.031274 ,
           'bond_length': [1.54,  1.30, 1.70, 0],
           'fa':[ 3.26697613e-001 , 7.17297000e-001 , 1.33212464e+000 ],
           'fb':[ 8.88813083e-002 , 1.11300198e+001 , 5.82141104e-001 ],
           'fc':[ 2.80801702e-001 , 1.15499241e+000 , 1.11984488e-002 ],
           'fd':[ 6.71583145e-001 , 1.26825395e+001 , 5.32334467e-002 ]},
    'Fe': {'Z':  26,   'chisq' : 0.031315 ,
           'bond_length': [1.47,  1.27, 1.70, 0],
           'fa':[ 3.13454847e-001 , 6.89290016e-001 , 1.47141531e+000 ],
           'fb':[ 8.99325756e-002 , 1.30366038e+001 , 6.33345291e-001 ],
           'fc':[ 1.03298688e+000 , 2.58280285e-001 , 1.03460690e-002 ],
           'fd':[ 1.16783425e+001 , 6.09116446e-001 , 4.81610627e-002 ]},
    'Co': {'Z':  27,   'chisq' : 0.031643 ,
           'bond_length': [1.45,  1.25, 1.70, 0],
           'fa':[ 3.15878278e-001 , 1.60139005e+000 , 6.56394338e-001 ],
           'fb':[ 9.46683246e-002 , 6.99436449e-001 , 1.56954403e+001 ],
           'fc':[ 9.36746624e-001 , 9.77562646e-003 , 2.38378578e-001 ],
           'fd':[ 1.09392410e+001 , 4.37446816e-002 , 5.56286483e-001 ]},
    'Ni': {'Z':  28,   'chisq' : 0.032245 ,
           'bond_length': [1.45,  1.25, 1.63, 0],
           'fa':[ 1.72254630e+000 , 3.29543044e-001 , 6.23007200e-001 ],
           'fb':[ 7.76606908e-001 , 1.02262360e-001 , 1.94156207e+001 ],
           'fc':[ 9.43496513e-003 , 8.54063515e-001 , 2.21073515e-001 ],
           'fd':[ 3.98684596e-002 , 1.04078166e+001 , 5.10869330e-001 ]},
    'Cu': {'Z':  29,   'chisq' : 0.010467 ,
           'bond_length': [1.48,  1.28, 1.40, 0],
           'fa':[ 3.58774531e-001 , 1.76181348e+000 , 6.36905053e-001 ],
           'fb':[ 1.06153463e-001 , 1.01640995e+000 , 1.53659093e+001 ],
           'fc':[ 7.44930667e-003 , 1.89002347e-001 , 2.29619589e-001 ],
           'fd':[ 3.85345989e-002 , 3.98427790e-001 , 9.01419843e-001 ]},
    'Zn': {'Z':  30,   'chisq' : 0.026698 ,
           'bond_length': [1.59,  1.39, 1.39, 0],
           'fa':[ 5.70893973e-001 , 1.98908856e+000 , 3.06060585e-001 ],
           'fb':[ 1.26534614e-001 , 2.17781965e+000 , 3.78619003e+001 ],
           'fc':[ 2.35600223e-001 , 3.97061102e-001 , 6.85657228e-003 ],
           'fd':[ 3.67019041e-001 , 8.66419596e-001 , 3.35778823e-002 ]},
    'Ga': {'Z':  31,   'chisq' : 0.008110 ,
           'bond_length': [1.61,  1.41, 1.87, 0],
           'fa':[ 6.25528464e-001 , 2.05302901e+000 , 2.89608120e-001 ],
           'fb':[ 1.10005650e-001 , 2.41095786e+000 , 4.78685736e+001 ],
           'fc':[ 2.07910594e-001 , 3.45079617e-001 , 6.55634298e-003 ],
           'fd':[ 3.27807224e-001 , 7.43139061e-001 , 3.09411369e-002 ]},
    'Ge': {'Z':  32,   'chisq' : 0.032198 ,
           'bond_length': [1.57,  1.37, 1.70, 0],
           'fa':[ 5.90952690e-001 , 5.39980660e-001 , 2.00626188e+000 ],
           'fb':[ 1.18375976e-001 , 7.18937433e+001 , 1.39304889e+000 ],
           'fc':[ 7.49705041e-001 , 1.83581347e-001 , 9.52190743e-003 ],
           'fd':[ 6.89943350e+000 , 3.64667232e-001 , 2.69888650e-002 ]},
    'As': {'Z':  33,   'chisq' : 0.034014 ,
           'bond_length': [1.59,  1.39, 1.85, 0],
           'fa':[ 7.77875218e-001 , 5.93848150e-001 , 1.95918751e+000 ],
           'fb':[ 1.50733157e-001 , 1.42882209e+002 , 1.74750339e+000 ],
           'fc':[ 1.79880226e-001 , 8.63267222e-001 , 9.59053427e-003 ],
           'fd':[ 3.31800852e-001 , 5.85490274e+000 , 2.33777569e-002 ]},
    'Se': {'Z':  34,   'chisq' : 0.035703 ,
           'bond_length': [1.60,  1.40, 1.90, 0],
           'fa':[ 9.58390681e-001 , 6.03851342e-001 , 1.90828931e+000 ],
           'fb':[ 1.83775557e-001 , 1.96819224e+002 , 2.15082053e+000 ],
           'fc':[ 1.73885956e-001 , 9.35265145e-001 , 8.62254658e-003 ],
           'fd':[ 3.00006024e-001 , 4.92471215e+000 , 2.12308108e-002 ]},
    'Br': {'Z':  35,   'chisq' : 0.039250 ,
           'bond_length': [1.80,  1.60, 2.10, 0],
           'fa':[ 1.14136170e+000 , 5.18118737e-001 , 1.85731975e+000 ],
           'fb':[ 2.18708710e-001 , 1.93916682e+002 , 2.65755396e+000 ],
           'fc':[ 1.68217399e-001 , 9.75705606e-001 , 7.24187871e-003 ],
           'fd':[ 2.71719918e-001 , 4.19482500e+000 , 1.99325718e-002 ]},
    'Kr': {'Z':  36,   'chisq' : 0.045421 ,
           'bond_length': [2.10,  1.90, 2.02, 0],
           'fa':[ 3.24386970e-001 , 1.31732163e+000 , 1.79912614e+000 ],
           'fb':[ 6.31317973e+001 , 2.54706036e-001 , 3.23668394e+000 ],
           'fc':[ 4.29961425e-003 , 1.00429433e+000 , 1.62188197e-001 ],
           'fd':[ 1.98965610e-002 , 3.61094513e+000 , 2.45583672e-001 ]},
    'Rb': {'Z':  37,   'chisq' : 0.130044 ,
           'bond_length': [2.75,  2.55, 1.70, 0],
           'fa':[ 2.90445351e-001 , 2.44201329e+000 , 7.69435449e-001 ],
           'fb':[ 3.68420227e-002 , 1.16013332e+000 , 1.69591472e+001 ],
           'fc':[ 1.58687000e+000 , 2.81617593e-003 , 1.28663830e-001 ],
           'fd':[ 2.53082574e+000 , 1.88577417e-002 , 2.10753969e-001 ]},
    'Sr': {'Z':  38,   'chisq' : 0.188055 ,
           'bond_length': [2.35,  2.15, 1.70, 0],
           'fa':[ 1.37373086e-002 , 1.97548672e+000 , 1.59261029e+000 ],
           'fb':[ 1.87469061e-002 , 6.36079230e+000 , 2.21992482e-001 ],
           'fc':[ 1.73263882e-001 , 4.66280378e+000 , 1.61265063e-003 ],
           'fd':[ 2.01624958e-001 , 2.53027803e+001 , 1.53610568e-002 ]},
    'Y' : {'Z':  39,   'chisq' : 0.174927 ,
           'bond_length': [2.00,  1.80, 1.70, 0],
           'fa':[ 6.75302747e-001 , 4.70286720e-001 , 2.63497677e+000 ],
           'fb':[ 6.54331847e-002 , 1.06108709e+002 , 2.06643540e+000 ],
           'fc':[ 1.09621746e-001 , 9.60348773e-001 , 5.28921555e-003 ],
           'fd':[ 1.93131925e-001 , 1.63310938e+000 , 1.66083821e-002 ]},
    'Zr': {'Z':  40,   'chisq' : 0.072078 ,
           'bond_length': [1.80,  1.60, 1.70, 0],
           'fa':[ 2.64365505e+000 , 5.54225147e-001 , 7.61376625e-001 ],
           'fb':[ 2.20202699e+000 , 1.78260107e+002 , 7.67218745e-002 ],
           'fc':[ 6.02946891e-003 , 9.91630530e-002 , 9.56782020e-001 ],
           'fd':[ 1.55143296e-002 , 1.76175995e-001 , 1.54330682e+000 ]},
    'Nb': {'Z':  41,   'chisq' : 0.011800 ,
           'bond_length': [1.67,  1.47, 1.70, 0],
           'fa':[ 6.59532875e-001 , 1.84545854e+000 , 1.25584405e+000 ],
           'fb':[ 8.66145490e-002 , 5.94774398e+000 , 6.40851475e-001 ],
           'fc':[ 1.22253422e-001 , 7.06638328e-001 , 2.62381591e-003 ],
           'fd':[ 1.66646050e-001 , 1.62853268e+000 , 8.26257859e-003 ]},
    'Mo': {'Z':  42,   'chisq' : 0.008976 ,
           'bond_length': [1.60,  1.40, 1.70, 0],
           'fa':[ 6.10160120e-001 , 1.26544000e+000 , 1.97428762e+000 ],
           'fb':[ 9.11628054e-002 , 5.06776025e-001 , 5.89590381e+000 ],
           'fc':[ 6.48028962e-001 , 2.60380817e-003 , 1.13887493e-001 ],
           'fd':[ 1.46634108e+000 , 7.84336311e-003 , 1.55114340e-001 ]},
    'Tc': {'Z':  43,   'chisq' : 0.023771 ,
           'bond_length': [1.56,  1.36, 1.70, 0],
           'fa':[ 8.55189183e-001 , 1.66219641e+000 , 1.45575475e+000 ],
           'fb':[ 1.02962151e-001 , 7.64907000e+000 , 1.01639987e+000 ],
           'fc':[ 1.05445664e-001 , 7.71657112e-001 , 2.20992635e-003 ],
           'fd':[ 1.42303338e-001 , 1.34659349e+000 , 7.90358976e-003 ]},
    'Ru': {'Z':  44,   'chisq' : 0.010613 ,
           'bond_length': [1.54,  1.34, 1.70, 0],
           'fa':[ 4.70847093e-001 , 1.58180781e+000 , 2.02419818e+000 ],
           'fb':[ 9.33029874e-002 , 4.52831347e-001 , 7.11489023e+000 ],
           'fc':[ 1.97036257e-003 , 6.26912639e-001 , 1.02641320e-001 ],
           'fd':[ 7.56181595e-003 , 1.25399858e+000 , 1.33786087e-001 ]},
    'Rh': {'Z':  45,   'chisq' : 0.012895 ,
           'bond_length': [1.54,  1.34, 1.70, 0],
           'fa':[ 4.20051553e-001 , 1.76266507e+000 , 2.02735641e+000 ],
           'fb':[ 9.38882628e-002 , 4.64441687e-001 , 8.19346046e+000 ],
           'fc':[ 1.45487176e-003 , 6.22809600e-001 , 9.91529915e-002 ],
           'fd':[ 7.82704517e-003 , 1.17194153e+000 , 1.24532839e-001 ]},
    'Pd': {'Z':  46,   'chisq' : 0.009172 ,
           'bond_length': [1.58,  1.38, 1.63, 0],
           'fa':[ 2.10475155e+000 , 2.03884487e+000 , 1.82067264e-001 ],
           'fb':[ 8.68606470e+000 , 3.78924449e-001 , 1.42921634e-001 ],
           'fc':[ 9.52040948e-002 , 5.91445248e-001 , 1.13328676e-003 ],
           'fd':[ 1.17125900e-001 , 1.07843808e+000 , 7.80252092e-003 ]},
    'Ag': {'Z':  47,   'chisq' : 0.006648 ,
           'bond_length': [1.64,  1.44, 1.72, 0],
           'fa':[ 2.07981390e+000 , 4.43170726e-001 , 1.96515215e+000 ],
           'fb':[ 9.92540297e+000 , 1.04920104e-001 , 6.40103839e-001 ],
           'fc':[ 5.96130591e-001 , 4.78016333e-001 , 9.46458470e-002 ],
           'fd':[ 8.89594790e-001 , 1.98509407e+000 , 1.12744464e-001 ]},
    'Cd': {'Z':  48,   'chisq' : 0.005588 ,
           'bond_length': [1.77,  1.57, 1.58, 0],
           'fa':[ 1.63657549e+000 , 2.17927989e+000 , 7.71300690e-001 ],
           'fb':[ 1.24540381e+001 , 1.45134660e+000 , 1.26695757e-001 ],
           'fc':[ 6.64193880e-001 , 7.64563285e-001 , 8.61126689e-002 ],
           'fd':[ 7.77659202e-001 , 1.66075210e+000 , 1.05728357e-001 ]},
    'In': {'Z':  49,   'chisq' : 0.002569 ,
           'bond_length': [1.86,  1.66, 1.93, 0],
           'fa':[ 2.24820632e+000 , 1.64706864e+000 , 7.88679265e-001 ],
           'fb':[ 1.51913507e+000 , 1.30113424e+001 , 1.06128184e-001 ],
           'fc':[ 8.12579069e-002 , 6.68280346e-001 , 6.38467475e-001 ],
           'fd':[ 9.94045620e-002 , 1.49742063e+000 , 7.18422635e-001 ]},
    'Sn': {'Z':  50,   'chisq' : 0.005051 ,
           'bond_length': [1.82,  1.62, 2.17, 0],
           'fa':[ 2.16644620e+000 , 6.88691021e-001 , 1.92431751e+000 ],
           'fb':[ 1.13174909e+001 , 1.10131285e-001 , 6.74464853e-001 ],
           'fc':[ 5.65359888e-001 , 9.18683861e-001 , 7.80542213e-002 ],
           'fd':[ 7.33564610e-001 , 1.02310312e+001 , 9.31104308e-002 ]},
    'Sb': {'Z':  51,   'chisq' : 0.004383 ,
           'bond_length': [1.79,  1.59, 2.20, 0],
           'fa':[ 1.73662114e+000 , 9.99871380e-001 , 2.13972409e+000 ],
           'fb':[ 8.84334719e-001 , 1.38462121e-001 , 1.19666432e+001 ],
           'fc':[ 5.60566526e-001 , 9.93772747e-001 , 7.37374982e-002 ],
           'fd':[ 6.72672880e-001 , 8.72330411e+000 , 8.78577715e-002 ]},
    'Te': {'Z':  52,   'chisq' : 0.004105 ,
           'bond_length': [1.80,  1.60, 2.06, 0],
           'fa':[ 2.09383882e+000 , 1.56940519e+000 , 1.30941993e+000 ],
           'fb':[ 1.26856869e+001 , 1.21236537e+000 , 1.66633292e-001 ],
           'fc':[ 6.98067804e-002 , 1.04969537e+000 , 5.55594354e-001 ],
           'fd':[ 8.30817576e-002 , 7.43147857e+000 , 6.17487676e-001 ]},
    'I' : {'Z':  53,   'chisq' : 0.004068 ,
           'bond_length': [1.90,  1.70, 2.15, 0],
           'fa':[ 1.60186925e+000 , 1.98510264e+000 , 1.48226200e+000 ],
           'fb':[ 1.95031538e-001 , 1.36976183e+001 , 1.80304795e+000 ],
           'fc':[ 5.53807199e-001 , 1.11728722e+000 , 6.60720847e-002 ],
           'fd':[ 5.67912340e-001 , 6.40879878e+000 , 7.86615429e-002 ]},
    'Xe': {'Z':  54,   'chisq' : 0.004381 ,
           'bond_length': [2.30,  2.10, 2.16, 0],
           'fa':[ 1.60015487e+000 , 1.71644581e+000 , 1.84968351e+000 ],
           'fb':[ 2.92913354e+000 , 1.55882990e+001 , 2.22525983e-001 ],
           'fc':[ 6.23813648e-002 , 1.21387555e+000 , 5.54051946e-001 ],
           'fd':[ 7.45581223e-002 , 5.56013271e+000 , 5.21994521e-001 ]},
    'Cs': {'Z':  55,   'chisq' : 0.042676 ,
           'bond_length': [2.93,  2.73, 1.70, 0],
           'fa':[ 2.95236854e+000 , 4.28105721e-001 , 1.89599233e+000 ],
           'fb':[ 6.01461952e+000 , 4.64151246e+001 , 1.80109756e-001 ],
           'fc':[ 5.48012938e-002 , 4.70838600e+000 , 5.90356719e-001 ],
           'fd':[ 7.12799633e-002 , 4.56702799e+001 , 4.70236310e-001 ]},
    'Ba': {'Z':  56,   'chisq' : 0.043267 ,
           'bond_length': [2.44,  2.24, 1.70, 0],
           'fa':[ 3.19434243e+000 , 1.98289586e+000 , 1.55121052e-001 ],
           'fb':[ 9.27352241e+000 , 2.28741632e-001 , 3.82000231e-002 ],
           'fc':[ 6.73222354e-002 , 4.48474211e+000 , 5.42674414e-001 ],
           'fd':[ 7.30961745e-002 , 2.95703565e+001 , 4.08647015e-001 ]},
    'La': {'Z':  57,   'chisq' : 0.033249 ,
           'bond_length': [2.08,  1.88, 1.70, 0],
           'fa':[ 2.05036425e+000 , 1.42114311e-001 , 3.23538151e+000 ],
           'fb':[ 2.20348417e-001 , 3.96438056e-002 , 9.56979169e+000 ],
           'fc':[ 6.34683429e-002 , 3.97960586e+000 , 5.20116711e-001 ],
           'fd':[ 6.92443091e-002 , 2.53178406e+001 , 3.83614098e-001 ]},
    'Ce': {'Z':  58,   'chisq' : 0.029355 ,
           'bond_length': [2.02,  1.82, 1.70, 0],
           'fa':[ 3.22990759e+000 , 1.57618307e-001 , 2.13477838e+000 ],
           'fb':[ 9.94660135e+000 , 4.15378676e-002 , 2.40480572e-001 ],
           'fc':[ 5.01907609e-001 , 3.80889010e+000 , 5.96625028e-002 ],
           'fd':[ 3.66252019e-001 , 2.43275968e+001 , 6.59653503e-002 ]},
    'Pr': {'Z':  59,   'chisq' : 0.029725 ,
           'bond_length': [2.03,  1.83, 1.70, 0],
           'fa':[ 1.58189324e-001 , 3.18141995e+000 , 2.27622140e+000 ],
           'fb':[ 3.91309056e-002 , 1.04139545e+001 , 2.81671757e-001 ],
           'fc':[ 3.97705472e+000 , 5.58448277e-002 , 4.85207954e-001 ],
           'fd':[ 2.61872978e+001 , 6.30921695e-002 , 3.54234369e-001 ]},
    'Nd': {'Z':  60,   'chisq' : 0.027597 ,
           'bond_length': [2.02,  1.82, 1.70, 0],
           'fa':[ 1.81379417e-001 , 3.17616396e+000 , 2.35221519e+000 ],
           'fb':[ 4.37324793e-002 , 1.07842572e+001 , 3.05571833e-001 ],
           'fc':[ 3.83125763e+000 , 5.25889976e-002 , 4.70090742e-001 ],
           'fd':[ 2.54745408e+001 , 6.02676073e-002 , 3.39017003e-001 ]},
    'Pm': {'Z':  61,   'chisq' : 0.025208 ,
           'bond_length': [2.01,  1.81, 1.70, 0],
           'fa':[ 1.92986811e-001 , 2.43756023e+000 , 3.17248504e+000 ],
           'fb':[ 4.37785970e-002 , 3.29336996e-001 , 1.11259996e+001 ],
           'fc':[ 3.58105414e+000 , 4.56529394e-001 , 4.94812177e-002 ],
           'fd':[ 2.46709586e+001 , 3.24990282e-001 , 5.76553100e-002 ]},
    'Sm': {'Z':  62,   'chisq' : 0.023540 ,
           'bond_length': [2.00,  1.80, 1.70, 0],
           'fa':[ 2.12002595e-001 , 3.16891754e+000 , 2.51503494e+000 ],
           'fb':[ 4.57703608e-002 , 1.14536599e+001 , 3.55561054e-001 ],
           'fc':[ 4.44080845e-001 , 3.36742101e+000 , 4.65652543e-002 ],
           'fd':[ 3.11953363e-001 , 2.40291435e+001 , 5.52266819e-002 ]},
    'Eu': {'Z':  63,   'chisq' : 0.022204 ,
           'bond_length': [2.24,  2.04, 1.70, 0],
           'fa':[ 2.59355002e+000 , 3.16557522e+000 , 2.29402652e-001 ],
           'fb':[ 3.82452612e-001 , 1.17675155e+001 , 4.76642249e-002 ],
           'fc':[ 4.32257780e-001 , 3.17261920e+000 , 4.37958317e-002 ],
           'fd':[ 2.99719833e-001 , 2.34462738e+001 , 5.29440680e-002 ]},
    'Gd': {'Z':  64,   'chisq' : 0.017492 ,
           'bond_length': [2.00,  1.80, 1.70, 0],
           'fa':[ 3.19144939e+000 , 2.55766431e+000 , 3.32681934e-001 ],
           'fb':[ 1.20224655e+001 , 4.08338876e-001 , 5.85819814e-002 ],
           'fc':[ 4.14243130e-002 , 2.61036728e+000 , 4.20526863e-001 ],
           'fd':[ 5.06771477e-002 , 1.99344244e+001 , 2.85686240e-001 ]},
    'Tb': {'Z':  65,   'chisq' : 0.020036 ,
           'bond_length': [1.98,  1.78, 1.70, 0],
           'fa':[ 2.59407462e-001 , 3.16177855e+000 , 2.75095751e+000 ],
           'fb':[ 5.04689354e-002 , 1.23140183e+001 , 4.38337626e-001 ],
           'fc':[ 2.79247686e+000 , 3.85931001e-002 , 4.10881708e-001 ],
           'fd':[ 2.23797309e+001 , 4.87920992e-002 , 2.77622892e-001 ]},
    'Dy': {'Z':  66,   'chisq' : 0.019351 ,
           'bond_length': [1.97,  1.77, 1.70, 0],
           'fa':[ 3.16055396e+000 , 2.82751709e+000 , 2.75140255e-001 ],
           'fb':[ 1.25470414e+001 , 4.67899094e-001 , 5.23226982e-002 ],
           'fc':[ 4.00967160e-001 , 2.63110834e+000 , 3.61333817e-002 ],
           'fd':[ 2.67614884e-001 , 2.19498166e+001 , 4.68871497e-002 ]},
    'Ho': {'Z':  67,   'chisq' : 0.018720 ,
           'bond_length': [1.98,  1.78, 1.70, 0],
           'fa':[ 2.88642467e-001 , 2.90567296e+000 , 3.15960159e+000 ],
           'fb':[ 5.40507687e-002 , 4.97581077e-001 , 1.27599505e+001 ],
           'fc':[ 3.91280259e-001 , 2.48596038e+000 , 3.37664478e-002 ],
           'fd':[ 2.58151831e-001 , 2.15400972e+001 , 4.50664323e-002 ]},
    'Er': {'Z':  68,   'chisq' : 0.018677 ,
           'bond_length': [1.96,  1.76, 1.70, 0],
           'fa':[ 3.15573213e+000 , 3.11519560e-001 , 2.97722406e+000 ],
           'fb':[ 1.29729009e+001 , 5.81399387e-002 , 5.31213394e-001 ],
           'fc':[ 3.81563854e-001 , 2.40247532e+000 , 3.15224214e-002 ],
           'fd':[ 2.49195776e-001 , 2.13627616e+001 , 4.33253257e-002 ]},
    'Tm': {'Z':  69,   'chisq' : 0.018176 ,
           'bond_length': [1.95,  1.75, 1.70, 0],
           'fa':[ 3.15591970e+000 , 3.22544710e-001 , 3.05569053e+000 ],
           'fb':[ 1.31232407e+001 , 5.97223323e-002 , 5.61876773e-001 ],
           'fc':[ 2.92845100e-002 , 3.72487205e-001 , 2.27833695e+000 ],
           'fd':[ 4.16534255e-002 , 2.40821967e-001 , 2.10034185e+001 ]},
    'Yb': {'Z':  70,   'chisq' : 0.018460 ,
           'bond_length': [2.10,  1.90, 1.70, 0],
           'fa':[ 3.10794704e+000 , 3.14091221e+000 , 3.75660454e-001 ],
           'fb':[ 6.06347847e-001 , 1.33705269e+001 , 7.29814740e-002 ],
           'fc':[ 3.61901097e-001 , 2.45409082e+000 , 2.72383990e-002 ],
           'fd':[ 2.32652051e-001 , 2.12695209e+001 , 3.99969597e-002 ]},
    'Lu': {'Z':  71,   'chisq' : 0.015021 ,
           'bond_length': [1.93,  1.73, 1.70, 0],
           'fa':[ 3.11446863e+000 , 5.39634353e-001 , 3.06460915e+000 ],
           'fb':[ 1.38968881e+001 , 8.91708508e-002 , 6.79919563e-001 ],
           'fc':[ 2.58563745e-002 , 2.13983556e+000 , 3.47788231e-001 ],
           'fd':[ 3.82808522e-002 , 1.80078788e+001 , 2.22706591e-001 ]},
    'Hf': {'Z':  72,   'chisq' : 0.012070 ,
           'bond_length': [1.78,  1.58, 1.70, 0],
           'fa':[ 3.01166899e+000 , 3.16284788e+000 , 6.33421771e-001 ],
           'fb':[ 7.10401889e-001 , 1.38262192e+001 , 9.48486572e-002 ],
           'fc':[ 3.41417198e-001 , 1.53566013e+000 , 2.40723773e-002 ],
           'fd':[ 2.14129678e-001 , 1.55298698e+001 , 3.67833690e-002 ]},
    'Ta': {'Z':  73,   'chisq' : 0.010775 ,
           'bond_length': [1.67,  1.47, 1.70, 0],
           'fa':[ 3.20236821e+000 , 8.30098413e-001 , 2.86552297e+000 ],
           'fb':[ 1.38446369e+001 , 1.18381581e-001 , 7.66369118e-001 ],
           'fc':[ 2.24813887e-002 , 1.40165263e+000 , 3.33740596e-001 ],
           'fd':[ 3.52934622e-002 , 1.46148877e+001 , 2.05704486e-001 ]},
    'W' : {'Z':  74,   'chisq' : 0.009479 ,
           'bond_length': [1.61,  1.41, 1.70, 0],
           'fa':[ 9.24906855e-001 , 2.75554557e+000 , 3.30440060e+000 ],
           'fb':[ 1.28663377e-001 , 7.65826479e-001 , 1.34471170e+001 ],
           'fc':[ 3.29973862e-001 , 1.09916444e+000 , 2.06498883e-002 ],
           'fd':[ 1.98218895e-001 , 1.35087534e+001 , 3.38918459e-002 ]},
    'Re': {'Z':  75,   'chisq' : 0.004620 ,
           'bond_length': [1.58,  1.38, 1.70, 0],
           'fa':[ 1.96952105e+000 , 1.21726619e+000 , 4.10391685e+000 ],
           'fb':[ 4.98830620e+001 , 1.33243809e-001 , 1.84396916e+000 ],
           'fc':[ 2.90791978e-002 , 2.30696669e-001 , 6.08840299e-001 ],
           'fd':[ 2.84192813e-002 , 1.90968784e-001 , 1.37090356e+000 ]},
    'Os': {'Z':  76,   'chisq' : 0.003085 ,
           'bond_length': [1.55,  1.35, 1.70, 0],
           'fa':[ 2.06385867e+000 , 1.29603406e+000 , 3.96920673e+000 ],
           'fb':[ 4.05671697e+001 , 1.46559047e-001 , 1.82561596e+000 ],
           'fc':[ 2.69835487e-002 , 2.31083999e-001 , 6.30466774e-001 ],
           'fd':[ 2.84172045e-002 , 1.79765184e-001 , 1.38911543e+000 ]},
    'Ir': {'Z':  77,   'chisq' : 0.003924 ,
           'bond_length': [1.56,  1.36, 1.70, 0],
           'fa':[ 2.21522726e+000 , 1.37573155e+000 , 3.78244405e+000 ],
           'fb':[ 3.24464090e+001 , 1.60920048e-001 , 1.78756553e+000 ],
           'fc':[ 2.44643240e-002 , 2.36932016e-001 , 6.48471412e-001 ],
           'fd':[ 2.82909938e-002 , 1.70692368e-001 , 1.37928390e+000 ]},
    'Pt': {'Z':  78,   'chisq' : 0.003817 ,
           'bond_length': [1.59,  1.39, 1.72, 0],
           'fa':[ 9.84697940e-001 , 2.73987079e+000 , 3.61696715e+000 ],
           'fb':[ 1.60910839e-001 , 7.18971667e-001 , 1.29281016e+001 ],
           'fc':[ 3.02885602e-001 , 2.78370726e-001 , 1.52124129e-002 ],
           'fd':[ 1.70134854e-001 , 1.49862703e+000 , 2.83510822e-002 ]},
    'Au': {'Z':  79,   'chisq' : 0.003143 ,
           'bond_length': [1.64,  1.44, 1.66, 0],
           'fa':[ 9.61263398e-001 , 3.69581030e+000 , 2.77567491e+000 ],
           'fb':[ 1.70932277e-001 , 1.29335319e+001 , 6.89997070e-001 ],
           'fc':[ 2.95414176e-001 , 3.11475743e-001 , 1.43237267e-002 ],
           'fd':[ 1.63525510e-001 , 1.39200901e+000 , 2.71265337e-002 ]},
    'Hg': {'Z':  80,   'chisq' : 0.002717 ,
           'bond_length': [1.77,  1.57, 1.55, 0],
           'fa':[ 1.29200491e+000 , 2.75161478e+000 , 3.49387949e+000 ],
           'fb':[ 1.83432865e-001 , 9.42368371e-001 , 1.46235654e+001 ],
           'fc':[ 2.77304636e-001 , 4.30232810e-001 , 1.48294351e-002 ],
           'fd':[ 1.55110144e-001 , 1.28871670e+000 , 2.61903834e-002 ]},
    'Tl': {'Z':  81,   'chisq' : 0.003492 ,
           'bond_length': [1.92,  1.72, 1.96, 0],
           'fa':[ 3.75964730e+000 , 3.21195904e+000 , 6.47767825e-001 ],
           'fb':[ 1.35041513e+001 , 6.66330993e-001 , 9.22518234e-002 ],
           'fc':[ 2.76123274e-001 , 3.18838810e-001 , 1.31668419e-002 ],
           'fd':[ 1.50312897e-001 , 1.12565588e+000 , 2.48879842e-002 ]},
    'Pb': {'Z':  82,   'chisq' : 0.001158 ,
           'bond_length': [1.95,  1.75, 2.02, 0],
           'fa':[ 1.00795975e+000 , 3.09796153e+000 , 3.61296864e+000 ],
           'fb':[ 1.17268427e-001 , 8.80453235e-001 , 1.47325812e+001 ],
           'fc':[ 2.62401476e-001 , 4.05621995e-001 , 1.31812509e-002 ],
           'fd':[ 1.43491014e-001 , 1.04103506e+000 , 2.39575415e-002 ]},
    'Bi': {'Z':  83,   'chisq' : 0.026436 ,
           'bond_length': [1.90,  1.70, 1.70, 0],
           'fa':[ 1.59826875e+000 , 4.38233925e+000 , 2.06074719e+000 ],
           'fb':[ 1.56897471e-001 , 2.47094692e+000 , 5.72438972e+001 ],
           'fc':[ 1.94426023e-001 , 8.22704978e-001 , 2.33226953e-002 ],
           'fd':[ 1.32979109e-001 , 9.56532528e-001 , 2.23038435e-002 ]},
    'Po': {'Z':  84,   'chisq' : 0.008962 ,
           'bond_length': [1.96,  1.76, 1.70, 0],
           'fa':[ 1.71463223e+000 , 2.14115960e+000 , 4.37512413e+000 ],
           'fb':[ 9.79262841e+001 , 2.10193717e-001 , 3.66948812e+000 ],
           'fc':[ 2.16216680e-002 , 1.97843837e-001 , 6.52047920e-001 ],
           'fd':[ 1.98456144e-002 , 1.33758807e-001 , 7.80432104e-001 ]},
    'At': {'Z':  85,   'chisq' : 0.033776 ,
           'bond_length': [2.00,  1.80, 1.70, 0],
           'fa':[ 1.48047794e+000 , 2.09174630e+000 , 4.75246033e+000 ],
           'fb':[ 1.25943919e+002 , 1.83803008e-001 , 4.19890596e+000 ],
           'fc':[ 1.85643958e-002 , 2.05859375e-001 , 7.13540948e-001 ],
           'fd':[ 1.81383503e-002 , 1.33035404e-001 , 7.03031938e-001 ]},
    'Rn': {'Z':  86,   'chisq' : 0.050132 ,
           'bond_length': [2.40,  2.20, 1.70, 0],
           'fa':[ 6.30022295e-001 , 3.80962881e+000 , 3.89756067e+000 ],
           'fb':[ 1.40909762e-001 , 3.08515540e+001 , 6.51559763e-001 ],
           'fc':[ 2.40755100e-001 , 2.62868577e+000 , 3.14285931e-002 ],
           'fd':[ 1.08899672e-001 , 6.42383261e+000 , 2.42346699e-002 ]},
    'Fr': {'Z':  87,   'chisq' : 0.056720 ,
           'bond_length': [3.00,  2.80, 1.70, 0],
           'fa':[ 5.23288135e+000 , 2.48604205e+000 , 3.23431354e-001 ],
           'fb':[ 8.60599536e+000 , 3.04543982e-001 , 3.87759096e-002 ],
           'fc':[ 2.55403596e-001 , 5.53607228e-001 , 5.75278889e-003 ],
           'fd':[ 1.28717724e-001 , 5.36977452e-001 , 1.29417790e-002 ]},
    'Ra': {'Z':  88,   'chisq' : 0.081498 ,
           'bond_length': [2.46,  2.26, 1.70, 0],
           'fa':[ 1.44192685e+000 , 3.55291725e+000 , 3.91259586e+000 ],
           'fb':[ 1.18740873e-001 , 1.01739750e+000 , 6.31814783e+001 ],
           'fc':[ 2.16173519e-001 , 3.94191605e+000 , 4.60422605e-002 ],
           'fd':[ 9.55806441e-002 , 3.50602732e+001 , 2.20850385e-002 ]},
    'Ac': {'Z':  89,   'chisq' : 0.077643 ,
           'bond_length': [2.09,  1.88, 1.70, 0],
           'fa':[ 1.45864127e+000 , 4.18945405e+000 , 3.65866182e+000 ],
           'fb':[ 1.07760494e-001 , 8.89090649e+001 , 1.05088931e+000 ],
           'fc':[ 2.08479229e-001 , 3.16528117e+000 , 5.23892556e-002 ],
           'fd':[ 9.09335557e-002 , 3.13297788e+001 , 2.08807697e-002 ]},
    'Th': {'Z':  90,   'chisq' : 0.048096 ,
           'bond_length': [2.00,  1.80, 1.70, 0],
           'fa':[ 1.19014064e+000 , 2.55380607e+000 , 4.68110181e+000 ],
           'fb':[ 7.73468729e-002 , 6.59693681e-001 , 1.28013896e+001 ],
           'fc':[ 2.26121303e-001 , 3.58250545e-001 , 7.82263950e-003 ],
           'fd':[ 1.08632194e-001 , 4.56765664e-001 , 1.62623474e-002 ]},
    'Pa': {'Z':  91,   'chisq' : 0.070186 ,
           'bond_length': [1.83,  1.63, 1.70, 0],
           'fa':[ 4.68537504e+000 , 2.98413708e+000 , 8.91988061e-001 ],
           'fb':[ 1.44503632e+001 , 5.56438592e-001 , 6.69512914e-002 ],
           'fc':[ 2.24825384e-001 , 3.04444846e-001 , 9.48162708e-003 ],
           'fd':[ 1.03235396e-001 , 4.27255647e-001 , 1.77730611e-002 ]},
    'U' : {'Z':  92,   'chisq' : 0.072478 ,
           'bond_length': [1.76,  1.56, 1.86, 0],
           'fa':[ 4.63343606e+000 , 3.18157056e+000 , 8.76455075e-001 ],
           'fb':[ 1.63377267e+001 , 5.69517868e-001 , 6.88860012e-002 ],
           'fc':[ 2.21685477e-001 , 2.72917100e-001 , 1.11737298e-002 ],
           'fd':[ 9.84254550e-002 , 4.09470917e-001 , 1.86215410e-002 ]},
    'Np': {'Z':  93,   'chisq' : 0.074792 ,
           'bond_length': [1.80,  1.60, 1.70, 0],
           'fa':[ 4.56773888e+000 , 3.40325179e+000 , 8.61841923e-001 ],
           'fb':[ 1.90992795e+001 , 5.90099634e-001 , 7.03204851e-002 ],
           'fc':[ 2.19728870e-001 , 2.38176903e-001 , 1.38306499e-002 ],
           'fd':[ 9.36334280e-002 , 3.93554882e-001 , 1.94437286e-002 ]},
    'Pu': {'Z':  94,   'chisq' : 0.071877 ,
           'bond_length': [1.84,  1.64, 1.70, 0],
           'fa':[ 5.45671123e+000 , 1.11687906e-001 , 3.30260343e+000 ],
           'fb':[ 1.01892720e+001 , 3.98131313e-002 , 3.14622212e-001 ],
           'fc':[ 1.84568319e-001 , 4.93644263e-001 , 3.57484743e+000 ],
           'fd':[ 1.04220860e-001 , 4.63080540e-001 , 2.19369542e+001 ]},
    'Am': {'Z':  95,   'chisq' : 0.062156 ,
           'bond_length': [2.01,  1.81, 1.70, 0],
           'fa':[ 5.38321999e+000 , 1.23343236e-001 , 3.46469090e+000 ],
           'fb':[ 1.07289857e+001 , 4.15137806e-002 , 3.39326208e-001 ],
           'fc':[ 1.75437132e-001 , 3.39800073e+000 , 4.69459519e-001 ],
           'fd':[ 9.98932346e-002 , 2.11601535e+001 , 4.51996970e-001 ]},
    'Cm': {'Z':  96,   'chisq' : 0.050111 ,
           'bond_length': [2.20,  2.00, 1.70, 0],
           'fa':[ 5.38402377e+000 , 3.49861264e+000 , 1.88039547e-001 ],
           'fb':[ 1.11211419e+001 , 3.56750210e-001 , 5.39853583e-002 ],
           'fc':[ 1.69143137e-001 , 3.19595016e+000 , 4.64393059e-001 ],
           'fd':[ 9.60082633e-002 , 1.80694389e+001 , 4.36318197e-001 ]},
    'Bk': {'Z':  97,   'chisq' : 0.044081 ,
           'bond_length': [2.20,  2.00, 1.70, 0],
           'fa':[ 3.66090688e+000 , 2.03054678e-001 , 5.30697515e+000 ],
           'fb':[ 3.84420906e-001 , 5.48547131e-002 , 1.17150262e+001 ],
           'fc':[ 1.60934046e-001 , 3.04808401e+000 , 4.43610295e-001 ],
           'fd':[ 9.21020329e-002 , 1.73525367e+001 , 4.27132359e-001 ]},
    'Cf': {'Z':  98,   'chisq' : 0.041053 ,
           'bond_length': [2.20,  2.00, 1.70, 0],
           'fa':[ 3.94150390e+000 , 5.16915345e+000 , 1.61941074e-001 ],
           'fb':[ 4.18246722e-001 , 1.25201788e+001 , 4.81540117e-002 ],
           'fc':[ 4.15299561e-001 , 2.91761325e+000 , 1.51474927e-001 ],
           'fd':[ 4.24913856e-001 , 1.90899693e+001 , 8.81568925e-002 ]}
    }


#From Appendix C of Kirkland, "Advanced Computing in Electron Microscopy", 2nd ed.
#Calculation of electron form factor for specific q:
# Using equation Kirkland C.15

def feq(element, q):
    # form factor parametrized in 1/Angstrom    
    # but we think in 1/nm
    q = q/10
    # q is now magnitude of scattering vector in 1/A -- (=> exp(-i*g.r), physics negative convention)
    param = electronFF[element]
    fL = 0
    fG = 0
    for i in range(3):
        fL += param['fa'][i]/(q**2  + param['fb'][i])
        fG += param['fc'][i]*np.exp(-q**2 *param['fd'][i])

    #Conversion factor from scattering factors to volts. h^2/(2pi*m0*e), see e.g. Kirkland eqn. C.5
    #!NB RVolume is already in A unlike RPlanckConstant
    #ScattFacToVolts=(PlanckConstant**2)*(AngstromConversion**2)/(2*np.pi*ElectronMass*ElectronCharge)
    return (fL+fG)#* ScattFacToVolts




# Jmol colors.  See: http://jmol.sourceforge.net/jscolors/#color_U
jmol_colors = np.array([
(1.000,0.000,0.000) ,# None
(1.000,1.000,1.000), # H
(0.851,1.000,1.000), # He
(0.800,0.502,1.000), # Li
(0.761,1.000,0.000), # Be
(1.000,0.710,0.710), # B
(0.565,0.565,0.565), # C
(0.188,0.314,0.973), # N
(1.000,0.051,0.051), # O
(0.565,0.878,0.314), # F
(0.702,0.890,0.961), # Ne
(0.671,0.361,0.949), # Na
(0.541,1.000,0.000), # Mg
(0.749,0.651,0.651), # Al
(0.941,0.784,0.627), # Si
(1.000,0.502,0.000), # P
(1.000,1.000,0.188), # S
(0.122,0.941,0.122), # Cl
(0.502,0.820,0.890), # Ar
(0.561,0.251,0.831), # K
(0.239,1.000,0.000), # Ca
(0.902,0.902,0.902), # Sc
(0.749,0.761,0.780), # Ti
(0.651,0.651,0.671), # V
(0.541,0.600,0.780), # Cr
(0.612,0.478,0.780), # Mn
(0.878,0.400,0.200), # Fe
(0.941,0.565,0.627), # Co
(0.314,0.816,0.314), # Ni
(0.784,0.502,0.200), # Cu
(0.490,0.502,0.690), # Zn
(0.761,0.561,0.561), # Ga
(0.400,0.561,0.561), # Ge
(0.741,0.502,0.890), # As
(1.000,0.631,0.000), # Se
(0.651,0.161,0.161), # Br
(0.361,0.722,0.820), # Kr
(0.439,0.180,0.690), # Rb
(0.000,1.000,0.000), # Sr
(0.580,1.000,1.000), # Y
(0.580,0.878,0.878), # Zr
(0.451,0.761,0.788), # Nb
(0.329,0.710,0.710), # Mo
(0.231,0.620,0.620), # Tc
(0.141,0.561,0.561), # Ru
(0.039,0.490,0.549), # Rh
(0.000,0.412,0.522), # Pd
(0.753,0.753,0.753), # Ag
(1.000,0.851,0.561), # Cd
(0.651,0.459,0.451), # In
(0.400,0.502,0.502), # Sn
(0.620,0.388,0.710), # Sb
(0.831,0.478,0.000), # Te
(0.580,0.000,0.580), # I
(0.259,0.620,0.690), # Xe
(0.341,0.090,0.561), # Cs
(0.000,0.788,0.000), # Ba
(0.439,0.831,1.000), # La
(1.000,1.000,0.780), # Ce
(0.851,1.000,0.780), # Pr
(0.780,1.000,0.780), # Nd
(0.639,1.000,0.780), # Pm
(0.561,1.000,0.780), # Sm
(0.380,1.000,0.780), # Eu
(0.271,1.000,0.780), # Gd
(0.188,1.000,0.780), # Tb
(0.122,1.000,0.780), # Dy
(0.000,1.000,0.612), # Ho
(0.000,0.902,0.459), # Er
(0.000,0.831,0.322), # Tm
(0.000,0.749,0.220), # Yb
(0.000,0.671,0.141), # Lu
(0.302,0.761,1.000), # Hf
(0.302,0.651,1.000), # Ta
(0.129,0.580,0.839), # W
(0.149,0.490,0.671), # Re
(0.149,0.400,0.588), # Os
(0.090,0.329,0.529), # Ir
(0.816,0.816,0.878), # Pt
(1.000,0.820,0.137), # Au
(0.722,0.722,0.816), # Hg
(0.651,0.329,0.302), # Tl
(0.341,0.349,0.380), # Pb
(0.620,0.310,0.710), # Bi
(0.671,0.361,0.000), # Po
(0.459,0.310,0.271), # At
(0.259,0.510,0.588), # Rn
(0.259,0.000,0.400), # Fr
(0.000,0.490,0.000), # Ra
(0.439,0.671,0.980), # Ac
(0.000,0.729,1.000), # Th
(0.000,0.631,1.000), # Pa
(0.000,0.561,1.000), # U
(0.000,0.502,1.000), # Np
(0.000,0.420,1.000), # Pu
(0.329,0.361,0.949), # Am
(0.471,0.361,0.890), # Cm
(0.541,0.310,0.890), # Bk
(0.631,0.212,0.831), # Cf
(0.702,0.122,0.831), # Es
(0.702,0.122,0.729), # Fm
(0.702,0.051,0.651), # Md
(0.741,0.051,0.529), # No
(0.780,0.000,0.400), # Lr
(0.800,0.000,0.349), # Rf
(0.820,0.000,0.310), # Db
(0.851,0.000,0.271), # Sg
(0.878,0.000,0.220), # Bh
(0.902,0.000,0.180), # Hs
(0.922,0.000,0.149), # Mt
])

# encoding: utf-8

### crystal data base cbd
cdb = {}
cdb['aluminum'] = {}
cdb['aluminum']['crystal_name'] = 'aluminum'
cdb['aluminum']['symmetry'] = 'FCC'
cdb['aluminum']['elements'] = ['Al']
cdb['aluminum']['a'] = 0.405 # nm
cdb['aluminum']['reference'] = 'W. Witt, Z. Naturforsch. A, 1967, 22A, 92'
cdb['aluminum']['link'] = 'http://doi.org/10.1515/zna-1967-0115'
cdb['Al'] = cdb['Aluminum'] = cdb['aluminum']

cdb['gold'] = {}
cdb['gold']['crystal_name'] = 'gold'
cdb['gold']['symmetry'] = 'FCC'
cdb['gold']['elements'] = ['Au']
cdb['gold']['a'] = 0.40782 # nm
cdb['gold']['reference'] = ''
cdb['gold']['link'] = ''
cdb['Au'] = cdb['Gold'] = cdb['gold']

cdb['silver'] = {}
cdb['silver']['crystal_name'] = 'silver'
cdb['silver']['symmetry'] = 'FCC'
cdb['silver']['elements'] = ['Ag']
cdb['silver']['a'] = 0.40853 # nm
cdb['silver']['reference'] = ''
cdb['silver']['link'] = ''
cdb['Ag'] = cdb['Silver'] = cdb['silver']

cdb['diamond'] = {}
cdb['diamond']['crystal_name'] = 'diamond'
cdb['diamond']['symmetry'] = 'zinc_blende'
cdb['diamond']['elements'] = ['C']
cdb['diamond']['a'] = 0.35668 # nm
cdb['diamond']['reference'] = ''
cdb['diamond']['link'] = ''
cdb['Diamond'] = cdb['diamond']

cdb['germanium'] = {}
cdb['germanium']['crystal_name'] = 'germanium'
cdb['germanium']['symmetry'] = 'zinc_blende'
cdb['germanium']['elements'] = ['Ge']
cdb['germanium']['a'] = 0.566806348   #nm for 300K 
cdb['germanium']['reference'] = 'H. P. Singh, Acta Crystallogr., 1968, 24A, 469'
cdb['germanium']['link'] = 'https://doi.org/10.1107/S056773946800094X'''
cdb['Ge'] = cdb['Germanium'] = cdb['germanium']

cdb['silicon'] = {}
cdb['silicon']['crystal_name'] = 'silicon'
cdb['silicon']['symmetry'] = 'zinc_blende'
cdb['silicon']['elements'] = ['Si']
cdb['silicon']['a'] = 0.566806348   #nm for 300K 
cdb['silicon']['reference'] = 'C. R. Hubbard, H. E. Swanson, and F. A. Mauer, J. Appl. Crystallogr., 1975, 8, 45'
cdb['silicon']['link'] = 'https://doi.org/10.1107/S0021889875009508'
cdb['Si'] = cdb['Silicon'] = cdb['silicon']  

cdb['GaAs'] = {}
cdb['GaAs']['crystal_name'] = 'GaAs'
cdb['GaAs']['symmetry'] = 'zinc_blende'
cdb['GaAs']['elements'] = ['Ga', 'As']
cdb['GaAs']['a'] = 0.565325   #nm for 300K 
cdb['GaAs']['reference'] = 'J.F.C. Baker, M. Hart, M.A.G. Halliwell, R. Heckingbottom, Solid-State Electronics, 19, 1976, 331-334,'
cdb['GaAs']['link'] = 'https://doi.org/10.1016/0038-1101(76)90031-9'
    
cdb['FCC Fe'] = {}
cdb['FCC Fe']['crystal_name'] = 'FCC Fe'
cdb['FCC Fe']['symmetry'] = 'FCC'
cdb['FCC Fe']['elements'] = ['Fe']
cdb['FCC Fe']['a'] = 0.3571   #nm 
cdb['FCC Fe']['reference'] = 'R. Kohlhaas, P. Donner, and N. Schmitz-Pranghe, Z. Angew. Phys., 1967, 23, 245'
cdb['FCC Fe']['link'] = ''
cdb['fcc fe'] = cdb['FCC Fe']


cdb['BCC Fe'] = {}
cdb['BCC Fe']['crystal_name'] = 'BCC Fe'
cdb['BCC Fe']['symmetry'] = 'BCC'
cdb['BCC Fe']['elements'] = ['Fe']
cdb['BCC Fe']['a'] = 0.2866   #nm 
cdb['BCC Fe']['reference'] = 'Z. S. Basinski, W. Hume-Rothery and A. L. Sutton, Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences Vol. 229, No. 1179 (May 24, 1955), pp. 459-467'
cdb['BCC Fe']['link'] = 'http://www.jstor.org/stable/99693'
cdb['bcc fe'] = cdb['BCC Fe']
                        


cdb['SrTiO3'] = {}
cdb['SrTiO3']['crystal_name'] = 'SrTiO3'
cdb['SrTiO3']['symmetry'] = 'perovskite'
cdb['SrTiO3']['elements'] = ['Sr','Ti','O']
cdb['SrTiO3']['a'] = 0.3905268   #nm 
cdb['SrTiO3']['reference'] =  'M. Schmidbauer, A. Kwasniewski and J. Schwarzkopf, Acta Cryst. (2012). B68, 8-14'
cdb['SrTiO3']['link'] = 'http://doi.org/10.1107/S0108768111046738'

cdb['ZnO Wurzite'] = {}
cdb['ZnO Wurzite']['crystal_name'] = 'ZnO Wurzite'
cdb['ZnO Wurzite']['symmetry'] = 'wurzite'
cdb['ZnO Wurzite']['elements'] = ['Zn','O']  
cdb['ZnO Wurzite']['a'] = 0.3278   #nm 
cdb['ZnO Wurzite']['c'] = 0.5292   #nm 
cdb['ZnO Wurzite']['u'] = 0.382   #nm 
cdb['ZnO Wurzite']['reference'] =  ''
cdb['ZnO Wurzite']['link'] = ''
cdb['ZnO'] = cdb['ZnO wurzite'] = cdb['wZnO'] = cdb['ZnO Wurzite']

cdb['GaN'] = {}
cdb['GaN']['crystal_name'] = 'GaN Wurzite'
cdb['GaN']['symmetry'] = 'wurzite'
cdb['GaN']['elements'] = ['Ga', 'N']  
cdb['GaN']['a'] = 0.3186   #nm 
cdb['GaN']['c'] = 0.5186   #nm 
cdb['GaN']['u'] = 0.376393   #nm 
cdb['GaN']['reference'] =  ''
cdb['GaN']['link'] = ''
cdb['GaN wurzite'] = cdb['wGaN'] = cdb['GaN Wurzite'] =  cdb['GaN']


cdb['MgO'] = {}
cdb['MgO']['crystal_name'] = 'MgO'
cdb['MgO']['symmetry'] = 'rocksalt'
cdb['MgO']['elements'] = ['Mg','O']
cdb['MgO']['a'] = 0.4256483   #nm 
cdb['MgO']['reference'] =  ''
cdb['MgO']['link'] = ''

cdb['MoS2'] = {}
cdb['MoS2']['crystal_name'] = 'MoS2'
cdb['MoS2']['symmetry'] = 'dichalcogenide'
cdb['MoS2']['elements'] = ['Mo','S']
cdb['MoS2']['a'] = 0.319031573 #nm
cdb['MoS2']['c'] = 1.487900430 #nm 
cdb['MoS2']['u'] = 0.105174    #nm
cdb['MoS2']['reference'] =  ''
cdb['MoS2']['link'] = ''

cdb['WS2'] = {}
cdb['WS2']['crystal_name'] = 'WS2'
cdb['WS2']['symmetry'] = 'dichalcogenide'
cdb['WS2']['elements'] = ['W','S']
cdb['WS2']['a'] = 0.319073051 #nm
cdb['WS2']['c'] = 1.420240204 #nm 
cdb['WS2']['u'] = 0.110759    #nm
cdb['WS2']['reference'] =  ''
cdb['WS2']['link'] = ''

cdb['WSe2'] = {}
cdb['WSe2']['crystal_name'] = 'WSe2'
cdb['WSe2']['symmetry'] = 'dichalcogenide'
cdb['WSe2']['elements'] = ['W','Se']
cdb['WSe2']['a'] = 0.332706918 #nm
cdb['WSe2']['c'] = 1.506895072 #nm 
cdb['WSe2']['u'] = 0.111569    #nm
cdb['WSe2']['reference'] =  ''
cdb['WSe2']['link'] = ''

cdb['MoSe2'] = {}
cdb['MoSe2']['crystal_name'] = 'MoSe2'
cdb['MoSe2']['elements'] = ['Mo','Se']
cdb['MoSe2']['a'] = 0.332694913 #nm
cdb['MoSe2']['c'] = 1.545142322 #nm 
cdb['MoSe2']['u'] = 0.108249    #nm
cdb['MoSe2']['reference'] =  ''
cdb['MoSe2']['link'] = ''



cdb['ZnO hexagonal'] = {}
cdb['ZnO hexagonal']['crystal_name'] = 'ZnO hexagonal'
#cdb['ZnO hexagonal']['symmetry'] = 'hexagonal'
cdb['ZnO hexagonal']['a']  = a = 0.3336     #nm
cdb['ZnO hexagonal']['c'] = c = 0.4754    # not np.sqrt(8/3)*1
cdb['ZnO hexagonal']['unit_cell'] = [[a,0.,0.],
                         [np.cos(120/180*np.pi)*a,np.sin(120/180*np.pi)*a,0. ],
                         [0.,0.,c]
                        ]
cdb['ZnO hexagonal']['elements'] =  ['Zn','Zn','O','O']
base = [(2./3.,1./3.,.500),(1./3.,2./3.,0.000),(2./3.,1./3.,0.0 ),(1./3.,2./3.,.5 )]
cdb['ZnO hexagonal']['base'] = np.array(base)

cdb['Graphite'] = {}
cdb['Graphite']['crystal_name'] = 'Graphite'
#cdb['Graphite']['symmetry'] = 'hexagonal'
#### Create graphite unit cell (or structure matrix)
cdb['Graphite']['a'] = a = b = 0.2464 #nm
cdb['Graphite']['c'] = c = 0.6711
gamma = 60
alpha = beta = 90

cdb['Graphite']['unit_cell'] =  from_parameters(a, b, c, alpha, beta, gamma)

#### Create graphite atom base
# Elements of base
cdb['Graphite']['elements'] =  ['C']*4
# atom positions in relative coordinates
base = np.array([(0,0,0), (0,0,0.5), (1./3.,1./3.,0.000),(2./3.,2./3.,0.5 )])
cdb['Graphite']['base'] = np.array(base)
cdb['Graphite']['reference'] = 'P. Trucano and R. Chen, Nature, 1975, 258, 136'
cdb['Graphite']['link'] = 'https://doi.org/10.1038/258136a0'

cdb['graphite'] = cdb['Graphite'] 

cdb['CsCl'] = {}
    # Create CsCl structure
cdb['CsCl']['a'] = a = 0.4209 #nm
cdb['CsCl']['crystal_name'] = 'CsCl'
cdb['CsCl']['unit_cell'] = np.identity(3)*a
cdb['CsCl']['elements'] =  ['Cs','Cl']  
cdb['CsCl']['base'] = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
cdb['CsCl']['reference'] = ''
cdb['CsCl']['link'] = ''

cdb['PdSe2'] = {}
    # Create CsCl structure
cdb['PdSe2']['crystal_name'] = 'PdSe2'
cdb['PdSe2']['unit_cell'] = (np.identity(3)*(.579441832,0.594542204,0.858506072) )
cdb['PdSe2']['elements'] =  ['Pd']*4 +['Se']*8  
cdb['PdSe2']['base'] = np.array([[.5, .0, .0], [.0, 0.5, 0.0], [.5, 0.5, 0.5], [.0, 0.5, 0.5],
                                [0.611300,  0.119356,  0.585891],
                                [0.111300,  0.380644,  0.414109],
                                [0.388700,  0.619356,  0.914109],
                                [0.888700,  0.880644,  0.085891],
                                [0.111300,  0.119356,  0.914109],
                                [0.611300,  0.380644,  0.085891],
                                [0.888700,  0.619356,  0.585891],
                                [0.388700,  0.880644,  0.414109]
                                ])
cdb['PdSe2']['reference'] = ''
cdb['PdSe2']['link'] = ''



crystal_data_base  = cdb
