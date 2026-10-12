"""

"""
import numpy as np
import scipy
import skimage

import matplotlib.patches as patches

import pyTEMlib.crystal_tools
from tqdm.auto import tqdm, trange

# from .graph_viz import *
QT_available = False

###########################################################################
# utility functions
###########################################################################

def interstitial_sphere_center(vertex_pos, atom_radii, optimize=True):
    """
        Function finds center and radius of the largest interstitial sphere of a simplex.
        Which is the center of the cirumsphere if all atoms have the same radius,
        but differs for differently sized atoms.
        In the last case, the circumsphere center is used as starting point for refinement.

        Parameters
        -----------------
        vertex_pos : numpy array
            The position of vertices of a tetrahedron
        atom_radii : float
            bond radii of atoms
        optimize: boolean
            whether atom bond lengths are optimized or not
        Returns
        ----------
        new_center : numpy array
            The center of the largest interstitial sphere
        radius : float
            The radius of the largest interstitial sphere
        """
    center, radius = circum_center(vertex_pos, tol=1e-4)

    def distance_deviation(sphere_center):
        return np.std(np.linalg.norm(vertex_pos - sphere_center, axis=1) - atom_radii)

    if np.std(atom_radii) == 0 or not optimize:
        return center, radius-atom_radii[0]
    else:
        center_new = scipy.optimize.minimize(distance_deviation, center)
        return center_new.x, np.linalg.norm(vertex_pos[0]-center_new.x)-atom_radii[0]


def circum_center(vertex_pos, tol=1e-4):
    """
    Function finds the center and the radius of the circumsphere of every simplex.
    Reference:
    Fiedler, Miroslav. Matrices and graphs in geometry. No. 139. Cambridge University Press, 2011.
    (p.29 bottom: example 2.1.11)
    Code started from https://github.com/spatala/gbpy
    with help of https://codereview.stackexchange.com/questions/77593/calculating-the-volume-of-a-tetrahedron

    Parameters
    -----------------
    vertex_pos : numpy array
        The position of vertices of a tetrahedron
    tol : float
        Tolerance defined  to identify co-planar tetrahedrons
    Returns
    ----------
    circum_center : numpy array
        The center of the circumsphere
    circum_radius : float
        The radius of the circumsphere
    """

    # Make Cayley-Menger Matrix
    number_vertices = len(vertex_pos)
    matrix_c = np.identity(number_vertices+1)*-1+1
    distances = scipy.spatial.distance.pdist(np.asarray(vertex_pos, dtype=float), metric='sqeuclidean')
    matrix_c[1:, 1:] = scipy.spatial.distance.squareform(distances)
    det_matrix_c = (np.linalg.det(matrix_c))
    if abs(det_matrix_c) < tol:
        return np.array(vertex_pos[0]*0), 0
    matrix = -2 * np.linalg.inv(matrix_c)

    center = vertex_pos[0, :]*0
    for i in range(number_vertices):
        center += matrix[0, i+1] * vertex_pos[i, :]
    center /= np.sum(matrix[0, 1:])

    circum_radius = np.sqrt(matrix[0, 0]) / 2

    return np.array(center), circum_radius


def voronoi_volumes(atoms):
    """
    Volumes of voronoi  cells from
    https://stackoverflow.com/questions/19634993/volume-of-voronoi-cell-python
    """
    points = atoms.positions
    v = scipy.spatial.Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices:  # some regions can be opened
            vol[i] = 0
        else:
            try:
                hull = scipy.spatial.ConvexHull(v.simplices[indices])
                vol[i] = hull.volume
            except:
                vol[i] = 0.

    if atoms.info is None:
        atoms.info = {}
    # atoms.info.update({'volumes': vol})
    return vol


def get_bond_radii(atoms, bond_type='bond'):
    """ get all bond radii from Kirkland 
    Parameter:
    ----------
    atoms ase.Atoms object
        structure information in ase format
    type: str
        type of bond 'covalent' or 'metallic'
    """
    
    r_a = []
    for atom in atoms:
        if atom.symbol == 'X':
            r_a.append(1.2)
        else:
            if bond_type == 'covalent':
                r_a.append(pyTEMlib.crystal_tools.electronFF[atom.symbol]['bond_length'][0])
            else:
                r_a.append(pyTEMlib.crystal_tools.electronFF[atom.symbol]['bond_length'][1])
    if atoms.info is None:
        atoms.info = {}
    atoms.info['bond_radii'] = r_a
    return r_a


def set_bond_radii(atoms, bond_type='bond'):
    """ set certain or all bond-radii taken from Kirkland 
    
    Bond_radii are also stored in atoms.info
    
    Parameter:
    ----------
    atoms ase.Atoms object
        structure information in ase format
    type: str
        type of bond 'covalent' or 'metallic'
    Return:
    -------
    r_a: list
        list of atomic bond-radii 
        
    """
    if atoms.info is None:
        atoms.info = {}
    if 'bond_radii' in atoms.info:
        r_a = atoms.info['bond_radii']
    else:
        r_a = np.ones(len(atoms))
        
    for atom in atoms:
        if bond_type == 'covalent':
            r_a[atom.index] = (pyTEMlib.crystal_tools.electronFF[atom.symbol]['bond_length'][0])
        else:
            r_a[atom.index] = (pyTEMlib.crystal_tools.electronFF[atom.symbol]['bond_length'][1])
    atoms.info['bond_radii'] = r_a
    return r_a


def get_voronoi(tetrahedra, atoms, bond_radii=None, optimize=True):
    """
    Find Voronoi vertices and keep track of associated tetrahedrons and interstitial radii

    Used in find_polyhedra function

    Parameters
    ----------
    tetrahedra: scipy.spatial.Delaunay object
        Delaunay tesselation
    atoms: ase.Atoms object
        the structural information
    optimize: boolean
        whether to use different atom radii or not

    Returns
    -------
    voronoi_vertices: list
        list of positions of voronoi vertices
    voronoi_tetrahedra:
        list of indices of associated vertices of tetrahedra
    r_vv: list of float
        list of all interstitial sizes
    """

    extent = atoms.cell.lengths()
    print('extent', extent)

    if np.abs(atoms.positions[:, 2]).sum() <= 0.01:
        positions = atoms.positions[:, :2]
        extent = extent[:2]
    else:
        positions = atoms.positions

    if atoms.info is None:
        atoms.info = {}

    if bond_radii is not None:
        bond_radii = [bond_radii]*len(atoms)
    elif 'bond_radii' in atoms.info:
        bond_radii = atoms.info['bond_radii']
    
    else:
        bond_radii = get_bond_radii(atoms)
        
    voronoi_vertices = []
    voronoi_tetrahedrons = []
    r_vv = []
    r_aa = []
    print('Find interstitials (finding centers for different elements takes a bit)')
    for vertices in tqdm(tetrahedra.simplices):
        r_a = []
        for vert in vertices:
            r_a.append(bond_radii[vert])
        voronoi, radius = interstitial_sphere_center(positions[vertices], r_a, optimize=optimize)
    
        r_a = np.average(r_a)  # np.min(r_a)
        r_aa.append(r_a)

        if (voronoi >= 0).all() and (extent - voronoi > 0).all() and radius > 0.01:
            voronoi_vertices.append(voronoi)
            voronoi_tetrahedrons.append(vertices)
            r_vv.append(radius)
    return voronoi_vertices, voronoi_tetrahedrons, r_vv, np.max(r_aa)


def find_overlapping_spheres(voronoi_vertices, r_vv, r_a, cheat=1.):
    """Find overlapping spheres"""

    vertex_tree = scipy.spatial.KDTree(voronoi_vertices)

    pairs = vertex_tree.query_pairs(r=r_a * 2)

    overlapping_pairs = []
    for (i, j) in pairs:
        if np.linalg.norm(voronoi_vertices[i] - voronoi_vertices[j]) < (r_vv[i] + r_vv[j]) * cheat:
            overlapping_pairs.append([i, j])

    return np.array(sorted(overlapping_pairs))


def find_interstitial_clusters(overlapping_pairs):
    """Make clusters
    Breadth first search to go through the list of overlapping spheres or circles to determine clusters
    """
    visited_all = []
    clusters = []
    for initial in overlapping_pairs[:, 0]:
        if initial not in visited_all:
            # breadth first search
            visited = []  # the atoms we visited
            queue = [initial]
            while queue:
                node = queue.pop(0)
                if node not in visited_all:
                    visited.append(node)
                    visited_all.append(node)
                    # neighbors = overlapping_pairs[overlapping_pairs[:,0]==node,1]
                    neighbors = np.append(overlapping_pairs[overlapping_pairs[:, 1] == node, 0],
                                          overlapping_pairs[overlapping_pairs[:, 0] == node, 1])

                    for i, neighbour in enumerate(neighbors):
                        if neighbour not in visited:
                            queue.append(neighbour)
            clusters.append(visited)
    return clusters, visited_all


def make_polygons(atoms, voronoi_vertices, voronoi_tetrahedrons, clusters, visited_all):
    """ make polygons from convex hulls of vertices around interstitial positions"""
    polyhedra = {}
    for index, cluster in tqdm(enumerate(clusters)):
        cc = []
        for c in cluster:
            cc = cc + list(voronoi_tetrahedrons[c])

        hull = scipy.spatial.ConvexHull(atoms.positions[list(set(cc)), :2])
        faces = []
        triangles = []
        for s in hull.simplices:
            faces.append(atoms.positions[list(set(cc))][s])
            triangles.append(list(s))
        polyhedra[index] = {'vertices': atoms.positions[list(set(cc))], 'indices': list(set(cc)),
                            'faces': faces, 'triangles': triangles,
                            'length': len(list(set(cc))),
                            'combined_vertices': cluster,
                            'interstitial_index': index,
                            'interstitial_site': np.array(voronoi_tetrahedrons)[cluster].mean(axis=0),
                            'atomic_numbers': atoms.get_atomic_numbers()[list(set(cc))]}   # , 'volume': hull.volume}
        # 'coplanar': hull.coplanar}

    print('Define conventional interstitial polyhedra')
    running_number = index + 0
    for index in trange(len(voronoi_vertices)):
        if index not in visited_all:
            vertices = voronoi_tetrahedrons[index]
            hull = scipy.spatial.ConvexHull(atoms.positions[vertices, :2])
            faces = []
            triangles = []
            for s in hull.simplices:
                faces.append(atoms.positions[vertices][s])
                triangles.append(list(s))

            polyhedra[running_number] = {'vertices': atoms.positions[vertices], 'indices': vertices,
                                         'faces': faces, 'triangles': triangles,
                                         'length': len(vertices),
                                         'combined_vertices': index,
                                         'interstitial_index': running_number,
                                         'interstitial_site': np.array(voronoi_tetrahedrons)[index],
                                         'atomic_numbers': atoms.get_atomic_numbers()[vertices]}
            # 'volume': hull.volume}

            running_number += 1

    return polyhedra


def make_polyhedrons(atoms, voronoi_vertices, voronoi_tetrahedrons, clusters, visited_all):
    """collect output data  and make dictionary"""

    polyhedra = {}
    import scipy.sparse
    connectivity_matrix = scipy.sparse.dok_matrix((len(atoms), len(atoms)), dtype=bool)

    print('Define clustered interstitial polyhedra')
    for index, cluster in tqdm(enumerate(clusters)):
        cc = []
        for c in cluster:
            cc = cc + list(voronoi_tetrahedrons[c])
        cc = list(set(cc))

        hull = scipy.spatial.ConvexHull(atoms.positions[cc])
        faces = []
        triangles = []
        for s in hull.simplices:
            faces.append(atoms.positions[cc][s])
            triangles.append(list(s))
            for k in range(len(s)):
                l = (k + 1) % len(s)
                if cc[s[k]] > cc[s[l]]:
                    connectivity_matrix[cc[s[l]], cc[s[k]]] = True
                else:
                    connectivity_matrix[cc[s[k]], cc[s[l]]] = True

        polyhedra[index] = {'vertices': atoms.positions[list(set(cc))], 'indices': list(set(cc)),
                            'faces': faces, 'triangles': triangles,
                            'length': len(list(set(cc))),
                            'combined_vertices': cluster,
                            'interstitial_index': index,
                            'interstitial_site': np.array(voronoi_tetrahedrons)[cluster].mean(axis=0),
                            'atomic_numbers': atoms.get_atomic_numbers()[list(set(cc))],
                            'volume': hull.volume}
        # 'coplanar': hull.coplanar}

    print('Define conventional interstitial polyhedra')
    running_number = index + 0
    for index in range(len(voronoi_vertices)):
        if index not in visited_all:
            vertices = voronoi_tetrahedrons[index]
            hull = scipy.spatial.ConvexHull(atoms.positions[vertices])
            faces = []
            triangles = []
            for s in hull.simplices:
                faces.append(atoms.positions[vertices][s])
                triangles.append(list(s))
                for k in range(len(s)):
                    l = (k + 1) % len(s)
                    if cc[s[k]] > cc[s[l]]:
                        connectivity_matrix[cc[s[l]], cc[s[k]]] = True
                    else:
                        connectivity_matrix[cc[s[k]], cc[s[l]]] = True

            polyhedra[running_number] = {'vertices': atoms.positions[vertices], 'indices': vertices,
                                         'faces': faces, 'triangles': triangles,
                                         'length': len(vertices),
                                         'combined_vertices': index,
                                         'interstitial_index': running_number,
                                         'interstitial_site': np.array(voronoi_tetrahedrons)[index],
                                         'atomic_numbers': atoms.get_atomic_numbers()[vertices],
                                         'volume': hull.volume}

            running_number += 1
    if atoms.info is None:
        atoms.info = {}
    atoms.info.update({'graph': {'connectivity_matrix': connectivity_matrix}})
    return polyhedra


##################################################################
# polyhedra functions
##################################################################

def get_non_periodic_supercell(super_cell):
    super_cell.wrap()
    atoms = super_cell*3
    atoms.positions -= super_cell.cell.lengths()
    atoms.positions[:,0] += super_cell.cell[0,0]*.0
    del(atoms[atoms.positions[: , 0]<-5])
    del(atoms[atoms.positions[: , 0]>super_cell.cell[0,0]+5])
    del(atoms[atoms.positions[: , 1]<-5])
    del(atoms[atoms.positions[: , 1]>super_cell.cell[1,1]+5])
    del(atoms[atoms.positions[: , 2]<-5])
    del(atoms[atoms.positions[: , 2]>super_cell.cell[2,2]+5])
    return atoms

def get_connectivity_matrix(crystal, atoms, polyhedra):
    crystal_tree = scipy.spatial.KDTree(crystal.positions)
    
    
    connectivity_matrix = np.zeros([len(atoms),len(atoms)], dtype=int)

    for polyhedron in polyhedra.values():
        vertices = polyhedron['vertices'] - crystal.cell.lengths()
        atom_ind = np.array(polyhedron['indices'])
        dd, polyhedron['atom_indices'] = crystal_tree.query(vertices , k=1)
        to_bond = np.where(dd<0.001)[0]
       
        for triangle in polyhedron['triangles']:
            triangle = np.array(triangle)
            for permut in [[0,1], [1,2], [0,2]]:
                vertex = [np.min(triangle[permut]), np.max(triangle[permut])]
                if vertex[0] in to_bond or vertex[1] in to_bond:
                    connectivity_matrix[atom_ind[vertex[1]], atom_ind[vertex[0]]] = 1 
                    connectivity_matrix[atom_ind[vertex[0]], atom_ind[vertex[1]]] = 1 
    return connectivity_matrix
    


def get_bonds(crystal, shift= 0., verbose = False, cheat=1.0):
    """
    Get polyhedra, and bonds from  and edges and lengths of edges for each polyhedron and store it in info dictionary of new ase.Atoms object
    
    Parameter:
    ----------
    crystal: ase.atoms_object
        information on all polyhedra
    """
    crystal.positions += shift * crystal.cell[0, 0]
    crystal.wrap()
   
    atoms = get_non_periodic_supercell(crystal)
    atoms = atoms[atoms.numbers.argsort()]

    
    atoms.positions += crystal.cell.lengths()
    polyhedra = find_polyhedra(atoms, cheat=cheat)

    connectivity_matrix = get_connectivity_matrix(crystal, atoms, polyhedra) 
    coord = connectivity_matrix.sum(axis=1)
    
    del(atoms[np.where(coord==0)])
    new_polyhedra = {}
    index = 0
    octahedra =[]
    tetrahedra = []
    other = []
    super_cell_atoms =[]
    
    atoms_tree = scipy.spatial.KDTree(atoms.positions-crystal.cell.lengths())
    crystal_tree = scipy.spatial.KDTree(crystal.positions)
    connectivity_matrix = np.zeros([len(atoms),len(atoms)], dtype=float)
    
    for polyhedron in polyhedra.values():
        polyhedron['vertices'] -= crystal.cell.lengths()
        vertices = polyhedron['vertices']
        center = np.average(polyhedron['vertices'], axis=0)
        
        dd, polyhedron['indices'] = atoms_tree.query(vertices , k=1)
        atom_ind = (np.array(polyhedron['indices']))
        dd, polyhedron['atom_indices'] = crystal_tree.query(vertices , k=1)

        to_bond = np.where(dd<0.001)[0]
        super_cell_atoms.extend(list(atom_ind[to_bond]))
        
        edges = []
        lengths = []
        for triangle in polyhedron['triangles']:
            triangle = np.array(triangle)
            for permut in [[0,1], [1,2], [0,2]]:
                vertex = [np.min(triangle[permut]), np.max(triangle[permut])]
                length = np.linalg.norm(vertices[vertex[0]]-vertices[vertex[1]])
                if vertex[0] in to_bond or vertex[1] in to_bond:
                    connectivity_matrix[atom_ind[vertex[1]], atom_ind[vertex[0]]] = length 
                    connectivity_matrix[atom_ind[vertex[0]], atom_ind[vertex[1]]] = length 
                    if vertex[0] not in to_bond:
                        atoms[atom_ind[vertex[0]]].symbol = 'Be'
                    if vertex[1] not in to_bond: 
                        atoms[atom_ind[vertex[1]]].symbol = 'Be'
                if vertex not in edges:
                    edges.append(vertex)
                    lengths.append(np.linalg.norm(vertices[vertex[0]]-vertices[vertex[1]] ))
        polyhedron['edges'] = edges
        polyhedron['edge_lengths'] = lengths
        if all(center > -0.000001) and all(center < crystal.cell.lengths()-0.01):
                new_polyhedra[str(index)]=polyhedron
                if polyhedron['length'] == 4:
                    tetrahedra.append(str(index)) 
                elif polyhedron['length'] == 6:
                    octahedra.append(str(index))
                else:
                    other.append(str(index))
                    if verbose:
                        print(polyhedron['length'])
                index += 1
    atoms.positions -= crystal.cell.lengths()
    coord = connectivity_matrix.copy()
    coord[np.where(coord>.1)] = 1
    coord = coord.sum(axis=1)
    
    super_cell_atoms = np.sort(np.unique(super_cell_atoms))
    atoms.info.update({'polyhedra': {'polyhedra': new_polyhedra, 
                                    'tetrahedra': tetrahedra,
                                    'octahedra': octahedra,
                                    'other' : other}})
    atoms.info.update({'bonds': {'connectivity_matrix': connectivity_matrix, 
                                 'super_cell_atoms': super_cell_atoms,
                                 'super_cell_dimensions': crystal.cell.array,
                                 'coordination': coord}})
    atoms.info.update({'supercell': crystal})
    return atoms    





def find_polyhedra(atoms, optimize=True, cheat=1.0, bond_radii=None):
    """ get polyhedra information from an ase.Atoms object

    This is following the method of Banadaki and Patala
    http://dx.doi.org/10.1038/s41524-017-0016-0

    We are using the bond radius according to Kirkland, which is tabulated in
        - pyTEMlib.crystal_tools.electronFF[atoms.symbols[vert]]['bond_length'][1]

    Parameter
    ---------
    atoms: ase.Atoms object
        the structural information
    cheat: float
        does not exist

    Returns
    -------
    polyhedra: dict
        dictionary with all information of polyhedra
    """
    if not isinstance(atoms, ase.Atoms):
        raise TypeError('This function needs an ase.Atoms object')

    if np.abs(atoms.positions[:, 2]).sum() <= 0.01:
        positions = atoms.positions[:, :2]
        print('2D')
    else:
        positions = atoms.positions
    tetrahedra = scipy.spatial.Delaunay(positions)

    voronoi_vertices, voronoi_tetrahedrons, r_vv, r_a = get_voronoi(tetrahedra, atoms, optimize=optimize, bond_radii=bond_radii)
    
    if positions.shape[1] < 3:
        r_vv = np.array(r_vv)*1.
    overlapping_pairs = find_overlapping_spheres(voronoi_vertices, r_vv, r_a, cheat=cheat)

    clusters, visited_all = find_interstitial_clusters(overlapping_pairs)

    if positions.shape[1] < 3:
        rings = get_polygons(atoms, clusters, voronoi_tetrahedrons)
        return rings
    else:
        polyhedra = make_polyhedrons(atoms, voronoi_vertices, voronoi_tetrahedrons, clusters, visited_all)
    return polyhedra


def polygon_sort(corners):
    center = np.average(corners[:, :2], axis=0)
    angles = (np.arctan2(corners[:,0]-center[0], corners[:,1]-center[1]) + 2.0 * np.pi)% (2.0 * np.pi)
    return corners[np.argsort(angles)]

def get_polygons(atoms, clusters, voronoi_tetrahedrons):
    polygons = []
    cyclicity = []
    centers = []
    corners =[]
    for index, cluster in (enumerate(clusters)):
        cc = []
        for c in cluster:
            cc = cc + list(voronoi_tetrahedrons[c])
        
        sorted_corners = polygon_sort(atoms.positions[list(set(cc)), :2])
        cyclicity.append(len(sorted_corners))
        corners.append(sorted_corners)
        centers.append(np.mean(sorted_corners[:,:2], axis=0))
        polygons.append(patches.Polygon(np.array(sorted_corners)[:,:2], closed=True, fill=True, edgecolor='red'))

    rings={'atoms': atoms.positions[:, :2],
           'cyclicity': np.array(cyclicity),
           'centers': np.array(centers),
           'corners': corners,
           'polygons': polygons}
    return rings


def sort_polyhedra_by_vertices(polyhedra, visible=range(4, 100), z_lim=[0, 100], verbose=False):
    indices = []

    for key, polyhedron in polyhedra.items():
        if 'length' not in polyhedron:
            polyhedron['length'] = len(polyhedron['vertices'])

        if polyhedron['length'] in visible:
            center = polyhedron['vertices'].mean(axis=0)
            if z_lim[0] < center[2] < z_lim[1]:
                indices.append(key)
                if verbose:
                    print(key, polyhedron['length'], center)
    return indices

# color_scheme = ['lightyellow', 'silver', 'rosybrown', 'lightsteelblue', 'orange', 'cyan', 'blue', 'magenta',
#                'firebrick', 'forestgreen']




