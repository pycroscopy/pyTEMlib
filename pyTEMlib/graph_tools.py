"""

"""
import numpy as np
import ase

# from scipy.spatial import cKDTree, Voronoi, ConvexHull
import scipy.spatial
import scipy.optimize

# from skimage.measure import grid_points_in_poly, points_in_poly

import plotly.graph_objects as go
import plotly.express as px
import pyTEMlib.crystal_tools
from tqdm.auto import tqdm, trange


###########################################################################
# utility functions
###########################################################################

def intersitital_sphere_center(vertex_pos, atom_radii):
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

    if np.std(atom_radii) == 0:
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


def voronoi_volumes(points):
    """
    Volumes of voronoi  cells from
    https://stackoverflow.com/questions/19634993/volume-of-voronoi-cell-python
    """
    v = scipy.spatial.Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices:  # some regions can be opened
            vol[i] = np.inf
        else:
            try:
                hull = scipy.spatial.ConvexHull(v.vertices[indices])
                vol[i] = hull.volume
            except:
                vol[i] = 0.
    return vol


def get_voronoi(tetrahedra, atoms):
    """
    Find Voronoi vertices and keep track of associated tetrahedrons and interstitial radii

    Used in find_polyhedra function

    Parameters
    ----------
    tetrahedra: scipy.spatial.Delaunay object
        Delaunay tesselation
    atoms: ase.Atoms object
        the structural information

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

    voronoi_vertices = []
    voronoi_tetrahedrons = []
    r_vv = []
    r_aa = []
    for vertices in tetrahedra.vertices:

        r_a = []
        for vert in vertices:
            r_a.append(pyTEMlib.crystal_tools.electronFF[atoms.symbols[vert]]['bond_length'][1])
        voronoi, radius = intersitital_sphere_center(atoms.positions[vertices], r_a)

        r_a = np.average(r_a)  # np.min(r_a)
        r_aa.append(r_a)

        if (voronoi >= 0).all() and (extent - voronoi > 0).all() and radius > 0.01:
            voronoi_vertices.append(voronoi)
            voronoi_tetrahedrons.append(vertices)
            r_vv.append(radius)
    return voronoi_vertices, voronoi_tetrahedrons, r_vv, np.max(r_aa)


def find_overlapping_spheres(voronoi_vertices, r_vv, r_a, cheat=1.):
    """Find overlapping spheres"""

    vertex_tree = scipy.spatial.cKDTree(voronoi_vertices)

    pairs = vertex_tree.query_pairs(r=r_a * 2)

    overlapping_pairs = []
    for (i, j) in pairs:
        if np.linalg.norm(voronoi_vertices[i] - voronoi_vertices[j]) < (r_vv[i] + r_vv[j]) * cheat:
            overlapping_pairs.append([i, j])

    return np.array(sorted(overlapping_pairs))


def find_intertitial_clusters(overlapping_pairs):
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
    polyhedra={}
    for index, cluster in tqdm(enumerate(clusters)):
        cc = []
        for c in cluster:
            cc = cc + list(voronoi_tetrahedrons[c])

        hull = scipy.spatial.ConvexHull(atoms.positions[list(set(cc)),:2])
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
                            'atomic_numbers': atoms.get_atomic_numbers()[list(set(cc))]}  #, 'volume': hull.volume}
        # 'coplanar': hull.coplanar}

    print('Define conventional interstitial polyhedra')
    running_number = index + 0
    for index in trange(len(voronoi_vertices)):
        if index not in visited_all:
            vertices = voronoi_tetrahedrons[index]
            hull = scipy.spatial.ConvexHull(atoms.positions[vertices,:2])
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
                                         'atomic_numbers': atoms.get_atomic_numbers()[vertices]}  # 'volume': hull.volume}

            running_number += 1

    return polyhedra


def make_polyhedrons(atoms, voronoi_vertices, voronoi_tetrahedrons, clusters, visited_all):
    """collect output data  and make dictionary"""

    polyhedra = {}
    print('Define clustered interstitial polyhedra')
    for index, cluster in tqdm(enumerate(clusters)):
        cc = []
        for c in cluster:
            cc = cc + list(voronoi_tetrahedrons[c])

        hull = scipy.spatial.ConvexHull(atoms.positions[list(set(cc))])
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
                            'atomic_numbers': atoms.get_atomic_numbers()[list(set(cc))],
                            'volume': hull.volume}
        # 'coplanar': hull.coplanar}
    print('Define conventional interstitial polyhedra')
    running_number = index + 0
    for index in trange(len(voronoi_vertices)):
        if index not in visited_all:
            vertices = voronoi_tetrahedrons[index]
            hull = scipy.spatial.ConvexHull(atoms.positions[vertices])
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
                                         'atomic_numbers': atoms.get_atomic_numbers()[vertices],
                                         'volume': hull.volume}

            running_number += 1

    return polyhedra

    print('Define conventional interstitial polyhedra')
    running_number = index + 0
    for index in trange(len(voronoi_vertices)):
        if index not in visited_all:
            vertices = voronoi_tetrahedrons[index]
            hull = scipy.spatial.ConvexHull(atoms.positions[vertices])
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
                                         'atomic_numbers': atoms.get_atomic_numbers()[vertices],
                                         'volume': hull.volume}

            running_number += 1

    return polyhedra


##################################################################
# polyhedra functions
##################################################################


def find_polyhedra(atoms, cheat=1.0):
    """ get polyhedra information from an ase.Atoms object

    This is following the method of Banadaki and Patala
    http://dx.doi.org/10.1038/s41524-017-0016-0

    We are using the bond radius according to Kirkland, which is tabulated in
        - pyTEMlib.crystal_tools.electronFF[atoms.symbols[vert]]['bond_length'][1]

    Parameter
    ---------
    atoms: ase.Atoms object
        the structural information

    Returns
    -------
    polyhedra: dict
        dictionary with all information of polyhedra
    """
    if not isinstance(atoms, ase.Atoms):
        raise TypeError('This function needs an ase.Atoms object')

    if np.abs(atoms.positions[:,2]).sum() <= 0.01:
        tetrahedra = scipy.spatial.Delaunay(atoms.positions[:,:2])
    else:
        tetrahedra = scipy.spatial.Delaunay(atoms.positions)

    voronoi_vertices, voronoi_tetrahedrons, r_vv, r_a = get_voronoi(tetrahedra, atoms)

    overlapping_pairs = find_overlapping_spheres(voronoi_vertices, r_vv, r_a, cheat=cheat)

    clusters, visited_all = find_intertitial_clusters(overlapping_pairs)
    if np.abs(atoms.positions[:, 2]).sum() <= 0.01:
        polyhedra = make_polygons(atoms, voronoi_vertices, voronoi_tetrahedrons, clusters, visited_all)
    else:
        polyhedra = make_polyhedrons(atoms, voronoi_vertices, voronoi_tetrahedrons, clusters, visited_all)

    return polyhedra


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

##################################################################
# plotting functions
##################################################################


def plot_super_cell(super_cell, shift_x=0.):
    """ make a super_cell to plot with extra atoms at periodic boundaries"""

    if not isinstance(super_cell, ase.Atoms):
        raise TypeError('Need an ase Atoms object')

    plot_boundary = super_cell * (2, 2, 3)
    plot_boundary.positions[:, 0] = plot_boundary.positions[:, 0] - super_cell.cell[0, 0] * shift_x

    del plot_boundary[plot_boundary.positions[:, 2] > super_cell.cell[2, 2] * 1.5 + 0.1]
    del plot_boundary[plot_boundary.positions[:, 1] > super_cell.cell[1, 1] + 0.1]
    del plot_boundary[plot_boundary.positions[:, 0] > super_cell.cell[0, 0] + 0.1]
    del plot_boundary[plot_boundary.positions[:, 0] < -0.1]
    plot_boundary.cell = super_cell.cell * (1, 1, 1.5)

    return plot_boundary


def plot_polyhedron(polyhedra, indices, center=False):
    if isinstance(indices, int):
        indices = [indices]
    if len(indices) == 0:
        print('Did not find any polyhedra')
        return {}

    center_point = np.mean(polyhedra[indices[0]]['vertices'], axis=0)

    if center:
        print(center_point)
        center = center_point
    else:
        center = [0, 0, 0]

    data = []
    for index in indices:
        polyhedron = polyhedra[index]

        vertices = polyhedron['vertices'] - center
        faces = np.array(polyhedron['triangles'])
        x, y, z = vertices.T
        i_i, j_j, k_k = faces.T

        mesh = dict(type='mesh3d',
                    x=x,
                    y=y,
                    z=z,
                    i=i_i,
                    j=j_j,
                    k=k_k,
                    name='',
                    opacity=0.2,
                    color=px.colors.qualitative.Light24[len(vertices) % 24]
                    )
        tri_vertices = vertices[faces]
        x_e = []
        y_e = []
        z_e = []
        for t_v in tri_vertices:
            x_e += [t_v[k % 3][0] for k in range(4)] + [None]
            y_e += [t_v[k % 3][1] for k in range(4)] + [None]
            z_e += [t_v[k % 3][2] for k in range(4)] + [None]

        # define the lines to be plotted
        lines = dict(type='scatter3d',
                     x=x_e,
                     y=y_e,
                     z=z_e,
                     mode='lines',
                     name='',
                     line=dict(color='rgb(70,70,70)', width=1.5))
        data.append(mesh)
        data.append(lines)
    return data

def plot_bonds(polyhedra,  center=False):
    indices = range(len(polyhedra))

    data = []
    for index in indices:
        polyhedron = polyhedra[index]

        vertices = polyhedron['vertices']
        faces = np.array(polyhedron['triangles'])
        x, y, z = vertices.T
        i_i, j_j, k_k = faces.T

        tri_vertices = vertices[faces]
        x_e = []
        y_e = []
        z_e = []
        for t_v in tri_vertices:
            x_e += [t_v[k % 3][0] for k in range(4)] + [None]
            y_e += [t_v[k % 3][1] for k in range(4)] + [None]
            z_e += [t_v[k % 3][2] for k in range(4)] + [None]

        # define the lines to be plotted
        lines = dict(type='scatter3d',
                     x=x_e,
                     y=y_e,
                     z=z_e,
                     mode='lines',
                     name='',
                     line=dict(color='rgb(70,70,70)', width=1.5))
        data.append(lines)
    return data


def plot_with_polyhedra(polyhedra, indices, atoms=None, title=''):
    data = plot_polyhedron(polyhedra, indices)
    data[0]['opacity'] = 0.05
    fig = go.Figure(data=data)

    if atoms is not None:
        fig.add_trace(go.Scatter3d(
            mode='markers',
            x=atoms.positions[:, 0], y=atoms.positions[:, 1], z=atoms.positions[:, 2],
            marker=dict(
                color=atoms.get_atomic_numbers(),
                size=5,
                sizemode='diameter',
                colorscale=["blue", "green", "red"])))

    fig.update_layout(width=1000, height=700, showlegend=False)
    fig.update_layout(scene_aspectmode='data',
                      scene_aspectratio=dict(x=1, y=1, z=1))

    camera = {'up': {'x': 1, 'y': 0, 'z': 0},
              'center': {'x': 0, 'y': 0, 'z': 0},
              'eye': {'x': 0, 'y': 0, 'z': 1}}

    fig.update_layout(scene_camera=camera, title=title)
    fig.update_scenes(camera_projection_type="orthographic")
    return fig


def plot_supercell(grain_boundary, size=(1, 1, 1), shift_x=0.25, title=''):
    plot_boundary = plot_super_cell(grain_boundary * size, shift_x=shift_x)

    grain_boundary.cell.volume
    grain_boundary_area = grain_boundary.cell.lengths()[1] / grain_boundary.cell.lengths()[2]
    print(grain_boundary.symbols)
    volume__bulk_atom = 16.465237835776012
    ideal_volume = len(grain_boundary.positions) * volume__bulk_atom
    print(len(grain_boundary.positions) * volume__bulk_atom, grain_boundary.cell.volume)
    x_0 = ideal_volume / grain_boundary.cell.lengths()[1] / grain_boundary.cell.lengths()[2]
    print(f'Zero volume expansion supercell length: {x_0 / 10:.2f} nm; '
          f' compared to actual {grain_boundary.cell.lengths()[0] / 10:.2f} nm')

    fig = go.Figure(data=[
        go.Scatter3d(x=plot_boundary.positions[:, 0], y=plot_boundary.positions[:, 1], z=plot_boundary.positions[:, 2],
                     mode='markers',
                     marker=dict(
                         color=plot_boundary.get_atomic_numbers(),
                         size=5,
                         sizemode='diameter',
                         colorscale=["blue", "green", "red"]))])

    fig.update_layout(width=700, margin=dict(r=10, l=10, b=10, t=10))
    fig.update_layout(scene_aspectmode='data',
                      scene_aspectratio=dict(x=1, y=1, z=1))

    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=1)
    )
    fig.update_layout(scene_camera=camera, title=title)
    fig.update_scenes(camera_projection_type="orthographic")
    return fig


def plot_supercell_bonds(polyhedra, atoms, volumes=None, atom_size=15, title=''):
    data = plot_bonds(polyhedra)
    if volumes is None:
        volumes = [10] * len(atoms.get_atomic_numbers())

    fig = go.Figure(data=data)
    fig.add_trace(go.Scatter3d(
        mode='markers',
        x=atoms.positions[:, 0], y=atoms.positions[:, 1], z=atoms.positions[:, 2],
        marker=dict(
            color=atoms.get_atomic_numbers(),
            size=np.asarray(volumes) ** 2 / 10,
            sizemode='diameter',
            colorscale=["blue", "green", "red"])))

    fig.update_layout(width=1000, height=700, showlegend=False)
    fig.update_layout(scene_aspectmode='data',
                      scene_aspectratio=dict(x=1, y=1, z=1))

    camera = {'up': {'x': 0, 'y': 1, 'z': 0},
              'center': {'x': 0, 'y': 0, 'z': 0},
              'eye': {'x': 0, 'y': 0, 'z': 1}}
    fig.update_layout(scene_camera=camera, title=title)
    fig.update_scenes(camera_projection_type="orthographic")
    return fig


def plot_supercell_polyhedra(polyhedra, indices, atoms, volumes=None, title=''):
    data = plot_polyhedron(polyhedra, indices)
    if volumes is None:
        volumes = [10] * len(atoms.get_atomic_numbers())

    fig = go.Figure(data=data)
    fig.add_trace(go.Scatter3d(
        mode='markers',
        x=atoms.positions[:, 0], y=atoms.positions[:, 1], z=atoms.positions[:, 2],
        marker=dict(
            color=atoms.get_atomic_numbers(),
            size=np.asarray(volumes)**2 / 10,
            sizemode='diameter',
            colorscale=["blue", "green", "red"])))

    fig.update_layout(width=1000, height=700, showlegend=False)
    fig.update_layout(scene_aspectmode='data',
                      scene_aspectratio=dict(x=1, y=1, z=1))

    camera = {'up': {'x': 0, 'y': 1, 'z': 0},
              'center': {'x': 0, 'y': 0, 'z': 0},
              'eye': {'x': 0, 'y': 0, 'z': 1}}
    fig.update_layout(scene_camera=camera, title=title)
    fig.update_scenes(camera_projection_type="orthographic")
    return fig


def show_polyhedra(polyhedra, boundary_polyhedra, atoms, volumes=None, title=f''):
    data = plot_polyhedron(polyhedra, boundary_polyhedra)
    atom_indices = []
    for poly in boundary_polyhedra:
        atom_indices.extend(polyhedra[poly]['indices'])
    atom_indices = list(set(atom_indices))
    atomic_numbers = []
    atomic_volumes = []
    for atom in atom_indices:
        atomic_numbers.append(atoms[atom].number)
        atomic_volumes.append(volumes[atoms[atom].index] ** 2 / 10)

    if volumes is None:
        atomic_volumes = [10] * len(atoms.get_atomic_numbers())
    fig = go.Figure(data=data)

    fig.add_trace(go.Scatter3d(
        mode='markers',
        x=atoms.positions[atom_indices, 0], y=atoms.positions[atom_indices, 1], z=atoms.positions[atom_indices, 2],
        marker=dict(
            color=atomic_numbers,
            size=atomic_volumes,
            sizemode='diameter',
            colorscale=["blue", "green", "red"])))

    fig.update_layout(width=1000, height=700, showlegend=False)
    fig.update_layout(scene_aspectmode='data',
                      scene_aspectratio=dict(x=1, y=1, z=1))

    camera = {'up': {'x': 1, 'y': 0, 'z': 0},
              'center': {'x': 0, 'y': 0, 'z': 0},
              'eye': {'x': 0, 'y': 0, 'z': 1}}
    fig.update_layout(scene_camera=camera, title=title)
    fig.update_scenes(camera_projection_type="orthographic")
    return fig


def get_grain_boundary_polyhedra(polyhedra, atoms, grain_boundary_x=0, grain_boundary_width=0.5,
                                 verbose=True, visible=range(4, 16), z_lim=[0, 100]):
    boundary_polyhedra = []
    for key, polyhedron in polyhedra.items():
        center = polyhedron['vertices'].mean(axis=0)
        if abs(center[0] - grain_boundary_x) < .5 and (z_lim[0] < center[2] < z_lim[1]):
            boundary_polyhedra.append(key)
            if verbose:
                print(key, polyhedron['length'], center)

    return boundary_polyhedra