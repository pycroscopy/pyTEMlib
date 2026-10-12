import numpy as np
import scipy
import skimage
import SimpleITK as sitk

import matplotlib.patches as patches

import pyTEMlib.crystal_tools
from tqdm.auto import tqdm, trange


def breadth_first_search(graph, initial_node_index, projected_crystal):
    """ breadth first search of atoms viewed as a graph

    the projection dictionary has to contain the following items
    'number_of_nearest_neighbours', 'rotated_cell', 'near_base', 'allowed_variation'

    Parameters
    ----------
    graph: numpy array (Nx2)
        the atom positions
    initial: int
        index of starting atom
    projection_tags: dict
        dictionary with information on projected unit cell (with 'rotated_cell' item)

    Returns
    -------
    graph[visited]: numpy array (M,2) with M<N
        positions of atoms hopped in unit cell lattice
    ideal: numpy array (M,2)
        ideal atom positions
    """

    projection_tags = projected_crystal.info['projection']
    if 'lattice_vector' in projection_tags:
        a_lattice_vector = projection_tags['lattice_vector']['a']
        b_lattice_vector = projection_tags['lattice_vector']['b']
        main = np.array([a_lattice_vector, -a_lattice_vector, b_lattice_vector, -b_lattice_vector])  # vectors of unit cell
        near =  main
    else:
        # get lattice vectors to hopp along through graph
        projected_unit_cell = projected_crystal.cell[:2, :2]
        a_lattice_vector = projected_unit_cell[0]
        b_lattice_vector = projected_unit_cell[1]
        main = np.array([a_lattice_vector, -a_lattice_vector, b_lattice_vector, -b_lattice_vector])  # vectors of unit cell
        near = projection_tags['near_base']  # all nearest atoms
        near = np.append(main, near, axis=0) 
    
    neighbour_tree = scipy.spatial.KDTree(graph)
    distances, indices = neighbour_tree.query(graph,  k=50) # let's get all neighbours
                                                
    
    visited = []  # the atoms we visited
    ideal = []  # atoms at ideal lattice
    sub_lattice = []  # atoms in base and disregarded
    queue = [initial_node_index]
    ideal_queue = [graph[initial_node_index]]

    while queue:
        node = queue.pop(0)
        ideal_node = ideal_queue.pop(0)

        if node not in visited:
            visited.append(node)
            ideal.append(ideal_node)
            # print(node,ideal_node)
            neighbors = indices[node]
            for i, neighbour in enumerate(neighbors):
                if neighbour not in visited:
                    distance_to_ideal = np.linalg.norm(near + graph[node] - graph[neighbour], axis=1)
                    if np.min(distance_to_ideal) < projection_tags['allowed_variation']:
                        direction = np.argmin(distance_to_ideal)
                        if direction > 3:  # counting starts at 0
                            sub_lattice.append(neighbour)
                        elif distances[node, i] < projection_tags['distance_unit_cell'] * 1.05:
                            queue.append(neighbour)
                            ideal_queue.append(ideal_node + near[direction])

    return graph[visited], ideal


def get_base_atoms(graph, origins, base, tolerance=3):
    """ get sublattices of atoms in a graph
    This function returns the indices of atoms in a graph that are close to the base atoms.
    Parameters
    ----------  
    graph: numpy array (Nx2)
        the atom positions
    origins: numpy array (Nx2)
        the origin positions
    base: numpy array (Mx2)
        the base atom positions
    tolerance: float
        the distance tolerance for finding base atoms
    Returns
    -------
    sublattices: list of numpy arrays
        list of indices of atoms in the graph that are close to each base atom
    """
    sublattices = []
    neighbour_tree = scipy.spatial.KDTree(graph)
    for base_atom in base:
         distances, indices = neighbour_tree.query(origins+base_atom[:2], k=50) 
         sublattices.append(indices[distances < tolerance]) 
    return sublattices

def analyze_atomic_structure(dataset, crystal, start_atom_index, tolerance=1.5):
    """ Analyze atomic structure of a crystal and return sublattices of atoms

    Parameters
    ----------
    dataset: pyTEMlib.Dataset
        dataset containing the atomic structure information
    crystal: ase.Atoms
        crystal structure to analyze
    start_atom_index: int
        index of the starting atom for the breadth-first search
    tolerance: float
        tolerance for determining the allowed variation in atom positions
    Returns
    -------
    sublattices: list of numpy arrays
        list of indices of atoms in the graph that are close to each base atom
    """
    if 'atoms' not in dataset.metadata:
        TypeError('dataset.metadata needs to contain atoms information')
        
    graph = dataset.metadata['atoms']['positions']
    
    layer = pyTEMlib.crystal_tools.get_projection(crystal)
    gamma = np.radians(layer.cell.angles()[2])
    rotation_angle = np.radians(crystal.info['experimental']['angle']
)
    length = (layer.cell.lengths() /10/dataset.x.slope)[:2]
    print(length, rotation_angle, gamma)
    a = np.array([np.cos(rotation_angle)*length[0], np.sin(rotation_angle)*length[0]])
    b = np.array([np.cos(rotation_angle+gamma)*length[1], np.sin(rotation_angle+gamma)*length[1]])
    base = layer.get_scaled_positions()
    base[:, :2] = np.dot(base[:, :2],[a,b])
    projection_tags = {'lattice_vector': {'a': a, 'b': b},
                       'allowed_variation': tolerance,
                       'distance_unit_cell':  np.max(length)*1.04,
                       'start_atom_index': start_atom_index,
                       'base': base}
    layer.info['projection'] = projection_tags

    origins, ideal = pyTEMlib.graph_tools.breadth_first_search(graph[:,:2], start_atom_index, layer)
    print(len(origins), 'origins found')
    dataset.metadata['atoms']['projection'] = layer
    sublattices = pyTEMlib.graph_tools.get_base_atoms(graph[:, :2], origins, base[:, :2], tolerance=3)
    
    dataset.metadata['atoms']['origins'] = origins
    dataset.metadata['atoms']['ideal_origins'] = ideal
    dataset.metadata['atoms']['sublattices'] = sublattices
    dataset.metadata['atoms']['projection_tags'] = projection_tags
    
    return sublattices


def breadth_first_search_felxible(graph, initial_node_index, lattice_parameter, tolerance=1):
    """ breadth first search of atoms viewed as a graph
        This is a rotational invariant search of atoms in a lattice, and returns the angles of unit cells.
        We only use the ideal lattice parameter to determine the lattice.  

    Parameters
    ----------
    graph: numpy array (Nx2)        

    """
    if isinstance(lattice_parameter, ase.Atoms):
        lattice_parameter = lattice_parameter.cell.lengths()[:2]
    elif isinstance(lattice_parameter, float):
        lattice_parameter = [lattice_parameter]
    lattice_parameter = np.array(lattice_parameter)

    neighbour_tree = scipy.spatial.KDTree(graph)
    distances, indices = neighbour_tree.query(graph,  k=50) # let's get all neighbours
    visited = []  # the atoms we visited
    angles = []  # atoms at ideal lattice
    sub_lattice = []  # atoms in base and disregarded
    queue = [initial_node_index]
    queue_angles=[0]
    
    while queue:
        node = queue.pop(0)
        angle = queue_angles.pop(0)
        if node not in visited:
            visited.append(node)
            angles.append(angle)
            neighbors = indices[node]
            for i, neighbour in enumerate(neighbors):
                if neighbour not in visited:
                    hopp = graph[node] - graph[neighbour]
                    distance_to_ideal = np.linalg.norm(hopp)
                    if np.min(np.abs(distance_to_ideal - lattice_parameter)) < tolerance:
                        queue.append(neighbour) 
                        queue_angles.append(np.arctan2(hopp[1], hopp[0]))
    angles[0] = angles[1]
    out_atoms = np.stack([graph[visited][:, 0], graph[visited][:, 1], angles])
    return out_atoms.T, visited

def delete_rim_atoms(atoms, extent, rim_distance):
    rim = np.where(atoms[:, :2] - extent > -rim_distance)[0]
    middle_atoms = np.delete(atoms, rim, axis=0)
    rim = np.where(middle_atoms[:, :2].min(axis=1)<rim_distance)[0]
    middle_atoms = np.delete(middle_atoms, rim, axis=0)
    return middle_atoms

####################
# Distortion Matrix
####################
def get_distortion_matrix(atoms, ideal_lattice):
    """    Calculates distortion matrix

    Calculates the distortion matrix by comparing ideal and distorted Voronoi tiles
    """

    vor = scipy.spatial.Voronoi(atoms)

    # determine a middle Voronoi tile
    ideal_vor = scipy.spatial.Voronoi(ideal_lattice)
    near_center = np.average(ideal_lattice, axis=0)
    index = np.argmin(np.linalg.norm(ideal_lattice - near_center, axis=0))

    # the ideal vertices fo such an Voronoi tile (are there crystals with more than one voronoi?)
    ideal_vertices = ideal_vor.vertices[ideal_vor.regions[ideal_vor.point_region[index]]]
    ideal_vertices = get_significant_vertices(ideal_vertices - np.average(ideal_vertices, axis=0))

    distortion_matrix = []
    for index in trange(vor.points.shape[0]):

        # determine vertices of Voronoi polygons of an atom with number index
        poly_point = vor.points[index]
        poly_vertices = get_significant_vertices(vor.vertices[vor.regions[vor.point_region[index]]] - poly_point)

        # where ATOM has to be moved (not pixel)
        ideal_point = ideal_lattice[index]

        # transform voronoi to ideal one and keep transformation matrix A
        uncorrected, corrected, aa = transform_voronoi(poly_vertices, ideal_vertices)

        # pixel positions
        corrected = corrected + ideal_point + (np.rint(poly_point) - poly_point)
        for i in range(len(corrected)):
            # original image pixels
            x, y = uncorrected[i] + np.rint(poly_point)
            # collect the two origin and target coordinates and store
            distortion_matrix.append([x, y, corrected[i, 0], corrected[i, 1]])
    print()
    return np.array(distortion_matrix)




def get_significant_vertices(vertices, distance=3):
    """Calculate average for  all points that are closer than distance apart, otherwise leave the points alone
        
        Parameters
        ----------
        vertices: numpy array (n,2)
            list of points
        distance: float
            (in same scale as points )
        
        Returns
        -------
        ideal_vertices: list of floats
            list of points that are all a minimum of 3 apart.
    """

    tt = scipy.spatial.KDTree(np.array(vertices))
    near = tt.query_ball_point(vertices, distance)
    ideal_vertices = []
    for indices in near:
        if len(indices) == 1:
            ideal_vertices.append(vertices[indices][0])
        else:
            ideal_vertices.append(np.average(vertices[indices], axis=0))
    ideal_vertices = np.unique(np.array(ideal_vertices), axis=0)
    angles = np.arctan2(ideal_vertices[:, 1], ideal_vertices[:, 0])
    ang_sort = np.argsort(angles)

    ideal_vertices = ideal_vertices[ang_sort]

    return ideal_vertices


def undistort(distortion_matrix, image_data):
    """ Undistort image according to distortion matrix
    
    Uses the griddata interpolation of scipy to apply distortion matrix to image.
    The distortion matrix contains in origin and target pixel coordinates
    target is where the pixel has to be moved (floats)
    
    Parameters
    ----------
    distortion_matrix: numpy array (Nx2)
        distortion matrix (format N x 2)
    image_data: numpy array or sidpy.Dataset
        image 
        
    Returns
    -------
    interpolated: numpy array
        undistorted image
    """

    intensity_values = image_data[(distortion_matrix[:, 0].astype(int), distortion_matrix[:, 1].astype(int))]

    corrected = distortion_matrix[:, 2:4]

    size_x, size_y = 2 ** np.round(np.log2(image_data.shape[0:2]))  # nearest power of 2
    size_x = int(size_x)
    size_y = int(size_y)
    grid_x, grid_y = np.mgrid[0:size_x - 1:size_x * 1j, 0:size_y - 1:size_y * 1j]
    print('interpolate')

    interpolated = scipy.interpolate.griddata(np.array(corrected), np.array(intensity_values), (grid_x, grid_y), method='linear')
    return interpolated


def transform_voronoi(vertices, ideal_voronoi):
    """ find transformation matrix A between a distorted polygon and a perfect reference one

    Returns
    -------
    uncorrected: list of points: 
        all points on a grid within original polygon
    corrected: list of points: 
        coordinates of these points where pixel have to move to
    aa: 2x2 matrix A:  
        transformation matrix
    """
    
    # Find Transformation Matrix, note polygons have to be ordered first.
    sort_vert = []
    for vert in ideal_voronoi:
        sort_vert.append(np.argmin(np.linalg.norm(vertices - vert, axis=1)))
    vertices = np.array(vertices)[sort_vert]

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix aa = A
    aa, res, rank, s = np.linalg.lstsq(vertices, ideal_voronoi, rcond=None)

    # expand polygon to include more points in distortion matrix
    vertices2 = vertices + np.sign(vertices)  # +np.sign(vertices)

    ext_v = int(np.abs(vertices2).max() + 1)

    polygon_grid = np.mgrid[0:ext_v * 2 + 1, :ext_v * 2 + 1] - ext_v
    polygon_grid = np.swapaxes(polygon_grid, 0, 2)
    polygon_array = polygon_grid.reshape(-1, polygon_grid.shape[-1])

    p = skimage.measure.points_in_poly(polygon_array, vertices2)
    uncorrected = polygon_array[p]

    corrected = np.dot(uncorrected, aa)

    return uncorrected, corrected, aa


def get_maximum_view(distortion_matrix):
    distortion_matrix_extent = np.ones(distortion_matrix.shape[1:], dtype=int)
    distortion_matrix_extent[distortion_matrix[0] == -1000.] = 0

    area = distortion_matrix_extent
    view_square = np.array([0, distortion_matrix.shape[1] - 1, 0, distortion_matrix.shape[2] - 1], dtype=int)
    while np.array(np.where(area == 0)).shape[1] > 0:
        view_square = view_square + [1, -1, 1, -1]
        area = distortion_matrix_extent[view_square[0]:view_square[1], view_square[2]:view_square[3]]

    change = [-int(np.sum(np.min(distortion_matrix_extent[:view_square[0], view_square[2]:view_square[3]], axis=1))),
              int(np.sum(np.min(distortion_matrix_extent[view_square[1]:, view_square[2]:view_square[3]], axis=1))),
              -int(np.sum(np.min(distortion_matrix_extent[view_square[0]:view_square[1], :view_square[2]], axis=0))),
              int(np.sum(np.min(distortion_matrix_extent[view_square[0]:view_square[1], view_square[3]:], axis=0)))]

    return np.array(view_square) + change

def transform_voronoi(vertices, ideal_voronoi):
    """
    find transformation matrix A between a polygon and a perfect one
    
    returns:
    list of points: all points on a grid within original polygon
    list of points: coordinates of these points where pixel have to move to
    2x2 matrix aa:  transformation matrix
    """
    # Find Transformation Matrix, note polygons have to be ordered first.
    sort_vert = []
    for vert in ideal_voronoi:
        sort_vert.append(np.argmin(np.linalg.norm(vertices - vert, axis=1)))
    vertices = np.array(vertices)[sort_vert]

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    aa, res, rank, s = np.linalg.lstsq(vertices, ideal_voronoi, rcond=None)

    # expand polygon to include more points in distortion matrix 
    vertices2 = vertices + np.sign(vertices)  # +np.sign(vertices)

    ext_v = int(np.abs(vertices2).max() + 1)

    polygon_grid = np.mgrid[0:ext_v * 2 + 1, :ext_v * 2 + 1] - ext_v
    polygon_grid = np.swapaxes(polygon_grid, 0, 2)
    polygon_array = polygon_grid.reshape(-1, polygon_grid.shape[-1])

    p = skimage.measure.points_in_poly(polygon_array, vertices2)
    uncorrected = polygon_array[p]

    corrected = np.dot(uncorrected, aa)

    return uncorrected, corrected, aa



def undistort_sitk(image_data, distortion_matrix):
    """    use simple ITK to undistort image
    
    Parameters
    ----------
    image_data: numpy array with size NxM 
    distortion_matrix: sidpy.Dataset or numpy array with size 2 x P x Q
    with P, Q >= M, N
    
    Returns
    -------
    image: numpy array MXN
      
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk.GetImageFromArray(image_data))
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)

    distortion_matrix2 = distortion_matrix[:, :image_data.shape[0], :image_data.shape[1]]

    displ2 = sitk.Compose(
        [sitk.GetImageFromArray(-distortion_matrix2[1]), sitk.GetImageFromArray(-distortion_matrix2[0])])
    out_tx = sitk.DisplacementFieldTransform(displ2)
    resampler.SetTransform(out_tx)
    out = resampler.Execute(sitk.GetImageFromArray(image_data))
    return sitk.GetArrayFromImage(out)


def undistort_stack_sitk(distortion_matrix, image_stack):
    """
    use simple ITK to undistort stack of image
    input:
    image: numpy array with size NxM 
    distortion_matrix: h5 Dataset or numpy array with size 2 x P x Q
    with P, Q >= M, N
    output:
    image M, N
      
    """

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk.GetImageFromArray(image_stack[0]))
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)

    displ2 = sitk.Compose(
        [sitk.GetImageFromArray(-distortion_matrix[1]), sitk.GetImageFromArray(-distortion_matrix[0])])
    out_tx = sitk.DisplacementFieldTransform(displ2)
    resampler.SetTransform(out_tx)

    interpolated = np.zeros(image_stack.shape)

    nimages = image_stack.shape[0]

    for i in trange(nimages):
        out = resampler.Execute(sitk.GetImageFromArray(image_stack[i]))
        interpolated[i] = sitk.GetArrayFromImage(out)

    return interpolated


def undistort_stack(distortion_matrix, data):
    """ Undistort stack with distortion matrix
    
    Use the griddata interpolation of scipy to apply distortion matrix to image
    The distortion matrix contains in each pixel where the pixel has to be moved (floats)

    Parameters
    ----------
    distortion_matrix: numpy array
        distortion matrix to undistort image (format image.shape[0], image.shape[2], 2)
    data: numpy array or sidpy.Dataset
        image
    """

    corrected = distortion_matrix[:, 2:4]
    intensity_values = data[:, distortion_matrix[:, 0].astype(int), distortion_matrix[:, 1].astype(int)]

    size_x, size_y = 2 ** np.round(np.log2(data.shape[1:]))  # nearest power of 2
    size_x = int(size_x)
    size_y = int(size_y)

    grid_x, grid_y = np.mgrid[0:size_x - 1:size_x * 1j, 0:size_y - 1:size_y * 1j]
    print('interpolate')

    interpolated = np.zeros([data.shape[0], size_x, size_y])
    nimages = data.shape[0]
    done = 0

    for i in trange(nimages):
        interpolated[i, :, :] = griddata(corrected, intensity_values[i, :], (grid_x, grid_y), method='linear')

    print(':-)')
    print('You have successfully completed undistortion of image stack')
    return interpolated

