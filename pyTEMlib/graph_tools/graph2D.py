from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import numpy as np
import scipy
import sklearn
import typing

def circum_center(vertex_pos: np.ndarray, tol: float = 1e-4) -> tuple[np.ndarray, float]:
    """
    Function finds the center and the radius of the circumsphere of every simplex.
    Reference:
    Fiedler, Miroslav. Matrices and graphs in geometry. No. 139. Cambridge University Press, 2011.
    (p.29 bottom: example 2.1.11)
    Code started from https://github.com/spatala/gbpy
    with help of 
    https://codereview.stackexchange.com/questions/77593/calculating-the-volume-of-a-tetrahedron

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
    distances = scipy.spatial.distance.pdist(np.asarray(vertex_pos, dtype=float),
                                             metric='sqeuclidean')
    matrix_c[1:, 1:] = scipy.spatial.distance.squareform(distances)
    det_matrix_c = np.linalg.det(matrix_c)
    if abs(det_matrix_c) < tol:
        return np.array(vertex_pos[0]*0), 0
    matrix = -2 * np.linalg.inv(matrix_c)

    center = vertex_pos[0, :]*0
    for i in range(number_vertices):
        center += matrix[0, i+1] * vertex_pos[i, :]
    center /= np.sum(matrix[0, 1:])

    circum_radius = np.sqrt(matrix[0, 0]) / 2

    return np.array(center), circum_radius

def interstitial_sphere_center(vertex_pos: np.ndarray, atom_radii: np.ndarray,
                               optimize: bool = True) -> typing.Tuple[np.ndarray, float]:
    """
        Function finds center and radius of the largest interstitial sphere of a simplex.
        Which is the center of the cirumsphere if all atoms have the same radius,
        but differs for differently sized atoms.
        In the last case, the circumsphere center is used as starting point for refinement.

        Parameters
        ----------
        vertex_pos : numpy array
            The position of vertices of a tetrahedron
        atom_radii : float
            bond radii of atoms
        optimize: boolean
            whether atom bond lengths are optimized or not
        Returns
        -------
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
    center_new = scipy.optimize.minimize(distance_deviation, center)
    return center_new.x, np.linalg.norm(vertex_pos[0]-center_new.x)-atom_radii[0]


def get_polygons(positions, min_voronoi_distance=1.):
    """
    Determines the rings in the plot according to planar graph theory
    """
    
    vor = scipy.spatial.Voronoi(positions)
    
    new_voronoi = get_new_vertices(vor, min_voronoi_distance)
    
    rings = get_rings(vor, new_voronoi)
    
    return rings

def get_new_vertices2(vor, min_voronoi_distance = 1.2):
    """
    Averages voronoi vertices that are too close together
    Not used
    """
    
    vorTree   = scipy.spatial.cKDTree(vor.vertices)
    pointTree = scipy.spatial.cKDTree(vor.points)

    nearVerticesIndex = []
    
    for i , vori in enumerate(vor.vertices):
        pv = pointTree.query(vor.vertices[i],10)
        vv = vorTree.query(vor.vertices[i],10)
        
        pvmin = pv[0][:].min() *min_voronoi_distance # smallest distance of this vertice to any atom
        # Find all vertices that are closer than this nearest atom.
        nn = vorTree.query_ball_point(vori, pvmin)
        
        # Put these vertices in the list
        if len (nearVerticesIndex) == 0:
            nearVerticesIndex.append(nn)
            #print('ring around vertices: ', nn)
        exist = 0
        for j in range(len(nearVerticesIndex)):
            if i in nearVerticesIndex[j]:
                exist = 1
                if len(nn) > len(nearVerticesIndex[j]):
                    nearVerticesIndex[j] = nn
                    
        if exist == 0:
            nearVerticesIndex.append(nn)

    new_voronoi = []       
    # get new vertices by calculating the center of the too close vertices and just keep the single ones
    for ver_ind in  nearVerticesIndex:
        new_ver_point = np.median(vor.vertices[ver_ind], axis=0) # center of all vertices
        new_voronoi.append(new_ver_point.flatten())
    
    return np.unique(new_voronoi, axis=0)
    
def get_rings(vor, new_voronoi):
    """
     Find rings by doing a vornoi analysis of the new Voronoi vertices
    """
    rings = {'atoms': vor.points,
             'centers': new_voronoi}
    
    pointTree = scipy.spatial.cKDTree(vor.points)
    vor2 = scipy.spatial.Voronoi(new_voronoi)
    vor2Tree   = scipy.spatial.cKDTree(new_voronoi)
    
    ##############################################
    ### Now we find which rings are associated with how many Voronoi vertices
    ##############################################
    cyclicity = []
    polygons = []
    for i, point in enumerate(vor2.points):
        region_index = vor2.point_region[i] 
        region = vor2.regions[region_index]

        #the vertice of the voronoi of a voronoi is the original lattice 
        rings[str(i)] = {'center': point}
        
        vv = vor2Tree.query(new_voronoi[i],10)
        # smallest distance to vertice 
        vertice_min_dist = (vv[0][:])[np.where(vv[0][:]>0)].min() /4 #0 is vertice itself
        
        nearest_neighbour_indices = []
        
        if len(region)>1 and (np.array(region)>-1).all():
            for vertice in region:
                dist, nn = pointTree.query(vor2.vertices[vertice], 2)
                if dist[1]-dist[0] < vertice_min_dist/5: #pathological case ()
                    ddd= scipy.spatial.distance.cdist(vor.points[nn], point[np.newaxis,:])
                    if abs(ddd[1]-ddd[0]) <  vertice_min_dist/20:
                        #add both
                        nearest_neighbour_indices.extend([nn[0]])
                        nearest_neighbour_indices.extend([nn[1]])
                    else:
                        #add closest point
                        if ddd[1] <ddd[0]:
                            nearest_neighbour_indices.extend([nn[1]])
                        else:
                            nearest_neighbour_indices.extend([nn[0]])
                else:
                    nearest_neighbour_indices.extend([nn[0]])
        
            nn = np.array(np.unique(nearest_neighbour_indices), dtype=int)
            center = np.median(vor.points[nn],axis=0)
            rel_positions =  vor.points[nn]-center
            
            angles = np.arctan2(rel_positions[:,0],rel_positions[:,1])
            
            sorted_ring = nn[np.argsort(angles)]
            
            rings[str(i)]['angles'] = angles[np.argsort(angles)]
            rings[str(i)]['positions'] = vor.points[sorted_ring]
            rings[str(i)]['cyclicity'] = len(rings[str(i)]['positions'])
            cyclicity.append(rings[str(i)]['cyclicity'])
            polygons.append(patches.Polygon(vor.points[sorted_ring], closed=True, fill=True, edgecolor='red', linewidth=2))
    rings['cyclicity'] = cyclicity
    rings['polygons'] = polygons
    return rings
    
def get_new_vertices(vor, min_voronoi_distance = 1.2):
    db = sklearn.cluster.DBSCAN(eps=min_voronoi_distance, min_samples=2).fit(vor.vertices)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(f'Found {n_clusters_} clusters and {np.sum(labels==-1)} single vertices')

    centers = list(vor.vertices[labels==-1])
    for k in set(labels):
        if k > -1:
            centers.append(np.mean(vor.vertices[labels==k], axis=0))

    return np.array(centers)