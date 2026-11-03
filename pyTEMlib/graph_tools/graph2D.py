from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import numpy as np
import scipy
import sklearn


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
    Averages voronoi vertces that are too close together
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