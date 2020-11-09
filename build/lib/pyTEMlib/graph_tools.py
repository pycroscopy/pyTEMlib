import numpy as np
import scipy.spatial as spatial

from skimage.measure import points_in_poly


def turning_function(corners, points):
    # sort corners in counter-clockwise direction
    # calculate centroid of the polygon
    corners1 = np.array(points[corners])
    corners2 = np.roll(corners1, 1)
    corners0 = np.roll(corners1, -1)

    v = corners1 - corners0
    an = (np.arctan2(v[:, 0], v[:, 1]) + 2.0 * np.pi) % (2.0 * np.pi) / np.pi * 180
    print(corners1)
    angles = []
    for i in range(len(corners1)):
        a = corners1[i] - corners0[i]
        b = corners1[i] - corners2[i]
        num = np.dot(a, b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        angles.append(np.arccos(num / denom) * 180 / np.pi)

    return angles


def polygon_sort2(corners, points):
    """
    # sort corners in counter-clockwise direction
    input:
            corners are indices in points array
            points is list or array of points
    output:
            corners_with_angles
    """
    # calculate centroid of the polygon
    n = len(corners)  # of corners
    cx = float(sum(x for x, y in points[corners])) / n
    cy = float(sum(y for x, y in points[corners])) / n

    # create a new list of corners which includes angles
    # angles from the positive x axis
    corners_with_angles = []
    for i in corners:
        x, y = points[i]
        an = (np.atan2(y - cy, x - cx) + 2.0 * np.pi) % (2. * np.pi)
        corners_with_angles.append([i, np.degrees(an)])

    # sort it using the angles
    corners_with_angles.sort(key=lambda tup: tup[1])

    return corners_with_angles


def polygons_inner(indices, points):
    pp = np.array(points)[indices, :]
    # Determine inner angle of polygon
    # Generate second array which is shifted by one
    pp2 = np.roll(pp, 1, axis=0)
    # and subtract it from former: this is now a list of vectors
    p_vectors = pp - pp2

    # angles of vectors with respect to positive x-axis
    ang = np.arctan2(p_vectors[:, 1], p_vectors[:, 0]) / np.pi * 180 + 360 % 360
    # shift angles array by one
    ang2 = np.roll(ang, -1, axis=0)

    # difference of angles is outer angle but we want the inner (inner + outer = 180)
    inner_angles = (180 - (ang2 - ang) + 360) % 360

    return inner_angles


# sort corners in counter-clockwise direction
def polygon_sort(corners):
    # calculate centroid of the polygon
    n = len(corners)  # of corners
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n

    # create a new list of corners which includes angles
    corners_with_angles = []
    for x, y in corners:
        an = (np.atan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        corners_with_angles.append((x, y, np.degrees(an)))

    # sort it using the angles
    corners_with_angles.sort(key=lambda tup: tup[2])

    return corners_with_angles


def polygon_area(corners):
    """
    # Area of Polygon using Shoelace formula
    # http://en.wikipedia.org/wiki/Shoelace_formula
    # FB - 20120218
    # corners must be ordered in clockwise or counter-clockwise direction
    """
    n = len(corners)  # of corners
    area = 0.0
    c_x = 0
    c_y = 0
    for i in range(n):
        j = (i + 1) % n
        nn = corners[i][0] * corners[j][1] - corners[j][0] * corners[i][1]
        area += nn
        c_x += (corners[i][0] + corners[j][0]) * nn
        c_y += (corners[i][1] + corners[j][1]) * nn

    area = abs(area) / 2.0

    # centeroid or arithmetic mean
    c_x = c_x / (6 * area)
    c_y = c_y / (6 * area)

    return area, c_x, c_y


def polygon_angles(corners):
    angles = []
    # calculate centroid of the polygon
    n = len(corners)  # of corners
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    # create a new list of angles
    # print (cx, cy)
    for x, y in corners:
        an = (np.atan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        angles.append((np.degrees(an)))

    return angles


def voronoi_tags(vor):
    sym = {'voronoi': vor, 'vertices': vor.vertices, 'ridge_points': vor.ridge_points,
           'ridge_vertices': vor.ridge_vertices, 'regions': vor.regions, 'point_region': vor.point_region}
    # Indices of the points between which each Voronoi ridge lies.
    # Indices of the Voronoi vertices forming each Voronoi ridge.
    # Indices of the Voronoi vertices forming each Voronoi region. -1 indicates vertex outside the Voronoi diagram.
    # Index of the Voronoi region for each input point. If qhull option “Qc” was not specified,
    # the list will contain -1 for points that are not associated with a Voronoi region.

    points = vor.points
    nn_tree = (points)

    rim = []
    regions = []

    # ##
    # We get all the vertice length

    lengths = []
    for vertice in vor.ridge_vertices:
        if not (-1 in vertice):
            p1 = vor.vertices[vertice[0]]
            p2 = vor.vertices[vertice[1]]
            lengths.append(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

    sym['lengths'] = lengths
    sym['median lengths'] = np.median(lengths)
    sym['Min Voronoi Edge'] = np.median(lengths) / 1.5
    # print ('median lengths', np.median(lengths))
    # print ('Min Voronoi Edge',np.median(lengths)/1.5)
    corners_hist = []
    nn_hist = []
    nn_dist_hist = []
    angle_hist = []
    area_hist = []
    deviation_hist = []

    for i, region in enumerate(vor.point_region):
        x, y = points[i]
        sym[str(i)] = {}
        vertices = vor.regions[region]

        # #
        # We get all the rim atoms
        # ##

        # if all(v >= 0  and all(vor.vertices[v] >0) and all(vor.vertices[v]<tags['data'].shape[0]) for v in vertices):
        if all(v >= 0 and all(vor.vertices[v] > 0) for v in vertices):
            # finite regions only now
            # negative and too large vertices (corners) are excluded

            regions.append(vertices)
            poly = []
            for v in vertices:
                poly.append(vor.vertices[v])

            area, cx, cy = polygon_area(poly)
            cx = abs(cx)
            cy = abs(cy)

            angles = polygon_angles(poly)
            angle_hist.append(angles)
            area_hist.append(area)
            deviation_hist.append(np.sqrt((cx - x) ** 2 + (cy - y) ** 2))

            sym[str(i)]['xy'] = [x, y]
            sym[str(i)]['geometric'] = [cx, cy]
            sym[str(i)]['area'] = area

            sym[str(i)]['angles'] = angles
            sym[str(i)]['off center'] = [cx - x, cy - y]

            sym[str(i)]['position'] = 'inside'
            sym[str(i)]['corner'] = vertices
            sym[str(i)]['vertices'] = poly
            sym[str(i)]['corners'] = len(vertices)
            corners_hist.append(len(vertices))
            nn = 0
            nn_vor = []
            length = []
            for j in range(len(vertices)):
                k = (j + 1) % len(vertices)
                p1 = vor.vertices[vertices[j]]
                p2 = vor.vertices[vertices[k]]
                leng = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                length.append(leng)
                sym[str(i)]['length'] = length
                if leng > sym['Min Voronoi Edge']:
                    nn += 1
                    nn_vor.append(vertices[j])
                sym[str(i)]['length'] = length
            nn_p = nn_tree.query(points[i], k=nn + 1)
            sym[str(i)]['neighbors'] = []
            sym[str(i)]['nn Distance'] = []
            sym[str(i)]['nn'] = nn
            if nn > 0:
                nn_hist.append(nn)
                for j in range(1, len(nn_p[0])):
                    sym[str(i)]['nn Distance'].append(nn_p[0][j])
                    sym[str(i)]['neighbors'].append(nn_p[1][j])
                    nn_dist_hist.append(nn_p[0][j])
            else:
                rim.append(i)
                sym[str(i)]['position'] = 'rim'
                sym[str(i)]['corners'] = 0
                print('weird nn determination', i)

        else:
            rim.append(i)
            sym[str(i)]['position'] = 'rim'
            sym[str(i)]['corners'] = 0
            sym[str(i)]['xy'] = [x, y]

    sym['average corners'] = np.median(corners_hist)
    sym['average area'] = np.median(area_hist)
    sym['num atoms at rim'] = len(rim)
    sym['num voronoi'] = len(points) - len(rim)
    sym['Median Coordination'] = np.median(nn_hist)
    sym['Median NN Distance'] = np.median(nn_dist_hist)

    sym['_hist corners'] = corners_hist
    sym['_hist area'] = area_hist
    sym['atoms at rim'] = rim
    sym['_hist Coordination'] = nn_hist
    sym['_hist NN Distance'] = nn_dist_hist
    sym['_hist deviation'] = deviation_hist

    return sym
    # print ('average corners', np.median(corners_hist))


def define_symmetry(tags):
    # make dictionary to store
    if 'symmetry' in tags:
        tags['symmetry'].clear()

    tags['symmetry'] = {}
    sym = tags['symmetry']
    if 'latticeType' in tags:
        lattice_types = ['None', 'Find Lattice', 'hexagonal', 'honeycomb', 'square', 'square centered',
                         'diamond', 'fcc']
        sym['lattice'] = lattice_types[tags['latticeType']]

    sym['number of atoms'] = len(tags['atoms'])

    points = []
    for i in range(sym['number of atoms']):
        sym[str(i)] = {}
        sym[str(i)]['index'] = i
        sym[str(i)]['x'] = tags['atoms'][i][0]
        sym[str(i)]['y'] = tags['atoms'][i][1]
        sym[str(i)]['intensity'] = tags['atoms'][i][3]
        sym[str(i)]['maximum'] = tags['atoms'][i][4]
        sym[str(i)]['position'] = 'inside'
        sym[str(i)]['Z'] = 0
        sym[str(i)]['Name'] = 'undefined'
        sym[str(i)]['Column'] = -1

        points.append([int(sym[str(i)]['x'] + 0.5), int(sym[str(i)]['y'] + 0.5)])

    # self.points = points.copy()


def voronoi(atoms, tags):
    im = tags['image']
    vor = spatial.Voronoi(np.array(atoms)[:, 0:2])  # Plot it:
    rim_vertices = []
    for i in range(len(vor.vertices)):

        if (vor.vertices[i, 0:2] < 0).any() or (vor.vertices[i, 0:2] > im.shape[0] - 5).any():
            rim_vertices.append(i)
    rim_vertices = set(rim_vertices)
    mid_vertices = list(set(np.arange(len(vor.vertices))).difference(rim_vertices))

    mid_regions = []
    for region in vor.regions:  # Check all Voronoi polygons
        # we get a lot of rim (-1) and empty and  regions
        if all(x in mid_vertices for x in region) and len(region) > 1:
            mid_regions.append(region)
    tags['atoms']['voronoi'] = vor
    tags['atoms']['voronoi_vertices'] = vor.vertices
    tags['atoms']['voronoi_regions'] = vor.regions
    tags['atoms']['voronoi_midVerticesIndices'] = mid_vertices
    tags['atoms']['voronoi_midVertices'] = vor.vertices[mid_vertices]
    tags['atoms']['voronoi_midRegions'] = mid_regions


def voronoi2(tags, atoms):
    sym = tags['symmetry']
    points = []

    for i in range(sym['number of atoms']):
        points.append([int(sym[str(i)]['x'] + 0.5), int(sym[str(i)]['y'] + 0.5)])

    # points = np.array(atoms[:][0:2])
    vor = spatial.Voronoi(points)

    sym['voronoi'] = vor

    nn_tree = spatial.cKDTree(points)

    rim = []
    regions = []

    # ##
    # We get all the vertice length

    lengths = []
    for vertice in vor.ridge_vertices:
        if all(v >= 0 for v in vertice):
            p1 = vor.vertices[vertice[0]]
            p2 = vor.vertices[vertice[1]]
            lengths.append(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

    sym['lengths'] = lengths
    sym['median lengths'] = np.median(lengths)
    sym['Min Voronoi Edge'] = np.median(lengths) / 1.5
    # print ('median lengths', np.median(lengths))
    # print ('Min Voronoi Edge',np.median(lengths)/1.5)
    corners_hist = []
    nn_hist = []
    nn_dist_hist = []
    angle_hist = []
    area_hist = []
    deviation_hist = []

    for i, region in enumerate(vor.point_region):
        x, y = points[i]

        vertices = vor.regions[region]

        # ##
        # We get all the rim atoms
        # ##

        if all(v >= 0 and all(vor.vertices[v] > 0) and all(vor.vertices[v] < tags['data'].shape[0]) for v in vertices):
            # finite regions only now
            # negative and too large vertices (corners) are excluded

            regions.append(vertices)
            poly = []
            for v in vertices:
                poly.append(vor.vertices[v])

            area, cx, cy = polygon_area(poly)
            cx = abs(cx)
            cy = abs(cy)

            angles = polygon_angles(poly)
            angle_hist.append(angles)
            area_hist.append(area)
            deviation_hist.append(np.sqrt((cx - x) ** 2 + (cy - y) ** 2))

            sym[str(i)]['xy'] = [x, y]
            sym[str(i)]['geometric'] = [cx, cy]
            sym[str(i)]['area'] = area

            sym[str(i)]['angles'] = angles
            sym[str(i)]['off center'] = [cx - x, cy - y]

            sym[str(i)]['position'] = 'inside'
            sym[str(i)]['corner'] = vertices
            sym[str(i)]['vertices'] = poly
            sym[str(i)]['corners'] = len(vertices)
            corners_hist.append(len(vertices))
            nn = 0
            nn_vor = []
            length = []
            for j in range(len(vertices)):
                k = (j + 1) % len(vertices)
                p1 = vor.vertices[vertices[j]]
                p2 = vor.vertices[vertices[k]]
                leng = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                length.append(leng)
                sym[str(i)]['length'] = length
                if leng > sym['Min Voronoi Edge']:
                    nn += 1
                    nn_vor.append(vertices[j])
                sym[str(i)]['length'] = length
            nn_p = nn_tree.query(points[i], k=nn + 1)
            sym[str(i)]['neighbors'] = []
            sym[str(i)]['nn Distance'] = []
            sym[str(i)]['nn'] = nn
            if nn > 0:
                nn_hist.append(nn)
                for j in range(1, len(nn_p[0])):
                    sym[str(i)]['nn Distance'].append(nn_p[0][j])
                    sym[str(i)]['neighbors'].append(nn_p[1][j])
                    nn_dist_hist.append(nn_p[0][j])
            else:
                rim.append(i)
                sym[str(i)]['position'] = 'rim'
                sym[str(i)]['corners'] = 0
                print('weird nn determination', i)
        else:
            rim.append(i)
            sym[str(i)]['position'] = 'rim'
            sym[str(i)]['corners'] = 0
            sym[str(i)]['xy'] = [x, y]

    sym['average corners'] = np.median(corners_hist)
    sym['average area'] = np.median(area_hist)
    sym['num atoms at rim'] = len(rim)
    sym['num voronoi'] = len(points) - len(rim)
    sym['Median Coordination'] = np.median(nn_hist)
    sym['Median NN Distance'] = np.median(nn_dist_hist)

    sym['_hist corners'] = corners_hist
    sym['_hist area'] = area_hist
    sym['atoms at rim'] = rim
    sym['_hist Coordination'] = nn_hist
    sym['_hist NN Distance'] = nn_dist_hist
    sym['_hist deviation'] = deviation_hist

    # print ('average corners', np.median(corners_hist))
