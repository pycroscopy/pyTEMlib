#!/usr/bin/env python

"""
From gb_code at https://github.com/oekosheri/GB_code
This module is a collection of functions that produce CSL properties.
When run from the terminal, the code runs in two modes.

 First mode:
  'csl_generator.py u v w [limit]' ----->  Where the u v w are the
  indices of the rotation axis such as 1 0 0, 1 1 1, 1 1 0 and so on. The limit
  is the maximum sigma of interest.
  (the limit by default: 100)

 Second mode:
 'csl_generator.py u v w basis sigma [limit]' -----> Where basis is
  either fcc, bcc, diamond or sc. You read the sigma of interest from the first
  mode run. The limit here refers to CSL GB inclinations. The bigger the limit,
  the higher the indices of CSL planes.
  (the limit by default: 2)

  your chosen axis, basis and sigma will be written to an io_file which will
  later be read by gb_generator.py.
"""

import numpy as np
import ase
import ase.lattice


def get_cubic_sigma(rotation_axis, m, n=1):
    """
    Grimmer Acta Cryst 1984 A40 p108-112
    https://doi-org.utk.idm.oclc.org/10.1107/S0108767384000246
    formula 2.
    originally by Ranganathan Acta Cryst. (1966). 21, 197-199
    https://doi.org/10.1107/S0365110X66002615

    Parameters
    ----------
    rotation_axis: list or numpy array of shape 3
        the rotation axis
    m, n: integer
        two integers that define grain boundary (n by default 1)
    """
    
    square_sum = np.inner(rotation_axis, rotation_axis)
    sigma = m*m + n*n * square_sum
    while sigma != 0 and sigma % 2 == 0:
        sigma /= 2
    return sigma if sigma > 1 else None


def get_cubic_theta(rotation_axis, m, n=1):
    """
    Grimmer Acta Cryst 1984 A40 p108-112
    https://doi-org.utk.idm.oclc.org/10.1107/S0108767384000246
    formula 1.
    originally by Ranganathan Acta Cryst. (1966). 21, 197-199
    https://doi.org/10.1107/S0365110X66002615

    Parameters
    ----------
    rotation_axis: list or numpy array of shape 3
        the rotation axis
    m,n:  integers
        two integers (n by default 1)
    """
    if m > 0:
        return 2 * np.arctan(np.linalg.norm(rotation_axis) * n / m)
    else:
        return np.pi


def get_theta_m_n_list(rotation_axis, sigma):
    """
    Finds integers m and n lists that match the input sigma.
    """
    if sigma == 1:
        return [(0., 0., 0.)]
    thetas = []
    max_m = int(np.ceil(np.sqrt(4*sigma)))

    for m in range(1, max_m):
        for n in range(1, max_m):
            if np.gcd(m, n) == 1:
                s = get_cubic_sigma(rotation_axis, m, n)
                if s == sigma:
                    theta = (get_cubic_theta(rotation_axis, m, n))
                    thetas.append((theta, m, n))
                    thetas.sort(key=lambda x: x[0])
    return np.array(thetas)


def print_list(rotation_axis, limit):
    """
    prints a list of smallest sigmas/angles for a given axis(rotation_matrix).
    """
    for i in range(limit):
        tt = get_theta_m_n_list(rotation_axis, i)
        if len(tt) > 0:
            theta, _, _ = tt[0]
            print(f"sigma:   {i:3d}  theta:  {np.degrees(theta):5.2f} ")
                  

def get_rotation_matrix(rotation_axis, theta):
    """
    produces a rotation matrix 
    
    according to 
    Grimmer Acta Cryst 1984 A40 p108-112
    https://doi-org.utk.idm.oclc.org/10.1107/S0108767384000246
    formula 10.

    Parameters
    ----------
    rotation_axis: list or numpy array of shape 3
        the rotation axis
    theta: float
        rotation angle in radians

    Returns 
    -------
    rotation_matrix: np.array of shape 3x3 
        rotation matrix
    """
    c = np.cos(theta)
    s = np.sin(theta)
    p1, p2, p3 = rotation_axis / np.linalg.norm(rotation_axis)
    rotation_matrix = np.array([[p1**2*(1-c)+c, p1*p2*(1-c)-p3*s, p1*p3*(1-c)+p2*s],
                                [p1*p2*(1-c)+p3*s, p2**2*(1-c)+c, p2*p3*(1-c)-p1*s],
                                [p1*p3*(1-c)-p2*s, p2*p3*(1-c)+p1*s, p3**2*(1-c)+c]])
    return rotation_matrix


# Helpful Functions:
# -------------------#
def integer_array(a, tol=1e-7):
    """
    returns True if an array is integer.
    """
    return np.all(abs(np.round(a) - a) < tol)


def angv(a, b):
    """
    returns the angle between two vectors.
    """
    angle = np.acos(np.round(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b), 8))
    return round(np.degrees(angle), 7)


def ang(a, b):
    """
    returns the cos(angle) between two vectors.
    """
    angle = np.round(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b), 7)
    return abs(angle)


def common_divisor(a):
    """
    returns the common factor of vector a and the reduced vector.
    """
    common_factor = []
    a = np.array(a)
    for i in range(2, 100):
        while a[0] % i == 0 and a[1] % i == 0 and a[2] % i == 0:
            a = a / i
            common_factor.append(i)
    return a.astype(int), (abs(np.prod(common_factor)))


def smallest_integer(a):
    """
    returns the smallest multiple integer to make an integer array.
    """
    a = np.array(a)
    for i in range(1, 200):
        test_v = i * a
        if integer_array(test_v):
            break
    return (test_v, i) if integer_array(test_v) else None


def integer_matrix(a):
    """
    returns an integer matrix from row vectors.
    """
    found = True
    b = np.zeros((3, 3))
    a = np.array(a)
    for i in range(3):
        for j in range(1, 2000):
            test_v = j * a[i]
            if integer_array(test_v):
                b[i] = test_v
                break
        if all(b[i] == 0):
            found = False
            print("Can not make integer matrix!")
    return b if found else None


def symmetry_equivalent(arr):
    """
    returns cubic symmetric equivalents of the given 2 dimensional vector.
    """
    sym = np.zeros([24, 3, 3])
    sym[0, :] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    sym[1, :] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    sym[2, :] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    sym[3, :] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    sym[4, :] = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
    sym[5, :] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    sym[6, :] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    sym[7, :] = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
    sym[8, :] = [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]
    sym[9, :] = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    sym[10, :] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    sym[11, :] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    sym[12, :] = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    sym[13, :] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
    sym[14, :] = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
    sym[15, :] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    sym[16, :] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    sym[17, :] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    sym[18, :] = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
    sym[19, :] = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
    sym[20, :] = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
    sym[21, :] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    sym[22, :] = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
    sym[23, :] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]

    arr = np.atleast_2d(arr)
    result = []
    for i in range(len(sym)):
        for j in range(len(arr)):
            result.append(np.dot(sym[i, :], arr[j]))
    result = np.array(result)
    return np.unique(result, axis=0)


def tilt_twist_components(v1, rotation_matrix, m, n):
    """
    returns the tilt and twist components of a given GB plane.
    arguments:
    v1 -- given gb plane
    rotation_matrix -- axis of rotation
    m,n -- the two necessary integers
    """
    theta = get_cubic_theta(rotation_matrix, m, n)
    rotation_matrix = get_rotation_matrix(rotation_matrix, theta)
    v2 = np.round(np.dot(rotation_matrix, v1), 6).astype(int)
    tilt = angv(v1, v2)
    if abs(tilt - np.degrees(theta)) < 10e-5:
        print("Pure tilt boundary with a tilt component: {0:6.2f}"
              .format(tilt))
    else:
        twist = 2 * np.arccos(np.cos(theta / 2) / np.cos(np.radians(tilt / 2)))
        print("Tilt component: {0:<6.2f} Twist component: {1:6.2f}"
              .format(tilt, twist))


def possible_grain_boundary_plane_list(rotation_matrix, m=5, n=1, lim=5):
    """
    generates GB planes and specifies the character.

    arguments:
    rotation_matrix -- axis of rotation.
    m,n -- the two necessary integers
    lim -- upper limit for the plane indices

    """
    rotation_matrix = np.array(rotation_matrix)
    theta = get_cubic_theta(rotation_matrix, m, n)
    sigma = get_cubic_sigma(rotation_matrix, m, n)
    rotation_matrix_2 = get_rotation_matrix(rotation_matrix, theta)

    # List and character of possible GB planes:
    x = np.arange(-lim, lim + 1, 1)
    y = x
    z = x
    indices = np.stack(np.meshgrid(x, y, z)).T.reshape(len(x) ** 3, 3)
    indices_0 = indices[np.where(np.sum(abs(indices), axis=1) != 0)]
    indices_0 = indices_0[np.argsort(np.linalg.norm(indices_0, axis=1))]

    # extract the minimal cell:
    min_1, min_2 = create_minimal_cell_method_1(sigma, rotation_matrix, rotation_matrix_2)
    v_1 = np.zeros([len(indices_0), 3])
    v_2 = np.zeros([len(indices_0), 3])
    grain_boundary_type = []
    tol = 0.001
    # Mirror planes cubic symmetry
    mp = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [1, 1, 0],
                   [0, 1, 1],
                   [1, 0, 1],
                   ], dtype='float')
    # Find GB plane coordinates:
    for i in range(len(indices_0)):
        if common_divisor(indices_0[i])[1] <= 1:
            v_1[i, :] = (indices_0[i, 0] * min_1[:, 0] +
                         indices_0[i, 1] * min_1[:, 1] +
                         indices_0[i, 2] * min_1[:, 2])
            v_2[i, :] = (indices_0[i, 0] * min_2[:, 0] +
                         indices_0[i, 1] * min_2[:, 1] +
                         indices_0[i, 2] * min_2[:, 2])

    v_1 = (v_1[~np.all(v_1 == 0, axis=1)]).astype(int)
    v_2 = (v_2[~np.all(v_2 == 0, axis=1)]).astype(int)
    mean_planes = (v_1 + v_2) / 2

    # Check the type of GB plane: Symmetric tilt, tilt, twist
    for i in range(len(v_1)):
        if ang(v_1[i], rotation_matrix) < tol:

            for j in range(len(symmetry_equivalent(mp))):
                if 1 - ang(mean_planes[i], symmetry_equivalent(mp)[j]) < tol:
                    grain_boundary_type.append('Symmetric Tilt')
                    break
            else:
                grain_boundary_type.append('Tilt')
        elif 1 - ang(v_1[i], rotation_matrix) < tol:
            grain_boundary_type.append('Twist')
        else:
            grain_boundary_type.append('Mixed')

    return v_1, v_2, mean_planes, grain_boundary_type


def create_minimal_cell_method_1(sigma, rotation_axis, rotation_matrix):
    """
    finds Minimal cell by means of a numerical search.
    (An alternative analytical method can be used too).
    arguments:
    sigma -- gb sigma
    rotation_matrix -- rotation axis
    rotation_matrix -- rotation matrix
    """
    rotation_axis = np.array(rotation_axis)
    minimal_cell_1 = np.zeros([3, 3])
    minimal_cell_1[:, 2] = rotation_axis

    lim = 20
    x = np.arange(-lim, lim + 1, 1)
    y = x
    z = x
    indices = np.stack(np.meshgrid(x, y, z)).T.reshape(len(x) ** 3, 3)

    # remove 0 vectors and rotation_axis from the list
    indices_0 = indices[np.where(np.sum(abs(indices), axis=1) != 0)]
    condition1 = ((abs(np.dot(indices_0, rotation_axis) / np.linalg.norm(indices_0, axis=1) /
                       np.linalg.norm(rotation_axis))).round(7))
    indices_0 = indices_0[np.where(condition1 != 1)]

    if minimal_cell_search(indices_0, minimal_cell_1, rotation_matrix, sigma):

        m1, m2 = minimal_cell_search(indices_0, minimal_cell_1, rotation_matrix, sigma)
        return m1, m2
    else:
        return None


def minimal_cell_search(indices, minimal_cell_1, rotation_matrix, sigma):

    tol = 0.001
    # norm1 = norm(indices, axis=1)
    new_indices = np.dot(rotation_matrix, indices.T).T
    nn = indices[np.all(abs(np.round(new_indices) - new_indices) < 1e-6, axis=1)]
    test_vecs = nn[np.argsort(np.linalg.norm(nn, axis=1))]
    # print(len(indices), len(test_vecs),test_vecs[:20])

    found = False
    count = 0
    while (not found) and count < len(test_vecs) - 1:
        if 1 - ang(test_vecs[count], minimal_cell_1[:, 2]) > tol:
            # and  (ang(test_vecs[i],rotation_matrix) > tol):
            minimal_cell_1[:, 1] = (test_vecs[count])
            count += 1
            for j in range(len(test_vecs)):
                if (1 - ang(test_vecs[j], minimal_cell_1[:, 2]) > tol) and (
                        1 - ang(test_vecs[j], minimal_cell_1[:, 1]) > tol):
                    if (ang(test_vecs[j],
                            np.cross(minimal_cell_1[:, 2], minimal_cell_1[:, 1])) > tol):
                        #  The condition that the third vector can not be
                        #  normal to any other two.
                        #  and (ang(test_vecs[i],rotation_matrix)> tol) and
                        # (ang(test_vecs[i],MiniCell[:,1])> tol)):
                        minimal_cell_1[:, 0] = (test_vecs[j]).astype(int)
                        det1 = abs(round(np.linalg.det(minimal_cell_1), 5))
                        minimal_cell_1 = minimal_cell_1.astype(int)
                        minimal_cell_2 = ((np.round(np.dot(rotation_matrix, minimal_cell_1), 7)).astype(int))
                        det2 = abs(round(np.linalg.det(minimal_cell_2), 5))

                        if ((abs(det1 - sigma)) < tol and
                                (abs(det2 - sigma)) < tol):
                            found = True
                            break
    if found:
        return minimal_cell_1, minimal_cell_2
    else:
        return found


def get_basis(basis):
    """
    defines the basis.
    """
    # Cubic basis
    if str(basis) == 'fcc':
        basis = np.array([[0, 0, 0],
                          [0.5, 0, 0.5],
                          [0.5, 0.5, 0],
                          [0, 0.5, 0.5]], dtype=float)
    elif str(basis) == 'bcc':
        basis = np.array([[0, 0, 0],
                          [0.5, 0.5, 0.5]], dtype=float)
    elif str(basis) == 'sc':
        basis = np.eye(3)

    elif str(basis) == 'diamond':
        basis = np.array([[0, 0, 0],
                          [0.5, 0, 0.5],
                          [0.5, 0.5, 0],
                          [0, 0.5, 0.5],
                          [0.25, 0.25, 0.25],
                          [0.75, 0.25, 0.75],
                          [0.75, 0.75, 0.25],
                          [0.25, 0.75, 0.75]], dtype=float)
    else:
        print('Sorry! For now only works for cubic lattices ...')

    return basis


def find_orthogonal_cell(basis, rotation_matrix, m, n, gb1):
    """
    finds Orthogonal cells from the CSL minimal cells.
    arguments:
    basis -- lattice basis
    rotation_matrix -- rotation axis
    m,n -- two necessary integers
    gb1 -- input plane orientation
    """
    # Changeable limit
    lim = 15
    rotation_matrix = np.array(rotation_matrix)
    theta = get_cubic_theta(rotation_matrix, m, n)
    sigma = get_cubic_sigma(rotation_matrix, m, n)
    rotation_matrix = get_rotation_matrix(rotation_matrix, theta)
    gb_2 = np.round((np.dot(rotation_matrix, gb1)), 6)
    x = np.arange(-lim, lim + 1, 1)
    y = x
    z = x
    indices = np.stack(np.meshgrid(x, y, z)).T.reshape(len(x) ** 3, 3)
    indices_0 = indices[np.where(np.sum(abs(indices), axis=1) != 0)]
    indices_0 = indices_0[np.argsort(np.linalg.norm(indices_0, axis=1))]
    ortho_cell_1 = np.zeros([3, 3])
    ortho_cell_1[:, 0] = np.array(gb1)
    ortho_cell_2 = np.zeros([3, 3])
    ortho_cell_2[:, 0] = np.array(gb_2)
    # extract the minimal cells:
    min_1, min_2 = create_minimal_cell_method_1(sigma, rotation_matrix, rotation_matrix)
    # Find Ortho vectors:
    tol = 0.001
    if ang(ortho_cell_1[:, 0], rotation_matrix) < tol:
        ortho_cell_1[:, 1] = rotation_matrix
        ortho_cell_2[:, 1] = rotation_matrix
    else:

        for i in range(len(indices_0)):

            v1 = (indices_0[i, 0] * min_1[:, 0] +
                  indices_0[i, 1] * min_1[:, 1] +
                  indices_0[i, 2] * min_1[:, 2])
            v2 = (indices_0[i, 0] * min_2[:, 0] +
                  indices_0[i, 1] * min_2[:, 1] +
                  indices_0[i, 2] * min_2[:, 2])
            if ang(v1, ortho_cell_1[:, 0]) < tol:
                ortho_cell_1[:, 1] = v1
                ortho_cell_2[:, 1] = v2
                break
    ortho_cell_1[:, 2] = np.cross(ortho_cell_1[:, 0], ortho_cell_1[:, 1])
    ortho_cell_2[:, 2] = np.cross(ortho_cell_2[:, 0], ortho_cell_2[:, 1])

    if (common_divisor(ortho_cell_1[:, 2])[1] ==
            common_divisor(ortho_cell_2[:, 2])[1]):
        ortho_cell_1[:, 2] = common_divisor(ortho_cell_1[:, 2])[0]
        ortho_cell_2[:, 2] = common_divisor(ortho_cell_2[:, 2])[0]
    # ortho_cell_1 = ortho_cell_1
    # ortho_cell_2 = ortho_cell_2
    # Test
    # OrthoCell_3 = (dot(rotation_matrix, ortho_cell_1))
    volume_1 = (round(np.linalg.det(ortho_cell_1), 5))
    volume_2 = (round(np.linalg.det(ortho_cell_2), 5))
    num = volume_1 * len(get_basis(basis)) * 2

    if volume_1 == volume_2:
        ortho_cell_1 = ortho_cell_1.astype(float)
        ortho_cell_2 = ortho_cell_2.astype(float)

        if basis == 'sc' or basis == 'diamond':

            return ((ortho_cell_1.astype(float),
                     ortho_cell_2.astype(float), num.astype(int)))

        elif basis == 'fcc' or basis == 'bcc':
            ortho1, ortho2 = ortho_fcc_bcc(basis, ortho_cell_1, ortho_cell_2)
            volume_1 = (round(np.linalg.det(ortho1), 5))
            num = volume_1 * len(get_basis(basis)) * 2
            return ortho1, ortho2, num.astype(int)
    else:
        return None


def print_list_grain_boundary_planes(rotation_matrix, basis, m, n, lim=3):
    """
    prints lists of GB planes given an axis, basis, m and n.
    """
    rotation_matrix = np.array(rotation_matrix)
    v_1, v_2, _, type1 = possible_grain_boundary_plane_list(rotation_matrix, m, n, lim)
    for i in range(len(v_1)):
        ortho = find_orthogonal_cell(basis, rotation_matrix, m, n, v_1[i])
        if ortho:
            print("{0:<20s}   {1:<20s}   {2:<20s}   {3:<10s}"
                  .format(str(v_1[i]), str(v_2[i]), type1[i], str(ortho[2])))


# ___CSL/DSC vector construction___#

# According to Grimmer et al. (Acta Cryst. (1974). A30, 197-207) ,
#  DSC and CSL lattices for bcc and fcc were made from the Sc lattice
# via body_centering and face_centering:
def odd_even(m1):
    """
    finds odd and even elements of a matrix.
    """
    d_e = np.array([['a', 'a', 'a'],
                    ['a', 'a', 'a'],
                    ['a', 'a', 'a']], dtype=str)
    for i in range(3):
        for j in range(3):
            if abs(m1[i][j]) % 2 == 0:
                d_e[i][j] = 'e'
            else:
                d_e[i][j] = 'd'
    return d_e


def self_test_b(a):
    z_b = np.eye(3, 3)
    m = a.copy()
    for i in range(3):
        if np.all(odd_even(m)[:, i] == ['d', 'd', 'd']):
            z_b[i][i] = 0.5
            break
    return z_b


def binary_test_b(a):
    count = 0
    z_b = np.eye(3, 3)
    for i in [0, 1]:
        for j in [1, 2]:
            if i != j and count < 1:
                m = a.copy()
                m[:, j] = m[:, i] + m[:, j]
                if np.all(odd_even(m)[:, j] == ['d', 'd', 'd']):
                    count = count + 1
                    z_b[i][j] = 0.5
                    z_b[j][j] = 0.5
    return z_b


def tertiary_test_b(a):
    z_b = np.eye(3, 3)
    m = a.copy()
    m[:, 2] = m[:, 0] + m[:, 1] + m[:, 2]
    for k in range(3):
        if np.all(odd_even(m)[:, k] == ['d', 'd', 'd']):
            z_b[0][k] = 0.5
            z_b[1][k] = 0.5
            z_b[2][k] = 0.5
            break
    return z_b


def body_centering(b):
    """
    converting a single crystal minimal cell to a bcc one.
    """
    z_b = np.eye(3, 3)
    while np.linalg.det(z_b) != 0.5:
        if np.linalg.det(self_test_b(b)) == 0.5:
            z_b = self_test_b(b)
            break
        if np.linalg.det(binary_test_b(b)) == 0.5:
            z_b = binary_test_b(b)
            break
        if np.linalg.det(tertiary_test_b(b)) == 0.5:
            z_b = tertiary_test_b(b)
            break
    return z_b


def face_centering(a):
    """
    converting a single crystal minimal cell to an fcc one.
    """
    z_f = np.eye(3, 3)
    m = a.copy()
    count = 0
    for i in range(3):
        if (np.all(odd_even(m)[:, i] == ['d', 'd', 'e']) or
                np.all(odd_even(m)[:, i] == ['e', 'd', 'd']) or
                np.all(odd_even(m)[:, i] == ['d', 'e', 'd'])):
            count = count + 1
            z_f[i][i] = 0.5
    if np.linalg.det(z_f) == 0.25:
        return z_f
    else:
        for i in [0, 1]:
            for j in [1, 2]:
                if i != j and count < 2:
                    m = a.copy()
                    m[:, j] = m[:, i] + m[:, j]
                    if (np.all(odd_even(m)[:, j] == ['d', 'd', 'e']) or
                            np.all(odd_even(m)[:, j] == ['e', 'd', 'd']) or
                            np.all(odd_even(m)[:, j] == ['d', 'e', 'd'])):
                        count = count + 1
                        z_f[i][j] = 0.5
                        z_f[j][j] = 0.5

    return z_f if np.linalg.det(z_f) == 0.25 else None


def dsc_vec(basis, sigma, mini_cell):
    """
    a discrete shift complete (DSC)
    network for the given sigma and minimal cell.
    arguments:
    basis -- a lattice basis(fcc or bcc)
    sigma -- gb sigma
    mini_cell -- gb minimal cell
    """
    d_sc = np.round(sigma * np.linalg.inv(mini_cell).T, 6).astype(int)
    if basis == 'sc':
        d = d_sc.copy()
    elif basis == 'bcc':
        d = np.dot(d_sc, body_centering(d_sc))
    elif basis == 'fcc' or basis == 'diamond':
        d = np.dot(d_sc, face_centering(d_sc))
    else:
        d = None
    return d


def csl_vec(basis, mini_cell):
    """
    CSL minimal cell for sc, fcc and bcc.
    arguments:
    basis -- a lattice basis(fcc or bcc)
    mini_cell -- gb minimal cell
    """
    c_sc = mini_cell.copy()
    if basis == 'sc':
        c = c_sc.copy()
    elif basis == 'bcc':
        c = np.dot(c_sc, body_centering(c_sc))
    elif basis == 'fcc':
        c = np.dot(c_sc, face_centering(c_sc))
    else:
        c = None
    return c


def dsc_on_plane(d, p_normal):
    """
    projects the given DSC network on a given plane.
    """
    d_proj = np.zeros((3, 3))
    p_normal = np.array(p_normal)
    for i in range(3):
        d_proj[:, i] = (d[:, i] - (np.dot(d[:, i], p_normal) / np.linalg.norm(p_normal)) *
                        p_normal/np.linalg.norm(p_normal))
    return d_proj


def csl_density(basis, mini_cell, plane):
    """
    returns the CSL density of a given plane and its d_spacing.
    """
    plane = np.array(plane)
    c = csl_vec(basis, mini_cell)
    h = np.dot(c.T, plane)
    h = smallest_integer(h)[0]
    h = common_divisor(h)[0]
    g = np.linalg.inv(np.dot(c.T, c))
    h_norm = np.sqrt(np.dot(h.T, np.dot(g, h)))
    density = 1/(h_norm * np.linalg.det(c))
    return abs(density), 1 / h_norm

# An auxiliary function to help reduce the size of the small orthogonal cell
# that is decided otherwise based on Sc for fcc and bcc


def ortho_fcc_bcc(basis, o1, o2):
    ortho1 = np.zeros((3, 3))
    ortho2 = np.zeros((3, 3))
    if basis == 'fcc':
        base = np.delete(get_basis('fcc').T, 0, 1)
    elif basis == 'bcc':
        base = ((np.array([[0.5, 0.5, 0.5], [0.5, 0.5, -0.5],
                           [-0.5, 0.5, 0.5]])).T)
    else:
        base = None
    for i in range(3):
        min_d = min(common_divisor(np.dot(o1[:, i], np.linalg.inv(base)))[1],
                    common_divisor(np.dot(o2[:, i], np.linalg.inv(base)))[1])
        ortho1[:, i] = o1[:, i] / min_d
        ortho2[:, i] = o2[:, i] / min_d
    return ortho1, ortho2
# Writing to a yaml file that will be read by gb_generator


def write_to_io(axis, m, n, basis):
    """
    an input file for gb_generator.py that can be customized.
    It also contains the 
    output from the usage of csl_generator.py.
    """

    my_dict = {'GB_plane': str([axis[0], axis[1], axis[2]]),
               'lattice_parameter': '4',
               'overlap_distance': '0.0', 'which_g': 'g1',
               'rigid_trans': 'no', 'a': '10', 'b': '5',
               'dimensions': '[1,1,1]',
               'File_type': 'LAMMPS'}

    with open('io_file', 'w') as f:
        f.write('### input parameters for gb_generator.py ### \n')
        f.write('# CSL plane of interest that you read from the output of '
                'csl_generator as gb1 \n')
        f.write(list(my_dict.keys())[0] + ': ' + list(my_dict.values())[0] +
                '\n\n')
        f.write('# lattice parameter in Angstrom \n')
        f.write(list(my_dict.keys())[1] + ': ' + list(my_dict.values())[1] +
                '\n\n')
        f.write('# atoms that are closer than this fraction of the lattice '
                'parameter will be removed \n')
        f.write('# either from grain1 (g1) or from grain2 (g2). If you choose '
                '0 no atoms will be removed \n')
        f.write(list(my_dict.keys())[2] + ': ' + list(my_dict.values())[2] +
                '\n\n')
        f.write('# decide which grain the atoms should be removed from \n')
        f.write(list(my_dict.keys())[3]+': ' + str(list(my_dict.values())[3]) +
                '\n\n')
        f.write('# decide whether you want rigid body translations to be done '
                'on the GB_plane or not (yes or no)\n')

        f.write('# When yes, for any GB aside from twist GBs, the two inplane \n'
                '# CSL vectors will be divided by integers a and b to produce a*b initial \n'
                '# configurations. The default values produce 50 initial structures \n'
                '# if you choose no for rigid_trans, you do not need to care about a and b. \n'
                '# twist boundaries are handled internally \n')

        f.write(list(my_dict.keys())[4] + ': ' +
                str(list(my_dict.values())[4]) + '\n')

        f.write(list(my_dict.keys())[5] + ': ' +
                str(list(my_dict.values())[5]) + '\n')

        f.write(list(my_dict.keys())[6] + ': ' +
                str(list(my_dict.values())[6]) + '\n\n')

        f.write('# dimensions of the supercell in: [l1,l2,l3],  where l1 is'
                'the direction along the GB_plane normal\n')
        f.write('#  and l2 and l3 are inplane dimensions \n')
        f.write(list(my_dict.keys())[7] + ': ' + list(my_dict.values())[7] +
                '\n\n')
        f.write('# File type, either VASP or LAMMPS input \n')
        f.write(list(my_dict.keys())[8] + ': ' + list(my_dict.values())[8] +
                '\n\n\n')
        f.write('# The following is your csl_generator output.'
                ' YOU DO NOT NEED TO CHANGE THEM! \n\n')
        f.write('axis'+': ' + str([axis[0], axis[1], axis[2]]) + '\n')
        f.write('m' + ': ' + str(m) + '\n')
        f.write('n' + ': ' + str(n) + '\n')
        f.write('basis' + ': ' + str(basis) + '\n')

    f.close()
    return


def fcc_grain_boundary(gb_plane, tilt_axis, element='Al', size=[1, 1, 1],
                       minimal_distance=None, lattice_constant=4.0384):
    """
    Super cell of a symmetric tilt grain boundary in an fcc structure.

    Returns the orthogonal supercell with periodic boundary condition containing
    two symmetric tilt grain boundaries.

    Parameters
    ----------
    gb_plane: numpy array or list of shape 3
        Miller indices of grain boundary plane
    tilt_axis: numpy array or list of shape 3
        Miller indices of tilt axis
    element: string
        Chemical symbol of element (default is 'Al')
    size: list, tuple or numpy array of shape 3 and of type int
        multiplicity of supercell in x , y and z direction
        (default is [1, 1, 1])
    minimal_distance: float
        Minimal distance required to be considered separate atoms.
        Atoms which are too close will be replaced with one in the middle
        (default =None: results in quarter of lattice parameter)
    lattice_constant: float
        lattice constant of fcc material (default ois lattice parameter of aluminum)

    Returns
    -------
    grain_boundary: ase.Atoms object
        Structure of grain boundary
    """
    x_direction = np.cross(gb_plane, tilt_axis)

    if minimal_distance is None:
        minimal_distance = lattice_constant / 4

    to_delete = []
    grain_1 = ase.lattice.cubic.FaceCenteredCubic(directions=[x_direction, gb_plane, tilt_axis],
                                                  size=size,
                                                  symbol=element,
                                                  pbc=True,
                                                  latticeconstant=lattice_constant)

    # move and delete atoms which are too close at grain boundary
    for i in range(len(grain_1.positions)):
        if grain_1.positions[i, 0] < minimal_distance:
            grain_1.positions[i, 0] = 0.  # grain_1.cell[0,0]
        if grain_1.positions[i, 0] > grain_1.cell[0, 0] - minimal_distance:
            to_delete.append(i)
    del grain_1[to_delete]

    grain_2 = grain_1.copy()
    grain_2.positions[:, 0] = grain_1.cell[0, 0] - grain_1.positions[:, 0]
    grain_2.wrap()

    # mirror atoms at zero x positions
    for i in range(len(grain_2)):
        if grain_2.positions[i, 0] < 0.01:
            grain_2.positions[i, 1] = grain_1.cell[1, 1] - grain_2.positions[i, 1]
    grain_boundary = ase.build.stack(grain_1, grain_2, axis=0)
    grain_boundary.wrap()
    grain_boundary.cell[2, 2] = abs(grain_boundary.cell[2, 2])
    grain_boundary.positions[:, 2] = abs(grain_boundary.positions[:, 2])

    return grain_boundary


def get_fcc_grain_boundary(sigma, tilt_axis, index=0, element='Al', size=[2, 1, 1], minimal_distance=1.,
                           lattice_constant=4.0384):
    gb_parameters = get_theta_m_n_list(tilt_axis, sigma)
    ind = np.argsort(gb_parameters[:, 0])
    gb_parameters = gb_parameters[ind, :]

    print(f"Possible symmetric Σ{sigma} grain boundaries")
    print(" tilt")
    print(" angle \t  m   \t n")
    for para in gb_parameters:
        print(f" {np.degrees(para[0]):3.12f} \t  {int(para[1])} \t {int(para[2])}")

    theta, m, n = gb_parameters[index]
    missing_l = int(-(m * tilt_axis[0] + n * tilt_axis[1]) / tilt_axis[2])
    boundary_plane = [int(m), int(n), missing_l]
    grain_boundary = fcc_grain_boundary(boundary_plane, tilt_axis, element=element, size=size,
                                        minimal_distance=minimal_distance, lattice_constant=lattice_constant)
    tilt_h, tilt_k, tilt_l = tilt_axis
    print(f'choosing possibility {index + 1} Σ{sigma}({m:.0f}{n:.0f}{missing_l})/[{tilt_h}{tilt_k}{tilt_l}]',
          f'with {grain_boundary.positions.shape[0]} atoms in supercell')

    return grain_boundary
