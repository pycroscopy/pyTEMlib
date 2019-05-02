import numpy as np
from math import gcd, ceil, atan
from fractions import Fraction


# SIGMA_SYMBOL = u'\u03A3'
UNIMODULAR_MATRIX = np.array([np.identity(3),
                              [[1, 0, 1],
                               [0, 1, 0],
                               [0, 1, 1]],
                              [[1, 0, 1],
                               [0, 1, 0],
                               [0, 1, -1]],
                              [[1, 0, 1],
                               [0, 1, 0],
                               [-1, 1, 0]],
                              [[1, 0, 1],
                               [1, 1, 0],
                               [1, 1, 1]]])
STRUCTURE_MATRIX = np.array([np.identity(3),
                             [[0.5, -0.5, 0],
                              [0.5, 0.5, 0],
                              [0.5, 0.5, 1]],
                             [[0.5, 0.5, 0],
                              [0, 0.5, 0.5],
                              [0.5, 0, 0.5]]])

# 0 is coprime only with 1
def coprime(a, b):
    return gcd(a,b) in (0, 1)


def get_cubic_sigma(hkl, m, n=1):
    sqsum = np.inner(hkl, hkl)
    sigma = m*m + n*n * sqsum
    while sigma != 0 and sigma % 2 == 0:
        sigma /= 2
    return (sigma if sigma > 1 else None)

def get_cubic_theta(hkl, m, n=1):
    h,k,l = hkl
    sqsum = h*h + k*k + l*l
    assert sqsum > 0
    if m > 0:
        return 2 * atan(np.sqrt(sqsum) * n / m)
    else:
        return pi


def get_theta_m_n_list(hkl, sigma, verbose=False):
    if sigma == 1:
        return [(0., 0, 0)]
    thetas = []

    # From Grimmer, Acta Cryst. (1984). A40, 108-112
    #    S = m^2 + (u^2+v^2+w^2) n^2     (eq. 2)
    #    S = alpha * Sigma               (eq. 4)
    #   where alpha = 1, 2 or 4.
    # Since (u^2+v^2+w^2) n^2 > 0,
    # thus alpha * Sigma > m^2    =>   m^2 < 4 * Sigma
    max_m = int(ceil(np.sqrt(4*sigma)))

    for m in range(max_m):
        for n in range(1, max_m):
            if not coprime(m, n):
                continue
            s = get_cubic_sigma(hkl, m, n)
            if s != sigma:
                continue
            theta = get_cubic_theta(hkl, m, n)
            if verbose:
                print("m=%i n=%i" % (m, n), "%.3f" % np.degrees(theta))
            thetas.append((theta, m, n))
    return np.array(thetas)


def rodrigues(a, angle, verbose=False):
    "use Rodrigues' rotation formula to get rotation matrix"
    a = np.array(a, dtype=float)
    a /= np.sqrt(np.inner(a, a)) # make unit vector
    #assert abs(sin_angle - sin(acos(cos_angle))) < 1e-6
    if verbose:
        print ("rotation angle:", np.degrees(angle))
        print ("rotation axis:", a)
    omega = np.array([[   0., -a[2],  a[1]],
                   [ a[2],    0., -a[0]],
                   [-a[1],  a[0],    0.]])
    rm = (np.identity(3) + omega * np.sin(angle)
                            + np.dot(omega, omega) * (1 - np.cos(angle)))
    if verbose:
        print("rotation matrix:", rm)
    return rm

def is_integer(a, epsilon=1e-7):
    "return true if numpy Float array consists off all integers"
    return (np.abs(a - np.round(a)) < epsilon).all()

def get_smallest_multiplier(a, max_n=1000):
    """return the smallest positive integer n such that matrix a multiplied
       by n is an integer matrix
    """
    for i in range(1, max_n):
        if is_integer(i*a):
            return i
        
def plus_minus_gen(start, end):
    """
    Generate a list of plus and minus alternating integers
    """
    for i in range(start, end):
        yield i
        yield -i
def get_csl_matrix(sigma, rotate_matrix):
    """\
    Find matrix that determines the coincidence site lattice
    for cubic structures.
    Parameters:
        sigma: CSL sigma
        R: rotation matrix
        centering: "f" for f.c.c., "b" for b.c.c. and None for p.c.
    Return value:
        matrix, which column vectors are the unit vectors of the CSL.
    Based on H. Grimmer et al., Acta Cryst. (1974) A30, 197
    https://doi.org/10.1107/S056773947400043X
    """

    s = STRUCTURE_MATRIX[0]
    for u in UNIMODULAR_MATRIX:
        t = np.eye(3) - np.dot(np.dot(np.dot(u, np.linalg.inv(s)),  np.linalg.inv(rotate_matrix)), s)
        if abs( np.linalg.det(t)) > 1e-6:
            break
    o_lattice = np.round( np.linalg.inv(t), 12)
    n = np.round(sigma /  np.linalg.det(o_lattice), 6)
    csl_matrix = o_lattice_to_csl(o_lattice, n)
    return csl_matrix
    

def o_lattice_to_csl(o_lattice, n):
    """
    The algorithm was borrowed from gosam project with slight changes.
    Link to the project: https://github.com/wojdyr/gosam

    There are two major steps: (1) Manipulate the columns of O-lattice to get
    an integral basis matrix for CSL: make two columns become integers and
    the remaining column can be multiplied by n whereby the determinant
    becomes sigma. (2) Simplify CSL so its vectors acquire the shortest length:
    decrease the integers in the matrix while keeping the determinant the same
    by adding other column vectors (row vectors in the following example) to a
    column vector. If after the addition or subtraction, the maximum value or
    absolute summation of added or subtracted vector is smaller than the
    original, then proceed the addition or subtraction.
    0 0 -1      0 0 -1      0 0 -1      0 0 -1      0 0 -1
    1 2 -1  ->  1 2 0   ->  1 2 0   ->  1 2 0   ->  1 2 0
    1 -3 2      1 -3 2      1 -3 1      1 -3 0      2 -1 0

    Args:
     o_lattice (3x3 array): O-lattice in crystal coordinates
     n (int): Number of O-lattice units per CSL unit

    Returns:
     CSL matrix (3x3 array) in crystal coordinates
    """
    csl = o_lattice.copy()
    if n < 0:
        csl[0] *= -1
        n *= -1
    while True:
        m = [get_smallest_multiplier(i) for i in csl]
        m_prod = np.prod(m)
        if m_prod <= n:
            for i in range(3):
                csl[i] *= m[i]
            if m_prod < n:
                assert n % m_prod == 0
                csl[0] *= n / m_prod
            break
        else:
            changed = False
            for i in range(3):
                for j in range(3):
                    if changed or i == j or m[i] == 1 or m[j] == 1:
                        continue
                    a, b = (i, j) if m[i] <= m[j] else (j, i)
                    for k in plus_minus_gen(1, m[b]):
                        handle = csl[a] + k * csl[b]
                        if get_smallest_multiplier(handle) < m[a]:
                            csl[a] += k * csl[b]
                            changed = True
                            break
            if not changed:
                # This situation rarely happens. Not sure if this solution is
                # legit, as det not equals to sigma. E.g. Sigma 115[113]
                for i in range(3):
                    csl[i] *= m[i]
                break
    csl = csl.round().astype(int)

    # Reshape CSL
    def simplify(l1, l2):
        x = abs(l1 + l2)
        y = abs(l1)
        changed = False
        while max(x) < max(y) or (max(x) == max(y) and sum(x) < sum(y)):
            l1 += l2
            changed = True
            x = abs(l1 + l2)
            y = abs(l1)
        return changed

    while True:
        changed = False
        for i in range(3):
            for j in range(3):
                if i != j and not changed:
                    changed = simplify(csl[i], csl[j]) or simplify(csl[i], -csl[j])
                    if changed:
                        break
        if not changed:
            break
    return csl


def orthogonalize_csl(csl, axis):
    """
    (1) Set the 3rd column of csl same as the rotation axis. The algorithm was
    borrowed from gosam project with slight changes. Link to the project:
    https://github.com/wojdyr/gosam
    (2) Orthogonalize CSL, which is essentially a Gram-Schmidt process. At the
    same time, we want to make sure the column vectors of orthogonalized csl
    has the smallest value possible. That's why we compared two different ways.
    csl = [v1, v2, v3], vi is the column vector
    u1 = v3/||v3||, y2 = v1 - (v1 . u1)u1
    u2 = y2/||y2||, y3 = v3 - [(v3 . u1)u1 + (v3 . u2)u2]
    u3 = y3/||y3||
    """
    axis = np.array(axis)
    c = np.linalg.solve(csl.transpose(), axis)
    if not is_integer(c):
        mult = get_smallest_multiplier(c)
        c *= mult
    c = c.round().astype(int)
    ind = min([(i, v) for i, v in enumerate(c) if not np.allclose(v, 0)],
              key=lambda x: abs(x[1]))[0]
    if ind != 2:
        csl[ind], csl[2] = csl[2].copy(), -csl[ind]
        c[ind], c[2] = c[2], -c[ind]

    csl[2] = np.dot(c, csl)
    if c[2] < 0:
        csl[1] *= -1

    def get_integer(vec):
        # Used vec = np.array(vec, dtype=float) before, but does not work for
        # [5.00000000e-01, -5.00000000e-01,  2.22044605e-16] in Sigma3[112]
        vec = np.round(vec, 12)
        vec_sign = np.array([1 if abs(i) == i else -1 for i in vec])
        vec = list(abs(vec))
        new_vec = []
        if 0.0 in vec:
            zero_ind = vec.index(0)
            vec.pop(zero_ind)
            if 0.0 in vec:
                new_vec = [get_smallest_multiplier(vec) * i for i in vec]
            else:
                frac = Fraction(vec[0] / vec[1]).limit_denominator()
                new_vec = [frac.numerator, frac.denominator]
            new_vec.insert(zero_ind, 0)
        else:
            for i in range(len(vec) - 1):
                frac = Fraction(vec[i] / vec[i + 1]).limit_denominator()
                new_vec.extend([frac.numerator, frac.denominator])
            if new_vec[1] == new_vec[2]:
                new_vec = [new_vec[0], new_vec[1], new_vec[3]]
            else:
                new_vec = reduce_vector([new_vec[0] * new_vec[2],
                                         new_vec[1] * new_vec[2],
                                         new_vec[3] * new_vec[1]])
        assert is_integer(new_vec)
        return new_vec * vec_sign

    u1 = csl[2] / np.linalg.norm(csl[2])
    y2_1 = csl[1] - np.dot(csl[1], u1) * u1
    c0_1 = get_integer(y2_1)
    y2_2 = csl[0] - np.dot(csl[0], u1) * u1
    c0_2 = get_integer(y2_2)
    if sum(abs(c0_1)) > sum(abs(c0_2)):
        u2 = y2_2 / np.linalg.norm(y2_2)
        y3 = csl[1] - np.dot(csl[1], u1) * u1 - np.dot(csl[1], u2) * u2
        csl[1] = get_integer(y3)
        csl[0] = c0_2
    else:
        u2 = y2_1 / np.linalg.norm(y2_1)
        y3 = csl[0] - np.dot(csl[0], u1) * u1 - np.dot(csl[0], u2) * u2
        csl[1] = c0_1
        csl[0] = get_integer(y3)
    for i in range(3):
        for j in range(i + 1, 3):
            if not np.allclose(np.dot(csl[i], csl[j]), 0):
                raise ValueError("Non-orthogonal basis: %s" % csl)
    return csl.round().astype(int)

def find_smallest_real_multiplier(a, max_n=1000):
    """return the smallest positive real f such that matrix `a' multiplied
       by f is an integer matrix
    """
    # |the smallest non-zero element|
    m = min(abs(i) for i in a if abs(i) > 1e-9)
    for i in range(1, max_n):
        t = i / float(m)
        if is_integer(t * a):
            return t
    raise ValueError("Sorry, we can't make this matrix integer:\n%s" % a)

def scale_to_integers(v):
    return np.array(v * find_smallest_real_multiplier(v)).round().astype(int)

