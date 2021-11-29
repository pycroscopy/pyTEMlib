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
