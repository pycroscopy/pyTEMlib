"""
##################################################################
# plotting functions for graph_tools
##################################################################

part of pyTEMlib
a pycrosccopy package

Author: Gerd Duscher
First Version: 2022-01-08
"""
import numpy as np
import ase

import plotly.graph_objects as go
import plotly.express as px

import pyTEMlib.crystal_tools
import pyTEMlib.graph_tools


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
    """
    Information to plot polyhedra with plotly

    Parameter
    ---------
    polyhedra: dict
        dictionary of all polyhedra
    indices: list or integer
        list or index of polyhedron to plot.
    center: boolean
        whether to center polyhedra on origin

    Returns
    -------
    data: dict
        instructions to plot for plotly
    """

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


def plot_bonds(polyhedra):
    """
    Information to plot bonds with plotly

    Parameter
    ---------
    polyhedra: dict
        dictionary of all polyhedra

    Returns
    -------
    data: dict
        instructions to plot for plotly
    """
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


def get_boundary_polyhedra(polyhedra, boundary_x=0, boundary_width=0.5, verbose=True, z_lim=[0, 100]):
    """
    get indices of polyhedra at boundary (assumed to be parallel to x-axis)

    Parameter
    ---------
    polyhedra: dict
        dictionary of all polyhedra
    boundary_x: float
        position of boundary in Angstrom
    boundary_width: float
        width of boundary where center of polyhedra are considered in Angstrom
    verbose: boolean
        optional
    z_lim: list
        upper and lower limit of polyhedra to plot

    Returns
    -------
    boundary_polyhedra: list
        list of polyhedra at boundary
    """
    boundary_polyhedra = []
    for key, polyhedron in polyhedra.items():
        center = polyhedron['vertices'].mean(axis=0)
        if abs(center[0] - boundary_x) < 0.5 and (z_lim[0] < center[2] < z_lim[1]):
            boundary_polyhedra.append(key)
            if verbose:
                print(key, polyhedron['length'], center)

    return boundary_polyhedra


def plot_with_polyhedra(polyhedra, indices, atoms=None, title=''):
    """
    plot atoms and polyhedra with plotly

    Parameter
    ---------
    polyhedra: dict
        dictionary of all polyhedra
    indices: list or integer
        list or index of polyhedron to plot.
    atoms: ase.Atoms
        optional structure info to plot atoms (with correct color)

    Returns
    -------
    fig: plotly.figure
        plotly figure instance
    """

    data = plot_polyhedron(polyhedra, indices)
    if not isinstance(atoms, ase.Atoms):
        atoms = None

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


def plot_supercell(supercell, size=(1, 1, 1), shift_x=0.25, title=''):
    """
    plot supercell with plotly

    Parameter
    ---------
    supercell: ase.Atoms
        optional structure info to plot atoms (with correct color)
    shift_x: float
        amount of shift in x direction of supercell
    title: str
        title of plot

    Returns
    -------
    fig: plotly.figure
        plotly figure instance
    """

    plot_cell = pyTEMlib.graph_tools.plot_super_cell(supercell * size, shift_x=shift_x)

    # grain_boundary.cell.volume
    supercell_area = supercell.cell.lengths()[1] / supercell.cell.lengths()[2]
    print(supercell.symbols)
    volume__bulk_atom = 16.465237835776012
    ideal_volume = len(supercell.positions) * volume__bulk_atom
    print(len(supercell.positions) * volume__bulk_atom, supercell.cell.volume)
    x_0 = ideal_volume / supercell.cell.lengths()[1] / supercell.cell.lengths()[2]
    print(f'Zero volume expansion supercell length: {x_0 / 10:.2f} nm; '
          f' compared to actual {supercell.cell.lengths()[0] / 10:.2f} nm')

    fig = go.Figure(data=[
        go.Scatter3d(x=plot_cell.positions[:, 0], y=plot_cell.positions[:, 1], z=plot_cell.positions[:, 2],
                     mode='markers',
                     marker=dict(
                         color=plot_cell.get_atomic_numbers(),
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
    """
    plot atoms and bonds with plotly

    Parameter
    ---------
    polyhedra: dict
        dictionary of all polyhedra
    atoms: ase.Atoms
        optional structure info to plot atoms (with correct color)
    volumes: list
        list of volumes, optional structure
    atoms_size: float
        sie of atoms to plot
    title: str
        title of plot

    Returns
    -------
    fig: plotly.figure
        plotly figure instance
    """

    data = plot_bonds(polyhedra)
    if volumes is None:
        volumes = [atom_size] * len(atoms.get_atomic_numbers())

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
    """
    plot atoms and polyhedra with plotly

    Parameter
    ---------
    polyhedra: dict
        dictionary of all polyhedra
    indices: list
        list of indices of polyhedra to plot
    atoms: ase.Atoms
        optional structure info to plot atoms (with correct color)
    volumes: list
        list of volumes, optional structure
    title: str
        title of plot

    Returns
    -------
    fig: plotly.figure
        plotly figure instance
    """
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
    """
    plot polyhedra and atoms of vertices with plotly

    Parameter
    ---------
    polyhedra: dict
        dictionary of all polyhedra
    boundary_polyhedra: list
        list of indices of polyhedra to plot
    atoms: ase.Atoms
        optional structure info to plot atoms (with correct color)
    volumes: list
        list of volumes, optional structure
    title: str
        title of plot

    Returns
    -------
    fig: plotly.figure
        plotly figure instance
    """

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
