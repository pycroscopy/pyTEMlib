##################################
#
# image_tools.py
# by Gerd Duscher, UTK
# part of pyTEMlib 
# MIT license except where stated differently
#
###############################
import numpy as np

import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.patches import Polygon # plotting of polygons -- graph rings

import matplotlib.widgets as mwidgets
from matplotlib.widgets import RectangleSelector



from PyQt5 import QtGui, QtWidgets
import pickle

import json
import struct

import sys, os

import math
import itertools
from itertools import product

from scipy import fftpack
from scipy import signal
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import leastsq

# Multidimensional Image library
import scipy.ndimage as ndimage

import scipy.spatial as sp
from scipy.spatial import Voronoi, KDTree, cKDTree
    
from skimage.feature import peak_local_max
# our blob detectors from the scipy image package
from skimage.feature import  blob_log #blob_dog, blob_doh

from sklearn.feature_extraction import image
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans



from pyTEMlib.dftregistration import *  # sup-pixel rigid registration


_SimpleITK_present = True
try:
    import SimpleITK as sitk
except:
     _SimpleITK_present = False

    
if _SimpleITK_present == False:
    print('SimpleITK not installed; Registration Functions for Image Stacks not available')



def plot_image2(image_tags,fig, axes):
    if 'color_map' not in image_tags: 
        image_tags['color_map'] = 'gray'
    color_map = image_tags['color_map']
    if 'origin' not in image_tags: 
        image_tags['origin'] = 'upper'
    origin = image_tags['origin']
    if 'extent' not in image_tags: 
        if 'FOV' in image_tags:
            FOV = image_tags['FOV']
            image_tags['extent'] = (0,FOV,FOV,0)
        else:
            image_tags['extent'] = (0,1,1,0)
    extent = image_tags['extent']
    if 'minimum_intensity' not in image_tags: 
        image_tags['minimum_intensity'] = image_tags['plotimage'].min()
    minimum_intensity = image_tags['minimum_intensity']
    if 'maximum_intensity' not in image_tags: 
        image_tags['maximum_intensity'] = image_tags['plotimage'].max()
    maximum_intensity = image_tags['maximum_intensity']
    
    
    
    ax1 = axes[0]

    ims = ax1.imshow(image_tags['plotimage'], cmap=color_map, origin = 'upper', extent=extent, vmin=minimum_intensity, vmax=maximum_intensity )
    plt.xlabel('distance [nm]')
    plt.colorbar(ims)
    
    ax2 = axes[1]
    def line_select_callback(eclick, erelease):
        pixel_size = out_tags['FOV']/data.shape[0]
        x0, y0 = eclick.xdata/pixel_size, eclick.ydata/pixel_size
        global eclick2
        eclick2 = eclick
        
        x1, y1 = erelease.xdata/pixel_size, erelease.ydata/pixel_size
        length_plot = np.sqrt((x1-x0)**2+(y1-y0)**2)
        
        num = length_plot
        x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

        # Extract the values along the line, using cubic interpolation
        zi2 = ndimage.map_coordinates(data, np.vstack((x,y)))
        x_axis = np.linspace(0,length_plot,len(zi2))*pixel_size
        line_plot.set_xdata(x_axis)
        line_plot.set_ydata(zi2)
        ax2.set_xlim(0,x_axis.max())
        ax2.set_ylim(zi2.min(),zi2.max())
        ax2.draw()
        
        return line_plot
    line_plot, = ax2.plot([],[])

    RS = RectangleSelector(ax1, line_select_callback,
           drawtype='line', useblit=False,
           button=[1, 3],  # don't use middle button
           minspanx=5, minspany=5,
           spancoords='pixels',
           interactive=True)
    plt.show()
    return RS, fig




def histogram_plot(image_tags):
    nbins = 75
    minbin = 0.
    maxbin = 1.
    color_map_list = ['gray','viridis','jet','hot']
    
    if 'minimum_intensity' not in image_tags: 
        image_tags['minimum_intensity'] = image_tags['plotimage'].min()
    minimum_intensity = image_tags['minimum_intensity']
    if 'maximum_intensity' not in image_tags: 
        image_tags['maximum_intensity'] = image_tags['plotimage'].max()
    data = image_tags['plotimage']
    vmin = image_tags['minimum_intensity']
    vmax = image_tags['maximum_intensity']
    if 'color_map' not in image_tags: 
        image_tags['color_map'] = color_map_list[0]
    cmap = plt.cm.get_cmap(image_tags['color_map'])

    colors = cmap(np.linspace(0.,1.,nbins))

    norm2 = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    hist, bin_edges = np.histogram(data, np.linspace(vmin,vmax,nbins),density=True)

    width = bin_edges[1]-bin_edges[0]

    def onselect(vmin, vmax):
        
        ax1.clear()
        cmap = plt.cm.get_cmap(image_tags['color_map'])

        colors = cmap(np.linspace(0.,1.,nbins))

        norm2 = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        hist2, bin_edges2 = np.histogram(data, np.linspace(vmin,vmax,nbins),density=True)

        width2 = (bin_edges2[1]-bin_edges2[0])

        for i in range(nbins-1):
            histogram[i].xy=(bin_edges2[i],0)
            histogram[i].set_height(hist2[i])
            histogram[i].set_width(width2)
            histogram[i].set_facecolor(colors[i])
        ax.set_xlim(vmin,vmax)
        ax.set_ylim(0,hist2.max()*1.01)
    
        
        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm = norm2,orientation='horizontal')

        image_tags['minimum_intensity']= vmin
        image_tags['maximum_intensity']= vmax
        
    def onclick(event):
        global event2
        event2 = event
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata)) 
        if event.inaxes == ax1:
            if event.button == 3:
                ind = color_map_list.index(image_tags['color_map'])+1
                if ind == len(color_map_list):
                    ind = 0
                image_tags['color_map']= color_map_list[ind]#'viridis'
                vmin = image_tags['minimum_intensity']
                vmax = image_tags['maximum_intensity']
            else:
                vmax = data.max()
                vmin = data.min()
        onselect(vmin,vmax)

    fig2 = plt.figure()

    ax = fig2.add_axes([0., 0.2, 0.9, 0.7])
    ax1 = fig2.add_axes([0., 0.15, 0.9, 0.05])

    histogram = ax.bar(bin_edges[0:-1], hist, width=width, color=colors, edgecolor = 'black',alpha=0.8)
    onselect(vmin,vmax)
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm = norm2,orientation='horizontal')

    rectprops = dict(facecolor='blue', alpha=0.5)

    span = mwidgets.SpanSelector(ax, onselect, 'horizontal',
                                  rectprops=rectprops)

    cid = fig2.canvas.mpl_connect('button_press_event', onclick)
    return span

def Fourier_Transform(current_channel):
    """
        Reads information into dictionary 'tags', performs 'FFT', and provides a smoothed FT and reciprocal and intensity
        limits for visualization. All information is stored in the 'fft' sub-dictionary of tags.
        
        Input
        -----
        current_channel: data group of pyUSID file
        
        Usage
        -----
        
        tags = Fourier_Transform(current_channel)
        fft = tags['fft']    
        fig = plt.figure()
        plt.imshow(np.log2(1+ 0.5*fft['magnitude_smooth']).T, extent=fft['extend'], origin = 'upper',
                   vmin=fft['minimum_intensity'], vmax=fft['maximum_intensity'])
        plt.xlabel('spatial frequency [1/nm]');
        
    """
    
    tags = ft.get_dictionary_from_pyUSID(current_channel)
    
    sizeX
    image = tags['data']- tags['data'].min()
    fft_mag = (np.abs((np.fft.fftshift(np.fft.fft2(image)))))
    
    tags['fft'] = {}
    fft = tags['fft']
    fft['magnitude'] = fft_mag

    fft['spatial_scale_x'] = 1/tags['FOV_x'] 
    fft['spatial_scale_y'] = 1/tags['FOV_y']  
    fft['spatial_offset_x'] = -1/tags['FOV_x']  * tags['data'].shape[0] /2.
    fft['spatial_offset_y'] = -1/tags['FOV_y']  * tags['data'].shape[1] /2.
      
    ## Field of View (FOV) in recipical space please note: rec_FOV_x = 1/(scaleX*2)
    fft['rec_FOV_x'] = 1/tags['FOV_x']  * sizeX /2.
    fft['rec_FOV_y'] = 1/tags['FOV_y']  * sizeY /2.

    ## Field ofView (FOV) in recipical space
    fft['extend'] = (fft['spatial_offset_x'],-fft['spatial_offset_x'],-fft['rec_FOV_y'],fft['rec_FOV_y'])
    
    # We need some smoothing (here with a Gaussian)
    smoothing = 3
    fft_mag2 = ndimage.gaussian_filter(fft_mag, sigma=(smoothing, smoothing), order=0)


    #prepare mask for low and high frequencies
    pixels = (np.linspace(0,image.shape[0]-1,image.shape[0])-image.shape[0]/2)* rec_scale_x
    x,y = np.meshgrid(pixels,pixels);
    mask = np.zeros(image.shape)

    mask_spot = x**2+y**2 > 2**2 
    mask = mask + mask_spot
    mask_spot = x**2+y**2 < 10**2 
    mask = mask + mask_spot

    mask[np.where(mask==1)]=0 # just in case of overlapping disks

    fft_mag3 = fft_mag2*mask
    
    fft['magnitude_smooth'] = fft_mag2
    fft['minimum_intensity'] = np.log2(1+fft_mag2)[np.where(mask==2)].min()*0.95
    #minimum_intensity = np.mean(fft_mag3)-np.std(fft_mag3)
    fft['maximum_intensity'] = np.log2(1+fft_mag2)[np.where(mask==2)].max()*1.05
    #maximum_intensity =  np.mean(fft_mag3)+np.std(fft_mag3)*2
    
    return tags 



def find_atoms(im, tags, verbose = False):
    from skimage.feature import  blob_log #blob_dog, blob_doh
    if 'rel_blob_size' not in tags:
        tags['rel_blob_size'] = .4 # between 0 and 1 nromally around 0.5
        tags['source_size'] = 0.06 #in nm gives the size of the atoms or resolution
        tags['nearest_neighbours'] = 7 # up to this number nearest neighbours are evaluated (normally 7)
        tags['threshold'] =  .15 # between 0.01 and 0.1 
        tags['rim_size'] = 2# size of rim in multiples of source size
        
    rel_blob_size = tags['rel_blob_size'] # between 0 and 1 nromally around 0.5
    source_size = tags['source_size']  #in nm gives the size of the atoms
    nearest_neighbours  = tags['nearest_neighbours'] # up to this number nearest neighbours are evaluated (normally 7)
    threshold = tags['threshold']  # between 0.01 and 0.1 
    rim_size = tags['rim_size'] # sizeof rim in multiples of resolution
    pixel_size = tags['pixel_size']
                      
    rim_width = rim_size*source_size/pixel_size
    
    ## Get a noise free image: reduced
    #pixel_size = FOV/im.shape[0]
    reduced_image = clean_svd(im,pixel_size=pixel_size,source_size=source_size)

    reduced_image = reduced_image-reduced_image.min()
    reduced_image = reduced_image/reduced_image.max()

    tags['reduced_image'] = reduced_image
    patch_size = im.shape[0]-reduced_image.shape[0]
    tags['patch_size'] = patch_size
    print(f' Use {patch_size} x {patch_size} pixels for image-patch of atoms')

    # Find atoms    
    thresh = reduced_image.std()*threshold
    blobs = blob_log(np.array(reduced_image), max_sigma=source_size/pixel_size, threshold=thresh)
    plot_image = im[int(patch_size/2):,int(patch_size/2):]

    atoms = []
    from skimage.feature import blob_log
    for blob in blobs:
        y, x, r = blob
        if r > patch_size*rel_blob_size:
            atoms.append([x+patch_size/2,y+patch_size/2,r])

    rim_atoms = []

    for i in range(len(atoms)):
        if (np.array(atoms[i][0:2])<rim_width).any() or (np.array(atoms[i]) > im.shape[0]-rim_width-5).any():
            rim_atoms.append(i)
    rim_atoms=np.unique(rim_atoms)
    mid_atoms_list = np.setdiff1d(np.arange(len(atoms)),rim_atoms)
    
    mid_atoms = np.array(atoms)[mid_atoms_list]
    if verbose:
        print(f'Evaluated {len(mid_atoms)} atom positions, out of {len(atoms)} atoms')
    tags['atoms'] = atoms
    tags['mid_atoms'] = mid_atoms
    tags['rim_atoms'] = rim_atoms
    tags['number_of_atoms'] = len(atoms)
    tags['number_of_evaluated_atoms' ]= len(mid_atoms)
    return tags
    
def atoms_clustering(atoms, mid_atoms, number_of_clusters = 3, nearest_neighbours  = 7):
    ## get distances
    T = cKDTree(np.array(atoms)[:,0:2])

    distances, indices = T.query(np.array(mid_atoms)[:,0:2], nearest_neighbours)


    ## CLustering
    k_means = KMeans(n_clusters=number_of_clusters, random_state=0) # Fixing the RNG in kmeans
    k_means.fit(distances)
    clusters = k_means.predict(distances)
    return clusters, distances, indices


def voronoi(atoms,tags):
    im = tags['image']
    vor = Voronoi(np.array(atoms)[:,0:2])# Plot it:
    rim_vertices = []
    for i in range(len(vor.vertices)):

        if (vor.vertices[i,0:2]<0).any() or (vor.vertices[i,0:2] > im.shape[0]-5).any():
            rim_vertices.append(i)
    rim_vertices=set(rim_vertices)
    mid_vertices = list(set(np.arange(len(vor.vertices))).difference(rim_vertices))

    mid_regions = []
    for region in vor.regions: #Check all Voronoi polygons
        if all(x in mid_vertices for x in region) and len(region)>1: # we get a lot of rim (-1) and empty and  regions
            mid_regions.append(region)
    tags['atoms']['voronoi']=vor
    tags['atoms']['voronoi_vertices']=vor.vertices
    tags['atoms']['voronoi_regions'] = vor.regions
    tags['atoms']['voronoi_midVerticesIndices']=mid_vertices
    tags['atoms']['voronoi_midVertices']=vor.vertices[mid_vertices]
    tags['atoms']['voronoi_midRegions'] = mid_regions




def clean_svd(im,pixel_size=1,source_size=5):
    patch_size = int(source_size/pixel_size)
    if patch_size < 3:
        patch_size = 3
    print(patch_size)
    
    patches = image.extract_patches_2d(im, (patch_size, patch_size))
    patches = patches.reshape(patches.shape[0],patches.shape[1]*patches.shape[2] )
    
    num_components = 32
    
    u, s, v = randomized_svd(patches, num_components)
    u_im_size = int(np.sqrt(u.shape[0]))
    reduced_image = u[:,0].reshape(u_im_size,u_im_size)
    reduced_image = reduced_image/reduced_image.sum()*im.sum()
    return reduced_image
    
def rebin(im,binning=2):
    """
    rebin an image by the number of pixels in x and y direction given by binning
    
    Input:
    ======
            image: numpy array in 2 dimensions
    
    Output:
    =======
            binned image 
    """
    if len(im.shape) == 2:
        return im.reshape((im.shape[0]//binning,binning,im.shape[1]//binning,binning)).mean(axis=3).mean(1)
    else:
        print('not a 2D image')
        return im



def power_spectrum(image, FOV_x, FOV_y):
    """
    Calculate power spectrum

    Input:
    ======
            image:
            FOV_x: field ofView in x direction
            FOV_x: field ofView in x direction
    Output:
    =======
            tags: dictionary with
                ['data']: fourier transformed image
                ['axis']: scale of reciprocal image
                ['power_spectrum']: power_spectrum
                ['FOV']: field of view for extent parameter in plotting
                ['minimum_intensity']: suggested minimum intensity for plotting
                ['maximum_intensity']: suggested maximum intensity for plotting

    """
    image = image-image.min()
    fft = np.fft.fftshift(np.fft.fft2(image))
    fft_mag = np.abs(fft)
    tags ={}
    tags['data'] = fft
    
    tags['axis']={}
    tags['axis']['0']={}
    tags['axis']['1']={}

    tags['axis']['0']['scale'] = 1/FOV_x  
    tags['axis']['1']['scale'] = 1/FOV_y
    tags['axis']['0']['unit'] = '1/nm'  
    tags['axis']['1']['unit'] = '1/nm'  
    tags['axis']['0']['pixels'] = image.shape[0]
    tags['axis']['1']['pixels'] = image.shape[1]
    tags['axis']['0']['origin'] = image.shape[0] /2.
    tags['axis']['1']['origin'] = image.shape[1] /2.
    
    rev_FOV_x = tags['axis']['0']['scale'] * tags['axis']['0']['pixels'] /2.
    rev_FOV_y = tags['axis']['1']['scale'] * tags['axis']['1']['pixels'] /2.
    tags['axis']['FOV'] = (-rev_FOV_x,rev_FOV_x,rev_FOV_y,-rev_FOV_y)
    
    # We need some smoothing (here with a Gaussian)
    smoothing = 3
    fft_mag2 = ndimage.gaussian_filter(fft_mag, sigma=(smoothing, smoothing), order=0)
    fft_mag2 = np.log2(1+fft_mag2)
    tags['power_spectrum'] = fft_mag2
    
    #prepare mask
    pixels = (np.linspace(0,2047,2048)-1023.5)* tags['axis']['0']['scale']
    x,y = np.meshgrid(pixels,pixels);
    mask = np.zeros(image.shape)

    mask_spot = x**2+y**2 > 2**2 
    mask = mask + mask_spot
    mask_spot = x**2+y**2 < 8**2 
    mask = mask + mask_spot
    
    tags['minimum_intensity'] = fft_mag2[np.where(mask==2)].min()*0.95
    tags['maximum_intensity'] = fft_mag2[np.where(mask==2)].max()*1.05

    # calculate angular averaged profile
    cent = [tags['axis']['0']['origin'], tags['axis']['1']['origin']]
    polar_projection = warp(tags['data'],cent)
    below_zero = polar_projection<0.
    polar_projection[below_zero]=0.

    # Sum over all angles (axis 1)
    profile = polar_projection.sum(axis=1)

    u =np.linspace(1,len(profile),len(profile))*tags['axis']['0']['scale']

    tags['polar_projection'] = polar_projection
    tags['polar_projection_profile'] = profile
    tags['polar_projection_frequencies'] = u

    return tags

def diffractogram_spots(fft_tags, spot_threshold):
    """
    Find spots in diffractogram and sort them by distance from center

    Input:
    ======
            fft_tags: dictionary with
                ['axis']: scale of reciprocal image
                ['power_spectrum']: power_spectrum
            spot_threshold: threshold for blob finder
    Output:
    =======
            spots: numpy array with sorted position (x,y) and radius (r) of all spots
    
    """
    ## Needed for conversion from pixel to Reciprocal space
    rev_scale = np.array([fft_tags['axis']['0']['scale'],  fft_tags['axis']['1']['scale'],1])
    center    = np.array([fft_tags['axis']['0']['origin'], fft_tags['axis']['1']['origin'],1] )

    ## spot detection ( for future referece there is no symmetry assumed here)

    spots_random =  (blob_dog(fft_tags['power_spectrum'],  max_sigma= 5 , threshold=spot_threshold)-center)*rev_scale
    print(f'Found {spots_random.shape[0]} reflections')
    spots_random[:,2] = np.linalg.norm(spots_random[:,0:2], axis=1)
    spots_index = np.argsort(spots_random[:,2])
    spots = spots_random[spots_index]
    
    return spots

def adaptive_Fourier_filter(tags, low_pass = 3, reflection_radius = 0.3):
    """
    Use spots in diffractogram for a Fourier Filter

    Input:
    ======
            tags: dictionary with
                ['axis']: scale of reciprocal image
                ['data']: Fourier transformed image
                ['diffractogram_spots']: sorted spots in diffractogram in 1/nm
            low_pass: low pass filter in center of diffractogrm
            
    Output:
    =======
            Fourier filtered image
    """
    #prepare mask
    size = tags['axis']['0']['pixels']
    pixels = (np.linspace(0,size-1,size)-tags['axis']['0']['origin'])* tags['axis']['0']['scale']
    x,y = np.meshgrid(pixels,pixels);
    mask = np.zeros(tags['data'].shape)


    # mask reflections
    #reflection_radius = 0.3 # in 1/nm
    for spot in tags['diffractogram_spots']:
        mask_spot = (x-spot[0])**2+(y-spot[1])**2 < reflection_radius**2 # make a spot 
        mask = mask + mask_spot# add spot to mask
    
    # mask zero region larger (low-pass filter = intensity variations)
    #low_pass = 3 # in 1/nm
    mask_spot = x**2+y**2 < low_pass**2 
    mask = mask + mask_spot
    mask[np.where(mask>1)]=1    
    return (tags['data'])*mask.T


def rotational_symmetry_diffractogram(spots):
    
    rotation_symmetry = []
    for n in [2,3,4,6]:
        C = np.array([[np.cos(2*np.pi/n), np.sin(2*np.pi/n),0],[-np.sin(2*np.pi/n), np.cos(2*np.pi/n),0], [0,0,1]])
        sym_spots = np.dot(spots,C)
        dif = []
        for p0, p1 in product(sym_spots[:,0:2], spots[:,0:2]):
            dif.append(np.linalg.norm(p0-p1))
        dif = np.array(sorted(dif))
        
        if dif[int(spots.shape[0]*.7)] < 0.2:
            rotation_symmetry.append(n)
    return(rotation_symmetry)

def cart2pol(points):
    rho = np.linalg.norm(points[:,0:2], axis=1)
    phi = np.arctan2(points[:,1], points[:,0])
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def xy2polar(points, rounding = 1e-3):
    """
    Conversion from carthesian to polar coordinates
    
    the angles and distances are sorted by r and then phi
    The indices of this sort is also returned
    
    input points: numpy array with number of points in axis 0 first two elements in axis 1 are x and y
    
    optional rounding in significant digits 
    
    returns r,phi, sorted_indices
    """
    
    r,phi = cart2pol(points)
    
    phi = phi-phi.min() # only positive angles
    r = (np.floor(r/rounding) )*rounding # Remove rounding error differences

    sorted_indices = np.lexsort((phi,r) ) # sort first by r and then by phi
    r = r[sorted_indices]
    phi = phi[sorted_indices]
    
    return r, phi,  sorted_indices
            



def cartesian2polar(x, y, grid, r, t, order=3):

    R,T = np.meshgrid(r, t)

    new_x = R*np.cos(T)
    new_y = R*np.sin(T)

    ix = interp1d(x, np.arange(len(x)))
    iy = interp1d(y, np.arange(len(y)))

    new_ix = ix(new_x.ravel())
    new_iy = iy(new_y.ravel())

    
    return ndimage.map_coordinates(grid, np.array([new_ix, new_iy]),
                            order=order).reshape(new_x.shape)

def warp(diff, center):
    # Define original polar grid
    nx = diff.shape[0]
    ny = diff.shape[1]

    

    x = np.linspace(1, nx, nx, endpoint = True)-center[1]
    y = np.linspace(1, ny, ny, endpoint = True)-center[0]
    z = np.abs(diff)

    # Define new polar grid
    nr = min([center[0], center[1], diff.shape[0]-center[0], diff.shape[1]-center[1]])-1
    nt = 360*3


    r = np.linspace(1, nr, nr)
    t = np.linspace(0., np.pi, nt, endpoint = False)
    return cartesian2polar(x,y, z, r, t, order=3).T

def calculateCTF(waveLength, Cs, defocus,k):
    """ Calculate Contrast Transfer Function
    everything in nm
    """
    ctf=np.sin(np.pi*defocus*waveLength*k**2+0.5*np.pi*Cs*waveLength**3*k**4)
    return ctf

def calculateScherzer(waveLength, Cs):
    """
    Calculate the Scherzer defocus. Cs is in mm, lambda is in nm
    # EInput and output in nm
    """
    scherzer=-1.155*(Cs*waveLength)**0.5 # in m
    return scherzer

def calibrate_imageScale(fft_tags,spots_reference,spots_experiment):
    gx = fft_tags['axis']['0']['scale']
    gy = fft_tags['axis']['1']['scale']

    dist_reference = np.linalg.norm(spots_reference, axis=1)
    distance_experiment = np.linalg.norm(spots_experiment, axis=1)

    first_reflections = abs(distance_experiment - dist_reference.min()) < .1
    print('Evaluate ', first_reflections.sum(), 'reflections')
    closest_exp_reflections = spots_experiment[first_reflections]

    import scipy.optimize as optimization
    def func(params, xdata, ydata):
        dgx , dgy = params
        return (np.sqrt((xdata*dgx)**2 + (ydata*dgy)**2 ) - dist_reference.min())

    x0 = [1.,0.999]
    dg, sig = optimization.leastsq(func, x0, args=(closest_exp_reflections[:,0], closest_exp_reflections[:,1]))
    return dg

def align_crystal_reflections(spots,crystals):
    crystal_reflections_polar=[]
    angles = []
    mask = np.ones(spots.shape[0], dtype=bool)
    exp_r, exp_phi = cart2pol(spots) # just in polar coordinates
    spots_polar= np.array([exp_r, exp_phi])
    number_spots_remain = len(mask)
        
    for i in range(len(crystals)):
        tags = crystals[i]
        r,phi,indices = xy2polar(tags['allowed']['g']) #sorted by r and phi , only positive angles
        ## we mask the experimental values that are found already
        angle = 0.
        if mask.sum()>1:
            angleI = np.argmin(np.abs((exp_r[mask])-r[0]) )
            angle = (exp_phi[mask])[angleI] - phi[0]
        angles.append(angle) ## Determine rotation angle
        crystal_reflections_polar.append([r, angle - phi, indices])
        tags['allowed']['g_rotated'] = pol2cart(r, angle + phi)
        for spot in tags['allowed']['g']:
            dif = np.linalg.norm(spots[:,0:2]-spot[0:2],axis=1)
            #print(dif.min())
            if dif.min() < 1.5:
                ind = np.argmin(dif)
                if mask[ind]:
                    mask[ind] = 0

        print(f'found {(number_spots_remain-mask.sum()):.0f} refletions in crystal {i}')         
        number_spots_remain -= (number_spots_remain-mask.sum())
        print(mask.sum())
        
    return crystal_reflections_polar, angles, mask



def plot_image(tags):
    if 'axis' in tags:
        pixel_size = tags['axis']['0']['scale']
        units = tags['axis']['0']['units']
    elif 'pixel_size' not in tags:
        pixel_size = 1
        units = 'px'
    else:
        pixel_size = tags['pixel_size']
        units = 'nm'
    
        
    image = tags['data'].T
    FOV = pixel_size*image.shape[0]
    plt.imshow(image, cmap='gray', extent=(0,FOV,0,FOV))
    if 'basename' in tags:
        plt.title(tags['basename'])

    plt.show()

def DemonReg(cube, verbose = False):
    """
    Diffeomorphic Demon Non-Rigid Registration 
    Usage:
    ------
    DemReg = DemonReg(cube, verbose = False)

    Input:
        cube: stack of image after rigid registration and cropping
    Output:
        DemReg: stack of images with non-rigid registration

    Dempends on:
        simpleITK and numpy
    
    Please Cite: http://www.simpleitk.org/SimpleITK/project/parti.html
    and T. Vercauteren, X. Pennec, A. Perchant and N. Ayache
    Diffeomorphic Demons Using ITK\'s Finite Difference Solver Hierarchy
    The Insight Journal, http://hdl.handle.net/1926/510 2007
    """
    
    
    DemReg =  np.zeros_like(cube)
    nimages = cube.shape[2]
    # create fixed image by summing over rigid registration

    fixed_np = np.sum(cube, axis=2)/float(nimages)

    fixed = sitk.GetImageFromArray(fixed_np, sitk.sitkFloat32)
    fixed = sitk.DiscreteGaussian(fixed, 2.0)

    #demons = sitk.SymmetricForcesDemonsRegistrationFilter()
    demons = sitk.DiffeomorphicDemonsRegistrationFilter()

    demons.SetNumberOfIterations(200)
    demons.SetStandardDeviations(1.0)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed);
    resampler.SetInterpolator(sitk.sitkGaussian)
    resampler.SetDefaultPixelValue(0)

    for i in range(nimages):
        moving = sitk.GetImageFromArray(cube[:,:,i], sitk.sitkFloat32)
        movingf = sitk.DiscreteGaussian(moving, 2.0)
        displacementField = demons.Execute(fixed,movingf)
        outTx = sitk.DisplacementFieldTransform( displacementField )
        resampler.SetTransform(outTx)
        out = resampler.Execute(moving)
        DemReg[:,:,i] = sitk.GetArrayFromImage(out)
        print('image ', i)
        
    
    print(':-)')
    print('You have succesfully completed Diffeomorphic Demons Registration')
    
    return DemReg

def dftRigReg(cube, verbose = False):
    """
    Implementation of sub-pixel rigid registration
    
    usage:
    import image_tools as it
    it.dftRigReg(cube, verbose = False)

    input:
        stack of images as 3dimensional numpy array with x,y as image axes.

    output:
        aligned stack
        drift

    For copyright information use:
    from dftregistration import *
    help(dftregistration1)
    """

    #help(dftregistration1)
    if len(cube.shape) !=3:
        print('Registration requires at least 2 images')
        return

    if cube.shape[2] <2:
        print('Registration requires at least 2 images')
        return
    nimages= cube.shape[2]
    RigReg = np.empty_like(cube)

    # select central image as fixed image 
    icent = int(nimages/2)
    fixed = cube[:,:,icent]

    # determine maximum shifts 
    xshift = []
    yshift = []
    drift = []

    usfac = 1000
    for i in range(nimages) :
        moving = cube[:,:,i]
        output, Greg = dftregistration1(np.fft.fft2(fixed),np.fft.fft2(moving),usfac)
        Greg= np.fft.ifft2(Greg)
        RigReg[:,:,i] = abs(Greg)
        xshift.append(output[3])
        yshift.append(output[2])
        drift.append([output[3],output[2]])
        print('Image number', i,' xshift = ',xshift[-1],' y-shift =',yshift[-1])

    return RigReg, drift

def CropImage(drift, image_shape, verbose = False):
    """
    # Images wrap around as they are shifted.
    # If image is shifted to the right (x +ve) we need to cropp pixels from the left
    # If image is shifted to the left (x -ve) we need to cropp pixels from the right
    # If image is shifted down (y +ve) we need to cropp pixels from the top
    # If image is shifted up (y -ve) we need to cropp pixels from the bottom

    Usage:
    ------
        image_limits = CropImage(drift, image_shape, verbose = False)
    Input:
    -----
        drift: nimages,2 array of sample drift
        ,image_shape: shape of image stack
    Output:
    -------
        [xpmin,xpmax,ypmin,ypmax]: list of image boundaries
    """
    
    xmax = max(np.array(drift)[:,0])
    xmin = min(np.array(drift)[:,0])
    ymax = max(np.array(drift)[:,1])
    ymin = min(np.array(drift)[:,1])

    # Round up or down as appropriate
    round_i = lambda x: (int(x+1), int(x-1))[x < 0]
    ixmin = round_i(xmin)
    ixmax = round_i(xmax)
    iymin = round_i(ymin)
    iymax = round_i(ymax)

    # Now determine the cropped area

    if ixmax < 0:
        xpmax = (image_shape[0]-1) + ixmin
        xpmin = 0
    else: 
        xpmin = ixmax
        if ixmin < 0:
            xpmax = (image_shape[0]-1) + ixmin
        else:
            xpmax = (image_shape[0]-1)
            
    if iymax < 0:
        ypmax = (image_shape[1]-1) + iymin
        ypmin = 0
    else: 
        ypmin = iymax
        if ixmin < 0:
            ypmax = (image_shape[1]-1) + iymin
        else:
            ypmax = (image_shape[1]-1)

    if verbose:
        print()        
        print ('Cropped area ranges',xpmin,':',xpmax, ' in the x-direction')         
        print ('Cropped area ranges',ypmin,':',ypmax, ' in the y-direction')   
        ixrange = xpmax-xpmin + 1 
        iyrange = ypmax-ypmin + 1
        print('Which results in a cropped image',ixrange,' pixels in the x direction and',iyrange, 'pixel in the y-direction' )

    return [xpmin,xpmax,ypmin,ypmax]        

def RigReg(cube, verbose = False):
    """**********************************************
    * RigReg rigid registration
    * This function alignes the images in stack 
    * which is called a rigid registration.
    * The stack of images should be in the 'cube' of the dictionary
    * output goes to the lists:
     - tags['aligned stack']
     - tags['drift']

    **********************************************"""
    # We need an image stack
    if len(cube.shape) !=3:
        print('Registration requires at least 2 images')
        return

    if cube.shape[2] <2:
        print('Registration requires at least 2 images')
        return

    # Define center image as fixed
    fixedID = int(cube.shape[2]/2)
    fixed = sitk.GetImageFromArray(cube[:,:,fixedID], sitk.sitkFloat64)
    moving = sitk.GetImageFromArray(cube[:,:,0], sitk.sitkFloat64)

    # Setup registration
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200 )
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    outTx = R.Execute(fixed, moving)
    resampler.SetTransform(outTx)
    #regimg = fixed

    # Do image registration
    aligned = []
    drift =[]
    for i in range(cube.shape[2]):
        moving = sitk.GetImageFromArray(cube[:,:,i], sitk.sitkFloat64)
        outTx = R.Execute(fixed, moving)
        out = resampler.Execute(moving)

        #regimg = regimg + out
        aligned.append(sitk.GetArrayFromImage(out))
        if verbose:
            print(i, 'Offset: ',outTx.GetParameters() )
        drift.append(outTx.GetParameters() )

    #tags['drift'] = drift
    #tags['aligned stack'] = np.array(aligned, dtype = float)
    

    if verbose:
        print('-------')
        #print(outTx)
        print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
        print(" Iteration: {0}".format(R.GetOptimizerIteration()))
        print(" Metric value: {0}".format(R.GetMetricValue()))

    return np.array(aligned, dtype = float), drift


def makechi1( phi, theta,wl,ab, C1include)  :
    """
    ###
    # Aberration function chi without defocus
    ###
    """
    t0 = np.power(theta,1)/1*(  float(ab['C01a']) * np.cos(1*phi)
                                + float(ab['C01b']) * np.sin(1*phi))

    if C1include == 1: #First and second terms
        t1 = np.power(theta,2)/2*(  ab['C10']
                                    + ab['C12a'] *np.cos(2*phi) 
                                    + ab['C12b'] *np.sin(2*phi))
    elif C1include == 2:#Second terms only
        t1 = np.power(theta,2)/2*(  ab['C12a'] *np.cos(2*phi) 
                                    + ab['C12b'] *np.sin(2*phi))
    else: # none for zero
        t1 = t0*0.
    t2 = np.power(theta,3)/3*(  ab['C21a'] * np.cos(1*phi)
                                + ab['C21b'] * np.sin(1*phi)
                                + ab['C23a'] * np.cos(3*phi)
                                + ab['C23b'] * np.sin(3*phi) )

    t3 = np.power(theta,4)/4*(  ab['C30']
                                + ab['C32a'] * np.cos(2*(phi))
                                + ab['C32b'] * np.sin(2*(phi))
                                + ab['C34a'] * np.cos(4*(phi))
                                + ab['C34b'] * np.sin(4*(phi)) )

    t4 = np.power(theta,5)/5*(  ab['C41a'] * np.cos(1*phi)
                                + ab['C41b'] * np.sin(1*phi)
                                + ab['C43a'] * np.cos(3*phi)
                                + ab['C43b'] * np.sin(3*phi)
                                + ab['C45a'] * np.cos(5*phi)
                                + ab['C45b'] * np.sin(5*phi) )

    t5 = np.power(theta,6)/6*(  ab['C50']
                                + ab['C52a'] * np.cos(2*phi)
                                + ab['C52b'] * np.sin(2*phi)
                                + ab['C54a'] * np.cos(4*phi)
                                + ab['C54b'] * np.sin(4*phi)
                                + ab['C56a'] * np.cos(6*phi)
                                + ab['C56b'] * np.sin(6*phi) )



    chi = t0 + t1+t2+t3+t4+t5
    if 'C70' in ab:
        chi += np.power(theta,8)/8*(   ab['C70'])
    return chi*2*np.pi/wl #np.power(theta,6)/6*(  ab['C50'] )



def Probe2( ab, sizeX, sizeY, tags, verbose= False):     
    """
    **********************************************
    * This function creates a incident STEM probe 
    * at position (0,0)
    * with parameters given in ab dictionary
    *
    * The following Abberation functions are being used:
    * 1) ddf = Cc*dE/E  but not  + Cc2*(dE/E)^2,    
    *    Cc, Cc2 = chrom. Abber. (1st, 2nd order) [1]
    * 2) chi(qx,qy) = (2*pi/lambda)*{0.5*C1*(qx^2+qy^2)+
    *                 0.5*C12a*(qx^2-qy^2)+
    *                 C12b*qx*qy+
    *                 C21a/3*qx*(qx^2+qy^2)+
    *                 ... 
    *                 +0.5*C3*(qx^2+qy^2)^2
    *                 +0.125*C5*(qx^2+qy^2)^3
    *                 ... (need to finish)
    *
    *
    *    qx = acos(kx/K), qy = acos(ky/K) 
    *
    * References:
    * [1] J. Zach, M. Haider, 
    *    "Correction of spherical and Chromatic Abberation 
    *     in a low Voltage SEM", Optik 98 (3), 112-118 (1995)
    * [2] O.L. Krivanek, N. Delby, A.R. Lupini,
    *    "Towards sub-Angstroem Electron Beams", 
    *    Ultramicroscopy 78, 1-11 (1999)
    *
    *********************************************'''
    ####
    # Internally reciprocal lattice vectors in 1/nm or rad.
    # All calculations of chi in angles.
    # All aberration coefficients in nm
    """  

    if 'FOV' not in ab:
        if 'FOV' not in tags:
            print(' need field of view in tags ' )
        else:
            ab['FOV'] = tags['FOV']

    if 'convAngle' not in ab:
        ab['convAngle'] = 30 # in mrad

    ApAngle=ab['convAngle']/1000.0 # in rad

    E0= ab['EHT'] = float( ab['EHT'])  # acceleration voltage in eV

    defocus = ab['C10'] 


    if 'C01a' not in ab:
        ab['C01a'] = 0.
    if 'C01b' not in ab:
        ab['C01b'] = 0.

    if 'C50' not in ab:
        ab['C50'] = 0.
    if 'C70' not in ab:
        ab['C70'] = 0.

    if 'Cc' not in ab:
        ab['Cc'] = 1.3e6            #// Cc in  nm


    def get_wl():
        h=6.626*10**-34
        m0=9.109*10**-31
        eV=1.602*10**-19*E0
        C=2.998*10**8
        return h/np.sqrt(2*m0*eV*(1+eV/(2*m0*C**2)))*10**9

    wl=get_wl()
    if verbose:
        print('Acceleration voltage {0:}kV => wavelength {1:.2f}pm'.format(int(E0/1000),wl*1000) )
    ab['wavelength'] = wl


    ## Reciprocal plane in 1/nm
    dk = 1/ab['FOV']
    kx = np.array(dk*(-sizeX/2.+ np.arange(sizeX)))
    ky = np.array(dk*(-sizeY/2.+ np.arange(sizeY)))
    Txv, Tyv = np.meshgrid(kx, ky)

    # define reciprocal plane in angles
    phi =  np.arctan2(Txv, Tyv)
    theta = np.arctan2(np.sqrt(Txv**2 + Tyv**2),1/wl)

    ## calculate chi but omit defocus
    chi = np.fft.ifftshift (makechi1(phi,theta,wl,ab, 2))
    probe = np.zeros((sizeX, sizeY))


    ## Aperture function 
    mask = theta >= ApAngle

    ## Calculate probe with Cc

    for i in range(len(ab['zeroLoss'])):
        df = ab['C10'] + ab['Cc']* ab['zeroEnergy'][i]/E0
        if verbose:
            print('defocus due to Cc: {0:.2f} nm with weight {1:.2f}'.format(df,ab['zeroLoss'][i]))
        # Add defocus
        chi2 = chi + np.power(theta,2)/2*(df)
        #Calculate exponent of - i * chi
        chiT = np.fft.ifftshift (np.vectorize(complex)(np.cos(chi2), -np.sin(chi2)) )
        ## Aply aperture function
        chiT[mask] = 0.
        ## inverse fft of aberration function
        i2  = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift (chiT)))
        ## add intensities
        probe = probe + np.real(i2 * np.conjugate(i2)).T*ab['zeroLoss'][i]

    ab0={}
    for key in ab:
        ab0[key] = 0.
    chiIA = np.fft.fftshift (makechi1(phi,theta,wl,ab0, 0))#np.ones(chi2.shape)*2*np.pi/wl
    chiI = np.ones((sizeY, sizeX))
    chiI[mask]=0.
    i2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift (chiI)))
    ideal = np.real(i2 * np.conjugate(i2))

    probeF = np.fft.fft2(probe,probe.shape)+1e-12
    idealF = np.fft.fft2(ideal,probe.shape)
    fourier_space_division = idealF/probeF
    probeR = (np.fft.ifft2(fourier_space_division,probe.shape))


    return probe/sum(ab['zeroLoss']), np.real(probeR)


    

def DeconLR2(  Oimage, probe, tags, verbose = False):

    if len(Oimage) < 1:
        return Oimage
    print(Oimage.shape)
    if Oimage.shape != probe.shape:
        print('Wierdness ',Oimage.shape,' != ',probe.shape)
    ## Input Image ###
    # read the input image
    img = sitk.GetImageFromArray(Oimage, sitk.sitkFloat32)
    img = sitk.MirrorPad( img, [128] *2, [128]*2)
    
    size = img.GetSize();
    # perform the FFT
    source = sitk.ForwardFFT( sitk.Cast( img, sitk.sitkFloat32 ) )

    

    ### Kernel Image ###
    # Read the kernel image file
    kernel= sitk.GetImageFromArray(probe, sitk.sitkFloat32)
    # flip kernel about all axis
    #kernel = sitk.Flip( kernel, [1]*2 )

    # normalize the kernel to sum to ~1
    stats = sitk.StatisticsImageFilter();
    stats.Execute( kernel )
    kernel = sitk.Cast( kernel / stats.GetSum(), sitk.sitkFloat32 )

    upadding = [0]*2
    upadding[0] = int( math.floor( (size[0] - kernel.GetSize()[0])/2.0 ) )
    upadding[1] = int( math.floor( (size[1] - kernel.GetSize()[1])/2.0 ) )

    lpadding = [0]*2
    lpadding[0] = int( math.ceil( (size[0] - kernel.GetSize()[0])/2.0 ) )
    lpadding[1] = int( math.ceil( (size[1] - kernel.GetSize()[1])/2.0 ) )
    
    # pad the kernel to prevent edge artifacts
    kernel = sitk.ConstantPad( kernel, upadding, lpadding, 0.0 )
    
    # perform FFT on kernel
    responseFT = sitk.ForwardFFT( sitk.FFTShift( kernel ) )
    

    error = sitk.GetImageFromArray(np.ones(size), sitk.sitkFloat32 )
    est = sitk.GetImageFromArray(np.ones(size), sitk.sitkFloat32 )
    

    verbose = True
    dE = 100
    dest = 100
    i=0
    while abs(dest) > 0.0001 :#or abs(dE)  > .025:
        i += 1

        error = source / sitk.InverseFFT( est*responseFT )
        est = est * sitk.InverseFFT( error*responseFT )

        #dest = np.sum(np.power((est - est_old).real,2))/np.sum(est)*100
        #print(np.sum((est.real - est_old.real)* (est.real - est_old.real) )/np.sum(est.real)*100 )

        
        print(' LR Deconvolution - Iteration: {0:d} Error: {1:.2f} = change: {2:.5f}%, {3:.5f}%'.format(i,error_new,dE,abs(dest)))
    
        if i > 10:
            dE = dest =  0.0
            print('terminate')
    
# This task generates a restored image from an input image and point spread function (PSF) using the algorithm developed independently by Lucy (1974, Astron. J. 79, 745) and Richardson (1972, J. Opt. Soc. Am. 62, 55) and adapted for HST imagery by Snyder (1990, in Restoration of HST Images and Spectra, ST ScI Workshop Proceedings; see also Snyder, Hammoud, & White, JOSA, v. 10, no. 5, May 1993, in press). Additional options developed by Rick White (STScI) are also included.
#
# The Lucy-Richardson method can be derived from the maximum likelihood expression for data with a Poisson noise distribution. Thus, it naturally applies to optical imaging data such as HST. The method forces the restored image to be positive, in accord with photon-counting statistics.
#
# The Lucy-Richardson algorithm generates a restored image through an iterative method. The essence of the iteration is as follows: the (n+1)th estimate of the restored image is given by the nth estimate of the restored image multiplied by a correction image. That is,
#
#                            original data
#       image    = image    ---------------  * reflect(PSF) 
#            n+1        n     image * PSF
#                                  n

# where the *'s represent convolution operators and reflect(PSF) is the reflection of the PSF, i.e. reflect((PSF)(x,y)) = PSF(-x,-y). When the convolutions are carried out using fast Fourier transforms (FFTs), one can use the fact that FFT(reflect(PSF)) = conj(FFT(PSF)), where conj is the complex conjugate operator. 

def DeconLR(  Oimage, probe, tags, verbose = False):

    if len(Oimage) < 1:
        return Oimage
    print(Oimage.shape)
    if Oimage.shape != probe.shape:
        print('Wierdness ',Oimage.shape,' != ',probe.shape)
    probeC = np.ones((probe.shape), dtype = np.complex64)
    probeC.real = probe


    error = np.ones((Oimage.shape), dtype = np.complex64)
    est = np.ones((Oimage.shape), dtype = np.complex64)
    source= np.ones((Oimage.shape), dtype = np.complex64)
    source.real = Oimage
    
    responseFT =fftpack.fft2(probeC)



    
    if 'ImageScanned' in tags:
        ab = tags['ImageScanned']
    elif 'aberrations' in  tags:
        ab = tags['aberrations']
    if 'convAngle' not in ab:
        ab['convAngle'] = 30
    ApAngle=ab['convAngle']/1000.0
    
    E0=  float( ab['EHT'])  

    def get_wl(E0):
        h=6.626*10**-34
        m0=9.109*10**-31
        eV=1.602*10**-19*E0
        C=2.998*10**8
        return h/np.sqrt(2*m0*eV*(1+eV/(2*m0*C**2)))*10**9

    wl=get_wl(E0)
    ab['wavelength'] = wl
    
    over_d = 2* ApAngle / wl

    dx = tags['pixel_size']
    dk = 1.0/ float(tags['FOV'])
    ScreenWidth = 1/dx

    
    aperture = np.ones((Oimage.shape), dtype = np.complex64)
    # Mask for the aperture before the Fourier transform
    N = Oimage.shape[0]
    sizeX = Oimage.shape[0]
    sizeY = Oimage.shape[1]
    App_ratio = over_d/ScreenWidth*N

    
    Thetax = np.array((-sizeX/2.+ np.arange(sizeX)))
    Thetay = np.array((-sizeY/2.+ np.arange(sizeY)))
    Txv, Tyv = np.meshgrid(Thetax, Thetay)

    tp1 = Txv**2 + Tyv**2 >= (App_ratio)**2
    aperture[tp1.T] = 0.
    print( App_ratio, ScreenWidth, dk)

    
    

    
    dE = 100
    dest = 100
    i=0
    while abs(dest) > 0.0001 :#or abs(dE)  > .025:
        i += 1

        error_old = np.sum(error.real)
        est_old = est.copy()
        error = source / np.real(fftpack.fftshift(fftpack.ifft2(fftpack.fft2(est)*responseFT)))
        est = est * np.real(fftpack.fftshift(fftpack.ifft2(fftpack.fft2(error)*np.conjugate(responseFT))))
        #est = est_old * est
        #est =  np.real(fftpack.fftshift(fftpack.ifft2(fftpack.fft2(est)*fftpack.fftshift(aperture) )))
    
        error_new = np.real(np.sum(np.power(error,2)))-error_old
        dest = np.sum(np.power((est - est_old).real,2))/np.sum(est)*100
        #print(np.sum((est.real - est_old.real)* (est.real - est_old.real) )/np.sum(est.real)*100 )

        if error_old!=0:
            dE = error_new / error_old *1.0
            
        else:
            dE = error_new
    
        if verbose:
            print(' LR Deconvolution - Iteration: {0:d} Error: {1:.2f} = change: {2:.5f}%, {3:.5f}%'.format(i,error_new,dE,abs(dest)))
    
        if i > 1000:
            dE = dest =  0.0
            print('terminate')
    
    print('\n Lucy-Richardson deconvolution converged in '+str(i)+ '  Iterations')
    est2 =  np.real(fftpack.ifft2(fftpack.fft2(est)*fftpack.fftshift(aperture) ))
    #plt.imshow(np.real(np.log10(np.abs(fftpack.fftshift(fftpack.fft2(est)))+1)+aperture), origin='lower',)
    #plt.show()
    print(est2.shape)
    return est2


##########################################
# Functions Used
##########################################
 
def MakeProbeG(sizeX,sizeY,widthi,xi,yi):
    sizeX = (sizeX/2)
    sizeY = (sizeY/2)
    width = 2*widthi**2
    x, y = np.mgrid[-sizeX:sizeX, -sizeY:sizeY]
    g = np.exp(-((x-xi)**2/float(width)+(y-yi)**2/float(width)))
    probe = g/g.sum()
        
    return probe

def MakeLorentz(sizeX,sizeY,width,xi,yi):
    sizeX = np.floor(sizeX/2)
    sizeY = np.floor(sizeY/2)
    gamma = width
    x, y = np.mgrid[-sizeX:sizeX, -sizeY:sizeY]
    g = gamma/(2*np.pi)/ np.power( ((x-xi)**2+(y-yi)**2+gamma**2),1.5)
    probe = g/g.sum()
        
    return probe



def ZLPWeight():
    x = np.linspace(-0.5,.9, 29)
    y = [0.0143,0.0193,0.0281,0.0440,0.0768,0.1447,0.2785,0.4955,0.7442,0.9380,1.0000,0.9483,0.8596,0.7620,0.6539,0.5515,0.4478,0.3500,0.2683,0.1979,0.1410,0.1021,0.0752,0.0545,0.0401,0.0300,0.0229,0.0176,0.0139]
    
    return (x,y)


    ##
    # All atom detection is done here
    # Everything is in unit of pixel!!
    ##


def findatoms(image, tags):
    """
    ######################################
    # Find atoms 
    ######################################
    """
    
    image = image-image.min()
    image = image/image.max()

    if 'sigma_min' not in tags:
        tags['sigma_min'] = 0.1
    if 'resolution' not in tags:
        tags['resolution'] = 0.1

    if 'ROIsize' not in tags:
        tags['ROIsize'] = 100.


    res = tags['resolution'] / tags['pixel_size']#* tags['ROIsize']/100.
    print('res',res)
    coordinates = peak_local_max(image, min_distance=int(res/2), threshold_rel=tags['sigma_min'], exclude_border =True)
    print('coor',len( coordinates))
    """
       peak_local_max(image, min_distance=10, threshold_abs=0, threshold_rel=0.1,
                   exclude_border=True, indices=True, num_peaks=np.inf,
                   footprint=None, labels=None):
    
        Find peaks in an image, and return them as coordinates or a boolean array.
        Peaks are the local maxima in a region of `2 * min_distance + 1
        
        (i.e. peaks are separated by at least `min_distance`).
        NOTE: If peaks are flat (i.e. multiple adjacent pixels have identical
        intensities), the coordinates of all such pixels are returned.
        """
        



    
    # We calculate the radius in pixel of a round area in which atoms are evaluated
    sc = tags['pixel_size']
    r= tags['resolution']/sc*tags['ROIsize']/100./2.
    tags['radius'] = r

    #######################################
    # Now we determine intensity #
    #######################################

    ###
    # Make a circular mask for integration of atom intensity
    ###
    rr = int(r+0.5)
    mask = np.zeros((2*rr+1,2*rr+1))
    
    for i in range (2*rr+1):
        for j  in range (2*rr+1):
            if (i-rr)**2+(j-rr)**2<rr**2+0.1:
                mask[i,j]=1

                
                
        
                    
    ###
    # Determine  pixel position and intensity  of all atoms 
    ###
    atoms = []
    for i in range(len( coordinates)):
        x,y = coordinates[i]
        
        if x>rr and y>rr and x<image.shape[1]-rr and y<image.shape[0]-rr:
            
            area = image[x-rr:x+rr+1,y-rr:y+rr+1]
            arr = area*mask
            atoms.append((x,y,rr, arr.sum(), arr.max()))

    print(' Detected ', len(atoms), ' atoms')
    atoms.sort()
    return atoms

# sort corners in counter-clockwise direction
def TurningFunction(corners,points):
    # calculate centroid of the polygon
    corners1 = np.array(points[corners])
    corners2 = np.roll(corners1,1)
    
    corners0 = np.roll(corners1,-1)

    v= corners1-corners0
    an = (np.arctan2(v[:,0],v[:,1]) + 2.0 * math.pi)% (2.0 * math.pi)/np.pi*180
    print(corners1)
    print('an',an,v)
    print(4*180/6)

    angles = []
    for i in range(len(corners1)):
        A = corners1[i] - corners0[i]
        B = corners1[i] - corners2[i] 
        num = np.dot(A, B)
        denom = np.linalg.norm(A) * np.linalg.norm(B)
        angles.append(np.arccos(num/denom) * 180 / np.pi)
        
    return angles


    
 
def PolygonSort2(corners,points):
    """
    # sort corners in counter-clockwise direction
    input:
            corners are indices in points array
            points is list or array of points
    output: 
            cornersWithAngles
    """
    # calculate centroid of the polygon
    n = len(corners) # of corners
    cx = float(sum(x for x, y in points[corners])) / n
    cy = float(sum(y for x, y in points[corners])) / n
    
    # create a new list of corners which includes angles
    # angles from the positive x axis
    cornersWithAngles = []
    for i in corners:
        x,y = points[i]       
        an = (math.atan2(y - cy, x - cx) + 2.0 * math.pi)% (2.0 * math.pi)
        cornersWithAngles.append([i, math.degrees(an)])
    
    #sort it using the angles
    cornersWithAngles.sort(key = lambda tup: tup[1])
    

    return  cornersWithAngles


def PolygonsInner(indices, points):
    pp = np.array(points)[indices,:]
    # Determine inner angle of polygon
    # Generate second array which is shifted by one 
    pp2 = np.roll(pp,1,axis=0)
    # and subtract it from former: this is now a list of vectors
    p_vectors = pp-pp2

    #angles of vectors with respect to positive x-axis
    ang = np.arctan2(p_vectors[:,1],p_vectors[:,0])/np.pi*180+360 % 360
    # shift angles array by one  
    ang2 = np.roll(ang,-1,axis=0)

    #difference of angles is outer angle but we want the inner (inner + outer = 180) 
    inner_angles = (180-(ang2-ang)+360 )% 360
    
    return inner_angles


 # sort corners in counter-clockwise direction
def PolygonSort(corners):
    # calculate centroid of the polygon
    n = len(corners) # of corners
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    
    # create a new list of corners which includes angles
    cornersWithAngles = []
    for x, y in corners:
        an = (math.atan2(y - cy, x - cx) + 2.0 * math.pi)% (2.0 * math.pi)
        cornersWithAngles.append((x, y, math.degrees(an)))
    
    #sort it using the angles
    cornersWithAngles.sort(key = lambda tup: tup[2])

    return  cornersWithAngles




def PolygonArea(corners):
    """
    # Area of Polygon using Shoelace formula
    # http://en.wikipedia.org/wiki/Shoelace_formula
    # FB - 20120218
    # corners must be ordered in clockwise or counter-clockwise direction
    """
    n = len(corners) # of corners
    area = 0.0
    C_x =0
    C_y =0
    for i in range(n):
        j = (i + 1) % n
        nn = corners[i][0] * corners[j][1] - corners[j][0] * corners[i][1]
        area += nn
        C_x +=  (corners[i][0] + corners[j][0])*nn
        C_y +=  (corners[i][1] + corners[j][1])*nn

    area = abs(area) / 2.0

    # centeroid or arithmetic mean 
    C_x = C_x/(6*area) 
    C_y = C_y/(6* area)

    return (area), C_x, C_y


def PolygonAngles( corners):
    angles = []
    # calculate centroid of the polygon
    n = len(corners) # of corners
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    # create a new list of angles
    #print (cx,cy)
    for x, y in corners:
        an = (math.atan2(y - cy, x - cx) + 2.0 * math.pi)% (2.0 * math.pi)
        angles.append((math.degrees(an)))

    return angles






def voronoi_tags(vor):
    sym = {}
    sym['voronoi'] = vor
    sym['vertices'] = vor.vertices #(ndarray of double, shape (nvertices, ndim)) Coordinates of the Voronoi vertices.
    sym['ridge_points'] = vor.ridge_points #ridge_points 	(ndarray of ints, shape (nridges, 2)) Indices of the points between which each Voronoi ridge lies.
    sym['ridge_vertices'] = vor.ridge_vertices #ridge_vertices 	(list of list of ints, shape (nridges, *)) Indices of the Voronoi vertices forming each Voronoi ridge.
    sym['regions'] = vor.regions #regions 	(list of list of ints, shape (nregions, *)) Indices of the Voronoi vertices forming each Voronoi region. -1 indicates vertex outside the Voronoi diagram.
    sym['point_region'] = vor.point_region #point_region 	(list of ints, shape (npoints)) Index of the Voronoi region for each input point. If qhull option “Qc” was not specified, the list will contain -1 for points that are not associated with a Voronoi region.

    
    points = vor.points
    nnTree = KDTree(points)
    
    rim = []
    regions=[]

    ###
    # We get all the vertice length
  
    lengths =[]
    for vertice in  vor.ridge_vertices:
        if not(-1 in vertice):
            p1 = vor.vertices[vertice[0]]
            p2 = vor.vertices[vertice[1]]
            lengths.append(np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1] )**2))
                
    sym['lengths'] = lengths
    sym['median lengths'] = np.median(lengths)
    sym['Min Voronoi Edge'] = np.median(lengths)/1.5
    #print ('median lengths', np.median(lengths))
    #print ('Min Voronoi Edge',np.median(lengths)/1.5)
    cornersHist=[]
    nnHist = []
    nnDistHist =[]
    angleHist = []
    areaHist=[]
    deviationHist =[]

    for i, region in enumerate(vor.point_region):
        x,y = points[i]
        sym[str(i)]={}
        vertices = vor.regions[region]

        ###
        # We get all the rim atoms
        ###

        #if all(v >= 0  and all(vor.vertices[v] >0) and all(vor.vertices[v]<tags['data'].shape[0]) for v in vertices):
        if all(v >= 0  and all(vor.vertices[v] >0) for v in vertices):
             # finite regions only now
            # negative and too large vertices (corners) are excluded

            regions.append(vertices)
            poly = []
            for v in vertices:
                poly.append(vor.vertices[v])

            area, cx,cy = PolygonArea(poly)
            cx = abs(cx)
            cy = abs(cy)

            angles = PolygonAngles(poly)
            angleHist.append(angles)
            areaHist.append(area)
            deviationHist.append(np.sqrt((cx-x)**2+ (cy-y)**2))

            sym[str(i)]['xy'] = [x, y]
            sym[str(i)]['geometric'] = [cx, cy]
            sym[str(i)]['area'] = area

            sym[str(i)]['angles'] = angles
            sym[str(i)]['off center'] = [cx-x, cy-y]

            sym[str(i)]['position'] = 'inside'
            sym[str(i)]['corner'] = vertices
            sym[str(i)]['vertices']=poly
            sym[str(i)]['corners'] = len(vertices)
            cornersHist.append(len(vertices))
            nn = 0
            nnVor = []
            length = []
            for j in range(len(vertices)):
                k = (j+1) % len(vertices)
                p1 = vor.vertices[vertices[j]]
                p2 = vor.vertices[vertices[k]]
                leng = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1] )**2)
                length.append(leng)
                sym[str(i)]['length'] = length
                if leng > sym['Min Voronoi Edge']:        
                    nn +=1
                    nnVor.append(vertices[j])
                sym[str(i)]['length'] = length    
            nnP = nnTree.query(points[i],k = nn+1)
            sym [str(i)]['neighbors'] = []
            sym [str(i)]['nn Distance'] = []
            sym [str(i)]['nn']=nn
            if nn>0:
                nnHist.append(nn)   
                for j in range (1,len(nnP[0])):
                    sym [str(i)]['nn Distance'].append(nnP[0][j])
                    sym [str(i)]['neighbors'].append(nnP[1][j])
                    nnDistHist.append(nnP[0][j])
            else:
                rim.append(i)
                sym[str(i)]['position'] = 'rim'
                sym[str(i)]['corners'] = 0
                print('weird nn determination',i)

        else:
            rim.append(i)
            sym[str(i)]['position'] = 'rim'
            sym[str(i)]['corners'] = 0
            sym[str(i)]['xy'] = [x, y]


    sym['average corners']= np.median(cornersHist)
    sym['average area']= np.median(areaHist)
    sym['num atoms at rim']= len(rim)
    sym['num voronoi']= len(points)-len(rim)
    sym['Median Coordination']= np.median(nnHist)
    sym['Median NN Distance']= np.median(nnDistHist)

    sym['Hist corners']= (cornersHist)
    sym['Hist area']= areaHist
    sym['atoms at rim']= (rim)
    sym['Hist Coordination']= (nnHist)
    sym['Hist NN Distance']= (nnDistHist)
    sym['Hist deviation']= (deviationHist)


    return sym
    #print ('average corners', np.median(cornersHist))


def defineSymmetry(tags):

    #make dictionary to store
    if 'symmetry' in tags:
        tags['symmetry'].clear()

    tags['symmetry'] = {}
    sym = tags['symmetry']
    if 'latticeType' in tags:
        latticeTypes = ['None', 'Find Lattice',  'hexagonal', 'honeycomb', 'square', 'square centered',
                'diamond', 'fcc']
        sym['lattice']=latticeTypes[tags['latticeType']]

    sym['number of atoms'] = len(self.tags['atoms'])
    
    points = []
    for i in range(sym['number of atoms']):            
        sym[str(i)] = {}
        sym[str(i)]['index']= i
        sym[str(i)]['x'] = self.tags['atoms'] [i][0]
        sym[str(i)]['y'] = self.tags['atoms'] [i][1]
        sym[str(i)]['intensity'] = self.tags['atoms'] [i][3]
        sym[str(i)]['maximum'] = self.tags['atoms'] [i][4]
        sym[str(i)]['position'] = 'inside'
        sym[str(i)]['Z'] = 0
        sym[str(i)]['Name'] = 'undefined'
        sym[str(i)]['Column'] = -1
        
        points.append([int(sym[str(i)]['x']+0.5),int(sym[str(i)]['y']+0.5)])

    self.points = points.copy()
    


def voronoi2(tags, atoms):

    sym = tags['symmetry']
    points = []
    
    for i in range(sym['number of atoms']):            
        points.append([int(sym[str(i)]['x']+0.5),int(sym[str(i)]['y']+0.5)])


    #points = np.array(atoms[:][0:2])
    vor = sp.Voronoi(points)

    
    sym['voronoi'] = vor

    nnTree = sp.KDTree(points)
    
    rim = []
    regions=[]

    ###
    # We get all the vertice length
  
    lengths =[]
    for vertice in  vor.ridge_vertices:
        if all(v >= 0 for v in vertice):
            p1 = vor.vertices[vertice[0]]
            p2 = vor.vertices[vertice[1]]
            lengths.append(np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1] )**2))
                
    sym['lengths'] = lengths
    sym['median lengths'] = np.median(lengths)
    sym['Min Voronoi Edge'] = np.median(lengths)/1.5
    #print ('median lengths', np.median(lengths))
    #print ('Min Voronoi Edge',np.median(lengths)/1.5)
    cornersHist=[]
    nnHist = []
    nnDistHist =[]
    angleHist = []
    areaHist=[]
    deviationHist =[]

    for i, region in enumerate(vor.point_region):
        x,y = points[i]

        vertices = vor.regions[region]

        ###
        # We get all the rim atoms
        ###

        if all(v >= 0  and all(vor.vertices[v] >0) and all(vor.vertices[v]<tags['data'].shape[0]) for v in vertices):
            # finite regions only now
            # negative and too large vertices (corners) are excluded

            regions.append(vertices)
            poly = []
            for v in vertices:
                poly.append(vor.vertices[v])

            area, cx,cy = PolygonArea(poly)
            cx = abs(cx)
            cy = abs(cy)

            angles = PolygonAngles(poly)
            angleHist.append(angles)
            areaHist.append(area)
            deviationHist.append(np.sqrt((cx-x)**2+ (cy-y)**2))

            sym[str(i)]['xy'] = [x, y]
            sym[str(i)]['geometric'] = [cx, cy]
            sym[str(i)]['area'] = area

            sym[str(i)]['angles'] = angles
            sym[str(i)]['off center'] = [cx-x, cy-y]

            sym[str(i)]['position'] = 'inside'
            sym[str(i)]['corner'] = vertices
            sym[str(i)]['vertices']=poly
            sym[str(i)]['corners'] = len(vertices)
            cornersHist.append(len(vertices))
            nn = 0
            nnVor = []
            length = []
            for j in range(len(vertices)):
                k = (j+1) % len(vertices)
                p1 = vor.vertices[vertices[j]]
                p2 = vor.vertices[vertices[k]]
                leng = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1] )**2)
                length.append(leng)
                sym[str(i)]['length'] = length
                if leng > sym['Min Voronoi Edge']:        
                    nn +=1
                    nnVor.append(vertices[j])
                sym[str(i)]['length'] = length    
            nnP = nnTree.query(points[i],k = nn+1)
            sym [str(i)]['neighbors'] = []
            sym [str(i)]['nn Distance'] = []
            sym [str(i)]['nn']=nn
            if nn>0:
                nnHist.append(nn)   
                for j in range (1,len(nnP[0])):
                    sym [str(i)]['nn Distance'].append(nnP[0][j])
                    sym [str(i)]['neighbors'].append(nnP[1][j])
                    nnDistHist.append(nnP[0][j])
            else:
                rim.append(i)
                sym[str(i)]['position'] = 'rim'
                sym[str(i)]['corners'] = 0
                print('weird nn determination',i)

        else:
            rim.append(i)
            sym[str(i)]['position'] = 'rim'
            sym[str(i)]['corners'] = 0
            sym[str(i)]['xy'] = [x, y]


    sym['average corners']= np.median(cornersHist)
    sym['average area']= np.median(areaHist)
    sym['num atoms at rim']= len(rim)
    sym['num voronoi']= len(points)-len(rim)
    sym['Median Coordination']= np.median(nnHist)
    sym['Median NN Distance']= np.median(nnDistHist)

    sym['Hist corners']= (cornersHist)
    sym['Hist area']= areaHist
    sym['atoms at rim']= (rim)
    sym['Hist Coordination']= (nnHist)
    sym['Hist NN Distance']= (nnDistHist)
    sym['Hist deviation']= (deviationHist)



    #print ('average corners', np.median(cornersHist))



def atomRefine(image, atoms, tags, maxDist = 2):
    
    rr = int(tags['radius']+0.5) # atom radius
    print('using radius ',rr, 'pixels')
    
    pixels = np.linspace(0,2*rr,2*rr+1)-rr
    x,y = np.meshgrid(pixels,pixels);
    mask = (x**2+y**2) < rr**2 #
    
    def func(params,  xdata, ydata):
        width = ydata.shape[0]/2
        Gauss_width = params[0]
        x0 = params[1]
        y0 = params[2]
        inten = params[3]

        x, y = np.mgrid[-width:width, -width:width]

        gauss = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / Gauss_width**2)*inten
        #self.img1b.setImage(gauss)
        return (ydata - gauss).flatten()


    ###
    # Determine sub pixel position and intensity  of all atoms within intensity range
    ###
    guess  = [rr, 0.0, 0.0 , 1]
    pout = [0.0, 0.0, 0.0 , 0.0]
    newatoms = []

    #tags['symmetry'] = {}
    sym = {}
    sym['number_of_atoms'] = len(atoms)
    Z=[]
    Name = []
    Column = []
    position = []
    intensity_area = []
    maximum_area = []
    Gauss_width = []
    Gauss_amplitude = []
    Gauss_volume = []

    for i in range(len( atoms)):
        
        y,x = atoms[i][0:2]
        x = int(x)
        y = int(y)
        append = False
        
        
        area = image[x-rr:x+rr+1,y-rr:y+rr+1]
                
        sym[str(i)] = {}
        sym[str(i)]['index']= i
        sym[str(i)]['x'] = x
        sym[str(i)]['y'] = y
        sym[str(i)]['Z'] = 0
        sym[str(i)]['Name'] = 'undefined'
        sym[str(i)]['Column'] = -1

        append = False
        
        if (x-rr) < 0 or y-rr <0 or x+rr+1 > image.shape[0] or y+rr+1 > image.shape[1]:
            sym[str(i)]['position'] = 'outside'
            sym[str(i)]['intensity area'] = 0 
            sym[str(i)]['maximum area'] = 0
        else:
            sym[str(i)]['position'] = 'inside'
            sym[str(i)]['intensity area'] = (area*mask).sum()
            sym[str(i)]['maximum area'] = (area*mask).max()
        
        if tags['MaxInt']>0:
            if area.sum()< tags['MaxInt']:                    
                if area.sum() > tags['MinInt']:
                    append = True
        elif area.sum()> tags['MinInt']:
            append = True
        
        if append: ## If possible do a Gaussian fit and update the x and y 
            if (x-rr) < 0 or y-rr <0 or x+rr+1 > image.shape[0] or y+rr+1 > image.shape[1]:
                pout[0] = 0 # width
                pout[1] = 0 # dx
                pout[2] = 0 # dy
                pout[3] = 0 # amplitude
            else:
                pout, res =  leastsq(func, guess, args=(area, area))
            # shift cannot be larger than two pixels
            if (abs(pout[1])> maxDist) or (abs(pout[2])> maxDist):
                #print(i,x,y,pout[1],pout[2])
                pout[0] = 0 # width
                pout[1] = 0 # dx
                pout[2] = 0 # dy
                pout[3] = 0 # amplitude

            sym[str(i)]['x'] = x+pout[1]
            sym[str(i)]['y'] = y+pout[2]

            volume = 2* np.pi * pout[3] * pout[0]*pout[0]

            newatoms.append([y+pout[2]+1, x+pout[1]+1])# ,pout[0],  volume)) #,pout[3]))

            sym[str(i)]['Gauss width'] =  pout[0]
            sym[str(i)]['Gauss amplitude'] = pout[3]
            sym[str(i)]['Gauss volume'] = volume
            
        #x.append(sym[str(i)]['x'])
        #y.append(sym[str(i)]['y'])
        Z.append(sym[str(i)]['Z'])
        Name.append(str(sym[str(i)]['Name']))
        Column.append(sym[str(i)]['Column'])
        if sym[str(i)]['position'] == 'inside':
            position.append(1)
        else:
            position.append(0)

        intensity_area.append(sym[str(i)]['intensity area'])
        maximum_area.append(sym[str(i)]['maximum area'])
        Gauss_width.append(sym[str(i)]['Gauss width'])
        Gauss_amplitude.append(sym[str(i)]['Gauss amplitude'])
        Gauss_volume.append(sym[str(i)]['Gauss volume'])
    tags2 = {}
    tags2['number_of_atoms'] = len(atoms)
    
    tags2['Z'] = np.array(Z)
    #out_tags2['Name'] = np.array(Name)
    tags2['Column'] = np.array(Column)
    tags2['position'] = np.array(position)
    tags2['intensity_area'] = np.array(intensity_area)
    tags2['maximum_area'] = np.array(maximum_area)

    tags2['Gauss_width'] = np.array(Gauss_width)
    tags2['Gauss_amplitude'] = np.array(Gauss_amplitude)
    tags2['Gauss_volume'] = np.array(Gauss_volume)        
    tags2['atoms'] = newatoms
    tags2['sym'] = sym
    return tags2

