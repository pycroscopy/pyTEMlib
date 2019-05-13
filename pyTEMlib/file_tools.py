##################################
#
# 2018 01 31 Included Nion Swift files to be opened
#
###############################

import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
try:
    import h5py
except:
    print('h5py not installed, cannot read Nion files')
    h5py_installed = False

import pyUSID as usid
import json
import struct
import ipywidgets as ipyw
import sys, os

# =============================================================
#   Include Quantifit Libraries                                      #
# =============================================================


import pyTEMlib.dm3lib_v1_0b as dm3 # changed this dm3 reader library to support SI and EELS

from pyTEMlib.config_dir import config_path

def savefile_dialog(ftype ="All files (*)"):
    from PyQt5 import QtGui, QtWidgets
    # Determine last path used
    try:
        fp = open(config_path+'\path.txt','r')
        path = fp.read()
        fp.close()
    except:
        path = ''
    
    if len(path)<2:
        path = '.'

    fname, file_filter = QtWidgets.QFileDialog.getSaveFileName(None, "Select a file...", path, filter=ftype)
    
    if len(fname) > 1:
        
        fp = open(config_path+'\path.txt','w')
        path, fileName = os.path.split(fname)
        fp.write(path)
        fp.close()
    return str(fname)

def openfile_dialog(ftype ="All files (*)", multiple_files = False):
    from PyQt5 import QtGui, QtWidgets

    """
    Opens a File dialog which is used in open_file() function
    This functon uses QWidgets from PyQt5.
    The usage of this function requries that you start an QApplication first with:
        app = QtWidgets.QApplication([dir])
    However, you can do that only once without crashing

    The file looks first for a path.txt file for the last directory you used.

    Parameters
    ----------
    ftype : string of the fie type filter 
    
    Returns
    -------
    filename : full filename with absolute path and extension as a string

    Examples
    --------
    >>> from PyQt5 import QtGui, QtWidgets
    >>> import sys, os
    >>> sys.path.append('../libQF')
    >>> from config_dir import config_path
    >>> import file_tools as ft
    >>>
    >>> app = QtWidgets.QApplication([dir])
    >>> filename = ft.openfile_dialog()
    >>> 
    >>> print(filename)

    """
    
    
    # Determine last path used
    try:
        fp = open(config_path+'\path.txt','r')
        path = fp.read()
        fp.close()
    except:
        path = ''
    
    if len(path)<2:
        path = '.'

    if multiple_files:
        fnames, file_filter = QtWidgets.QFileDialog.getOpenFileNames(None, "Select a file...", path, filter=ftype)
        if len(fnames) >0:
            fname = fnames[0]
        else:
            return 
    else:
        fname, file_filter = QtWidgets.QFileDialog.getOpenFileName(None, "Select a file...", path, filter=ftype)
    
    if len(fname) > 1:
        
        fp = open(config_path+'\path.txt','w')
        path, fileName = os.path.split(fname)
        fp.write(path)
        fp.close()
    else:
        return ''
        
    if multiple_files:
         return fnames
    else:
        return str(fname)



def getDictionary(intags):
    
    tags = {}
    tags['type']='DM'
    tags['file version'] = 3
    
    for line in intags.keys():
        if '.ImageList.1.' in line:
            keys = line[17:].split('.')
            tags_before = tags
            for i in range(len(keys)-1):
                if keys[i] not in tags_before:
                    tags_before[keys[i]] = {}
                tags_before = tags_before[keys[i]]
            
            tags_before[keys[-1]] = intags[line]
        if 'AnnotationGroupList' in line:
            if 'Label' in line:            
                if  self.__tagDict[line] == 'Spectrum Image':
                     tags['spectrum_image'] ={}
                     base = line[0:-5]
                     tags['spectrum_image']['rectangle'] = self.__tagDict[base+'Rectangle']
            
    return tags

def save_file(tags):
    file_name = savefile_dialog('QF files ( *.qf3);;All files (*)')
    if not (file_name[-4:]  == '.qf3'):
        file_name = file_name + '.qf3'
    print(file_name)
    f = open(file_name, 'wb')
    pickle.dump(tags, f)
    f.close()



def get_dictionary_from_pyUSID(current_channel):
    tags = {}
    current_channel_tags = dict(current_channel.attrs)
    for key in current_channel_tags:
        if 'original' not in key:
            tags[key]=current_channel_tags[key]
    tags['data_type'] = current_channel['data_type'][()]
    if tags['data_type']== 'EELS_spectrum':
        tags['data'] = current_channel['Raw_Data'][0,:]
        tags['dispersion'] = current_channel['spectral_scale_x'][()]
        tags['units'] = current_channel['spectral_units_x'][()]
        tags['offset'] = current_channel['spectral_origin_x'][()]
        tags['spectrum_length'] = float(current_channel['spectral_size_x'][()])
        tags['energy_scale'] = np.arange(tags['spectrum_length'])*tags['dispersion']+tags['offset']
    elif tags['data_type']== 'image':
        tags['data'] = np.reshape(current_channel['Raw_Data'][:, 0], (current_channel['spatial_size_x'][()],current_channel['spatial_size_y'][()]))
        tags['image_size_x'] = current_channel['spatial_size_x'][()]
        tags['image_size_y'] = current_channel['spatial_size_y'][()]
        tags['pixel_size_x'] = current_channel['spatial_scale_x'][()]
        tags['pixel_size_y'] = current_channel['spatial_scale_y'][()]
        tags['FOV_x'] = tags['pixel_size_y'][()] * tags['image_size_x'][()]
        tags['FOV_y'] = tags['pixel_size_y'][()] * tags['image_size_y'][()]
        tags['extent']=(0,tags['FOV_x'],tags['FOV_y'],0)
        tags['units']=current_channel['spatial_units'][()]
        
        
                        
    elif tags['data_type']== 'spectrum_image':
        tags['cube'] = np.reshape(current_channel['Raw_Data'][:, :], (current_channel['spatial_size_x'][()],current_channel['spatial_size_y'][()],current_channel['spectral_size_x'][()]))
        tags['data'] = tags['cube'].sum(axis=2)
        tags['image_size_x'] = current_channel['spatial_size_x'][()]
        tags['image_size_y'] = current_channel['spatial_size_y'][()]
        tags['pixel_size_x'] = current_channel['spatial_scale_x'][()]
        tags['pixel_size_y'] = current_channel['spatial_scale_y'][()]
        tags['FOV_x'] = tags['pixel_size_y'][()] * tags['image_size_x'][()]
        tags['FOV_y'] = tags['pixel_size_y'][()] * tags['image_size_y'][()]
        tags['extent']=(0,tags['FOV_x'],tags['FOV_y'],0)
        if current_channel['spatial_units'][()] == '':
            tags['units']='nm'
        else:
            tags['units']=current_channel['spatial_units'][()]
        
        tags['dispersion'] = current_channel['spectral_scale_x'][()]
        tags['spectral_units'] = current_channel['spectral_units_x'][()]
        tags['offset'] = current_channel['spectral_origin_x'][()]
        tags['spectrum_length'] = float(current_channel['spectral_size_x'][()])
        tags['energy_scale'] = np.arange(tags['spectrum_length'])*tags['dispersion']+tags['offset']
    elif tags['data_type']== 'image_stack':
        tags['image_stack'] = np.array(current_channel['image_stack'][()])#[:, ], (current_channel['spatial_size_x'][()],current_channel['spatial_size_y'][()],current_channel['spatial_size_y'][()]))
        tags['data'] = np.reshape(current_channel['Raw_Data'][:, 0], (current_channel['spatial_size_x'][()],current_channel['spatial_size_y'][()]))
        tags['image_size_x'] = current_channel['spatial_size_x'][()]
        tags['image_size_y'] = current_channel['spatial_size_y'][()]
        tags['pixel_size_x'] = current_channel['spatial_scale_x'][()]
        tags['pixel_size_y'] = current_channel['spatial_scale_y'][()]
        tags['FOV_x'] = tags['pixel_size_y'][()] * tags['image_size_x'][()]
        tags['FOV_y'] = tags['pixel_size_y'][()] * tags['image_size_y'][()]
        tags['extent']=(0,tags['FOV_x'],tags['FOV_y'],0)
        tags['units']=current_channel['spatial_units'][()]
        
    return tags

def h5open_file(filename = None):
    """
    Opens a file if the extension is .hf5, .dm3 or .dm4
    If no filename is provided the qt5 open_file windows opens
    if you used the magic comand: "% gui qt"

    Everything will be stored in a pyUSID style hf5 file.

    Subbroutines used:
        - dm_to_pyUSID
            - get_main_tags
            - get_additional tags
            - several pyUSID io fucntions
    

    """
    
    if filename == None:
        filename = openfile_dialog('TEM files (*.dm3 *.qf3 *.ndata *.h5 *.hf5);;pyUSID files (*.hf5);;QF files ( *.qf3);;DM files (*.dm3);;Nion files (*.ndata *.h5);;All files (*)')
        if filename == '':
            return
    path, file_name = os.path.split(filename)
    basename, extension = os.path.splitext(file_name)
     
    if extension ==  '.hf5':
        return h5py.File(filename, 'a')
    if extension  in ['.dm3','.dm4']:
        h5_file = dm_to_pyUSID(filename)
        return h5_file
    else:
        print('file type not handled yet.')
        return


def h5log_calculation(h5_file,current_channel,tags):
    measurement_group = h5_file[current_channel.name.split('/')[1]]
    i = 0
    for key in measurement_group:
        if 'Calculation'in key:
            i+=1
    name = f'Calculation_{i:03d}'
    if 'data' not in tags:
        print('no data to log')
        return
    
    calc_grp = measurement_group.create_group(name)
    calc_grp['time_stamp']= usid.io.io_utils .get_time_stamp()
    for key in tags:
        calc_grp[key]= tags[key]

    h5_file.flush()
    return calc_grp

def h5add_measurement(h5_file,current_channel,title):
    new_measurement_group = usid.io.hdf_utils.create_indexed_group(h5_file,'Measurement')

    return new_measurement_group




def h5add_channels(h5_file,current_channel,title):
    file_filters = 'TEM files (*.dm3 *.qf3 *.ndata *.h5 *.hf5);;pyUSID files (*.hf5);;QF files ( *.qf3);;DM files (*.dm3);;Nion files (*.ndata *.h5);;All files (*)'
    filenames = openfile_dialog(file_filters, multiple_files=True)
    if len(filenames) == 0:
        return current_channel
    for file in filenames:
        current_channel = h5add_channel(h5_file,current_channel,title,filename = file)
    return current_channel    

     

    
def h5add_channel(h5_file,current_channel,title,filename=None):

    #Open file
   
    if filename == None:
        file_filters = 'TEM files (*.dm3 *.qf3 *.ndata *.h5 *.hf5);;pyUSID files (*.hf5);;QF files ( *.qf3);;DM files (*.dm3);;Nion files (*.ndata *.h5);;All files (*)'
        filename = openfile_dialog(file_filters)
        if filename == '':
            return
    path, file_name = os.path.split(filename)
    basename, extension = os.path.splitext(file_name)
    time_last_modified = os.path.getmtime(filename)
    
    #if extension ==  '.hf5':
    #    return h5py.File(filename, 'a')
    if extension !=  '.dm3':
        print('file type is not handled yet')
    
    ### 
    # Read DM file and extract the main information
    ###
    si = dm3.DM3(filename)
    main_tags, channel_tags, data_tags  = get_main_tags(si)
    
    ###
    # Open pyUSID file and write data and main meta-data
    ###
    
    tran = usid.NumpyTranslator()
    h5_file_name = filename[:-4]+'.hf5'
    
    quantity = 'distance'
    units = 'nm'
    if current_channel == None:
        return
        
    else:
        '''
        measurement_group = h5_file[current_channel.name.split('/')[1]]
        i = 0
        for key in measurement_group:
            if 'Channel'in key:
                i+=1
        name = f'Channel_{i:03d}'
        if 'rawData' not in data_tags:
            print('no data to add')
            return
        '''
        name = usid.io.hdf_utils.assign_group_index(measurement_group,'Channel')
        
        current_channel = measurement_group.create_group(name)
        _ = usid.io.hdf_utils.write_main_dataset(current_channel, data_tags['rawData'], 'Raw_Data', 
                                                 'distance', 'nm',  data_tags['pos_dims'], data_tags['spec_dims'])
    
    current_channel.create_dataset('title', data = basename)
    current_channel.create_dataset('data_type', data = data_tags['data_type'])
    for key in channel_tags:
        current_channel.create_dataset(key, data=channel_tags[key])
    
    ###
    # Read Additional Meta_Data
    ###
    channel_tags['data_type'] = data_tags['data_type']
    meta_tags = get_additional_tags(si,channel_tags)
    meta_tags['time_last_modified'] = time_last_modified
    
	###
	# Write additional important metadata and original_metadata to current_channel attributes
    ###
    current_channel_tags = current_channel.attrs
    for key in meta_tags:
        #print(key,meta_tags[key])
        if 'DM' in key:
            pass
        else:
            current_channel_tags[key]= meta_tags[key]
    
    h5_file.flush()
    return current_channel

def h5_add_Data2Log(log_group, info_dictionary):
    for key in info_dictionary:
        log_group[key] = info_dictionary[key]
def h5_add_Log(current_channel, name):
    log_group = usid.io.hdf_utils.create_indexed_group(current_channel,'Log')
    log_group['title'] = name
    log_group['_'+name] = name ## just easier to read the file-tree that way 
    log_group['time_stamp']= usid.io.io_utils.get_time_stamp()
    try:
        log_group['notebook'] = __notebook__
        log_group['notebook_version'] = __notebook_version__
    except:
        pass
    return log_group



def dm_to_pyUSID(filename = None):
    
    #Open file
   
    
    path, file_name = os.path.split(filename)
    basename, extension = os.path.splitext(file_name)
    time_last_modified = os.path.getmtime(filename)
    
    if extension not in ['.dm3','.dm4']:
        print('file type is not handled yet')
        return 
    
    ### 
    # Read DM file and extract the main information
    ###
    si = dm3.DM3(filename)
    main_tags, channel_tags, data_tags  = get_main_tags(si)
    
    ###
    # Open pyUSID file and write data and main meta-data
    ###
    
    tran = usid.NumpyTranslator()

    h5_file_name = filename[:-4]+'.hf5'
    
    quantity = 'distance'
    units = 'nm'
    h5_path = tran.translate(h5_file_name, data_tags['data_type'], data_tags['rawData'],  quantity, units,
                             data_tags['pos_dims'], data_tags['spec_dims'], translator_name='pyTEMlib', parm_dict={})#parm_dict})

    h5_file =  h5py.File(h5_path, mode='r+')

    for key in main_tags:
        h5_file.attrs[key] =  main_tags[key]

    current_channel = h5_file['Measurement_000/Channel_000']
        
    
    current_channel.create_dataset('title', data = basename)
    current_channel.create_dataset('data_type', data = data_tags['data_type'])
    for key in channel_tags:
        current_channel.create_dataset(key, data=channel_tags[key])
    
    ###
    # Read Additional Meta_Data
    ###
    channel_tags['data_type'] = data_tags['data_type']
    meta_tags = get_additional_tags(si,channel_tags)
    meta_tags['time_last_modified'] = time_last_modified
    
	###
	# Write additional important metadata and original_metadata to current_channel attributes
    ###
    current_channel_tags = current_channel.attrs
    original_group = current_channel.create_group('original_metadata')
    original_group_tags = original_group.attrs
    for key in meta_tags:
        #print(key,meta_tags[key])
        if 'DM' in key:
            pass
        elif 'original_meta' in key:
            original_group_tags[key]= meta_tags[key]
        else:
            current_channel_tags[key]= meta_tags[key]
            
    
    h5_file.flush()
    return h5_file
    
	
def plt_pyUSID(current_channel,ax=None, ax2=None):
    ## Start plotting
    tags = dict(current_channel.attrs)
    basename = current_channel['title'][()]
    if ax == None:
        if  current_channel['data_type'][()] == 'spectrum_image':
            fig, [ax, ax2] = plt.subplots(nrows=1, ncols=2)          
        else:
            fig, ax, = plt.subplots(nrows=1, ncols=1)

            
    # plot according to what data type is in your file
    if current_channel['data_type'][()] == 'EELS_spectrum':
        if ax == None:
            ax = fig.add_subplot(1,2,1)
        ## spectrall data
        spec_sizeX = current_channel['spectral_size_x'][()]
        spec_scaleX = current_channel['spectral_scale_x'][()]
        spec_offsetX = current_channel['spectral_origin_x'][()]

        data = current_channel['Raw_Data'][0,:]
        energy_scale = np.arange(spec_sizeX)*spec_scaleX+spec_offsetX
        if ax== None:
            ax = fig.add
        ax.plot(energy_scale,data);
        ax.set_title('spectrum: '+basename)
        ax.set_xlabel('energy loss [eV]')
        ax.set_ylim(0);
    elif current_channel['data_type'][()] == 'image':
        ## spatial data
        if ax == None:
            ax = fig.add_subplot(1,1,1)
        sizeX = current_channel['spatial_size_x'][()]
        sizeY = current_channel['spatial_size_y'][()]
        scaleX = current_channel['spatial_scale_x'][()]
        scaleY = current_channel['spatial_scale_y'][()]
        basename = current_channel['title'][()]

        extent = (0,sizeX*scaleX,sizeY*scaleY,0)
        data = np.reshape(current_channel['Raw_Data'][:,0],(sizeX,sizeY))
        ax.set_title('image: '+basename)

        ax.imshow(data,extent= extent)
        ax.set_xlabel('distance ['+current_channel['spatial_units'][()]+']');
        annotation_done = []

        for key in tags:
            if 'annotations' in key:
                annotation_number = key[12]
                if annotation_number not in annotation_done:
                    annotation_done.append(annotation_number)

                    if tags['annotations_'+annotation_number+'_type'] == 'text':
                        x =tags['annotations_'+annotation_number+'_x'] 
                        y = tags['annotations_'+annotation_number+'_y']
                        text = tags['annotations_'+annotation_number+'_text'] 
                        ax.text(x,y,text,color='r')

                    elif tags['annotations_'+annotation_number+'_type'] == 'circle':
                        radius = 20 * scaleX#tags['annotations'][key]['radius']
                        xy = tags['annotations_'+annotation_number+'_position']
                        circle = plt.patches.Circle(xy, radius, color='r',fill = False)
                        ax.add_artist(circle)

                    elif tags['annotations_'+annotation_number+'_type'] == 'spectrum image':
                        width = tags['annotations_'+annotation_number+'_width'] 
                        height = tags['annotations_'+annotation_number+'_height']
                        position = tags['annotations_'+annotation_number+'_position']
                        rectangle = plt.patches.Rectangle(position, width, height, color='r',fill = False)
                        ax.add_artist(rectangle)
                        ax.text(position[0],position[1],'Spectrum Image',color='r')
    elif  current_channel['data_type'][()] == 'spectrum_image':
        if ax == None:
            ax = fig.add_subplot(1,2,1)

            ax2 = fig.add_subplot(1,2,2)
        ax.set_title('spectrum image: '+tags['basename'])
        sizeX = current_channel['spatial_size_x'][()]
        sizeY = current_channel['spatial_size_y'][()]
        scaleX = current_channel['spatial_scale_x'][()]
        scaleY = current_channel['spatial_scale_y'][()]

        spec_sizeX = current_channel['spectral_size_x'][()]
        spec_scaleX = current_channel['spectral_scale_x'][()]
        spec_offsetX = current_channel['spectral_origin_x'][()]

        data = np.reshape(current_channel['Raw_Data'][:,:],(sizeX,sizeY,spec_sizeX))
        energy_scale = np.arange(spec_sizeX)*spec_scaleX+spec_offsetX

        extent = (0,sizeX*scaleX,sizeY*scaleY,0)


        ax.imshow(data.sum(axis=2),extent= extent)
        ax.set_xlabel('distance ['+current_channel['spatial_units'][()]+']');

        if ax2 != None:
            ax2.set_title('Spectrum 0, 0')
            ax2.plot(energy_scale,data[0,0,:])
            ax2.set_xlabel('energy loss [eV]')
    elif current_channel['data_type'][()] == 'image_stack':
        ## spatial data
        if ax == None:
            ax = fig.add_subplot(1,1,1)
        sizeX = current_channel['spatial_size_x'][()]
        sizeY = current_channel['spatial_size_y'][()]
        scaleX = current_channel['spatial_scale_x'][()]
        scaleY = current_channel['spatial_scale_y'][()]
        extent = (0,sizeX*scaleX,sizeY*scaleY,0)
        data = np.reshape(current_channel['Raw_Data'][:,0],(sizeX,sizeY))

        ax.set_title('image stack: '+basename)
        ax.imshow(data,extent= extent)
        ax.set_xlabel('distance ['+current_channel['spatial_units'][()]+']');


    
def open_file(file_name = None):


    #Open file
   
    if file_name == None:
        file_name = openfile_dialog('TEM files (*.dm3 *.qf3 *.ndata *.h5 *.hf5);;pyUSID files (*.hf5);;QF files ( *.qf3);;DM files (*.dm3);;Nion files (*.ndata *.h5);;All files (*)')

    tags = {}
    tags['filename']= file_name
    head, tail = os.path.split(file_name)
    tags['path'] = head
    path, tags['extension'] = os.path.splitext(file_name)
    tags['basename'] = tail[:-len(tags['extension'])]

    if tags['extension'] == '.ndata':
        open_nion_file(file_name,tags)
        if 'file_type' not in tags:
            tags['file_type'] = 'image'
    elif tags['extension'] == '.h5':
        if h5py_installed:
            open_h5_nion_file(file_name,tags)
        else:
            print(' You need to install h5py the HDF5 package for python to open Nion data files')
            print(' type "pip install h5py" in the command window')
    elif tags['extension'] == '.dm3':
        open_dm3_file(file_name,tags)
    elif tags['extension'] == '.qf3':
        open_qf3_file(file_name,tags)
        if 'data_type' not in tags:
            
            if len(tags['ene'])> 12:
                tags['data_type'] = 'EELS'
            else:
                tags['data_type'] = 'Image'

        

    else:
        tags['filename'] == 'None'
        print('io 1 no')

    
  
    tags['filename']= file_name
    head, tail = os.path.split(file_name)
    tags['path'] = head
    path, tags['extension'] = os.path.splitext(file_name)
    tags['basename'] = tail[:-len(tags['extension'])]

    #plot_tags(tags, 'new')
    return tags

def open_h5_nion_file(file_name,tags):
    fp = h5py.File(file_name)
    if 'data' in fp:
        json_properties = fp['data'].attrs.get("properties", "")
        data = fp['data'][:]
        tags['shape'] = data.shape
        dic = json.loads(json_properties)

        tags['original_metadata'] = dic
        tags.update(get_nion_tags(dic,data))

        return tags
def get_nion_tags(dic,data):

    tags={}
    tags['original_name'] = dic['description']['title']
    tags['created_date'] = dic['created'][0:10]
    tags['created_time'] = dic['created'][11:19]
    tags['shape'] = dic['data_source']['data_shape']
    
    spectrum = -1
    if 'dimensional_calibrations' in dic['data_source']:
        tags['axis'] = {}
        for i in range(len(dic['data_source']['dimensional_calibrations'])):
            tags['axis'][str(i)]= {}
            tags['axis'][str(i)]['offset'] = dic['data_source']['dimensional_calibrations'][i]['offset']
            tags['axis'][str(i)]['scale']  = dic['data_source']['dimensional_calibrations'][i]['scale']
            tags['axis'][str(i)]['units']  = dic['data_source']['dimensional_calibrations'][i]['units']
            tags['axis'][str(i)]['pixels']  = tags['shape'][i]
            if tags['axis'][str(i)]['units'] == 'nm':
                tags['axis'][str(i)]['offset'] = 0.0
            if tags['axis'][str(i)]['units'] == 'eV':
                spectrum = i
        if spectrum > 0:
            tags['EELS_dimension'] = spectrum
            tags['dispersion'] = tags['axis'][str(spectrum)]['scale']
            tags['offset'] = tags['axis'][str(spectrum)]['offset']
            if len(tags['shape'])==1:
                tags['data_type'] = 'EELS_spectrum'
                tags['spec'] = data
            elif len(tags['shape'])==2:
                tags['data_type'] = 'EELS_linescan'
                tags['spec'] = data[0,:]
                tags['cube'] = data
                tags['SI'] = {}
                tags['SI']['data'] = data
            elif len(tags['shape'])==3:
                tags['data_type'] = 'spectrum_image'
                tags['spec'] = data[0,0,:]
                tags['SI'] = {}
                tags['SI']['data'] = data
                tags['cube'] = data
        else:    
            if tags['axis']['0']['units'] == tags['axis']['1']['units']:
                tags['data_type'] = 'image'
        tags['pixel_size'] = tags['axis']['0']['scale']
        tags['FOV'] = tags['shape'][0] * tags['pixel_size']
    if 'metadata' in  dic['data_source']:
        if 'hardware_source' in dic['data_source']['metadata']:
            hs = dic['data_source']['metadata']['hardware_source']
    
            tags['pixel_size'] = tags['axis']['0']['scale']
    
            if 'fov_nm' in hs: 
                tags['FOV'] = hs['fov_nm']
                tags['exposure'] = hs['exposure']
                tags['pixel_time_us'] = hs['pixel_time_us']
                tags['unit'] = 'nm'
                tags['pixel_size'] = tags['FOV']/tags['shape'][0]
                tags['acceleration_voltage'] = hs['autostem']['ImageScanned:EHT']
                if tags['acceleration_voltage'] > 100000.:
                    tags['microscope'] = 'UltraSTEM200'
                else:
                    tags['microscope'] = 'UltraSTEM100'
                tags['type'] = hs['channel_name']
                tags['aberrations'] = {}
                tags['image']={}
                tags['image']['exposure'] = hs['exposure']
                tags['image']['pixel_time_us'] = hs['pixel_time_us']
                tags['image']['ac_line_sync'] = hs['ac_line_sync']
                tags['image']['rotation_deg'] = hs['rotation_deg']
                
                for key2 in hs['autostem']:
                    if 'ImageScanned' in key2:
                         ## For consistency we remove the . in the aberrations 
                         name = key2[13:].replace(".", "")
                         if name[0] == 'C':
                             tags['aberrations'][name] =  hs['autostem'][key2]*1e9 # aberrations in nm
                         elif name[0:4] == 'BP2^':
                                 tags['image'][name[4:]] = hs['autostem'][key2]

                         else: 
                            tags['image'][name] =  hs['autostem'][key2]

                                    
    assume_corrected = ['C10','C12a','C12b']
    if 'aberrations' in tags:
        for key2 in assume_corrected:
            tags['aberrations'][key2] = 0.
    if 'image' in tags:
        if tags['image']['EHT'] > 101000:
            tags['aberrations']['source_size'] = 0.051
        elif tags['image']['EHT'] < 99000:
            tags['aberrations']['source_size'] = 0.081
        else:
            tags['aberrations']['source_size'] = 0.061

    return tags


def open_nion_file(file_name,tags):
    fp = open(file_name, "rb")
    local_files, dir_files, eocd = parse_zip(fp)
    
    contains_data = b"data.npy" in dir_files
    contains_metadata = b"metadata.json" in dir_files
    file_count = contains_data + contains_metadata # use fact that True is 1, False is 0
    
    fp.seek(local_files[dir_files[ b"data.npy"][1]][1])
    
    tags['data'] = np.load(fp)
    tags['shape'] = tags['data'].shape
    
    if len(tags['shape']) > 2:
        tags['cube'] = tags['data']
        tags['data'] = tags['cube'][0,:,:]
        tags['shape'] = tags['data'].shape
        print(tags['shape'] )
    
    json_pos = local_files[dir_files[b"metadata.json"][1]][1]
    json_len = local_files[dir_files[b"metadata.json"][1]][2]
    fp.seek(json_pos)
    json_properties = fp.read(json_len)
    fp.close()
    dic = json.loads(json_properties.decode("utf-8"))
    
    tags['original_metadata'] = dic
    tags.update(get_nion_tags(dic,tags['data']))


def open_dm3_file(file_name,tags):
    
    si = dm3.DM3(file_name)
    data = si.data_cube
    dm = si.getDictionary()
    
    dmtags = getTagsFromDM3(dm)
    tags.update(dmtags)
    tags['shape'] = data.shape
    tags['original_metadata'] = si.tags
    if 'data_type' not in tags:
        tags['data_type'] = 'unknown'
    print('Found ',tags['data_type'],' in dm3 file')
    
    if tags['data_type'] == 'image':
        tags['data'] = data
        ## Find annotations
        annotations = {}
        for key in si.tags:
            if 'AnnotationGroupList' in key:
                #print(key, dict(current_channel.attrs)[key])
                split_keys= key.split('.')
                if split_keys[4] not in annotations:
                    annotations[split_keys[4]] = {}
                if split_keys[5] in ['AnnotationType','Text','Rectangle','Name', 'Label']:
                    annotations[split_keys[4]][split_keys[5]]=si.tags[key]
        tags['annotations'] = {}
        for key in annotations:
            if annotations[key]['AnnotationType']==13: 
                tags['annotations'][key] = {}
                if 'Label' in annotations[key]:
                    tags['annotations'][key]['label'] = annotations[key]['Label']
                tags['annotations'][key]['type'] = 'text'
                rect = np.array(annotations[key]['Rectangle'])* np.array([tags['axis']['0']['scale'],tags['axis']['1']['scale'],tags['axis']['0']['scale'],tags['axis']['1']['scale']])
                tags['annotations'][key]['x'] = rect[1]
                tags['annotations'][key]['y'] = rect[0]
                tags['annotations'][key]['text'] = annotations[key]['Text']

            elif annotations[key]['AnnotationType']==6:
                tags['annotations'][key] = {}
                if 'Label' in annotations[key]:
                    tags['annotations'][key]['label'] = annotations[key]['Label']
                tags['annotations'][key]['type'] = 'circle'
                rect = np.array(annotations[key]['Rectangle'])* np.array([tags['axis']['0']['scale'],tags['axis']['1']['scale'],tags['axis']['0']['scale'],tags['axis']['1']['scale']])
                
                tags['annotations'][key]['radius'] =rect[3]-rect[1]
                tags['annotations'][key]['position'] = [rect[1],rect[0]]
                
                

            elif annotations[key]['AnnotationType']==23:
                if 'Name' in annotations[key]:
                    if annotations[key]['Name'] == 'Spectrum Image':
                        tags['annotations'][key] = {}
                        if 'Label' in annotations[key]:
                            tags['annotations'][key]['label'] = annotations[key]['Label']
                        tags['annotations'][key]['type'] = 'spectrum image'
                        rect = np.array(annotations[key]['Rectangle'])* np.array([tags['axis']['0']['scale'],tags['axis']['1']['scale'],tags['axis']['0']['scale'],tags['axis']['1']['scale']])

                        tags['annotations'][key]['width'] =rect[3]-rect[1]
                        tags['annotations'][key]['height'] =rect[2]-rect[0]
                        position = tags['annotations'][key]['position'] = [rect[1],rect[0]]
                        
    elif tags['data_type'] == 'image_stack':
        tags['data'] = data[:,:,0]
        tags['cube'] = data
    elif tags['data_type'] == ['spectrum_image']:
        tags['spec'] = data[0,0,:]
        tags['SI'] = {}
        tags['SI']['data'] = data
        tags['cube'] = data
    elif tags['data_type'] == 'EELS_linescan':
        ## A linescan is a spectrum image with the second spatial dimension being 1
        ## The third dimension is the spectrum
        if tags['axis']['0']['units']=='eV':
            data = data.T
            tags['axis']['2']= tags['axis']['0'].copy()
            tags['axis']['0']= tags['axis']['1'].copy()
        else:
            tags['axis']['2']= tags['axis']['1'].copy()
            tags['axis']['1']= tags['axis']['0'].copy()

        for key in tags['axis']['0']:
                temp  = tags['axis']['2'][key]
                tags['axis']['2'][key] = tags['axis']['0'][key]
                tags['axis']['0'][key] = temp
        data = np.reshape(data,(data.shape[0],1,data.shape[1]))    
        tags['spec'] = data[0,0,:]
        tags['data'] = data
        tags['cube'] = data
        tags['SI'] = {}
        tags['SI']['data'] = data

        
    elif tags['data_type'] == 'EELS_spectrum':
        tags['spec'] = data
    else:
        tags['data'] = data
        print('OOPSY-Dasy that is not handled correctly') #Gerd!


def get_main_tags(si):
    """
	Reads specific most meta data and data from the DM3 file
	The information is interpreted to determine data type of measurement_group
	The 'rawdata' tag in data_tags dictionary provides the data in the correct format for pyUSID.
	Input:
		basic tags from DM3 or 4 file
		
	Output:
		main_tags, channel_tags, data_tags Three dictionaries in a format that allows comprehensive code for writing to pyUSID
    """


    ###
    # The calibrations tags are used to determine the kind of data and their units 
    ###
    eVdim = -1  # if and where is there a eV in Units
    main_tags = si.tags['DM'].copy()
    main_tags['instrument']=''
    channel_tags = {}
    data_tags = {}
    for key in si.tags:
        if 'Calibrations.Dimension.' in key:
            if 'Units' in key:
                
                if si.tags[key] == 'µm':## Fix annoying scale of SI in Zeiss
                    si.tags[key] = 'nm'
                    si.tags[key[0:52]+'Scale'] = si.tags[key[0:52]+'Scale']*1000.
                

                if 'eV' in si.tags[key]:
                    channel_tags['spectral_size_x']=si.data_cube.shape[int(key[50])]
                    
                    channel_tags['spectral_scale_x'] = float(si.tags[key[0:52]+'Scale'])
                    channel_tags['spectral_origin_x'] = float(si.tags[key[0:52]+'Origin'] )
                    channel_tags['spectral_units_x'] = (si.tags[key])
                    eVdim = int(key[50])
                
                else:  
                    if si.tags[key] != '' and int(key[15]) > 0 : # Not foolproof
                        if 'spatial_scale_x' not in channel_tags.keys():
                            channel_tags['spatial_scale_x'] = float(si.tags[key[0:52]+'Scale'])
                            channel_tags['spatial_origin_x'] = float(si.tags[key[0:52]+'Origin'])
                            channel_tags['spatial_units'] = si.tags[key]
                        else:
                            channel_tags['spatial_scale_y'] = float(si.tags[key[0:52]+'Scale'])
                            channel_tags['spatial_origin_y'] = float(si.tags[key[0:52]+'Origin'] )
                            #print(channel_tags['spatial_scale_y'], si.tags[key[0:52]+'Scale'],key)
                        
                number_of_dimensions = int(key[50])
        if 'Microscope' in key:
            main_tags['instrument'] = si.tags[key]
        if 'SuperScan' in key:
                main_tags['instrument'] = 'UltraSTEM'
                if 'ChannelName' in key:
                    channel_tags['image_type'] = si.tags[key]
                    channel_tags['detector_type'] = si.tags[key]
                    channel_tags['image_mode'] = 'STEM'

                if 'PixelTime(us)' in key:
                    channel_tags['seconds_per_pixel'] = si.tags[key] *1e-6
                if 'BlankTime(us)' in key:
                    channel_tags['seconds_per_flyback'] = si.tags[key] *1e-6
                if 'nmPerPixel' in key:
                    channel_tags['spatial_scale_x'] = float(si.tags[key])
                    channel_tags['spatial_units'] = 'nm'
                if 'nmPerLine' in key:
                    channel_tags['spatial_scale_y'] = float(si.tags[key]) 


    #Determine type of data  by 'Calibrations' tags 
    data = si.data_cube
    
        
    if eVdim  > -1: # This data set has some spectra component

        if number_of_dimensions ==2:
            data_tags['data_type'] = 'spectrum_image'

            if eVdim != number_of_dimensions: # This should never happen
                data = np.swapaxes(si.data_cube, eVdim, 2)
            
            data = np.swapaxes(data, 0, 1)  # Make fast moving axis the Oth order
            rawData = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2]),order='C' )
            pos_dims = [usid.write_utils.Dimension('X', 'nm', data.shape[0]),
                usid.write_utils.Dimension('Y', 'nm', data.shape[1])]

            spec_dims = usid.write_utils.Dimension('energy loss', 'eV', data.shape[2])


        elif number_of_dimensions ==1:
            data_tags['data_type'] = 'EELS_linescan'
            if eVdim == 1: 
                rawData = np.swapaxes(si.data_cube, 0, 1)
            else:
                rawData = data

            pos_dims = [usid.write_utils.Dimension('X', 'nm', data.shape[0]),
                    usid.write_utils.Dimension('Y', 'nm', 1)]

            spec_dims = usid.write_utils.Dimension('energy loss', 'eV', data.shape[1])    

        else:
            data_tags['data_type'] = 'EELS_spectrum'
            pos_dims = [usid.write_utils.Dimension('X', 'a. u.', 1),
                    usid.write_utils.Dimension('Y', 'a. u.', 1)]
            spec_dims = usid.write_utils.Dimension('energy loss', 'eV', data.shape[0])
            rawData =  np.reshape(data,(1,data.shape[0]),order='C' )

            
            
            
    if number_of_dimensions== 2:
        data_tags['data_type'] = 'image_stack'
        data = data.sum(axis=2)
        pos_dims = [usid.write_utils.Dimension('X', 'nm', data.shape[0]),
                usid.write_utils.Dimension('Y', 'nm', data.shape[1])]
        spec_dims = usid.write_utils.Dimension('none', 'none',1)


        rawData = np.reshape(data,(data.shape[0]*data.shape[1],1),order='F' )
        channel_tags['image_stack'] = si.data_cube
        
    elif number_of_dimensions == 1: #starts at 0!!!!
        data_tags['data_type'] = 'image'

        rawData = np.reshape(data,(data.shape[0]*data.shape[1],1),order='F' )
        pos_dims = [usid.write_utils.Dimension('X', 'nm', data.shape[0]),
                usid.write_utils.Dimension('Y', 'nm', data.shape[1])]
        spec_dims = usid.write_utils.Dimension('none', 'none', 1)


   
    data_tags['rawData'] = rawData
    data_tags['pos_dims'] = pos_dims
    data_tags['spec_dims'] = spec_dims
    
    if 'spatial_scale_x' not in channel_tags:
        channel_tags['spatial_scale_x'] = 1
        channel_tags['spatial_origin_x'] = 0
        channel_tags['spatial_units'] = ''
        
        
        if data_tags['data_type'] in ['image', 'image_stack', 'spectrum_image']:
            channel_tags['spatial_scale_y'] = 1
            channel_tags['spatial_origin_y'] = 0
    
    if 'spectral_origin_x' in channel_tags:
        channel_tags['spectral_origin_x'] = -channel_tags['spectral_scale_x'] *channel_tags['spectral_origin_x'] 



    channel_tags['spatial_size_x'] = data.shape[0]
    if 'spatial_scale_y' in channel_tags:
        channel_tags['spatial_size_y'] = data.shape[1]
    
    main_tags['pyUSID_version'] =  usid.__version__
    #main_tags['instrument']#'Nion US200'
    main_tags['user_name'] = 'Gerd'
    main_tags['sample_name'] = ''
    main_tags['sample_description'] = ''
    main_tags['project_name'] = 'STEM-FWP'
    main_tags['project_id'] = ''
    main_tags['data_tool'] = 'pyTEM-0.983'
    main_tags['experiment_date'] = ''
    main_tags['experiment_time'] = ''
    main_tags['data_type'] = data_tags['data_type']
    main_tags['comments'] = ''
    
    main_tags['translator'] = 'pyTEM'
    
    return main_tags, channel_tags, data_tags

def get_additional_tags(si, channel_tags):
    """
	Reads specific additional meta data from the DM3 file
	Input:
		basic tags from DM3 or 4 file
		channe_tags with information on data_type and such
		
	Output:
		dictionary of tags and all of the tags of the original dm file.
		This is a format that allows comprehensive code for writing to pyUSID
    """

    time = date = ''        

    out_tags = {}
    ##### Image type data in Nion microscope
    aberrations = ['C10','C12','C21','C23','C30','C32','C34','C41','C43','C45','C50','C52','C54','C56']

    for key in si.tags:
        if 'date' in key.lower():
            date = si.tags[key]
            #print(date)
        if 'start time' in key.lower():
            #print(key, tags['original_metadata'][key])
            time = si.tags[key]
        if 'ImageScanned' in  key: ## Nion type file read out relevant parameters
            if 'EHT' in key:
                out_tags['acceleration_voltage'] = float(si.tags[key])
                if out_tags['acceleration_voltage'] >100000:
                    out_tags['microscope'] = 'UltraSTEM 200'
                else:
                    out_tags['microscope'] = 'UltraSTEM 100'
            if 'HAADF' in channel_tags['detector_type']:
                if 'PMTDF_gain' in key:
                    out_tags['detector_gain'] = si.tags[key]
            else:
                if 'PMTBF_gain' in key:
                    out_tags['detector_gain'] = si.tags[key]
            if 'StageOutX' in key:
                out_tags['stage_x'] =  si.tags[key]
            if 'StageOutY' in key:
                out_tags['stage_y'] =  si.tags[key]
            if 'StageOutZ' in key:
                out_tags['stage_z'] =  si.tags[key]
            if 'StageOutA' in key:
                out_tags['stage_alpha'] =  si.tags[key]
            if 'StageOutB' in key:
                out_tags['stage_beta'] =  si.tags[key]

            if  key[-3:] in aberrations:
                out_tags['aberrations_'+key[-3:]] = si.tags[key]
            elif key[-5:-2] in aberrations:    
                out_tags['aberrations_'+key[-5:]] = si.tags[key]

        if 'Microscope Info' in key:
            if 'Voltage'  in key:
                out_tags['acceleration_voltage']  =  si.tags[key]
            if 'Illumination Mode' in key:

                if channel_tags['spatial_units']  == '':
                    if si.tags[key] == 'STEM':
                        out_tags['image_type']= 'Ronchigram'

                if '1/' in channel_tags['spatial_units']  or 'mrad'  in channel_tags['spatial_units']:
                    out_tags['image_type'] = 'Diffraction'
                else:
                    out_tags['image_type'] = 'Image'

                if 'Operation Mode' in key:
                    if  si.tags[key] =='SCANNING':
                        out_tags['image_mode'] = 'STEM'
                    else:
                        out_tags['image_mode'] = 'TEM'
                if 'Cs(mm)'in key:
                    out_tags['aberrations_C30'] = si.tags[key]*1e6
            if 'STEM Camera Length' in key:
                out_tags['camera_length'] = si.tags[key]
        if 'EELS' in key:
            if 'Exposure (s)' in key:
                out_tags['exposure_spectrum'] = si.tags[key]
            if 'Number of frames' in key:
                out_tags['number_of_frames'] = si.tags[key]

            if 'Convergence semi-angle (mrad)' in key:
                out_tags['convergence_angle'] = si.tags[key]

            if 'Collection semi-angle (mrad)' in key:
                out_tags['collection_angle'] = si.tags[key]
    
    if 'number_of_frames' in out_tags:
        out_tags['integration_time'] = out_tags['number_of_frames']*out_tags['exposure_spectrum']

    if 'microscope' in out_tags:
        if 'UltraSTEM' in out_tags['microscope']:      
            if out_tags['acceleration_voltage'] == 200000:
                out_tags['aberrations_source_size'] = 0.051
            elif out_tags['acceleration_voltage'] == 100000:
                out_tags['aberrations_source_size'] = 0.061
            elif out_tags['acceleration_voltage'] == 60000:
                out_tags['aberrations_source_size'] = 0.081

                out_tags['aberrations_zeroLoss'] = [0.0143,0.0193,0.0281,0.0440,0.0768,	0.1447,	0.2785,	0.4955,	0.7442,	0.9380,	1.0000,	0.9483,	0.8596,	0.7620,	0.6539,	0.5515,0.4478,	0.3500,	0.2683,	0.1979,	0.1410,	0.1021,	0.0752,	0.0545,	0.0401,	0.0300,	0.0229,	0.0176,	0.0139]
                out_tags['aberrations_zeroEnergy'] = np.linspace(-.5,.9,len(out_tags['aberrations_zeroLoss']))
        if 'Libra' in out_tags['microscope'] :
            if out_tags['image_mode'] == 'STEM':
                if 'camera_length' in out_tags:
                    out_tags['collection_angle'] = np.tan(0.65/out_tags['camera_length'])*1000*23.73527943
                    out_tags['convergence_angle']= 9

    if channel_tags['data_type'] == 'image':

        ## Find annotations
        annotations = {}
        for key in si.tags:
            if 'AnnotationGroupList' in key:
                #print(key, dict(current_channel.attrs)[key])
                split_keys= key.split('.')
                if split_keys[4] not in annotations:
                    annotations[split_keys[4]] = {}
                if split_keys[5] in ['AnnotationType','Text','Rectangle','Name', 'Label']:
                    annotations[split_keys[4]][split_keys[5]]=si.tags[key]
        #out_tags['annotations'] = {}
        rec_scale=  np.array([channel_tags['spatial_scale_x'],channel_tags['spatial_scale_y'],channel_tags['spatial_scale_x'],channel_tags['spatial_scale_y']])  
                
        for key in annotations:
            if annotations[key]['AnnotationType']==13: 
                #out_tags['annotations'][key] = {}
                if 'Label' in annotations[key]:
                    out_tags['annotations_'+str(key)+'_label'] = annotations[key]['Label']
                out_tags['annotations_'+str(key)+'_type'] = 'text'
                rect = np.array(annotations[key]['Rectangle'])* rec_scale
                out_tags['annotations_'+str(key)+'_x'] = rect[1]
                out_tags['annotations_'+str(key)+'_y'] = rect[0]
                out_tags['annotations_'+str(key)+'_text'] = annotations[key]['Text']

            elif annotations[key]['AnnotationType']==6:
                #out_tags['annotations'][key] = {}
                if 'Label' in annotations[key]:
                    out_tags['annotations_'+str(key)+'_label'] = annotations[key]['Label']
                out_tags['annotations_'+str(key)+'_type'] = 'circle'
                rect = np.array(annotations[key]['Rectangle'])* rec_scale
                
                out_tags['annotations_'+str(key)+'_radius'] =rect[3]-rect[1]
                out_tags['annotations_'+str(key)+'_position'] = [rect[1],rect[0]]



            elif annotations[key]['AnnotationType']==23:
                if 'Name' in annotations[key]:
                    if annotations[key]['Name'] == 'Spectrum Image':
                        #tags['annotations'][key] = {}
                        if 'Label' in annotations[key]:
                            out_tags['annotations_'+str(key)+'_label'] = annotations[key]['Label']
                        out_tags['annotations_'+str(key)+'_type'] = 'spectrum image'
                        rect = np.array(annotations[key]['Rectangle'])* rec_scale
                        
                        out_tags['annotations_'+str(key)+'_width'] =rect[3]-rect[1]
                        out_tags['annotations_'+str(key)+'_height'] =rect[2]-rect[0]
                        out_tags['annotations_'+str(key)+'_position'] = [rect[1],rect[0]]
    
    for key in si.tags:
        out_tags['original_metadata.'+key] = si.tags[key]
    return out_tags
    

def getTagsFromDM3 (dm):
    tags = {}
    ## Fix annoying scale of SI in Zeiss
    for key in  dm['ImageData']['Calibrations']['Dimension']:
        if dm['ImageData']['Calibrations']['Dimension'][key]['Units'] == 'µm':
            dm['ImageData']['Calibrations']['Dimension'][key]['Units'] = 'nm'
            dm['ImageData']['Calibrations']['Dimension'][key]['Scale'] *= 1000.0
    
    #Determine type of data  by 'Calibrations' tags 
    if len(dm['ImageData']['Calibrations']['Dimension']) >1:
        units1 = dm['ImageData']['Calibrations']['Dimension']['1']['Units']
    else:
        units1=''
     
    units = dm['ImageData']['Calibrations']['Dimension']['0']['Units']
    if 'ImageTags' in dm:
        if 'SI' in dm['ImageTags']:
            if len(dm['ImageData']['Calibrations']['Dimension']) == 3:
                tags['data_type'] = 'spectrum_image'
            else:
                if units == 'eV' or units1 == 'eV':
                    tags['data_type'] = 'EELS_linescan'
                    
                else:
                    tags['data_type'] = 'image'
                    tags['image_type'] = 'survey image'
        elif 'EELS' in dm['ImageTags']:
            tags['data_type'] = 'EELS_spectrum'
        elif len(dm['ImageData']['Calibrations']['Dimension']) == 3:
            tags['data_type'] = 'image_stack'
        else:
            tags['data_type'] = 'image'
            
    
        tags['microscope']= 'unknown'
    
        ## General Information from Zeiss like TEM
        if 'Microscope Info' in dm['ImageTags']:
            if 'Microscope' in dm['ImageTags']['Microscope Info']:
                tags['microscope'] =  (dm['ImageTags']['Microscope Info']['Microscope'])
                if 'Libra' in tags['microscope']:
                    tags['microscope'] = 'Libra 200'
            if 'Voltage'  in dm['ImageTags']['Microscope Info']:
                tags['acceleration_voltage']  = dm['ImageTags']['Microscope Info']['Voltage']
    
        ## General Information from Nion STEM
        if 'ImageRonchigram' in dm['ImageTags']:
            tags['image_type']= 'Ronchigram'
            tags['microscope'] = 'UltraSTEM'    
        if 'SuperScan' in  dm['ImageTags']:
            tags['microscope'] = 'UltraSTEM'
            if 'ChannelName' in dm['ImageTags']['SuperScan']:
                tags['image_type'] = dm['ImageTags']['SuperScan']['ChannelName']
                tags['detector_type'] = tags['image_type']
                tags['image_mode'] = 'STEM'
                
                    
            if 'PixelTime(us)' in dm['ImageTags']['SuperScan']:
                tags['seconds per pixel'] = dm['ImageTags']['SuperScan']['PixelTime(us)'] *1e6
            if 'BlankTime(us)' in dm['ImageTags']['SuperScan']:
                tags['seconds per flyback'] = dm['ImageTags']['SuperScan']['BlankTime(us)'] *1e6   
    
            
        if 'ImageScanned' in  dm['ImageTags']:
            if 'EHT' in dm['ImageTags']['ImageScanned']:
                tags['acceleration_voltage'] = float(dm['ImageTags']['ImageScanned']['EHT'])
                if tags['acceleration_voltage'] >100000:
                    tags['microscope'] = 'UltraSTEM 200'
                else:
                    tags['microscope'] = 'UltraSTEM 100'
            if tags['detector_type'] == 'HAADF':
                if 'PMTDF_gain' in dm['ImageTags']['ImageScanned']:
                    tags['detector_gain'] = dm['ImageTags']['ImageScanned']['PMTDF_gain']
            else:
                if 'PMTBF_gain' in dm['ImageTags']['ImageScanned']:
                    tags['detector_gain'] = dm['ImageTags']['ImageScanned']['PMTBF_gain']
            if 'StageOutX' in dm['ImageTags']['ImageScanned']:
                tags['stage'] = {}
                tags['stage']['x'] =  dm['ImageTags']['ImageScanned']['StageOutX']
                tags['stage']['y'] =  dm['ImageTags']['ImageScanned']['StageOutY']
                tags['stage']['z'] =  dm['ImageTags']['ImageScanned']['StageOutZ']
                
                tags['stage']['alpha'] =  dm['ImageTags']['ImageScanned']['StageOutA']
                tags['stage']['beta'] =  dm['ImageTags']['ImageScanned']['StageOutB']
    

    #if tags['data_type'] == 'Image':

        
    ##### Image type data in Zeiss like microscope
    tags['aberrations'] = {}
    if 'data_type' in tags:
        if tags['data_type'] == 'Image':
            if 'Microscope Info' in dm['ImageTags']:
                if units=='':
                    if 'Illumination Mode' in dm['ImageTags']['Microscope Info']:
                        if tags['ImageTags']['Microscope Info']['Illumination Mode'] == 'STEM':
                            tags['image_type']= 'Ronchigram'
                if units[:2] == '1/' or units =='mrad' :
                    tags['image_type'] = 'Diffraction'
                else:
                    tags['image_type'] = 'Image'
    
                if 'Microscope' in dm['ImageTags']['Microscope Info']:
                    tags['microscope'] =  (dm['ImageTags']['Microscope Info']['Microscope'])
                if 'Operation Mode' in dm['ImageTags']['Microscope Info']:
                    if dm['ImageTags']['Microscope Info']['Operation Mode'] =='SCANNING':
                        tags['image_mode'] = 'STEM'
                    else:
                        tags['image_mode'] = 'TEM'
                if 'Cs(mm)'in dm['ImageTags']['Microscope Info']:
                    tags['aberrations']['C30'] = float(dm['ImageTags']['Microscope Info']['Cs(mm)'])*1e6

        
    ##### Image type data in Nion microscope
    aberrations = ['C10','C12','C21','C23','C30','C32','C34','C41','C43','C45','C50','C52','C54','C56']
    assume_corrected = ['C10','C12a','C12b']
    debug = 0
    if 'ImageTags' in dm:
        if 'ImageScanned' in dm['ImageTags']:
            for key in aberrations:
                if key in dm['ImageTags']['ImageScanned']:
                    if  isinstance(dm['ImageTags']['ImageScanned'][key],dict): # if element si a dictionary
                        tags['aberrations'][key+'a'] = dm['ImageTags']['ImageScanned'][key]['a']
                        tags['aberrations'][key+'b'] = dm['ImageTags']['ImageScanned'][key]['b']
                    else:
                        tags['aberrations'][key] = dm['ImageTags']['ImageScanned'][key]
                
    #for key in tags['SuperScan']:
    #    if key == 'Rotation':
    #        print(key, tags['SuperScan'][key], ' = ' , np.rad2deg(tags['SuperScan'][key]))#/np.pi*180)
    
            if tags['acceleration_voltage'] == 200000:
                tags['aberrations']['source_size'] = 0.051
            elif tags['acceleration_voltage'] == 100000:
                tags['aberrations']['source_size'] = 0.061
            elif tags['acceleration_voltage'] == 60000:
                tags['aberrations']['source_size'] = 0.081
        
            tags['aberrations']['zeroLoss'] = [0.0143,0.0193,0.0281,0.0440,0.0768,	0.1447,	0.2785,	0.4955,	0.7442,	0.9380,	1.0000,	0.9483,	0.8596,	0.7620,	0.6539,	0.5515,0.4478,	0.3500,	0.2683,	0.1979,	0.1410,	0.1021,	0.0752,	0.0545,	0.0401,	0.0300,	0.0229,	0.0176,	0.0139]
            tags['aberrations']['zeroEnergy'] = np.linspace(-.5,.9,len(tags['aberrations']['zeroLoss']))

        tags['axis']=dm['ImageData']['Calibrations']['Dimension']
        for dimension in tags['axis']:
            for key in tags['axis'][dimension]:
                tags['axis'][dimension][key.lower()] = tags['axis'][dimension].pop(key)
            #tags['axis'][dimension]['pixels'] = tags['shape'][int(dimension)]
        if 'SuperScan' in dm['ImageTags']:  
            if 'nmPerLine' in dm['ImageTags']['SuperScan']['Calibration']:
                tags['axis']['0']['scale']  = dm['ImageTags']['SuperScan']['Calibration']['nmPerPixel']
                tags['axis']['0']['units'] = 'nm'
                tags['axis']['0']['origin'] = 0
                tags['axis']['1']['scale']  = dm['ImageTags']['SuperScan']['Calibration']['nmPerLine']
                tags['axis']['1']['units'] = 'nm'
                tags['axis']['1']['origin'] = 0
    if 'axis' in tags:    
        tags['pixel_size'] = tags['axis']['0']['scale']
        tags['FOV'] = tags['axis']['0']['scale']*dm['ImageData']['Dimensions']['0']
    
    
        for key in tags['axis']:
            if tags['axis'][key]['units'] == 'eV':
                
                tags['dispersion'] = float(tags['axis'][key]['scale'])
                tags['offset'] = -float(tags['axis'][key]['origin'])  *tags['dispersion']                
                tags['axis'][key]['origin'] = tags['offset']
    if 'spectrum_image' in dm:
        tags['spectrum_image'] = dm['spectrum_image']
        print('spectrum_image')
    if 'ImageTags' in dm:    
        if 'EELS' in dm['ImageTags']:
            eels = dm['ImageTags']['EELS']
            #print(eels)
            if 'Exposure (s)' in eels['Acquisition']:
                
                if 'Number of frames' in eels['Acquisition']:
                    tags['integration_time'] = eels['Acquisition']['Exposure (s)']*eels['Acquisition']['Number of frames']
                    tags['number_of_frames'] = eels['Acquisition']['Number of frames']
            
        #if 'Dispersion (eV/ch)' in eels:
        #    tags['dispersion'] = float(eels['Dispersion (eV/ch)'])
        #if 'Energy loss (eV)' in eels:
        #    tags['offset'] = float(eels['Energy loss (eV)'])
        #    #Gatan measures offset at channel 100, but QF at channel 1
        #    tags['offset'] = tags['offset']- 100.0*tags['dispersion']
            
            if  'Convergence semi-angle (mrad)' in eels:
                tags['convAngle']= float(eels['Convergence semi-angle (mrad)'])
            if 'Collection semi-angle (mrad)' in eels:
                tags['collAngle'] = float(eels['Collection semi-angle (mrad)'])
    
            ## Determine angles for Zeiss Libra 200 MC at UTK
            if tags['microscope'] == 'Libra 200 MC':
                
                if 'Microscope Info' in dm['ImageTags']:
                    if 'Illumination Mode' in dm['ImageTags']['Microscope Info']:
                        if dm['ImageTags']['Microscope Info']['Illumination Mode'] == 'STEM':
                            if dm['ImageTags']['Microscope Info']['Operation Mode'] == 'SCANNING':
                                if 'STEM Camera Length' in dm['ImageTags']['Microscope Info']:
                                    tags['collAngle'] = np.tan(0.65/dm['ImageTags']['Microscope Info']['STEM Camera Length'])*1000*23.73527943
                                    tags['convAngle']= 9
                                    tags['STEM camera length'] = dm['ImageTags']['Microscope Info']['STEM Camera Length']

    
    return tags

def parse_zip(fp):
    
    """
        Parse the zip file headers at fp
        :param fp: the file pointer from which to parse the zip file
        :return: A tuple of local files, directory headers, and end of central directory
        The local files are dictionary where the keys are the local file offset and the
        values are each a tuple consisting of the name, data position, data length, and crc32.
        The directory headers are a dictionary where the keys are the names of the files
        and the values are a tuple consisting of the directory header position, and the
        associated local file position.
        The end of central directory is a tuple consisting of the location of the end of
        central directory header and the location of the first directory header.
        This method will seek to location 0 of fp and leave fp at end of file.

        This function is copied from  nionswift/nion/swift/model/NDataHandler.py

    """
    local_files = {}
    dir_files = {}
    eocd = None
    fp.seek(0)
    while True:
        pos = fp.tell()
        signature = struct.unpack('I', fp.read(4))[0]
        if signature == 0x04034b50:
            fp.seek(pos + 14)
            crc32 = struct.unpack('I', fp.read(4))[0]
            fp.seek(pos + 18)
            data_len = struct.unpack('I', fp.read(4))[0]
            fp.seek(pos + 26)
            name_len = struct.unpack('H', fp.read(2))[0]
            extra_len = struct.unpack('H', fp.read(2))[0]
            name_bytes = fp.read(name_len)
            fp.seek(extra_len, os.SEEK_CUR)
            data_pos = fp.tell()
            fp.seek(data_len, os.SEEK_CUR)
            local_files[pos] = (name_bytes, data_pos, data_len, crc32)
        elif signature == 0x02014b50:
            fp.seek(pos + 28)
            name_len = struct.unpack('H', fp.read(2))[0]
            extra_len = struct.unpack('H', fp.read(2))[0]
            comment_len = struct.unpack('H', fp.read(2))[0]
            fp.seek(pos + 42)
            pos2 = struct.unpack('I', fp.read(4))[0]
            name_bytes = fp.read(name_len)
            fp.seek(pos + 46 + name_len + extra_len + comment_len)
            dir_files[name_bytes] = (pos, pos2)
        elif signature == 0x06054b50:
            fp.seek(pos + 16)
            pos2 = struct.unpack('I', fp.read(4))[0]
            eocd = (pos, pos2)
            break
        else:
            raise IOError()
    return local_files, dir_files, eocd



def open_qf3_file(file_name,tags):
    pkl_file = open(file_name,'rb')    
    qf= pickle.load(pkl_file)
    pkl_file.close()
    
    
    if 'QF' in qf:
        if qf['QF']['version'] < 0.982:
            #we need to update of dictionary structure
            print('updating file to new format')
            if 'DM' in qf['mem']:
                dmtemp  = qf['mem'].pop('DM', None)
                
                dm = getDictionary(dmtemp)
                outtags = getTagsFromDM3 (dm)
                
            outtags.update(qf['mem'])    
            tags.update(outtags)    

            if 'pixel_size' in tags:
                tags['pixel_size'] = tags['pixel_size']
                for key in tags['axis']:
                    if tags['axis']['0']['units'] == 'nm':
                        tags['axis']['0']['scale'] = tags['pixel_size']
                    
                tags.pop('pixel_size')
            if 'data' not in tags:
                tags['data'] = tags['spec']
            for dimension in tags['axis']:
                tags['axis'][dimension]['pixels'] = tags['data'].shape[int(dimension)]
        
            tags['original_metadata'] = dm
            tags.pop('cal', None)
            tags.pop('dm', None)
            tags.pop('EELS', None)
            tags.pop('Calibrations', None)
            tags.pop('DZM', None)

            tags.pop('Microscope', None)
            tags.pop('Nion', None)
            tags.pop('SI', None)
            tags.pop('DM', None)

            if 'SI' in qf['QF']:
                si  = qf['QF']['SI']
                
                try:
                    si.pop('specTags')
                    tags['SI'] = si
                except:
                    pass
                
            
        else:
            print('new file format')
            tags = qf
    
    else:
        tags.update(qf) ## This is a CAAC file and we need to do soemthing about that.
    #iplot_tags(tags, which = 'new')
    print(tags.keys())

    
    if tags['data_type'] == 'image':
        
    
        if 'cube' in tags:
            tags['shape'] = tags['cube'].shape
        else:
            tags['shape'] = tags['data'].shape
        
           
        tags['circs']=[]
        if 'summed' in qf:
            qf['data'] = tags['summed']
        if 'pixel_size' in tags:
            tags['pixel_size'] = tags['pixel_size']
        if 'FOV' not in tags:
            tags['FOV'] = tags['data'].shape[0]*tags['pixel_size']
        
        print ('Field of View: {0:.2f}nm'.format( tags['FOV']) )
    print('in qf3')
    print(tags['data_type'])
   
    return tags



def plot_tags(tags, which = 'All'):

   
    if which == 'All':
        dm = tags.pop('original_metadata', None)
        tree = DictionaryTreeBrowser(tags)
        
        
    else:
        if which in ['new', 'original']:
            temptags = tags.copy()
            dm = temptags.pop('original_metadata', None)
            if which == 'original':
                tree = DictionaryTreeBrowser(dm)
            else:
                tree = DictionaryTreeBrowser(temptags)
        else:
            if which in tags:
                tree = DictionaryTreeBrowser(tags[which])
    print(tree._get_print_items())


def str2num(string, **kargs):
    """Transform a a table in string form into a numpy array
    Parameters
    ----------
    string : string
    Returns
    -------
    numpy array
    """
    stringIO = StringIO(string)
    return np.loadtxt(stringIO, **kargs)


_slugify_strip_re_data = ''.join(
    c for c in map(
        chr, np.delete(
            np.arange(256), [
                95, 32])) if not c.isalnum()).encode()


import unicodedata
import types
def ensure_unicode(stuff, encoding='utf8', encoding2='latin-1'):
    if not isinstance(stuff, (bytes, np.string_)):
        return stuff
    else:
        string = stuff
    try:
        string = string.decode(encoding)
    except:
        string = string.decode(encoding2, errors='ignore')
    return string


def slugify(value, valid_variable_name=False):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    Adapted from Django's "django/template/defaultfilters.py".
    """
    if not isinstance(value, str):
        try:
            # Convert to unicode using the default encoding
            value = str(value)
        except:
            # Try latin1. If this does not work an exception is raised.
            value = str(value, "latin1")
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = value.translate(None, _slugify_strip_re_data).decode().strip()
    value = value.replace(' ', '_')
    if valid_variable_name and not value.isidentifier():
        value = 'Number_' + value
    return value




class DictionaryTreeBrowser(object):

    """A class to comfortably browse a dictionary using a CLI.
    In addition to accessing the values using dictionary syntax
    the class enables navigating  a dictionary that constains
    nested dictionaries as attribures of nested classes.
    Also it is an iterator over the (key, value) items. The
    `__repr__` method provides pretty tree printing. Private
    keys, i.e. keys that starts with an underscore, are not
    printed, counted when calling len nor iterated.
    Methods
    -------
    export : saves the dictionary in pretty tree printing format in a text
        file.
    keys : returns a list of non-private keys.
    as_dictionary : returns a dictionary representation of the object.
    set_item : easily set items, creating any necessary node on the way.
    add_node : adds a node.
    Examples
    --------
    >>> tree = DictionaryTreeBrowser()
    >>> tree.set_item("Branch.Leaf1.color", "green")
    >>> tree.set_item("Branch.Leaf2.color", "brown")
    >>> tree.set_item("Branch.Leaf2.caterpillar", True)
    >>> tree.set_item("Branch.Leaf1.caterpillar", False)
    >>> tree
    └── Branch
        ├── Leaf1
        │   ├── caterpillar = False
        │   └── color = green
        └── Leaf2
            ├── caterpillar = True
            └── color = brown
    >>> tree.Branch
    ├── Leaf1
    │   ├── caterpillar = False
    │   └── color = green
    └── Leaf2
        ├── caterpillar = True
        └── color = brown
    >>> for label, leaf in tree.Branch:
    ...     print("%s is %s" % (label, leaf.color))
    Leaf1 is green
    Leaf2 is brown
    >>> tree.Branch.Leaf2.caterpillar
    True
    >>> "Leaf1" in tree.Branch
    True
    >>> "Leaf3" in tree.Branch
    False
    >>>
    """

    def __init__(self, dictionary=None, double_lines=False):
        self._double_lines = double_lines
        if not dictionary:
            dictionary = {}
        super(DictionaryTreeBrowser, self).__init__()
        self.add_dictionary(dictionary, double_lines=double_lines)

    def add_dictionary(self, dictionary, double_lines=False):
        """Add new items from dictionary.
        """
        for key, value in dictionary.items():
            if key == '_double_lines':
                value = double_lines
            #print(key,value)
            self.__setattr__(key, value)
            

    def export(self, filename, encoding='utf8'):
        """Export the dictionary to a text file
        Parameters
        ----------
        filename : str
            The name of the file without the extension that is
            txt by default
        encoding : valid encoding str
        """
        f = codecs.open(filename, 'w', encoding=encoding)
        f.write(self._get_print_items(max_len=None))
        f.close()

    def _get_print_items(self, padding='', max_len=78):
        """Prints only the attributes that are not methods
        """
        

        def check_long_string(value, max_len):
            if not isinstance(value, (str, np.string_)):
                value = repr(value)
            value = ensure_unicode(value)
            strvalue = str(value)
            _long = False
            if max_len is not None and len(strvalue) > 2 * max_len:
                right_limit = min(max_len, len(strvalue) - max_len)
                strvalue = '%s ... %s' % (
                    strvalue[:max_len], strvalue[-right_limit:])
                _long = True
            return _long, strvalue

        string = ''
        eoi = len(self)
        j = 0
        #if preferences.General.dtb_expand_structures and self._double_lines:
        if self._double_lines:
            s_end = '╚══ '
            s_middle = '╠══ '
            pad_middle = '║   '
        else:
            s_end = '└── '
            s_middle = '├── '
            pad_middle = '│   '
        for key_, value in iter(sorted(self.__dict__.items())):
            if key_.startswith("_"):
                continue
            if not isinstance(key_, types.MethodType):
                key = ensure_unicode(value['key'])
                value = value['_dtb_value_']
                if j == eoi - 1:
                    symbol = s_end
                else:
                    symbol = s_middle
                if True:#preferences.General.dtb_expand_structures:
                    if isinstance(value, list) or isinstance(value, tuple):
                        iflong, strvalue = check_long_string(value, max_len)
                        if iflong:
                            key += (" <list>"
                                    if isinstance(value, list)
                                    else " <tuple>")
                            value = DictionaryTreeBrowser(
                                {'[%d]' % i: v for i, v in enumerate(value)},
                                double_lines=True)
                        else:
                            string += "%s%s%s = %s\n" % (
                                padding, symbol, key, strvalue)
                            j += 1
                            continue

                if isinstance(value, DictionaryTreeBrowser):
                    string += '%s%s%s\n' % (padding, symbol, key)
                    if j == eoi - 1:
                        extra_padding = '    '
                    else:
                        extra_padding = pad_middle
                    string += value._get_print_items(
                        padding + extra_padding)
                else:
                    _, strvalue = check_long_string(value, max_len)
                    string += "%s%s%s = %s\n" % (
                        padding, symbol, key, strvalue)
            j += 1
        return string

    def __repr__(self):
        return self._get_print_items()

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getattribute__(self, name):
        if isinstance(name, bytes):
            name = name.decode()
        name = slugify(name, valid_variable_name=True)
        item = super(DictionaryTreeBrowser, self).__getattribute__(name)
        if isinstance(item, dict) and '_dtb_value_' in item and "key" in item:
            return item['_dtb_value_']
        else:
            return item

    def __setattr__(self, key, value):
        if key.startswith('_sig_'):
            key = key[5:]
            #from hyperspy.signal import BaseSignal
            #value = BaseSignal(**value)
        slugified_key = str(slugify(key, valid_variable_name=True))
        if isinstance(value, dict):
            if self.has_item(slugified_key):
                self.get_item(slugified_key).add_dictionary(
                    value,
                    double_lines=self._double_lines)
                return
            else:
                value = DictionaryTreeBrowser(
                    value,
                    double_lines=self._double_lines)
        super(DictionaryTreeBrowser, self).__setattr__(
            slugified_key,
            {'key': key, '_dtb_value_': value})

    def __len__(self):
        return len(
            [key for key in self.__dict__.keys() if not key.startswith("_")])

    def keys(self):
        """Returns a list of non-private keys.
        """
        return sorted([key for key in self.__dict__.keys()
                       if not key.startswith("_")])

    def as_dictionary(self):
        """Returns its dictionary representation.
        """
        from hyperspy.signal import BaseSignal
        par_dict = {}
        for key_, item_ in self.__dict__.items():
            if not isinstance(item_, types.MethodType):
                key = item_['key']
                if key in ["_db_index", "_double_lines"]:
                    continue
                if isinstance(item_['_dtb_value_'], DictionaryTreeBrowser):
                    item = item_['_dtb_value_'].as_dictionary()
                elif isinstance(item_['_dtb_value_'], BaseSignal):
                    item = item_['_dtb_value_']._to_dictionary()
                    key = '_sig_' + key
                elif hasattr(item_['_dtb_value_'], '_to_dictionary'):
                    item = item_['_dtb_value_']._to_dictionary()
                else:
                    item = item_['_dtb_value_']
                par_dict.__setitem__(key, item)
        return par_dict

    def has_item(self, item_path):
        """Given a path, return True if it exists.
        The nodes of the path are separated using periods.
        Parameters
        ----------
        item_path : Str
            A string describing the path with each item separated by
            full stops (periods)
        Examples
        --------
        >>> dict = {'To' : {'be' : True}}
        >>> dict_browser = DictionaryTreeBrowser(dict)
        >>> dict_browser.has_item('To')
        True
        >>> dict_browser.has_item('To.be')
        True
        >>> dict_browser.has_item('To.be.or')
        False
        """
        if isinstance(item_path, str):
            item_path = item_path.split('.')
        else:
            item_path = copy.copy(item_path)
        attrib = item_path.pop(0)
        if hasattr(self, attrib):
            if len(item_path) == 0:
                return True
            else:
                item = self[attrib]
                if isinstance(item, type(self)):
                    return item.has_item(item_path)
                else:
                    return False
        else:
            return False

    def get_item(self, item_path, default=None):
        """Given a path, return it's value if it exists, or default
        value if missing.
        The nodes of the path are separated using periods.
        Parameters
        ----------
        item_path : Str
            A string describing the path with each item separated by
            full stops (periods)
        default :
            The value to return if the path does not exist.
        Examples
        --------
        >>> dict = {'To' : {'be' : True}}
        >>> dict_browser = DictionaryTreeBrowser(dict)
        >>> dict_browser.has_item('To')
        True
        >>> dict_browser.has_item('To.be')
        True
        >>> dict_browser.has_item('To.be.or')
        False
        """
        if isinstance(item_path, str):
            item_path = item_path.split('.')
        else:
            item_path = copy.copy(item_path)
        attrib = item_path.pop(0)
        if hasattr(self, attrib):
            if len(item_path) == 0:
                return self[attrib]
            else:
                item = self[attrib]
                if isinstance(item, type(self)):
                    return item.get_item(item_path, default)
                else:
                    return default
        else:
            return default

    def __contains__(self, item):
        return self.has_item(item_path=item)

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def set_item(self, item_path, value):
        """Given the path and value, create the missing nodes in
        the path and assign to the last one the value
        Parameters
        ----------
        item_path : Str
            A string describing the path with each item separated by a
            full stops (periods)
        Examples
        --------
        >>> dict_browser = DictionaryTreeBrowser({})
        >>> dict_browser.set_item('First.Second.Third', 3)
        >>> dict_browser
        └── First
           └── Second
                └── Third = 3
        """
        if not self.has_item(item_path):
            self.add_node(item_path)
        if isinstance(item_path, str):
            item_path = item_path.split('.')
        if len(item_path) > 1:
            self.__getattribute__(item_path.pop(0)).set_item(
                item_path, value)
        else:
            self.__setattr__(item_path.pop(), value)

    def add_node(self, node_path):
        """Adds all the nodes in the given path if they don't exist.
        Parameters
        ----------
        node_path: str
            The nodes must be separated by full stops (periods).
        Examples
        --------
        >>> dict_browser = DictionaryTreeBrowser({})
        >>> dict_browser.add_node('First.Second')
        >>> dict_browser.First.Second = 3
        >>> dict_browser
        └── First
            └── Second = 3
        """
        keys = node_path.split('.')
        dtb = self
        for key in keys:
            if dtb.has_item(key) is False:
                dtb[key] = DictionaryTreeBrowser()
            dtb = dtb[key]

    def __next__(self):
        """
        Standard iterator method, updates the index and returns the
        current coordiantes
        Returns
        -------
        val : tuple of ints
            Returns a tuple containing the coordiantes of the current
            iteration.
        """
        if len(self) == 0:
            raise StopIteration
        if not hasattr(self, '_db_index'):
            self._db_index = 0
        elif self._db_index >= len(self) - 1:
            del self._db_index
            raise StopIteration
        else:
            self._db_index += 1
        key = list(self.keys())[self._db_index]
        return key, getattr(self, key)

    def __iter__(self):
        return self

def file_select(dp1,dp2,current_directory_output,dir = '/home/jovyan/notebooks/TEMdata'):    
    os.chdir(dir)

    global resetChange
    resetChange = 0

    def setFileDialog():
        
        current_directory = os.getcwd()
        
        directory_list = os.listdir(current_directory)
        directories = [x for x in directory_list if os.path.isdir(os.path.join(current_directory,x))]
        directories.append('.')
        directories.append('..')

        files=[]
        for x in directory_list:    
            if os.path.isfile(os.path.join(current_directory,x)):
                root, ext = os.path.splitext(x)
                if ext in ['.dm3', '.qf3','hf5']:
                    files.append(x)

        dp1.options = directories
        if len(files) == 0:
            files = [' ']
        dp2.options = files

        dp1.value = '.'
        dp2.value = files[0]

        current_directory_output.value = current_directory
    setFileDialog()


   
    def on_change(change):
        global resetChange
        if change['type'] == 'change' and change['name'] == 'value':
            if resetChange == 0:
                resetChange=1
                current_directory = os.path.normpath(os.path.join(os.getcwd(), change['new']))
                os.chdir(current_directory)
                file_name = ''
                setFileDialog()
            else:
                if change['new'] == '.': # if we are done with the update
                    resetChange=0            

    
    dp1.observe(on_change)
    
    
import ipywidgets as widgets

class FileBrowser(object):
    def __init__(self):
        self.path = os.getcwd()
        self._update_files()

    def _update_files(self):
        self.files = list()
        self.dirs = list()
        if(os.path.isdir(self.path)):
            for f in os.listdir(self.path):
                ff = os.path.join(self.path, f)
                if os.path.isdir(ff):
                    self.dirs.append(f)
                else:
                    self.files.append(f)

    def widget(self):
        box = widgets.VBox()
        self._update(box)
        return box

    def _update(self, box):

        def on_click(b):
            if b.description == '..':
                self.path = os.path.split(self.path)[0]
            else:
                self.path = os.path.join(self.path, b.description)
            self._update_files()
            self._update(box)

        buttons = []
        if self.files:
            button = widgets.Button(description='..', background_color='#d0d0ff')
            button.on_click(on_click)
            buttons.append(button)
        for f in self.dirs:
            button = widgets.Button(description=f, background_color='#d0d1ffff')
            button.on_click(on_click)
            buttons.append(button)
        for f in self.files:
            button = widgets.Button(description=f)
            button.on_click(on_click)
            buttons.append(button)
        box.children = tuple([widgets.HTML("<h2>%s</h2>" % (self.path,))] + buttons)

