import numpy as np

from scipy import integrate
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage.filters import gaussian_filter

from scipy import constants
import matplotlib.pyplot as plt

from scipy.optimize import leastsq  ## leastsqure fitting routine fo scipy

import pkg_resources, pickle

### And we use the image tool library of Quantifit
import pyTEMlib.file_tools  as ft
from pyTEMlib.config_dir import data_path

major_edges = ['K1', 'L3', 'M5', 'N5']
all_edges = ['K1','L1','L2','L3','M1','M2','M3','M4','M5','N1', 'N2','N3','N4','N5','N6','N7','O1','O2','O3','O4','O5','O6','O7', 'P1', 'P2', 'P3']
first_close_edges = ['K1', 'L3', 'M5', 'M3', 'N5', 'N3']


# KroegerCore(edata,adata,epsdata,ee,thick, relativistic =True)
#KroegerCore2(edata,adata,epsdata,acceleration_voltage_keV,thickness, relativistic =True)
# get_waveLength(E0)

#plotDispersion(plotdata, units, adata, edata, title, maxP, ee, EF = 4., Ep= 16.8, Es = 0, IBT = [])
# drude(tags, e, ep, ew, tnm, eb)
# Drude(Ep, Eb, gamma, e)
# DrudeLorentz(epsInf,leng, Ep, Eb, gamma, e, Amplitude)
# ZLfunc( p,  x)

def get_Xsections(Z=0):
    """
    ####
    # reads X-ray fluoresecent cross sections from a pickle file.
    ####
    Input: nothing or atomic number
    Output: dictionary
            of a element or of all elements if Z = 0

    """
    pkl_file = open(data_path+'/edges_db.pkl', 'rb')

    Xsections = pickle.load(pkl_file)
    pkl_file.close()
    Z = int(Z)
    if Z <1:
        return Xsections
    else:
        Z = int(Z)
        return Xsections[str(Z)]

def get_Z(Z):
    """
    returns the atomic number independent of input as a string or number
    
    input:
    Z: atomic number of chemical symbol (0 if not valid)
    """
    Xsections = get_Xsections()
    
    ZZ_out = 0
    if str(Z).isdigit(): 
        Z_out = Z
    elif isinstance(Z, str):
        for key in Xsections:
            if Xsections[key]['name'].lower() == Z.lower(): ## Well one really should know how to write elemental 
                Z_out = int(key)
    return Z_out

class elemental_edges(object):
    """
        The necessary initializtion parameter are:
        ax: matplotlib axis
     
        
        There is an optional parameter maximum_chemical_shift which allows to change 
        the energy range in which the edges are searched.
        
        available functions:
        - update(): updates the drawing of ionization edges
        - set_edge(Z) : changes atomic number and updates everything accordingly
        - disconnect: makes everythign invisible and stops drawing
        - reconnect: undo of disconnect
        
        usage:
        >>> fig, ax = plt.subplots()
        >>> ax.plot(energy_scale, spectrum) 
        >>> Z= 42
        >>> cursor = elemental_edges(ax, Z)
        
        
        see Chapter4 'CH4-Working_with_X-Sections' notebook
    """
    def __init__(self, ax, Z):
        self.ax = ax
        self.labels = None
        self.lines = None
        self.Z = get_Z(Z)
        self.color = 'black'
        self.Xsections = get_Xsections()
        self.cid = ax.figure.canvas.mpl_connect('draw_event', self.onresize)
        #self.update() is not necessary because of a drawing event is issued
        
    def set_edge(self,Z):
        self.Z = get_Z(Z)
        if self.cid == None:
            self.cid = ax.figure.canvas.mpl_connect('draw_event', self.onresize)
        self.update()
        
    def onresize(self, event):
        self.update()
        
    def update(self):
        if self.labels != None:
            for label in self.labels:
                label.remove()
        if self.lines != None:
            for line in self.lines:
                line.remove()
        self.labels = [] ; self.lines =[] 
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
  
        element = str(self.Z)
        Xsections = self.Xsections
        for key in all_edges:
            if key in Xsections[element]:
                if 'onset' in Xsections[element][key]:
                    x = Xsections[element][key]['onset']
                    if x > x_min and x < x_max:
                        if key in first_close_edges:
                            label2 = self.ax.text(x, y_max,f"{Xsections[element]['name']}-{key}",
                                                  verticalalignment='top', rotation = 0, color = self.color)
                        else:
                            label2 = self.ax.text(x, y_max,f"\n{Xsections[element]['name']}-{key}",
                                                  verticalalignment='top', color = self.color)
                        line2 = self.ax.axvline(x,ymin = 0,ymax = 1,color=self.color)
                        
                        self.labels.append(label2)
                        self.lines.append(line2)
                    
        
    def reconnect(self):
        self.cid = ax.figure.canvas.mpl_connect('draw_event', self.onresize)
        self.update()
        
    def disconnect(self):
        if self.labels != None:
            for label in self.labels:
                label.remove()
        if self.lines != None:
            for line in self.lines:
                line.remove()
        self.labels = None
        self.lines = None
        self.ax.figure.canvas.mpl_disconnect(self.cid)
        
        
class Region_Selector(object):
    """
        Selects fitting region and the regions that are excluded for each edge.
        
        Select a region with a spanSelector and then type 'a' for all of the fitting region or a number for the edge you want to define the region excluded from the fit (solid state effects).

        see Chapter4 'CH4-Working_with_X-Sections,ipynb' notebook 

    """
    def __init__(self, ax):
        self.ax = ax
        self.regions = {}
        self.rect = None
        self.xmin = 0
        self.xwidth = 0
         
        self.span = SpanSelector(ax, self.onselect1, 'horizontal', useblit = True,
                    rectprops=dict(alpha=0.5, facecolor='red'), span_stays=True)
        self.cid = ax.figure.canvas.mpl_connect('key_press_event', self.click)
        self.draw = ax.figure.canvas.mpl_connect('draw_event', self.onresize)
        
    def onselect1(self, xmin, xmax):
        self.xmin =  xmin
        self.width = xmax-xmin
    
    def onresize(self, event):
        self.update()
        
    def delete_region(self, key):
        if key in self.regions:
            if 'Rect' in self.regions[key]:
                self.regions[key]['Rect'].remove()
                self.regions[key]['Text'].remove()
            del(self.regions[key]) 
        
    def update(self):
        
        y_min, y_max = self.ax.get_ylim()
        for key in self.regions:
            if 'Rect' in self.regions[key]:
                self.regions[key]['Rect'].remove()
                self.regions[key]['Text'].remove()
       
            xmin = self.regions[key]['xmin']
            width = self.regions[key]['width']
            height = y_max-y_min 
            alpha = self.regions[key]['alpha']
            color = self.regions[key]['color']
            self.regions[key]['Rect'] =  Rectangle((xmin,y_min), width,height, 
                                                    edgecolor=color,alpha=alpha, facecolor=color)
            self.ax.add_patch(self.regions[key]['Rect'])

            self.regions[key]['Text'] = self.ax.text(xmin, y_max, self.regions[key]['text'],verticalalignment='top')
       
            
    def click(self, event):
        if str(event.key) in ['1','2','3','4','5','6']:
            key = str(event.key)
            text = 'exclude \nedge ' + key
            alpha = 0.5
            color = 'red'
        elif str(event.key) in ['a', 'A', 'b', 'B']:
            key = '0'
            color = 'blue'
            alpha = 0.2
            text = 'fit region'
        else:
            return
       
        if key not in self.regions:
            self.regions[key] = {}

        self.regions[key]['xmin'] = self.xmin
        self.regions[key]['width'] = self.width
        self.regions[key]['color'] = color
        self.regions[key]['alpha'] = alpha
        self.regions[key]['text'] = text
        
        self.update()
    
    
    def set_regions(self, region, start_x, width):
        if 'fit' in str(region):
            key = '0'
        if region in ['0', '1','2','3','4','5','6']:
            key = region
        if region in [0,1,2,3,4,5,6]:
            key = str(region)
             
            
        if key not in self.regions:
            self.regions[key] = {}
            if key in ['1','2','3','4','5','6']:
                self.regions[key]['text'] = 'exclude \nedge ' + key
                self.regions[key]['alpha'] = 0.5
                self.regions[key]['color'] = 'red'
            elif key == '0':
                self.regions[key]['text'] = 'fit region'
                self.regions[key]['alpha'] = 0.2
                self.regions[key]['color'] = 'blue'
            
        self.regions[key]['xmin'] = start_x
        self.regions[key]['width'] = width
            
        self.update()
            
    def get_regions(self):
        tags = {}
        for key in self.regions:
            if key == '0':
                area = 'fit_area'
            else:
                area =  key
            tags[area] = {}
            tags[area]['start_x'] = self.regions[key]['xmin']
            tags[area]['width_x'] = self.regions[key]['width']

        return tags
    
    def disconnect(self):
        del(self.span)
        self.ax.figure.canvas.mpl_disconnect(self.cid)
        #self.ax.figure.canvas.mpl_disconnect(self.draw)
        pass
        
 

    
class EdgesatCursor(object):
    """ 
        Adds a Cursor to a plot, which plots all major (possible) ionization edges at
        the cursor location if left (right) mosue button is clicked.
        
        The necessary initializtion parameter are:
        ax: matplotlib axis
        x: energy_scale of spectrum
        y: intensities of spectrum
        
        There is an optional parameter maximum_chemical_shift which allows to change 
        the energy range in which the edges are searched.
        
        usage:
        fig, ax = plt.subplots()
        ax.plot(energy_scale, spectrum)
        cursor = EdgesatCursor(ax, energy_scale, spectrum)
        
        see Chapter4 'CH4-Working_with_X-Sections' notebook
        
    """
    def __init__(self, ax, x, y,maximal_chemical_shift = 5 ):
        self.ax = ax
        self.ly = ax.axvline(x[0], color='k', alpha=0.2)  # the vert line
        self.marker, = ax.plot(x[0],y[0], marker="o", color="crimson", zorder=3) 
        self.x = x
        self.y = y
        self.txt = ax.text(0.7, 0.9, '', verticalalignment='bottom')
        self.select = 0
        self.label = None
        self.line =None
        self.cid = ax.figure.canvas.mpl_connect('button_press_event', self.click)
        self.mouse_cid = ax.figure.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.maximal_chemical_shift = maximal_chemical_shift
    def click(self, event):
        
        #print('click', event)
        if not event.inaxes: return
        x, y = event.xdata, event.ydata
        
        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        self.select = x
        
        y_min, y_max =self.ax.get_ylim()
        
        if self.label != None:
            self.label.remove()
            self.line.remove()
        if event.button == 1:
            self.label = self.ax.text(x, y_max,find_major_edges(event.xdata,self.maximal_chemical_shift),verticalalignment='top')
            self.line, = self.ax.plot([x,x],[y_min, y_max],color='black')
        if event.button == 3:
            self.line, = self.ax.plot([x,x],[y_min, y_max],color='black')
            self.label = self.ax.text(x, y_max,find_all_edges(event.xdata,self.maximal_chemical_shift),verticalalignment='top')
        self.ax.set_ylim(y_min, y_max)    
            
    def mouse_move(self, event):
        
        if not event.inaxes: return
        
        
        x, y = event.xdata, event.ydata
        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        self.select = x
        self.ly.set_xdata(x)
        self.marker.set_data([x],[y])
        self.txt.set_text(f'\n x={x:1.2f}, y={y:1.2g}\n')
        
        
        #self.ax.text(x, y*2,find_major_edges(x))
        self.txt.set_position((x,y))
        self.ax.figure.canvas.draw_idle()

    def del_edges(self):
        if self.label != None:
            self.label.remove()
            self.line.remove()
            self.label = None
    def disconnect(self):    
        self.ly.remove()
        self.marker.remove()
        self.txt.remove()

        self.ax.figure.canvas.mpl_disconnect(self.cid)
        self.ax.figure.canvas.mpl_disconnect(self.mouse_cid)
        
        
def list_all_edges(Z):
    element = str(Z)
    Xsections = get_Xsections()
    print('Major edges')
    for key in all_edges:
        if key in Xsections[element]:
            if 'onset' in Xsections[element][key]:
                print(f" {Xsections[element]['name']}-{key}: {Xsections[element][key]['onset']:8.1f} eV ")

def find_major_edges(edge_onset, maximal_chemical_shift = 5):
    text = ''
    Xsections = get_Xsections()
    for element in Xsections:
        for key in Xsections[element]:
            
            #if isinstance(Xsections[element][key], dict):
            if key in major_edges:
                
                #if 'onset' in Xsections[element][key]:
                #print(key, Xsections[element][key])
                if abs(Xsections[element][key]['onset'] - edge_onset) < maximal_chemical_shift:
                    #print(element, Xsections[element]['name'], key, Xsections[element][key]['onset'])
                    text = text+ f"\n {Xsections[element]['name']:2s}-{key}: {Xsections[element][key]['onset']:8.1f} eV "
                    
                    
    return text


def find_all_edges(edge_onset, maximal_chemical_shift = 5):
    text = ''
    Xsections = get_Xsections()
    for element in Xsections:
        for key in Xsections[element]:
            
            if isinstance(Xsections[element][key], dict):
            
                
                if 'onset' in Xsections[element][key]:
                
                    if abs(Xsections[element][key]['onset'] - edge_onset) < maximal_chemical_shift:
                        #print(element, Xsections[element]['name'], key, Xsections[element][key]['onset'])
                        text = text+ f"\n {Xsections[element]['name']:2s}-{key}: {Xsections[element][key]['onset']:8.1f} eV "
                    
    return text


                    
                    
def find_maxima(y,number_of_peaks):
    """
    find the first most prominent peaks 
    peaks are then sorted by energy

    input:
        y: array of (part) of spectrum
        number_of_peaks: int
    output:
        array of indices of peaks
    """
    blurred2 = gaussian_filter(y, sigma=2)
    peaks, _ = find_peaks(blurred2)
    prominences = peak_prominences(blurred2, peaks)[0]
    prominences_sorted = np.argsort(prominences)
    peaks = peaks[prominences_sorted[-number_of_peaks:]]
    
    peak_indices = np.argsort(peaks)
    return peaks[peak_indices]



def gauss(x, p): # p[0]==mean, p[1]= area p[2]==fwhm, 
    return p[1] * np.exp(-(x- p[0])**2/(2.0*( p[2]/2.3548)**2))
def lorentz(x, p):
    lorentz_peak = 0.5 * p[2]/np.pi/((x- p[0])**2+( p[2]/2)**2)
    return p[1]*lorentz_peak/lorentz_peak.max()
def ZL(x,p,pZL):
    pZl_local = pZL.copy()
    pZl_local[2] += p[0]
    pZl_local[5] += p[0]
    zero_loss = ZLfunc(pZl_local, x)
    return p[1]*zero_loss/zero_loss.max()



def model3(x, p, number_of_peaks, peak_shape, pZL, pin=[], restrictPos=0,restrictWidth=0):
    
    if pin == []:
        pin = p
    
    #if len([restrictPos]) == 1:
    #    restrictPos = [restrictPos]*number_of_peaks
    #if len([restrictWidth]) == 1:
    #    restrictWidth = [restrictWidth]*number_of_peaks
    y = np.zeros(len(x))
    
    for i in range(number_of_peaks):
        index = int(i*3)
        if restrictPos >0:
            if p[index] > pin[index]*(1.0+restrictPos):
                p[index] = pin[index]*(1.0+restrictPos)
            if p[index] <  pin[index]*(1.0-restrictPos):
                p[index] = pin[index]*(1.0-restrictPos)
            #print(p[index] , pin[index])
        
        p[index+1] = abs(p[index+1])
        print(p[index+1])
        p[index+2] = abs(p[index+2])
        if restrictWidth >0:
            if p[index+2] > pin[index+2]*(1.0+restrictWidth):
                p[index+2] = pin[index+2]*(1.0+restrictWidth)
                
        if peak_shape[i] == 'Lorentzian':
            y  = y + lorentz(x, p[index:])
        elif peak_shape[i] == 'ZL':
            
            y = y+ ZL(x,p[index:],pZL)
        else:
            y  = y + gauss(x, p[index:])
    return y

def sort_peaks(p, peak_shape):
    number_of_peaks = int(len(p)/3)
    p3  = np.reshape(p, (number_of_peaks, 3))
    sort_pin = np.argsort(p3[:,0])
    #print(sort_pin)

    p = p3[sort_pin].flatten()
    peak_shape = np.array(peak_shape)[sort_pin].tolist()
    
    return (p, peak_shape)



def addPeaks(x,y,peaks, pin_in=[], peak_shape_in=[], shape = 'Gaussian'):
    
    pin = pin_in.copy()
    peak_shape = peak_shape_in.copy()
    if isinstance(shape, str):  # if peak_shae is only a string make a list of it.
        shape = [shape]
    
    if len(shape)==1:
        shape = shape*len(peaks)
    for i, peak in enumerate(peaks):
        pin.append(x[peak])
        pin.append(y[peak])
        pin.append(.3)
        peak_shape.append(shape[i])

    return pin, peak_shape
    
def fit_model(x,y,pin, number_of_peaks, peak_shape, pZL, restrictPos = 0, restrictWidth=0):
    
    pin_original  = pin.copy()
    def residuals3(p, x, y  ):
        err = (y-model3(x,p, number_of_peaks, peak_shape, pZL, pin_original, restrictPos,restrictWidth))/np.sqrt(np.abs(y))
        return err 

    p, cov = leastsq(residuals3, pin,  args = (x,y) )
    p2 = p.tolist()
    p3  = np.reshape(p2, (number_of_peaks, 3))
    sort_pin = np.argsort(p3[:,0])
    #print(sort_pin)

    #p = p3[sort_pin].flatten()
    #peak_shape = np.array(peak_shape)[sort_pin].tolist()

    return p, peak_shape

def EELSdictionary(zero_loss_fit_width = 0, verbose = False):
    spectrum_tags = ft.open_file()#os.path.join(current_directory,filename))
    
    spectrum_tags['data'] = spectrum_tags['spec']
    spectrum_tags['axis']['0']['pixels'] = len(spectrum_tags['spec'])
    scale_p = spectrum_tags['axis']['0']
    spectrum_tags['energy_scale'] = np.arange(scale_p['pixels'])*scale_p['scale']+scale_p['Origin']
    if verbose:
        print('Energy Scale')
        print(f"Dispersion [eV/pixel] : {scale_p['scale']:.2f} eV ")
        print(f"Offset [eV] : {scale_p['Origin']:.2f} eV ")
        print(f"Maximum energy [eV] : {spectrum_tags['energy_scale'][-1]:.2f} eV ")


    energy_scale = spectrum_tags['energy_scale'] 
    spectrum = spectrum_tags['spec']
    spectrum[np.where(spectrum<1e-10)] = 1e-10

    FWHM, dE = fixE(spectrum, energy_scale)
    if verbose:
        print(f'FWHM {FWHM:.3f} eV')
        print(f'Energy scale is shifted by {dE:.3f} eV')
    energy_scale = energy_scale-dE
    specSum= spectrum.sum()
    spectrum = spectrum/specSum*1e6 # Now in ppm
    spectrum_tags['axis']['0']['Origin'] = energy_scale[0]
    spectrum_tags['spectrum_ppm'] = spectrum
    spectrum_tags['energy_scale'] = energy_scale
    spectrum_tags['resolution'] = FWHM

    if zero_loss_fit_width == 0:
        zero_loss_fit_width = FWHM * 4.
    

    zLoss, pZL = resolution_function(energy_scale, spectrum, zero_loss_fit_width)

    spectrum_tags['pZL'] = pZL
    spectrum_tags['zero_loss'] = zLoss

    #######################################
    ## Important Experimental Parameters ##
    #######################################

    spectrum_tags['eels_parameter'] = {}
    spectrum_tags['eels_parameter']['integration_time_s'] = spectrum_tags['integration_time']
    spectrum_tags['eels_parameter']['number_of_frames'] = spectrum_tags['number_of_frames']
    spectrum_tags['eels_parameter']['acceleration_voltage_V'] = spectrum_tags['acceleration_voltage']
    spectrum_tags['eels_parameter']['collectionAngle_mrad'] =30.
    spectrum_tags['eels_parameter']['convergenceAngle_mrad'] =10.
    spectrum_tags['eels_parameter']['integration_time'] = spectrum_tags['integration_time']

    return spectrum_tags
def fixE( spec, energy):
        
    start = np.searchsorted(energy,-2)
    end   = np.searchsorted(energy, 2)
    startx = np.argmax(spec[start:end])+start

    end = startx+3
    start = startx-3
    for i in range(10):
        if spec[startx-i]<0.3*spec[startx]:
            start = startx-i
        if spec[startx+i]<0.3*spec[startx]:
            end = startx+i
    if end-start<3:
        end = startx+2
        start = startx2
    
    x = np.array(energy[start:end])
    y = np.array(spec[start:end]).copy()
    
    y[np.nonzero(y<=0)] = 1e-12

    def gauss(x, p): # p[0]==mean, p[1]= area p[2]==fwhm, 
        return p[1] * np.exp(-(x- p[0])**2/(2.0*( p[2]/2.3548)**2))
    
    p0 = [energy[startx],1000.0,(energy[end]-energy[start])/3.] # Inital guess is a normal distribution
    errfunc = lambda p, x, y: (gauss(x, p) - y)/np.sqrt(y) # Distance to the target function
    p1, success = leastsq(errfunc, p0[:], args=(x, y))

    fit_mu, area, FWHM = p1
    
    return FWHM, fit_mu

def resolution_function(energy_scale, spectrum, width):
    guess = [ 0.2, 1000,0.02,0.2, 1000,0.2 ]
    p0 = np.array(guess)

    start = np.searchsorted(energy_scale,-width/2.)
    end   = np.searchsorted(energy_scale, width/2.)
    x = energy_scale[start:end]
    y = spectrum[start:end]

    def ZL2(p, y, x):
        err = (y - ZLfunc(p,  x))#/np.sqrt(y)
        return err

    def ZL(p, y, x):
            
            if p[2]>x[-1]*.8:
                p[2]=x[-1]*.8
            if p[2]<x[0]*.8:
                p[2]=x[0]*.8

            if p[5]>x[-1]*.8:
                p[5]=x[-1]*.8
            if p[5]<x[0]*.8:
                p[5]=x[0]*.8

            if len(p) > 6:
                p[7] = abs(p[7])
                if abs(p[7])>(p[1]+p[4])/10:
                    p[7] = abs(p[1]+p[4])/10
                if abs(p[8])>1:
                    p[8]=p[8]/abs(p[8])
                p[6]=abs(p[6])
                p[9]=abs(p[9])
            
            p[0] = abs(p[0])
            p[3] = abs(p[3])
            if  p[0]> (x[-1]-x[0])/2.0:
                p[0] = x[-1]-x[0]/2.0
            if  p[3]> (x[-1]-x[0])/2.0:
                p[3] = x[-1]-x[0]/2.0
                
            y[y<0]= 0.   ## no negative numbers in sqrt below
            err = (y - ZLfunc(p,  x))/np.sqrt(y)
            
            return err

    pZL, lsq = leastsq(ZL2, p0, args=(y, x), maxfev=2000)
    print('Fit of a Product of two Lorentzians')
    print('Positions: ',pZL[2],pZL[5], 'Distance: ',pZL[2]-pZL[5])
    print('Width: ', pZL[0],pZL[3])
    print('Areas: ', pZL[1],pZL[4])
    err = (y - ZLfunc(pZL,  x))/np.sqrt(y)
    print (f'Goodness of Fit: {sum(err**2)/len(y)/sum(y)*1e2:.5}%')

    zLoss = ZLfunc(pZL,  energy_scale)
    
    return zLoss, pZL


def get_waveLength(E0):
    eV = constants.e * E0 
    return constants.h/np.sqrt(2*constants.m_e*eV*(1+eV/(2*constants.m_e*constants.c**2)))

def KroegerCore(edata,adata,epsdata,ee,thick, relativistic =True):
    """
    
    This function calculates the differential scattering probability
     .. math::
        \\frac{d^2P}{d \\Omega dE}
    of the low-loss region for total loss and volume plasmon loss
      
    Args:
       edata (array): energy scale [eV]
       adata (array): angle or momentum range [rad]
       psdata (array): dielectric function
       ee (float): acceleration voltage [keV]
       thick (float): thickness in m
    
    Returns:
       P (numpy array 2d): total loss probability
       Pvol (numpy array 2d): volume loss probability

    #d^2P/(dEd\Omega) = \frac{1}{\pi^2 a_0 m_0 v^2} \Im \left[ \frac{t\mu^2}{\varepsilon \phi^2 } \right]

    
    # ee = 200 #keV
    # thick = 32.0# nm

    """

    thick  = thick * 1e-9; #% input thickness now in m 
    #%Define constants
    #%ec = 14.4;
    m_0 = constants.value(u'electron mass') #% REST electron mass in kg
    h = constants.Planck; #% Planck's constant
    hbar = constants.hbar;
    
    c = constants.speed_of_light #% speed of light m/s
    bohr = constants.value(u'Bohr radius'); #% Bohr radius in meters
    e = constants.value(u'elementary charge')#% electron charge in Coulomb
    print('hbar =', hbar ,' [Js] =', hbar/e ,'[ eV s]')

    
    #%Calculate fixed terms of equation
    va = 1 - (511./(511.+ee))**2; #% ee is incident energy in keV
    v = c*np.sqrt(va);
    beta =  v/c; # non relativistic for =1

    if relativistic:
        gamma = 1./np.sqrt(1-beta**2); #
    else:
        gamma = 1 #set = 1 to correspond to E+B & Siegle

    momentum = m_0*v*gamma; #%used for xya, E&Bnhave no gamma 
    
    ###### Define mapped variables
    
    #%Define independant variables E, Theta
    [E,Theta] = np.meshgrid(edata+1e-12,adata);
    #%Define CONJUGATE dielectric function variable eps
    [eps,ignore] = np.meshgrid(np.conj(epsdata),adata)

    ###### Calculate lambda in equation EB 2.3
    Theta2 = Theta**2+1e-15
    ThetaE = E *e/ momentum / v;
    ThetaE2 = ThetaE**2;
    
    lambda2 = Theta2 - eps * ThetaE2 * beta**2; #%Eq 2.3

    lambd = np.sqrt(lambda2);
    if (np.real(lambd) < 0).any() :
        print(' error negative lambda');

    ###### Calculate lambda0 in equation EB 2.4
    #% According to Kröger real(lambda0) is defined as positive!
    
    phi2 = lambda2 + ThetaE2; #%Eq. 2.2
    lambda02 = Theta2 - ThetaE2 * beta**2; # %eta=1 %Eq 2.4
    lambda02[lambda02<0]=0
    lambda0=np.sqrt(lambda02);
    if not(np.real(lambda0) >= 0).any() :
        print(' error negative lambda0');
    
    de = thick* E *e/2.0/hbar / v; #%Eq 2.5

    xya = lambd*de/ThetaE; #%used in Eqs 2.6, 2.7, 4.4
    

    lplus = lambda0*eps + lambd*np.tanh(xya); # %eta=1 %Eq 2.6
    lminus = lambda0*eps + lambd/np.tanh(xya); #  %eta=1 %Eq 2.7

    mue2 = 1 - (eps*beta**2); #%Eq. 4.5
    phi20 = lambda02 + ThetaE2; #%Eq 4.6
    phi201 = Theta2 + ThetaE2 *(1-(eps+1)*beta**2); #%eta=1, eps-1 in E+B Eq.(4.7)
    

    
    
    #%Eq 4.2
    A1 = phi201**2/eps;
    A2 = np.sin(de)**2/lplus + np.cos(de)**2/lminus;
    A = A1*A2;
    
    #%Eq 4.3
    B1 = beta**2*lambda0*ThetaE*phi201;
    B2 = (1./lplus - 1./lminus)*np.sin(2.*de);
    B = B1*B2;
    
    #%Eq 4.4
    C1 = -beta**4*lambda0*lambd*ThetaE2;
    C2 = np.cos(de)**2*np.tanh(xya)/lplus;
    C3 = np.sin(de)**2/np.tanh(xya)/lminus;
    C = C1*(C2+C3);
    

    #%Put all the pieces together...
    Pcoef = e/(bohr*np.pi**2*m_0*v**2);
    
    Pv = thick*mue2/eps/phi2;        
    
    Ps1 = 2.*Theta2*(eps-1)**2/phi20**2/phi2**2; #%ASSUMES eta=1
    Ps2 = hbar/momentum;
    Ps3 = A + B + C;

    Ps = Ps1*Ps2*Ps3;
    
    #print(Pv.min(),Pv.max(),Ps.min(),Ps.max())
    #%Calculate P and Pvol (volume only)
    dTheta = adata[1]-adata[0]
    scale = np.sin(np.abs(Theta))*dTheta*2*np.pi
    
    P = Pcoef*np.imag(Pv - Ps); #%Eq 4.1
    Pvol = Pcoef*np.imag(Pv)*scale

    lplusMin = edata[np.argmin(np.real(lplus), axis=1)]
    lminusMin = edata[np.argmin(np.imag(lminus), axis=1)]

    
    Psimple  = Pcoef*np.imag(1/eps) *thick/(Theta2+ThetaE2)*scale ## Watchit eps is conjugated dielectric function


    return P, P*scale*1e2,Pvol*1e2, Psimple*1e2#,lplusMin,lminusMin

def KroegerCore2(edata,adata,epsdata,acceleration_voltage_keV,thickness, relativistic =True):
    """
    
    This function calculates the differential scattering probability
     .. math::
        \\frac{d^2P}{d \\Omega dE}
    of the low-loss region for total loss and volume plasmon loss
      
    Args:
       edata (array): energy scale [eV]
       adata (array): angle or momentum range [rad]
       psdata (array): dielectric function
       acceleration_voltage_keV (float): acceleration voltage [keV]
       thickness (float): thickness in nm
    
    Returns:
       P (numpy array 2d): total loss probability
       Pvol (numpy array 2d): volume loss probability

       return P, P*scale*1e2,Pvol*1e2, Psimple*1e2

    #d^2P/(dEd\Omega) = \frac{1}{\pi^2 a_0 m_0 v^2} \Im \left[ \frac{t\mu^2}{\varepsilon \phi^2 } \right]

    # Internally everything is calculated in SI units
    # acceleration_voltage_keV = 200 #keV
    # thick = 32.0*10-9 # m

    """

    #%adjust input to SI units
    wavelength = get_waveLength(acceleration_voltage_keV*1e3) #in m
    thickness  = thickness * 1e-9; #% input thickness now in m 
    
    #%Define constants
    #%ec = 14.4;
    m_0 = constants.value(u'electron mass') #% REST electron mass in kg
    h = constants.Planck; #% Planck's constant
    hbar = constants.hbar;
    
    c = constants.speed_of_light #% speed of light m/s
    bohr = constants.value(u'Bohr radius'); #% Bohr radius in meters
    e = constants.value(u'elementary charge')#% electron charge in Coulomb
    #print('hbar =', hbar ,' [Js] =', hbar/e ,'[ eV s]')

    
    #%Calculate fixed terms of equation
    va = 1 - (511./(511.+acceleration_voltage_keV))**2; #% acceleration_voltage_keV is incident energy in keV
    v = c*np.sqrt(va);

    if relativistic:
        beta =  v/c; # non relativistic for =1
        gamma = 1./np.sqrt(1-beta**2); #
        
    else:
        beta = 1 
        gamma = 1 #set = 1 to correspond to E+B & Siegle

    momentum = m_0*v*gamma; #%used for xya, E&Bnhave no gamma 
    
    ###### Define mapped variables
    
    #%Define independant variables E, Theta
    [E,Theta] = np.meshgrid(edata+1e-12,adata);
    #%Define CONJUGATE dielectric function variable eps
    [eps,ignore] = np.meshgrid(np.conj(epsdata),adata)

    ###### Calculate lambda in equation EB 2.3
    Theta2 = Theta**2+1e-15

    ThetaE = E *e/ momentum / v; # critical angle
       
    lambda2 = Theta2 - eps * ThetaE**2 * beta**2; #%Eq 2.3

    lambd = np.sqrt(lambda2);
    if (np.real(lambd) < 0).any() :
        print(' error negative lambda');

    ###### Calculate lambda0 in equation EB 2.4
    #% According to Kröger real(lambda0) is defined as positive!
    
    phi2 = lambda2 + ThetaE**2; #%Eq. 2.2
    lambda02 = Theta2 - ThetaE**2 * beta**2; # %eta=1 %Eq 2.4
    lambda02[lambda02<0]=0
    lambda0=np.sqrt(lambda02);
    if not(np.real(lambda0) >= 0).any() :
        print(' error negative lambda0');
    
    de = thickness* E *e/(2.0 * hbar * v); #%Eq 2.5
    xya = lambd*de/ThetaE; #%used in Eqs 2.6, 2.7, 4.4
    
    lplus = lambda0*eps + lambd*np.tanh(xya); # %eta=1 %Eq 2.6
    lminus = lambda0*eps + lambd/np.tanh(xya); #  %eta=1 %Eq 2.7

    mue2 = 1 - (eps*beta**2); #%Eq. 4.5
    phi20 = lambda02 + ThetaE**2; #%Eq 4.6
    phi201 = Theta2 + ThetaE**2 *(1-(eps+1)*beta**2); #%eta=1, eps-1 in E+B Eq.(4.7)
      
    
    #%Eq 4.2
    A1 = phi201**2/eps;
    A2 = np.sin(de)**2/lplus + np.cos(de)**2/lminus;
    A = A1*A2;
    
    #%Eq 4.3
    B1 = beta**2*lambda0*ThetaE*phi201;
    B2 = (1./lplus - 1./lminus)*np.sin(2.*de);
    B = B1*B2;
    
    #%Eq 4.4
    C1 = -beta**4*lambda0*lambd*ThetaE**2;
    C2 = np.cos(de)**2*np.tanh(xya)/lplus;
    C3 = np.sin(de)**2/np.tanh(xya)/lminus;
    C = C1*(C2+C3);
    

    #%Put all the pieces together...
    Pcoef = e/(bohr*np.pi**2*m_0*v**2);
    
    Pv = thickness*mue2/eps/phi2;        
    
    Ps1 = 2.*Theta2*(eps-1)**2/phi20**2/phi2**2; #%ASSUMES eta=1
    Ps2 = hbar/momentum;
    Ps3 = A + B + C;

    Ps = Ps1*Ps2*Ps3;
    
    #print(Pv.min(),Pv.max(),Ps.min(),Ps.max())
    #%Calculate P and Pvol (volume only)
    dTheta = adata[1]-adata[0]
    scale = np.sin(np.abs(Theta))*dTheta*2*np.pi
    
    P = Pcoef*np.imag(Pv - Ps); #%Eq 4.1
    Pvol = Pcoef*np.imag(Pv)*scale

    lplusMin = edata[np.argmin(np.real(lplus), axis=1)]
    lminusMin = edata[np.argmin(np.imag(lminus), axis=1)]

    
    Psimple  = Pcoef*np.imag(1/eps) *thickness/(Theta2+ThetaE**2)*scale ## Watch it: eps is conjugated dielectric function


    return P, P*scale*1e2,Pvol*1e2, Psimple*1e2#,lplusMin,lminusMin

def Drude(Ep, Eb, gamma, e):

    eps = 1 - (Ep**2-Eb*e*1j)/(e**2+2*e*gamma*1j) #Mod Drude term
    return eps    

def DrudeLorentz(epsInf,leng, Ep, Eb, gamma, e, Amplitude):
    eps = epsInf
    for i in range(leng):
        eps = eps+Amplitude[i]*(1/(e+Ep[i]+gamma[i]*1j)-1/(e-Ep[i]+gamma[i]*1j))

    return eps
    
def plotDispersion(plotdata, units, adata, edata, title, maxP, ee, EF = 4., Ep= 16.8, Es = 0, IBT = []):

    [X,Y] = np.meshgrid(edata+1e-12,adata[1024:2048]*1000);
    
    Z=plotdata
    lev= np.array([0.01,0.05,0.1,0.25,0.5,1,2,3,4,4.9])*maxP/5


    wavelength = get_waveLength(ee)
    q = adata[1024:2048]/(wavelength *1e9) #in [1/nm]
    scale = np.array([0, adata[-1], edata[0] , edata[-1]])
    eV2Hertz = constants.value('electron volt-hertz relationship')
    
    if units[0] == 'mrad':
        units[0] = 'scattering angle [mrad]'
        scale[1] = scale[1] *1000.
        light_line = constants.c * adata # for mrad
    elif units[0] == '1/nm':
        units[0] ='scattering vector [1/nm]'
        scale[1] = scale[1] /(wavelength*1e9)
        light_line=1/(constants.c  / eV2Hertz)*1e-9

    if units[1] == 'eV':
        units[1] = 'energy loss [eV]'

    if units[2] == 'ppm':
        units[2] = 'probability [ppm]'
    if units[2] == '1/eV':
        units[2] = 'probability [eV$^{-1}$ srad$^{-1}$]'

    alpha = 3./5.*EF/Ep

    ax2 = plt.gca()
    fig2 = plt.gcf()
    im = ax2.imshow(Z.T, clim= (0,maxP), origin='lower', aspect='auto',extent=scale)
    co = ax2.contour(Y,X,Z, levels=lev, colors='k', origin='lower')#,extent=(-ang*1000.,ang*1000.,edata[0],edata[-1]))#, vmin = Pvol.min(), vmax = 1000)

    
    fig2.colorbar(im, ax=ax2,label=units[2])
    
    ax2.plot(adata,light_line, c='r', label='light line')
    #ax2.plot(edata*light_line*np.sqrt(np.real(epsdata)),edata, color='steelblue', label='$\omega = c q \sqrt{\epsilon_2}$')

    ax2.plot(q,Ep_disp,c='r')
    ax2.plot([11.5*light_line,0.12],[11.5,11.5], c='r')

    ax2.text(.05,11.7,'surface plasmon', color = 'r')
    ax2.plot([0.0,0.12],[16.8,16.8], c='r')
    ax2.text(.05,17,'volume plasmon', color = 'r')
    ax2.set_xlim(0, scale[1])
    ax2.set_ylim(0, 20)
    # Interband transitions
    ax2.plot([0.0,0.25],[4.2,4.2], c='g', label='interband transitions')
    ax2.plot([0.0,0.25],[5.2,5.2], c='g')
    ax2.set_ylabel (units[1])
    ax2.set_xlabel (units[0])
    ax2.legend(loc='lower right')

def ZLfunc( p,  x):

        p[0] = abs(p[0])
        
        gauss1 = np.zeros(len(x))
        gauss2 = np.zeros(len(x))
        lorentz3 = np.zeros(len(x))
        lorentz = ((0.5 *  p[0]* p[1]/3.14)/((x- p[2])**2+(( p[0]/2)**2)))
        lorentz2 = ((0.5 *  p[3]* p[4]/3.14)/((x- (p[5]))**2+(( p[3]/2)**2)))
        if len(p)> 6:
            lorentz3 = (0.5 *  p[6]* p[7]/3.14)/((x- p[8])**2+( p[6]/2)**2)
            gauss2 = p[10]*np.exp(-(x- p[11])**2/(2.0*( p[9]/2.3548)**2))#((0.5 *  p[9]* p[10]/3.14)/((x- (p[11]))**2+(( p[9]/2)**2)))
        y = (lorentz*lorentz2)+gauss1+gauss2+lorentz3
        
                
        return y

def Drude2( tags, e, p):
        return self.drude( e, p[0], p[1], p[2],  p[3])
def drude(tags, e, ep, ew, tnm, eb):
    #function Drude(ep,ew,eb,epc,e0,beta,nn,tnm)
    #%Given the plasmon energy (ep), plasmon FWHM (ew) and binding energy(eb), 
    #%this program generates:
    #%EPS1, EPS2 from modified Eq. (3.40), ELF=Im(-1/EPS) from Eq. (3.42),
    #%single scattering from Eq. (4.26) and SRFINT from Eq. (4.31)
    #%The output is e,ssd into the file Drude.ssd (for use in Flog etc.) 
    #%and e,eps1 ,eps2 into Drude.eps (for use in Kroeger etc.)
    #% Gives probabilities relative to zero-loss integral (I0 = 1) per eV
    #% Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011)
    #% Version 10.11.26

    '''
    B.7 Drude Simulation of a Low-Loss Spectrum
    The program DRUDE calculates a single-scattering plasmon-loss spectrum for
    a specimen of a given thickness tnm (in nm), recorded with electrons of a
    specified incident energy e0 by a spectrometer that accepts scattering up to a
    specified collection semi-angle beta. It is based on the extended Drude model
    (Section 3.3.2), with a volume energy-loss function elf in accord with Eq. (3.64) and
    a surface-scattering energy-loss function srelf as in Eq. (4.31). Retardation effects
    and coupling between the two surface modes are not included. The surface term can
    be made negligible by entering a large specimen thickness (tnm > 1000).
    Surface intensity srfint and volume intensity volint are calculated from
    Eqs. (4.31) and (4.26), respectively. The total spectral intensity ssd is written to
    the file DRUDE.SSD, which can be used as input for KRAKRO. These intensities are
    all divided by I0, to give relative probabilities (per eV). The real and imaginary parts
    of the dielectric function are written to DRUDE.EPS and can be used for comparison
    with the results of Kramers–Kronig analysis (KRAKRO.DAT).
    Written output includes the surface-loss probability Ps, obtained by integrating
    srfint (a value that relates to two surfaces but includes the negative begrenzungs
    term), for comparison with the analytical integration represented by Eq. (3.77). The
    volume-loss probability Pv is obtained by integrating volint and is used to calculate
    the volume plasmon mean free path (lam = tnm/Pv). The latter is listed and
    compared with the MFP obtained from Eq. (3.44), which represents analytical integration
    assuming a zero-width plasmon peak. The total probability (Pt = Pv+Ps) is
    calculated and used to evaluate the thickness (lam.Pt) that would be given by the formula
    t/λ = ln(It/I0), ignoring the surface-loss probability. Note that Pv will exceed
    1 for thicker specimens (t/λ > 1), since it represents the probability of plasmon
    scattering relative to that of no inelastic scattering.
    The command-line usage is Drude(ep,ew,eb,epc,beta,e0,tnm,nn), where ep is the
    plasmon energy, ew the plasmon width, eb the binding energy of the electrons (0 for
    a metal), and nn is the number of channels in the output spectrum. An example of
    the output is shown in Fig. B.1a,b.

    '''
    epc = tags['dispersion']#input('ev per channel : ');
    e0 = tags['E0']#input('incident energy E0(kev) : ');
    beta = tags['collAngle']#input('collection semiangle beta(mrad) : ');
    #tnm = input('thickness(nm) : ');

    b = beta/1000.0 # %rad
    T = 1000.0*e0*(1.+e0/1022.12)/(1.0+e0/511.06)**2;# %eV # equ.5.2a or Appendix E p 427 
    tgt = 1000*e0*(1022.12 + e0)/(511.06 + e0);# %eV  Appendix E p 427 
    rk0 = 2590*(1.0+e0/511.06)*np.sqrt(2.0*T/511060);
    os = e[0]
    ewMod = eb

    #eps = 1 - ep**2/(e**2-eb**2+2*e*ew*1j)   #eq 3.64
    #eps = 1 - ep**2/(e**2+2*e*ew*1j)   #eq 3.64
    #eps = 1 - (ep**2)/(e**2+e*ew*1j-ep**2) #Lorentz Term
    eps = 1 - (ep**2-ewMod*e*1j)/(e**2+2*e*ew*1j) #Mod Drude term
    #eps1 = np.real(eps);
    #eps2 = np.imag(eps);
    #%eps1 = 1. - ep.^2./(e.^2+ew.^2);  #eq 3.40
    #%eps2 = ew.*ep.^2./e./(e.^2+ew.^2);#eq 3.40
    #elf = ep**2*e*ew/((e**2-ep**2)**2+(e*ew)**2);    #eq 3.40?
    eps[np.nonzero(eps==0.0)]= 1e-19
    elf = np.imag(-1/eps)

    the = e/tgt; #% varies with energy loss! # Appendix E p 427 
    #%srfelf = 4..*eps2./((1+eps1).^2+eps2.^2) - elf; %equivalent
    srfelf=np.imag(-4./(1.0+eps))-elf; #% for 2 surfaces
    angdep = np.arctan(b/the)/the - b/(b*b+the*the);
    srfint = angdep*srfelf/(3.1416*0.05292*rk0*T); #% probability per eV
    anglog = np.log(1.0+ b*b/the/the);
    I0 = tags['spec'].sum()#*tags['counts2e']
    #print('counts2e',1/tags['counts2e'])
    

    # 2 * T = m_0 v**2 !!!  a_0 = 0.05292 nm
    volint = abs(tnm/(np.pi*0.05292*T*2.0)*elf*anglog); #S equ 4.26% probability per eV
    volint = volint *I0/ epc #S probability per channel
    ssd = volint #+ srfint;

    if os <-1.0:
        xs = int(abs(-os/epc))
   
        ssd[0:xs]=0.0
        volint[0:xs]=0.0
        srfint[0:xs]=0.0
        
        #if os <0:
        Ps = np.trapz(e,srfint); #% 2 surfaces but includes negative begrenzungs contribn.
        Pv = abs(np.trapz(e,abs(volint/tags['spec'].sum()))); #% integrated volume probability
        Pv = (volint/I0).sum() ## our data have he same epc and the trapz formula does not include 
        lam = tnm/Pv; #% does NOT depend on free-electron approximation (no damping). 
        lamfe = 4.0*0.05292*T/ep/np.log(1+(b* tgt / ep) **2); #% Eq.(3.44) approximation
    
        #print('Ps(2surfaces+begrenzung terms) =', Ps, 'Pv=t/lambda(beta)= ',Pv,'\n');
        #print('Volume-plasmon MFP(nm) = ', lam,' Free-electron MFP(nm) = ',lamfe,'\n');
        #print('--------------------------------\n');

        tags['eps'] = eps
        
        tags['lam'] = lam
        tags['lamfe'] = lamfe
        tags['Pv'] = Pv
        
        
    return ssd#/np.pi
    
