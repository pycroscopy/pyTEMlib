
# This code modified for Python Jan 2016 by Mark Oxley
import numpy as np
from numpy.lib.scimath import *
import matplotlib.pyplot as plt


def dftregistration1(buf1ft,buf2ft,usfac):
   """
   function [output, Greg] = dftregistration(buf1ft,buf2ft,usfac)
   #
   # Efficient subpixel image registration by crosscorrelation. This code
   # gives the same precision as the FFT upsampled cross correlation in a
   # small fraction of the computation time and with reduced memory 
   # requirements. It obtains an initial estimate of the crosscorrelation peak
   # by an FFT and then refines the shift estimation by upsampling the DFT
   # only in a small neighborhood of that estimate by means of a 
   # matrix-multiply DFT. With this procedure all the image points are used to
   # compute the upsampled crosscorrelation.
   # Manuel Guizar - Dec 13, 2007
   #
   # Rewrote all code not authored by either Manuel Guizar or Jim Fienup
   # Manuel Guizar - May 13, 2016
   #
   # Citation for this algorithm:
   # Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, 
   # "Efficient subpixel image registration algorithms," Opt. Lett. 33, 
   # 156-158 (2008).
   #
   # Inputs
   ----------------------------------------------
   # buf1ft    Fourier transform of reference image, 
   #           DC in (1,1)   [DO NOT FFTSHIFT]
   # buf2ft    Fourier transform of image to register, 
   #           DC in (1,1) [DO NOT FFTSHIFT]
   # usfac     Upsampling factor (integer). Images will be registered to 
   #           within 1/usfac of a pixel. For example usfac = 20 means the
   #           images will be registered within 1/20 of a pixel. (default = 1)
   #
   # Outputs
   ----------------------------------------------
   # output =  [error,diffphase,net_row_shift,net_col_shift]
   # error     Translation invariant normalized RMS error between f and g
   # diffphase     Global phase difference between the two images (should be
   #               zero if images are non-negative).
   # net_row_shift net_col_shift   Pixel shifts between images
   # Greg      (Optional) Fourier transform of registered version of buf2ft,
   #           the global phase difference is compensated for.
   #
   #
   # Copyright (c) 2016, Manuel Guizar Sicairos, James R. Fienup, University of Rochester
   # All rights reserved.
   # 
   # Redistribution and use in source and binary forms, with or without
   # modification, are permitted provided that the following conditions are
   # met:
   # 
   #     * Redistributions of source code must retain the above copyright
   #       notice, this list of conditions and the following disclaimer.
   #     * Redistributions in binary form must reproduce the above copyright
   #       notice, this list of conditions and the following disclaimer in
   #       the documentation and/or other materials provided with the distribution
   #     * Neither the name of the University of Rochester nor the names
   #       of its contributors may be used to endorse or promote products derived
   #       from this software without specific prior written permission.
   # 
   # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
   # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   # POSSIBILITY OF SUCH DAMAGE.
   #   from scipy.fftpack import fft2, ifft2
   #   import numpy as np
   #   from numpy.lib.scimath import *
   """
   if usfac < 0 :
    usfac = 1
    
#if ~exist('usfac','var')
#    usfac = 1;
#end
   
   [nr, nc] = buf2ft.shape
#[nr,nc]=size(buf2ft);
   Nr = np.arange(-np.floor(nr/2), np.ceil(nr/2),1.0 )
   Nc = np.arange(-np.floor(nc/2), np.ceil(nc/2),1.0 )
   Nr = np.fft.ifftshift(Nr)
   Nc = np.fft.ifftshift(Nc)

#Nr = ifftshift(-fix(nr/2):ceil(nr/2)-1);
#Nc = ifftshift(-fix(nc/2):ceil(nc/2)-1);

   if usfac == 0:
      # Simple computation of error and phase difference without registration
      CCmax = sum(buf1ft*np.conjugate(buf2ft))
   #    CCmax = sum(buf1ft(:).*conj(buf2ft(:)));
      row_shift = 0
      col_shift = 0
   elif usfac >0:#  == 1: # changed by gerd
      # Single pixel registration
      CC = np.fft.ifft2(buf1ft*np.conjugate(buf2ft))
      CCabs = abs(CC)
      ## changed by Gerd TO GET CLOSEST OF 5 MAXIMA

      #x = np.argmax(CCabs)
      #dims = CCabs.shape
      #[row_shift, col_shift] = np.unravel_index(x,dims)

      x = np.argsort(CCabs, axis=None)
      dims = CCabs.shape
      [row_shift, col_shift] = np.array(np.unravel_index(x,dims))[:,-10:]

      ## end changed by Gerd 
      
      CCmax = CC[row_shift,col_shift]*nr*nc
      # Now change shifts so that they represent relative shifts and not indices
      row_shift = Nr[row_shift]
      col_shift = Nc[col_shift]
      ## changed by GerdTO GET CLOSEST OF 5 MAXIMA
      maxima_cc = np.array(list(zip(row_shift, col_shift)))
      #print(maxima_cc)
      closest_maximum = np.argmin(np.linalg.norm(maxima_cc, axis=1))
      row_shift = maxima_cc[closest_maximum,0]
      col_shift = maxima_cc[closest_maximum,1]
      CCmax = CCmax[closest_maximum]

      ## end changed by Gerd
   """
      elif usfac > 1:
      # Start with usfac == 2
   #   CC = ifft2(FTpad(buf1ft.*conj(buf2ft),[2*nr,2*nc]));
      outsize = np.array((1,2),np.int32)
      outsize[0] = 2*nr
      outsize[1] = 2*nc

      CC = np.fft.ifft2(FTpad(buf1ft*np.conjugate(buf2ft),outsize))
   #    CCabs = abs(CC);
      CCabs = abs(CC)
      ## changed by Gerd TO GET CLOSEST OF 5 MAXIMA

      #x = np.argmax(CCabs)
      #dims = CCabs.shape
      #[row_shift, col_shift] = np.unravel_index(x,dims)

      x = np.argsort(CCabs, axis=None)
      dims = CCabs.shape
      [row_shift, col_shift] = np.array(np.unravel_index(x,dims))[:,-5:]

      ## end changed by Gerd 
      #
      CCmax = CC[row_shift,col_shift]*nr*nc
      # Now change shifts so that they represent relative shifts and not indices
      Nr2 = np.arange(-np.floor(nr), np.ceil(nr),1.0 )
      Nc2 = np.arange(-np.floor(nc), np.ceil(nc),1.0 )
      Nr2 = np.fft.ifftshift(Nr2)
      Nc2= np.fft.ifftshift(Nc2)
      
      row_shift = Nr2[row_shift]/2;
      col_shift = Nc2[col_shift]/2;

      ## changed by GerdTO GET CLOSEST OF 5 MAXIMA
      maxima_cc = np.array(list(zip(row_shift, col_shift)))
      
      closest_maximum = np.argmin(np.linalg.norm(maxima_cc, axis=1))
      row_shift = maxima_cc[closest_maximum,0]
      col_shift = maxima_cc[closest_maximum,1]
      CCmax = CCmax[closest_maximum]

      ## end changed by Gerd
   """
# If upsampling > 2, then refine estimate with matrix multiply DFT
   if usfac > 2:
      ### DFT computation ###
      # Initial shift estimate in upsampled grid
      row_shift = np.round(row_shift*usfac)/usfac; 
      col_shift = np.round(col_shift*usfac)/usfac;     
      dftshift = np.fix(np.ceil(usfac*1.5)/2); ## Center of output array at dftshift+1
      # Matrix multiply DFT around the current shift estimate
      CC = np.conjugate(dftups(buf2ft*np.conjugate(buf1ft),np.int(np.ceil(usfac*1.5)),np.int(np.ceil(usfac*1.5)),usfac, dftshift-row_shift*usfac,dftshift-col_shift*usfac))
         # Locate maximum and map back to original pixel grid 
      CCabs = abs(CC)
      x = np.argmax(CCabs)
      dims = CCabs.shape
      [rloc, cloc] = np.unravel_index(x,dims)
      #
      #        [rloc, cloc] = find(CCabs == max(CCabs(:)),1,'first');
      CCmax = CC[rloc,cloc]
      rloc = rloc - dftshift - 1
      cloc = cloc - dftshift - 1
      row_shift = row_shift + rloc/usfac
      col_shift = col_shift + cloc/usfac
      
      
      #    end

      # If its only one row or column the shift along that dimension has no
      # effect. Set to zero.
      if nr == 1:
         row_shift = 0
      
      if nc == 1:
         col_shift = 0
    
    
#end  

   rg00 = np.sum(abs(buf1ft)**2)
   rf00 = np.sum(abs(buf2ft)**2)
   error = 1.0 - abs(CCmax)**2/(rg00*rf00)
   error = sqrt(abs(error))
   diffphase = np.angle(CCmax)

   output = [error,diffphase,row_shift,col_shift,maxima_cc]

# Compute registered version of buf2ft
#if (nargout > 1)&&(usfac > 0),
   if usfac > 1:
#    [Nc,Nr] = meshgrid(Nc,Nr);
#    Greg = buf2ft.*exp(1i*2*pi*(-row_shift*Nr/nr-col_shift*Nc/nc));
#    Greg = Greg*exp(1i*diffphase);
    Nc, Nr = np.meshgrid(Nc,Nr)
    Greg = buf2ft*np.exp(1j*2*np.pi*(-row_shift*Nr/nr-col_shift*Nc/nc))
    Greg = Greg.dot(np.exp(1j*diffphase))
   elif usfac == 0:
    Greg = buf2ft.dot(np.exp(1j*diffphase))
     
#elseif (nargout > 1)&&(usfac == 0)
#    Greg = buf2ft*exp(1i*diffphase);
#end
   return output, Greg
def shift_image(image, reference_image, row_shift, col_shift):
   """shift image by adding to the phase of the Fourier transform

   INPUT:
      image to be shifted 
      reference image ususally the middle image of stack
      row_shift shift of rows 
      col_shift shift of columns
   """

 
   buf2ft = np.fft.fft2(image)
   buf1ft = np.fft.fft2(reference_image)
   [nr, nc] = buf2ft.shape
   Nr = np.arange(-np.floor(nr/2), np.ceil(nr/2),1.0 )
   Nc = np.arange(-np.floor(nc/2), np.ceil(nc/2),1.0 )
   Nr = np.fft.ifftshift(Nr)
   Nc = np.fft.ifftshift(Nc)

   usfac = 1000
   dftshift = np.fix(np.ceil(usfac*1.5)/2);


   CC = np.conjugate(dftups(buf2ft*np.conjugate(buf1ft),np.int(np.ceil(usfac*1.5)),np.int(np.ceil(usfac*1.5)),usfac, dftshift-row_shift*usfac,dftshift-col_shift*usfac))
   # Locate maximum and map back to original pixel grid 
   CCabs = abs(CC)

   rloc= int((row_shift-int(row_shift))*usfac)
   cloc= int((col_shift-int(col_shift))*usfac)

   CCmax = CC[rloc,cloc]

   diffphase = np.angle(CCmax)

   Nc, Nr = np.meshgrid(Nc,Nr)
   Greg = buf2ft*np.exp(1j*2*np.pi*(-row_shift*Nr/nr-col_shift*Nc/nc))

   Greg = Greg.dot(np.exp(1j*diffphase))

   return np.abs(np.fft.ifft2(Greg))


def dftups(inp,nor,noc,usfac,roff,coff):
# function out=dftups(in,nor,noc,usfac,roff,coff);
# Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just
# a small region.
# usfac         Upsampling factor (default usfac = 1)
# [nor,noc]     Number of pixels in the output upsampled DFT, in
#               units of upsampled pixels (default = size(in))
# roff, coff    Row and column offsets, allow to shift the output array to
#               a region of interest on the DFT (default = 0)
# Recieves DC in upper left corner, image center must be in (1,1) 
# Manuel Guizar - Dec 13, 2007
# Modified from dftus, by J.R. Fienup 7/31/06

# This code is intended to provide the same result as if the following
# operations were performed
#   - Embed the array "in" in an array that is usfac times larger in each
#     dimension. ifftshift to bring the center of the image to (1,1).
#   - Take the FFT of the larger array
#   - Extract an [nor, noc] region of the result. Starting with the 
#     [roff+1 coff+1] element.

# It achieves this result by computing the DFT in the output array without
# the need to zeropad. Much faster and memory efficient than the
# zero-padded FFT approach if [nor noc] are much smaller than [nr*usfac nc*usfac]

#[nr,nc]=size(in);
 [nr,nc]=inp.shape

# Set defaults
#if exist('roff', 'var')~=1, roff=0;  end
#if exist('coff', 'var')~=1, coff=0;  end
#if exist('usfac','var')~=1, usfac=1; end
#if exist('noc',  'var')~=1, noc=nc;  end
#if exist('nor',  'var')~=1, nor=nr;  end
# Compute kernels and obtain DFT by matrix products
#kernc=exp((-1i*2*pi/(nc*usfac))*( ifftshift(0:nc-1).' - floor(nc/2) )*( (0:noc-1) - coff ));
# kernc=np.exp((-1j*2*np.pi/(nc*usfac))*( np.fft.ifftshift(np.arange(nc)).T - np.floor(nc/2) )*( np.arange(noc) - coff ))
#old
# temp = np.zeros((noc,nc), np.float , 'F')
# tempc = ( np.arange(noc) - coff )
# for ic in range(noc):
#   temp[ic,:] = tempc[ic]
 
# temp2 = temp*( np.fft.ifftshift(np.arange(nc)).T - np.floor(nc/2)  )  
 
# kernc=np.exp((-1j*2*np.pi/(nc*usfac))*temp2).T
# new
 temp = np.zeros((nc,noc), np.float , 'F')
 V = ( np.arange(noc) - coff )
 for ic in range(noc):
   temp[:,ic] = V[ic]

 Y = np.fft.ifftshift(np.arange(nc)).T - np.floor(nc/2) 
 for ic in range(noc):
   temp[:,ic] = temp[:,ic]*Y

 kernc=np.exp((-1j*2*np.pi/(nc*usfac))*temp)

#kernr=exp((-1i*2*pi/(nr*usfac))*( (0:nor-1).' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
# kernr=np.exp((-1j*2*np.pi/(nr*usfac))*( np.arange(nor).T - roff )*( np.fft.ifftshift(np.arange(nr)) - np.floor(nr/2)  ))
#old 
# temp = np.zeros((nor,nr), np.float , 'F')
# tempc = ( np.arange(nor).T - roff )
# for ir in range(nor):
#   temp[ir,:] = tempc[ir]
 
# temp = temp*( np.fft.ifftshift(np.arange(nr)) - np.floor(nr/2)  )  
 
# kernr=np.exp((-1j*2*np.pi/(nr*usfac))*temp)
#new
 temp = np.zeros((nor,nr), np.float , 'F')
 U = ( np.arange(nor).T - roff )
 for ir in range(nor):
   temp[ir,:] = U[ir]
 X =  np.fft.ifftshift(np.arange(nr)) - np.floor(nr/2)
 for ir in range(nor):
   temp[ir,:] = temp[ir,:]*X

 kernr=np.exp((-1j*2*np.pi/(nr*usfac))*temp)


# out=kernr*in*kernc;
 out=kernr.dot(inp.dot(kernc))

 return out



def FTpad(imFT,outsize):
# imFTout = FTpad(imFT,outsize)
# Pads or crops the Fourier transform to the desired ouput size. Taking 
# care that the zero frequency is put in the correct place for the output
# for subsequent FT or IFT. Can be used for Fourier transform based
# interpolation, i.e. dirichlet kernel interpolation. 
#
#   Inputs
# imFT      - Input complex array with DC in [1,1]
# outsize   - Output size of array [ny nx] 
#
#   Outputs
# imFTout   - Output complex image with DC in [1,1]
# Manuel Guizar - 2014.06.02
# This Python version Mark Oxley January 2017
 Nout = outsize
 temp = imFT.shape
 Nin = np.array((1,2),np.int32)
 Nin[0] = temp[0]
 Nin[1] = temp[1]
 
# print('Nout = ', Nout)
# print('Nin = ', Nin)
 imFT = np.fft.fftshift(imFT)
# print(imFT)

 center = np.array((1,2),np.int32)
 center[0] = np.floor(Nin[0]/2)
 center[1] = np.floor(Nin[1]/2)
   
 imFTout = np.zeros(Nout, np.cfloat , 'F') 

 centerout = np.array((1,2),np.int32)

 centerout[0] = np.floor(Nout[0]/2)
 centerout[1] = np.floor(Nout[1]/2)
 
 cenout_cen = np.array((1,2),np.int32)

 cenout_cen = centerout - center
# print(center)
# print(centerout)
 
# print(cenout_cen)
 for iy in range(Nin[0]):
   for ix in range(Nin[1]):
    imFTout[cenout_cen[0]+iy,cenout_cen[1]+ix]=imFT[iy,ix] 

 imFTout = np.fft.ifftshift(imFTout)*Nout[0]*Nout[1]/(Nin[0]*Nin[1])
 
 return imFTout

