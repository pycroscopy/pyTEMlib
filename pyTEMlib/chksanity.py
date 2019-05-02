import numpy as np
# Making sure the variables are loaded and that
# they are of the correct type, i.e, float, int, etc
# If the variables are not assigned, then assign their default values

def sanityoptics():

	from experiment import ApAngle, Detectors, DetShift, V0, OAM_value
	from experiment import df, C0a, C0b, C12a, C12b, C21a, C21b, C23a, C23b, C3
	from experiment import C32a, C32b, C34a, C34b, C41a, C43a, C45a, C5, C7

	try:
		ApAngle
	except NameError:
		ApAngle = float(30) # default value
	else:
		ApAngle = float(ApAngle)

	try:
		V0
	except NameError:
		V0 = float(100) # default value
	else:
		V0   = float(V0)    

	try:
		df
	except NameError:
		df = float(0) # default value
	else:
		df   = float(df)    

	try:
		C0a
	except NameError:
		C0a = float(0) # default value
	else:
		C0a  = float(C0a)   

	try:
		C0b
	except NameError:
		C0b = float(0) # default value
	else:
		C0b  = float(C0b)   

	try:
		C12a
	except NameError:
		C12a = float(0) # default value
	else:
		C12a = float(C12a)  

	try:
		C12b
	except NameError:
		C12b = float(0) # default value
	else:
		C12b = float(C12b)  

	try:
		C21a
	except NameError:
		C21a = float(0) # default value
	else:
		C21a = float(C21a)  

	try:
		C21b
	except NameError:
		C21b = float(0) # default value
	else:
		C21b = float(C21b)  

	try:
		C23a
	except NameError:
		C23a = float(0) # default value
	else:
		C23a = float(C23a)  

	try:
		C23b
	except NameError:
		C23b = float(0) # default value
	else:
		C23b = float(C23b)  

	try:
		C3
	except NameError:
		C3 = float(4000) # default value
	else:
		C3   = float(C3*1000)   

	try:
		C32a
	except NameError:
		C32a = float(0) # default value
	else:
		C32a = float(C32a) 

	try:
		C32b
	except NameError:
		C32b = float(0) # default value
	else:
		C32b = float(C32b) 

	try:
		C34a
	except NameError:
		C34a = float(0) # default value
	else:
		C34a = float(C34a) 

	try:
		C34b
	except NameError:
		C34b = float(0) # default value
	else:
		C34b = float(C34b) 

	try:
		C41a
	except NameError:
		C41a = float(0) # default value
	else:
		C41a = float(C41a) 

	try:
		C43a
	except NameError:
		C43a = float(0) # default value
	else:
		C43a = float(C43a) 

	try:
		C45a
	except NameError:
		C45a = float(0) # default value
	else:
		C45a = float(C45a) 

	try:
		C5
	except NameError:
		C5 = float(1000000) # default value
	else:
		C5 = float(C5*1000000) 

	try:
		C7
	except NameError:
		C7 = float(1000000) # default value
	else:
		C7 = float(C7*1000000) 

	try:
		OAM_value
	except NameError:
		OAM_value = int(0)
	else:
		OAM_value = int(OAM_value)

	try:
		Detectors
	except NameError:
		Detectors = np.array([81, 200], dtype = float)
	else:
		Detectors = np.array(Detectors, dtype = float)

	try:
		DetShift
	except NameError:
		DetShift = np.array([0, 0], dtype = float)
	else:
		DetShift = np.array(DetShift, dtype = float)  

	aberrations = np.array([df, C0a, C0b, C12a, C12b, C21a, C21b, C23a, C23b, C3,\
		C32a, C32b, C34a, C34b, C41a, C43a, C45a, C5, C7])
	return ApAngle, Detectors, DetShift, V0, aberrations, OAM_value

def sanityimaging():
	from experiment import Thickness, FieldofView, ImgPixelsX, ImgPixelsY 
   
	try:
		Thickness
	except NameError:
		Thickness = float(0)
	else:
		Thickness = float(Thickness)

	try:
		FieldofView
	except NameError:
		FieldofView = float(4)
	else:
		FieldofView = float(FieldofView)

	try:
		ImgPixelsX
	except NameError:
		ImgPixelsX = int(512)
	else:
		ImgPixelsX = int(ImgPixelsX)

	try:
		ImgPixelsY
	except NameError:
		ImgPixelsY = int(512)
	else:
		ImgPixelsY = int(ImgPixelsY)
	
	return FieldofView, ImgPixelsX, ImgPixelsY, Thickness

def sanityoutput():
	from experiment import PlotAmpProbe, PlotAngProbe, SaveCell, SaveChaProbe, SavePot, PlotSTEM

	try:
		PlotAmpProbe
	except NameError:
		PlotAmpProbe = False
	else:
		PlotAmpProbe = PlotAmpProbe

	try:
		PlotAngProbe
	except NameError:
		PlotAngProbe = False
	else:
		PlotAngProbe = PlotAngProbe

	try:
		SaveCell
	except NameError:
		SaveCell = True
	else:
		SaveCell = SaveCell

	try:
		SaveChaProbe
	except NameError:
		SaveChaProbe = False
	else:
		SaveChaProbe = SaveChaProbe

	try:
		SavePot
	except NameError:
		SavePot = False
	else:
		SavePot = SavePot

	try:
		PlotSTEM
	except NameError:
		PlotSTEM = True
	else:
		PlotSTEM = PlotSTEM

	return PlotAmpProbe, PlotAngProbe, SaveCell, SaveChaProbe, SavePot, PlotSTEM

def sanitycalculations():
	
	from experiment import OnlyCheck, OnlyProbe, Channeling   # Loading the experiment variables

	try:
		OnlyCheck
	except NameError:
		OnlyCheck = False
	else:
		OnlyCheck = OnlyCheck

	try:
		OnlyProbe
	except NameError:
		OnlyProbe = False
	else:
		OnlyProbe = OnlyProbe

	try:
		Channeling
	except NameError:
		Channeling = False
	else:
		Channeling = Channeling

	return OnlyCheck, OnlyProbe, Channeling

def sanitymisc():
	from experiment import nmax, MaxOAM, Maxradius, ProPixelSize, PosProbChan, TransVect
	try:
		nmax
	except NameError:
		nmax = int(2)
	else:
		nmax = int(nmax)

	try:
		MaxOAM
	except NameError:
		MaxOAM = int(4)
	else:
		MaxOAM = int(MaxOAM)

	try:
		Maxradius
	except NameError:
		Maxradius = float(0.1)
	else:
		Maxradius = float(Maxradius)

	try:
		ProPixelSize
	except NameError:
		ProPixelSize = int(128) # default value
	else:
		ProPixelSize = int(ProPixelSize)

	try:
		PosProbChan
	except NameError:
		PosProbChan = np.array([0, 0], dtype = float)
	else:
		PosProbChan = np.array(PosProbChan, dtype = float)

	try:
		TransVect
	except NameError:
		TransVect = np.array([0., 0.])
	else:
		TransVect = np.array(TransVect)

	return nmax, MaxOAM, Maxradius, ProPixelSize, PosProbChan, TransVect
