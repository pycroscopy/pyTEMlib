import numpy as np

from lxml import etree
import xml.etree.ElementTree as ET


def exciting_getSpectra(file):
    
    tags = {}
    tags['data'] = {}
    
    tree = ET.ElementTree(file=file)

    root = tree.getroot()
    
    
    data = tags['data'] 
    supportedMap = ['loss', 'dielectric']
    if root.tag in supportedMap :
        
        
        print(' reading ', root.tag ,' function from file ', file)
        #print(root[0].tag, root[0].text)
        mapdef = root[0]
        i = 0
        v = {}
        for child_of_root in mapdef:
            data[child_of_root.tag] = child_of_root.attrib
            v[child_of_root.tag]=[]
            i+=1
        #print(data)
        #print(i, ' variables')
        for elem in tree.iter(tag='map'):
            mdic = elem.attrib
            for key in mdic:
                v[key].append(float(mdic[key]))
        
        for key in data:
            data[key]['data'] = np.array(v[key])
        data['type']= root.tag+' function'
    
        return tags


def FinalStateBroadening(x,y,start, instrument):
    ### Getting the smearing
    Ai = 107.25*5
    Bi = 0.04688*2.
    x = np.array(x)-start
    zero= int(-x[0]/(x[1]-x[0]))+1
    SmearI = x*0.0
    SmearI[zero:-1] =(Ai/x[zero:-1]**2)+Bi*np.sqrt(x[zero:-1])
    hbar = 6.58e-16 #h/2pi
    pre = 1.0
    m = 6.58e-31
    Smear = x*0.0
    Smear[zero:-1]=pre*(hbar/(SmearI[zero:-1]*0.000000001))*np.sqrt((2*x[zero:-1]*1.6E-19)/m)

    def Lorenzian(x,p):
        y = ((0.5 *  p[1]/3.14)/((x- p[0])**2+(( p[1]/2)**2)))
        return y/sum(y)

    p = [0,instrument]
    pi = p.copy()
    indat = y.copy()
    outdat = np.array(y)*0.0
    for i in range (zero+5, len(x)):
        p[0] = x[i]
        p[1] = Smear[i]/1.0
        lor = Lorenzian(x+1e-9,p)
        outdat[i] = sum(indat* lor)
        if np.isnan(outdat[i]):
            outdat[i] = 0.0
        
    p[1] = instrument
    indat = outdat.copy()   
    for i in range (zero-5, len(x)):
        p[0] = x[i]
        lor = Lorenzian(x+1e-9,p)
        outdat[i] = sum(indat* lor)
        #print(outdat[i],indat[i], lor[i],indat[i-1], lor[i-1], )
    return outdat
