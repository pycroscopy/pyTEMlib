#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-


################################################################################
## Python script for parsing GATAN DM3 (DigitalMicrograph) files
## and extracting various metadata
## --
## tested on EELS spectra, spectrum images and single-image files only*
## --
## based on the DM3_Reader plug-in (v 1.3.4) for ImageJ by Greg Jefferis <jefferis@stanford.edu>
## http://rsb.info.nih.gov/ij/plugins/DM3_Reader.html
## --
## Python adaptation: Pierre-Ivan Raynal <raynal@med.univ-tours.fr>
## http://microscopies.med.univ-tours.fr/
##
## Extended for EELS support by Gerd Duscher, UTK 2012
## Added EELSinfo and data_cube support for EELS spectra and spectrum images
## See sample python program readDM3.py for usage.
## Data_cube could be changed to imagedata, but may not be backwards compatible
##
## Works for python 3.2 and 3.3 now
##
################################################################################

import sys, os, time
import struct
import numpy
import scipy.misc

__all__ = ["DM3","version"]

version='0.1beta'

debugLevel = 0   # 0=none, 1-3=basic, 4-5=simple, 6-10 verbose


### utility fuctions ###
# Image to Array
def im2ar( im ):
        if im.mode in ('L','I','F'):
            # Warning: only works with PIL.Image.Image whose mode is 'L', 'I' or 'F'
                #          => error if mode == 'I;16' for instance
                a = scipy.misc.fromimage( im )
        return a
#       else:
#               return False

## Array to image file
def ar2imfile(filename, a):
    scipy.misc.imsave(filename, a)


### binary data reading functions ###

def readLong(f):
        '''Read 4 bytes as integer in file f'''
        read_bytes = f.read(4)
        return struct.unpack('>l', read_bytes)[0]

def readShort(f):
        '''Read 2 bytes as integer in file f'''
        read_bytes = f.read(2)
        return struct.unpack('>h', read_bytes)[0]

def readByte(f):
        '''Read 1 byte as integer in file f'''
        read_bytes = f.read(1)
        return struct.unpack('>b', read_bytes)[0]

def readBool(f):
        '''Read 1 byte as boolean in file f'''
        read_val = readByte(f)
        return (read_val!=0)

def readChar(f):
        '''Read 1 byte as char in file f'''
        read_bytes = f.read(1)
        return struct.unpack('c', read_bytes)[0]

def readString(f, len=1):
        '''Read len bytes as a string in file f'''
        read_bytes = f.read(len)
        str_fmt = '>'+str(len)+'s'
        return struct.unpack( str_fmt, read_bytes )[0]

def readLEShort(f):
        '''Read 2 bytes as *little endian* integer in file f'''
        read_bytes = f.read(2)
        return struct.unpack('<h', read_bytes)[0]

def readLELong(f):
        '''Read 4 bytes as *little endian* integer in file f'''
        read_bytes = f.read(4)
        return struct.unpack('<l', read_bytes)[0]

def readLEUShort(f):
        '''Read 2 bytes as *little endian* unsigned integer in file f'''
        read_bytes = f.read(2)
        return struct.unpack('<H', read_bytes)[0]

def readLEULong(f):
        '''Read 4 bytes as *little endian* unsigned integer in file f'''
        read_bytes = f.read(4)
        return struct.unpack('<L', read_bytes)[0]

def readLEFloat(f):
        '''Read 4 bytes as *little endian* float in file f'''
        read_bytes = f.read(4)
        return struct.unpack('<f', read_bytes)[0]

def readLEDouble(f):
        '''Read 8 bytes as *little endian* double in file f'''
        read_bytes = f.read(8)
        return struct.unpack('<d', read_bytes)[0]


## constants for encoded data types ##
SHORT = 2
LONG = 3
USHORT = 4
ULONG = 5
FLOAT = 6
DOUBLE = 7
BOOLEAN = 8
CHAR = 9
OCTET = 10
STRUCT = 15
STRING = 18
ARRAY = 20

# - association data type <--> reading function
readFunc = {
        SHORT: readLEShort,
        LONG: readLELong,
        USHORT: readLEUShort,
        ULONG: readLEULong,
        FLOAT: readLEFloat,
        DOUBLE: readLEDouble,
        BOOLEAN: readBool,
        CHAR: readChar,
        OCTET: readChar,    # difference with char???
}

## other constants ##
IMGLIST = "root.ImageList."
OBJLIST = "root.DocumentObjectList."
MAXDEPTH = 64

## END constants ##


class DM3(object):
        ## utility functions
        def __makeGroupString(self):
                tString = self.__curGroupAtLevelX[0]
                for i in range( 1, self.__curGroupLevel+1 ):
                        tString += '.' + self.__curGroupAtLevelX[i]
                return tString

        def __makeGroupNameString(self):
                tString = self.__curGroupNameAtLevelX[0]
                for i in range( 1, self.__curGroupLevel+1 ):
                        tString += '.' + str( self.__curGroupNameAtLevelX[i] )
                return tString

        def __readTagGroup(self):
                # go down a level
                self.__curGroupLevel += 1
                # increment group counter
                self.__curGroupAtLevelX[self.__curGroupLevel] += 1
                # set number of current tag to -1 --- readTagEntry() pre-increments => first gets 0
                self.__curTagAtLevelX[self.__curGroupLevel] = -1
                #if ( debugLevel > 5):
                #       print "rTG: Current Group Level:", self.__curGroupLevel
                # is the group sorted?
                sorted = readByte(self.__f)
                isSorted = (sorted == 1)
                # is the group open?
                opened = readByte(self.__f)
                isOpen = (opened == 1)
                # number of Tags
                nTags = readLong(self.__f)
                #if ( debugLevel > 5):
                #       print "rTG: Iterating over the", nTags, "tag entries in this group"
                # read Tags
                for i in range( nTags ):
                        self.__readTagEntry()
                # go back up one level as reading group is finished
                self.__curGroupLevel += -1
                return 1

        def     __readTagEntry(self):
                # is data or a new group?
                data = readByte(self.__f)
                isData = (data == 21)
                self.__curTagAtLevelX[self.__curGroupLevel] += 1
                # get tag label if exists
                lenTagLabel = readShort(self.__f)
                if ( lenTagLabel != 0 ):
                        tagLabel = readString(self.__f, lenTagLabel).decode('latin-1')
                        #print(tagLabel)
                else:
                        tagLabel = str( self.__curTagAtLevelX[self.__curGroupLevel] )
                #if ( debugLevel > 5):
                #       print str(self.__curGroupLevel)+"|"+__makeGroupString()+": Tag label = "+tagLabel
                #elif ( debugLevel > 0 ):
                #       print str(self.__curGroupLevel)+": Tag label = "+tagLabel
                if isData:
                        # give it a name
                        self.__curTagName = self.__makeGroupNameString()+"."+tagLabel#.decode('utf8')
                        # read it
                        self.__readTagType()
                else:
                        # it is a tag group
                        self.__curGroupNameAtLevelX[self.__curGroupLevel+1] = tagLabel
                        self.__readTagGroup()  # increments curGroupLevel
                return 1

        def __readTagType(self):
                delim = readString(self.__f, 4)
                if ( delim != b"%%%%" ):
                        raise Exception( hex( self.__f.tell() )+": Tag Type delimiter not %%%%")
                nInTag = readLong(self.__f)
                self.__readAnyData()
                return 1

        def __encodedTypeSize(self, eT):
                # returns the size in bytes of the data type
                if eT == 0:
                        width = 0
                elif eT in (BOOLEAN, CHAR, OCTET):
                        width = 1
                elif eT in (SHORT, USHORT):
                        width = 2
                elif eT in (LONG, ULONG, FLOAT):
                        width = 4
                elif eT == DOUBLE:
                        width = 8
                else:
                        # returns -1 for unrecognised types
                        width=-1
                return width

        def __readAnyData(self):
                ## higher level function dispatching to handling data types to other functions
                # - get Type category (short, long, array...)
                encodedType = readLong(self.__f)
                # - calc size of encodedType
                etSize = self.__encodedTypeSize(encodedType)
                #if ( debugLevel > 5):
                #       print "rAnD, " + hex( f.tell() ) + ": Tag Type = " + str(encodedType) +  ", Tag Size = " + str(etSize)
                if ( etSize > 0 ):
                        self.__storeTag( self.__curTagName, self.__readNativeData(encodedType, etSize) )
                elif ( encodedType == STRING ):
                        stringSize = readLong(self.__f)
                        data = self.__readStringData(stringSize)
                        if ( debugLevel > 5):
                                print ('String')
                                print (data)
                elif ( encodedType == STRUCT ):
                        # GD does  store tags  now
                        structTypes = self.__readStructTypes()
                        data = self.__readStructData(structTypes)
                        #print('Struct ',self.__curTagName)
                        if ( debugLevel > 5):
                                print ('Struct')
                                print (data)
                        self.__storeTag( self.__curTagName, data )
                        
                elif ( encodedType == ARRAY ):
                        # GD does  store tags now
                        # indicates size of skipped data blocks
                        arrayTypes = self.__readArrayTypes()
                        data = self.__readArrayData(arrayTypes)
                        #print('Array ',self.__curTagName)
                        if ( debugLevel > 5):
                                print ('Array')
                                print (data)
                        self.__storeTag( self.__curTagName, data )
                        
                else:
                        raise Exception("rAnD, " + hex(self.__f.tell()) + ": Can't understand encoded type")
                return 1

        def __readNativeData(self, encodedType, etSize):
                # reads ordinary data types
                if encodedType in readFunc.keys():
                        val = readFunc[encodedType](self.__f)
                else:
                        raise Exception( "rND, " + hex(self.__f.tell()) + ": Unknown data type " + str(encodedType))
                #if ( debugLevel > 3 ):
                #       print "rND, " + hex(self.__f.tell()) + ": " + str(val)
                #elif ( debugLevel > 0 ):
                ##      print val
                return val

        def __readStringData(self, stringSize):
                # reads string data
                if ( stringSize <= 0 ):
                        rString = ""
                else:
                        #if ( debugLevel > 3 ):
                                #print "rSD @ " + str(f.tell()) + "/" + hex(f.tell()) +" :",
                        ## !!! *Unicode* string (UTF-16)... convert to Python unicode str
                        rString = readString(self.__f, stringSize)
                        rString = str(rString, "utf_16_le")
                        #if ( debugLevel > 3 ):
                        #       print rString + "   <"  + repr( rString ) + ">"
                #if ( debugLevel > 0 ):
                #       print "StringVal:", rString
                self.__storeTag( self.__curTagName, rString )
                return rString

        def __readArrayTypes(self):
                # determines the data types in an array data type
                arrayType = readLong(self.__f)
                itemTypes=[]
                if ( arrayType == STRUCT ):
                        itemTypes = self.__readStructTypes()
                elif ( arrayType == ARRAY ):
                        itemTypes = self.__readArrayTypes()
                else:
                        itemTypes.append( arrayType )
                return itemTypes

        def __readArrayData(self, arrayTypes):
                # reads array data

                arraySize = readLong(self.__f)

                #if ( debugLevel > 3 ):
                #       print "rArD, " + hex( f.tell() ) + ": Reading array of size = " + str(arraySize)

                itemSize = 0
                encodedType = 0

                for i in range( len(arrayTypes) ):
                        encodedType = int( arrayTypes[i] )
                        etSize = self.__encodedTypeSize(encodedType)
                        itemSize += etSize
                        #if ( debugLevel > 5 ):
                        #       print "rArD: Tag Type = " + str(encodedType) + ", Tag Size = " + str(etSize)
                        ##! readNativeData( encodedType, etSize ) !##

                #if ( debugLevel > 5 ):
                #       print "rArD: Array Item Size = " + str(itemSize)

                bufSize = arraySize * itemSize

                if ( (not self.__curTagName.endswith("ImageData.Data"))
                                and  ( len(arrayTypes) == 1 )
                                and  ( encodedType == USHORT )
                                and  ( arraySize < 256 ) ):
                        # treat as string
                        val = self.__readStringData( bufSize )
                else:
                        # treat as binary data
                        # - store data size and offset as tags
                        self.__storeTag( self.__curTagName + ".Size", bufSize )
                        self.__storeTag( self.__curTagName + ".Offset", self.__f.tell() )
                        # - skip data w/o reading
                        self.__f.seek( self.__f.tell() + bufSize )
                        val = 1

                return val

        def __readStructTypes(self):
                # analyses data types in a struct

                #if ( debugLevel > 3 ):
                #       print "Reading Struct Types at Pos = " + hex(self.__f.tell())

                structNameLength = readLong(self.__f)
                nFields = readLong(self.__f)

                #if ( debugLevel > 5 ):
                #       print "nFields = ", nFields

                #if ( nFields > 100 ):
                #       raise Exception, hex(self.__f.tell())+": Too many fields"

                fieldTypes = []
                nameLength = 0
                for i in range( nFields ):
                        nameLength = readLong(self.__f)
                        #if ( debugLevel > 9 ):
                        #       print i + "th namelength = " + nameLength
                        fieldType = readLong(self.__f)
                        fieldTypes.append( fieldType )

                return fieldTypes

        def __readStructData(self, structTypes):
                # reads struct data based on type info in structType
                data = []
                for i in range( len(structTypes) ):
                        encodedType = structTypes[i]
                        etSize = self.__encodedTypeSize(encodedType)

                        #if ( debugLevel > 5 ):
                        #       print "Tag Type = " + str(encodedType) + ", Tag Size = " + str(etSize)

                        # get data
                        data.append(self.__readNativeData(encodedType, etSize))
                        
                return data

        def __storeTag(self, tagName, tagValue):
                # NB: all tag values (and names) stored as unicode objects;
                #     => can then be easily converted to any encoding
                # - /!\ tag names may not be ascii char only (e.g. '\xb5', i.e. MICRO SIGN)
                tagName = str(tagName)#, 'latin-1')

                # GD: Changed this over to store real values and not strings in dictionary
                self.__tagDict[tagName] = tagValue
                # - convert tag value to unicode if not already unicode object (as for string data)
                tagValue = str(tagValue)
                # store Tags as list and dict
                self.__storedTags.append( tagName + " = " + tagValue )
                

        ### END utility functions ###

        def __init__(self, filename, dump=False, dump_dir='/tmp', debug=0):
                '''DM3 object: parses DM3 file and extracts Tags; dumps Tags in a txt file if dump==True.'''

                ## initialize variables ##
                self.debug = debug
                self.__filename = filename
                self.__chosenImage = 1
                # - track currently read group
                self.__curGroupLevel = -1
                self.__curGroupAtLevelX = [ 0 for x in range(MAXDEPTH) ]
                self.__curGroupNameAtLevelX = [ '' for x in range(MAXDEPTH) ]
                # - track current tag
                self.__curTagAtLevelX = [ '' for x in range(MAXDEPTH) ]
                self.__curTagName = ''
                # - open file for reading
                self.__f = open( self.__filename, 'rb' )
                # - create Tags repositories
                self.__storedTags = []
                self.__tagDict = {}
                self.__tagDict['DM'] ={}
                

                if self.debug>0:
                        t1 = time.time()
                isDM3 = True
                ## read header (first 3 4-byte int)
                # get version
                fileVersion = readLong(self.__f)
                if ( fileVersion not in (3, 4) ):
                        isDM3 = False
                # get indicated file size
                fileSize = readLong(self.__f)
                # get byte-ordering
                lE = readLong(self.__f)
                littleEndian = (lE == 1)
                if not littleEndian:
                        isDM3 = False
                # check file header, raise Exception if not DM3
                if not isDM3:
                        raise Exception("%s does not appear to be a DM3 or DM4 file."%os.path.split(self.__filename)[1])
                #elif self.debug > 0:
                        #print "%s appears to be a DM3 file"%(self.__filename)
                self.__tagDict['DM']['file version'] = fileVersion
                self.__tagDict['DM']['fileSize'] = fileSize

                if ( debugLevel > 5 or self.debug > 1):
                        print( "Header info.:")
                        print("- file version:", fileVersion)
                        print("- lE:", lE)
                        print("- file size:", fileSize, "bytes")
                
                # set name of root group (contains all data)...
                self.__curGroupNameAtLevelX[0] = "root"
                # ... then read it
                self.__readTagGroup()
                #if self.debug > 0:
                #       print "-- %s Tags read --"%len(self.__storedTags)

                #if self.debug>0:
                #       t2 = time.time()
                #       print "| parse DM3 file: %.3g s"%(t2-t1)

                # dump Tags in txt file if requested
                if dump:
                        dump_file = os.path.join(dump_dir, os.path.split(self.__filename)[1]+".tagdump.txt")
                        try:
                                dumpf = open( dump_file, 'w' )
                        except:
                                pass#print "Warning: cannot generate dump file."
                        else:
                                for tag in self.__storedTags:
                                        dumpf.write( tag.encode('latin-1') + "\n" )
                                dumpf.close

        def getFilename(self):
                return self.__filename
        filename = property(getFilename)

        def getTags(self):
            return self.__tagDict
        tags = property(getTags)


        def getRaw(self):
                '''Extracts  data as np array'''

                # DataTypes for image data <--> PIL decoders
                data_types = {
                    '1' :  '<u2', # 2 byte integer signed ("short")
                    '2' :  '<f4', # 4 byte real (IEEE 754)
                    '3' :  '<c8', # 8 byte complex (real, imaginary)
                    '4' :  '',    # ?
                    # 4 byte packed complex (see below)
                    '5' :  (numpy.int16, {'real':(numpy.int8,0), 'imaginary':(numpy.int8,1)}),
                    '6' :  '<u1', # 1 byte integer unsigned ("byte")
                    '7' :  '<i4', # 4 byte integer signed ("long")
                    # I do not have any dm3 file with this format to test it.
                    '8' :  '',    # rgb view, 4 bytes/pixel, unused, red, green, blue?
                    '9' :  '<i1', # byte integer signed
                    '10' : '<u2', # 2 byte integer unsigned
                    '11' : '<u4', # 4 byte integer unsigned
                    '12' : '<f8', # 8 byte real
                    '13' : '<c16', # byte complex
                    '14' : 'bool', # 1 byte binary (ie 0 or 1)
                     # Packed RGB. It must be a recent addition to the format because it does
                     # not appear in http://www.microscopy.cen.dtu.dk/~cbb/info/dmformat/
                    '23' :  (numpy.float32,
                    {'R':('<u1',0), 'G':('<u1',1), 'B':('<u1',2), 'A':('<u1',3)}),
                }
                # get relevant Tags

                data_dim = 0 # 1 = spectrum, 2 = image, 3 = SI
                if 'root.ImageList.1.ImageData.Data.Offset' in self.tags:
                        data_offset = int( self.tags['root.ImageList.1.ImageData.Data.Offset'] )
                if 'root.ImageList.1.ImageData.Data.Size' in self.tags:
                        data_size = int( self.tags['root.ImageList.1.ImageData.Data.Size'] )
                if 'root.ImageList.1.ImageData.Data.DataType' in self.tags:
                        data_type = int( self.tags['root.ImageList.1.ImageData.DataType'] )
                if 'root.ImageList.1.ImageData.Dimensions.0' in self.tags:
                        im_width = int( self.tags['root.ImageList.1.ImageData.Dimensions.0'] )
                try:
                        im_height = int( self.tags['root.ImageList.1.ImageData.Dimensions.1'] )
                        #if self.debug>0:
                        #       print "Notice: image  data with dimesnions %s x %s"%(im_width,im_height)
                        data_dim = 2
                except:
                        #if self.debug>0:
                        #       print "Notice: spectrum data with spectrum length %s channels "%(im_width)
                        im_height = 1
                        data_dim = 1
                try:
                        im_length = int( self.tags['root.ImageList.1.ImageData.Dimensions.2'] )
                        #if self.debug>0:
                        #       print "Notice: spectrum image data with spectra of length %s channels"%(im_length)
                        data_dim = 3

                except:
                        pass#if self.debug>0:
                        #       print "Notice: Not a spectrum image "


                #if self.debug>0:
                        #print "Notice: image data in %s starts at %s"%(os.path.split(self.__filename)[1], hex(data_offset))
                        #print "Notice: image size: %sx%s px"%(im_width,im_height)

                # check if DataType is implemented, then read
                dt = data_types[str(self.tags['root.ImageList.1.ImageData.DataType'])]
                if dt == '':
                #       print('The datatype is not supported')
                        return
                #if data_type in dataTypes.keys():
                else:
                        #decoder = dataTypes[data_type]
                        #if self.debug>0:
                        #       print "Notice: image data read as %s"%decoder
                        #       t1 = time.time()

                        self.__f.seek( data_offset )
                        rawdata = self.__f.read(data_size)

                        if data_dim >2:
                                #print rawdata[0],rawdata[1],rawdata[2],rawdata[3]
                                shape = (im_width,im_height,im_length)
                        else:
                                shape = (im_width,im_height)
                        if data_dim == 1:
                                shape = (im_width)

                        raw_data = numpy.fromstring(rawdata, dtype = dt, count =numpy.cumprod(shape)[-1]).reshape(shape, order = 'F')
                        #raw_data = numpy.array(rawdata).reshape(im_width,im_height,im_length)
                        #print raw_data[0],raw_data[1],raw_data[2],raw_data[3]#raw_data = numpy.array(rawdata).reshape((im_width,im_height,im_length), order = 'F')
                return raw_data
        data_cube = property(getRaw)


if __name__ == '__main__':
        pass#print "DM3lib v.%s"%version

