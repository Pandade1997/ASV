import os
import wave
import numpy
import struct

def makedir(filename):
    '''
    create directory for filename
    '''
    dirname=os.path.dirname(filename)

    if not os.path.exists(dirname):
       try:
           os.makedirs(dirname)
       except:
           pass
           
def float2raw(data,w=2):
    if w == 1:
        sdata = numpy.clip(data*127+127,-127,127).astype(int)
        return struct.pack('%dB'%len(sdata),*sdata)
    elif w == 2:
        sdata = numpy.clip(data*32768,-32767,32767).astype(int).flatten()
        return struct.pack('%dh'%len(sdata),*sdata)
    else:
        return b''


def raw2float(data, w=2):
    if w == 1:
        return numpy.array(struct.unpack('%dB'%len(data),data))/127-1.0
    elif w == 2:
        return numpy.array(struct.unpack('%dh'%(len(data)/2),data))/32768
    else:
        return numpy.empty((0,0))
    
    
def readwav(filename):
    with wave.open(filename, "r") as fi:
        c = fi.getnchannels()
        w = fi.getsampwidth()
        fs = fi.getframerate()
        n = fi.getnframes()
        data = fi.readframes(n)        
    return c, w, fs, n, raw2float(data, w)
    
def writewav(c, w, fs, data, filename):
    with wave.open(filename, "w") as wf:
        wf.setnchannels(c)
        wf.setsampwidth(w)
        wf.setframerate(fs)
        wf.writeframes(float2raw(data,w))
