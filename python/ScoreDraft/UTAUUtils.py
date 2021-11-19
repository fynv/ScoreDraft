from .VoiceSampler import FrqDataPoint,FrqData
import struct
import os
import math
import re

def LoadFrq(filename):
    with open(filename, 'rb') as f:
        f.seek(8,0)
        interval= struct.unpack('i',f.read(4))[0]
        f.seek(12,0)
        key = struct.unpack('d', f.read(8))[0]
        f.seek(36,0)
        count = struct.unpack('i', f.read(4))[0]
        f.seek(40,0)
        data=[]
        for i in range(count):
            (freq, dyn) = struct.unpack('dd', f.read(16))
            data+=[FrqDataPoint(freq,dyn)]
            
        frq = FrqData()
        frq.set(interval, key, data)
        return frq

def LoadOtoINIPath(otoMap, path, encoding):
    otoIniPath=path+'/oto.ini'
    with open(otoIniPath,'r', encoding=encoding) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line=line.strip('\n')

            p = line.find('=')
            if p==-1:
                continue
            fn=line[0:p]

            p+=1

            # lyric
            lyric=''
            p2 = line.find(',',p)
            if p2==-1:
                continue
            if p2>p:
                lyric=line[p:p2]
            p=p2+1

            if len(lyric)==0:
                lyric = fn[0:len(fn)-4]

            # offset
            offset=0
            p2 = line.find(',',p)
            if p2==-1:
                continue
            if p2>p:
                offset=float(line[p:p2])
            p=p2+1

            # consonant
            consonant = 0
            p2 = line.find(',',p)
            if p2==-1:
                continue
            if p2>p:
                consonant=float(line[p:p2])
            p=p2+1

            # cutoff
            cutoff = 0
            p2 = line.find(',',p)
            if p2==-1:
                continue
            if p2>p:
                cutoff=float(line[p:p2])
            p=p2+1

            # preutter
            preutterance = 0
            p2 = line.find(',',p)
            if p2==-1:
                continue
            if p2>p:
                preutterance=float(line[p:p2])
            p=p2+1

            # overlap
            overlap = 0
            if len(line[p:])>0:
                overlap=float(line[p:])

            properties={
                'filename': path+'/'+fn,
                'offset': offset,
                'consonant': consonant,
                'cutoff': cutoff,
                'preutterance': preutterance,
                'overlap': overlap
            }

            otoMap[lyric]=properties

def LoadPrefixMap(filename):
    prefixMap={}
    with open(filename,'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            words=re.findall(r"[^\s]+", line)
            prefix=''
            if len(words)>1:
                prefix=words[1]
            if len(words)>0:
                prefixMap[words[0]]=prefix
    return prefixMap

centerC=440.0 * (2.0** (- 9.0 / 12.0)) # C4
lowest = centerC * (2.0** (- 3.0)) # C1
highest = centerC * (2.0**( 3.0 + 11.0 / 12.0)) #B7
nameMap= [ 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def LookUpPrefixMap(prefixMap, freq):
    name=''
    if freq<=lowest:
        name='C1'
    elif freq>=highest:
        name='B7'
    else:
        fPitch = math.log(freq / centerC) / math.log(2.0) + 4.0
        octave = int(fPitch)
        pitchInOctave = int((fPitch-octave)*12.0)
        if pitchInOctave==12:
            pitchInOctave = 0
            octave+=1
        name = nameMap[pitchInOctave]+str(octave)
    if name in prefixMap:
        return prefixMap[name]
    else:
        return ''

class VoiceBank:
    def __init__ (self, path):
        self.path=path
        self.otoMap={}
        self.prefixMap={}
        self.encoding='shift-jis'
        if os.name == 'nt':
            self.fsEncoding = 'mbcs'
        else:
            self.fsEncoding = 'gbk'
        self.wavFileNameTranscode=True
        self.frqFileNameTranscode=True
        self.initialized=False

    def buildOtoMap(self,path):
        for item in os.walk(path):
            if os.path.isfile(item[0]+'/oto.ini'):
                LoadOtoINIPath(self.otoMap, item[0], self.encoding)

    def initialize(self):
        self.buildOtoMap(self.path)
        if os.path.isfile(self.path+'/prefix.map'):
            self.prefixMap=LoadPrefixMap(self.path+'/prefix.map')
        self.initialized=True

    def getWavFrq(self,lyric):
        if not (lyric in self.otoMap):
            print("missed lyic: "+ lyric)
            return None
        wav=self.otoMap[lyric].copy()
        wavFileName=wav['filename']
        if self.wavFileNameTranscode:
            wav['filename']=wavFileName.encode(self.encoding).decode(self.fsEncoding)
        frqFileName=wavFileName[0:len(wavFileName)-4]+'_wav.frq'
        if self.frqFileNameTranscode:
            frqFileName=frqFileName.encode(self.encoding).decode(self.fsEncoding)
        frq=LoadFrq(frqFileName)
        return (wav,frq)

    def getWavFrq_PrefixMap(self,lyric, freq):
        if len(self.prefixMap)>0:
            lyric+=LookUpPrefixMap(self.prefixMap, freq)
        return self.getWavFrq(lyric)

