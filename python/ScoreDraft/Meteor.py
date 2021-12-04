import os
from cffi import FFI
from .ScoreDraftCore import ObjArray

ffi = FFI()
ffi.cdef("""
void* CtrlPntCreate(double freq, double fduration);
void CtrlPntDestroy(void* ptr);
void* SyllableCreate(const char* lyric, void* ptr_lst_ctrl_pnts);
void SyllableDestroy(void* ptr);
void EventDestroy(void* ptr);
void EventSetOffset(void* ptr, float offset);
void* EventInstCreate(unsigned instrument_id, double freq, float fduration);
void* EventPercCreate(unsigned instrument_id, float fduration);
void* EventSingCreate(unsigned instrument_id, void* ptr_syllable_list);
void* MeteorCreate0();
void* MeteorCreate(void* ptr_event_list);
void MeteorDestroy(void* ptr);
void MeteorSaveToFile(void* ptr, const char* filename);
void MeteorLoadFromFile(void* ptr, const char* filename);
void MeteorPlay(void* ptr_meteor, void* ptr_track);
void* Base64Create(void* ptr_meteor);
void Base64Destroy(void* ptr);
const char* Base64Get(void* ptr);
""")

if os.name == 'nt':
    fn_shared_lib = 'Meteor.dll'
    fs_encoding = "mbcs"    
elif os.name == "posix":
    fn_shared_lib = 'libMeteor.so'
    fs_encoding = "utf-8"

path_shared_lib = os.path.dirname(__file__)+"/"+fn_shared_lib
Native = ffi.dlopen(path_shared_lib)

class CtrlPnt:
    def __init__(self, freq, fduration):
        self.m_cptr = Native.CtrlPntCreate(freq, fduration)
        
    def __del__(self):
        Native.CtrlPntDestroy(self.m_cptr)
        
class Syllable:
    def __init__(self, lyric, ctrl_pnts):
        obj_ctrl_pnts = ObjArray(ctrl_pnts)
        self.m_cptr = Native.SyllableCreate(lyric.encode('utf-8'), obj_ctrl_pnts.m_cptr)
    
    def __del__(self):
        Native.SyllableDestroy(self.m_cptr)
        
class Event:       
    def __del__(self):
        Native.EventDestroy(self.m_cptr)
        
    def set_offset(self, offset):
        Native.EventSetOffset(self.m_cptr, offset)
        
class EventInst(Event):
    def __init__(self, instrument_id, freq, fduration):
        self.m_cptr = Native.EventInstCreate(instrument_id, freq, fduration)
        
class EventPerc(Event):
    def __init__(self, instrument_id, fduration):
        self.m_cptr = Native.EventPercCreate(instrument_id, fduration)
        
class EventSing(Event):
    def __init__(self, instrument_id, syllable_list):
        obj_syllable_list = ObjArray(syllable_list)
        self.m_cptr = Native.EventSingCreate(instrument_id, obj_syllable_list.m_cptr)
        
class Meteor:
    def __init__(self, event_list = None):
        if event_list is None:
            self.m_cptr = Native.MeteorCreate0()
        else:
            obj_event_list = ObjArray(event_list)
            self.m_cptr = Native.MeteorCreate(obj_event_list.m_cptr)
        
    def __del__(self):
        Native.MeteorDestroy(self.m_cptr)
        
    def save_to_file(self, filename):
        Native.MeteorSaveToFile(self.m_cptr, filename.encode(fs_encoding))
        
    def load_from_file(self, filename):
        Native.MeteorLoadFromFile(self.m_cptr, filename.encode(fs_encoding))
        
    def to_base64(self):
        p_b64 = Native.Base64Create(self.m_cptr)
        b64 = ffi.string(Native.Base64Get(p_b64)).decode('utf-8')
        Native.Base64Destroy(p_b64)
        return b64
        
def MeteorPlay(meteor, track):
    Native.MeteorPlay(meteor.m_cptr, track.m_cptr)

from .ScoreDraftCore import TrackBuffer
from .ScoreDraftCore import MixTrackBufferList
from .ScoreDraftCore import WriteTrackBufferToWav

from .Instrument import Instrument
from .Percussion import Percussion
from .Singer import Singer

import math

EventType_Inst=0
EventType_Perc=1
EventType_Sing=2

class DummyTrackBuffer(TrackBuffer):
    def __init__ (self, eventList, chn=-1):
        TrackBuffer.__init__(self, chn)
        self.eventList=eventList

    def writeBlend(self, wavBuf):
        TrackBuffer.writeBlend(self,wavBuf)
        if hasattr(wavBuf, 'event'):
            event = wavBuf.event
            event.set_offset(self.getCursor())
            self.eventList += [event]
            
class DummyInstrumentEngine:
    def __init__(self, inst_id, engine, isGMDrum):
        self.inst_id=inst_id
        self.engine = engine
        self.isGMDrum = isGMDrum

    def tune(self, cmd):
        return self.engine.tune(cmd)

    def generateWave(self, freq, fduration, sampleRate):
        wavBuf=self.engine.generateWave(freq, fduration, sampleRate)
        if not self.isGMDrum:
            event = EventInst(self.inst_id, freq, fduration)
        else:
            midiPitch = int(math.log(freq/261.626)*12.0 / math.log(2.0) + 0.5)  + 60;
            if midiPitch<0:
                midiPitch = 0
            elif midiPitch>127:
                midiPitch = 127
            event = EventPerc(midiPitch, fduration)
        wavBuf.event = event
        return wavBuf

class DummyInstrument(Instrument):
    def __init__(self, inst_id, inst):
        self.shell = inst.shell
        self.engine= DummyInstrumentEngine(inst_id, inst.engine, inst.isGMDrum())
        
class DummyInstrumentCreator:
    def __init__(self):
        self.count=0
        self.map={}

    def Create(self, inst):
        if not inst in self.map:
            self.map[inst] = DummyInstrument(self.count, inst)
            self.count+=1
        return self.map[inst]

class DummyPercussionEngine:
    def __init__(self, inst_id, engine):
        self.inst_id=inst_id
        self.engine=engine

    def tune(self, cmd):
        return self.engine.tune(cmd)

    def generateWave(self, fduration, sampleRate):
        wavBuf=self.engine.generateWave(fduration, sampleRate)
        event = EventPerc(self.inst_id, fduration)
        wavBuf.event = event
        return wavBuf

class DummyPercussion(Percussion):
    def __init__(self, inst_id, perc):
        self.shell = perc.shell
        self.engine= DummyPercussionEngine(inst_id, perc.engine)    


class DummyPercussionCreator:
    def __init__(self):
        self.count=0
        self.map={}

    def Create(self, perc):
        if not perc in self.map:
            self.map[perc] = DummyPercussion(self.count, perc)
            self.count+=1
        return self.map[perc]
        
class DummySingerEngine:
    def __init__(self, inst_id, engine):
        self.inst_id=inst_id
        self.engine=engine

    def tune(self, cmd):
        return self.engine.tune(cmd)

    def generateWave(self, syllableList, sampleRate):
        wavBuf=self.engine.generateWave(syllableList, sampleRate)
        syllable_obj_list = [Syllable(syllable['lyric'], [CtrlPnt(ctrl_pnt[0], ctrl_pnt[1]) for ctrl_pnt in syllable['ctrlPnts']]) for syllable in syllableList]        
        event = EventSing(self.inst_id, syllable_obj_list)
        wavBuf.event = event
        return wavBuf

class DummySinger(Singer):
    def __init__(self, inst_id, singer):
        self.shell = singer.shell
        self.engine= DummySingerEngine(inst_id, singer.engine)   


class DummySingerCreator:
    def __init__(self):
        self.count=0
        self.map={}

    def Create(self, singer):
        if not singer in self.map:
            self.map[singer] = DummySinger(self.count, singer)
            self.count+=1
        return self.map[singer]
        
class Document:
    def __init__ (self):
        self.bufferList=[]
        self.tempo=80
        self.refFreq=261.626
        self.eventList=[]
        self.instCreator=DummyInstrumentCreator()
        self.percCreator=DummyPercussionCreator()
        self.singerCreator=DummySingerCreator()

    def getBuffer(self, bufferIndex):
        return self.bufferList[bufferIndex]

    def getTempo(self):
        return self.tempo

    def setTempo(self,tempo):
        self.tempo=tempo

    def getReferenceFrequency(self):
        return self.refFreq

    def setReferenceFrequency(self,refFreq):
        self.refFreq=refFreq

    def newBuf(self, chn=-1):
        buf=DummyTrackBuffer(self.eventList,chn)
        self.bufferList.append(buf)
        return len(self.bufferList)-1

    def setTrackVolume(self, bufferIndex, volume):
        self.bufferList[bufferIndex].setVolume(volume)

    def setTrackPan(self, bufferIndex, pan):
        self.bufferList[bufferIndex].setPan(pan)

    def playNoteSeq(self, seq, instrument, bufferIndex=-1):
        dummyInst = self.instCreator.Create(instrument)
        if bufferIndex==-1:
            bufferIndex= self.newBuf()      
        buf=self.bufferList[bufferIndex]
        dummyInst.play(buf, seq, self.tempo, self.refFreq)
        return bufferIndex  

    def playBeatSeq(self, seq, percList, bufferIndex=-1):
        dummyPercList =[self.percCreator.Create(perc) for perc in percList]
        if bufferIndex==-1:
            bufferIndex= self.newBuf()      
        buf=self.bufferList[bufferIndex]            
        Percussion.play(dummyPercList, buf, seq, self.tempo)
        return bufferIndex

    def sing(self, seq, singer, bufferIndex=-1):
        dummySinger = self.singerCreator.Create(singer)
        if bufferIndex==-1:
            bufferIndex= self.newBuf()      
        buf=self.bufferList[bufferIndex]
        dummySinger.sing( buf, seq, self.tempo, self.refFreq)
        return bufferIndex

    def trackToWav(self, bufferIndex, filename):
        WriteTrackBufferToWav(self.bufferList[bufferIndex], filename)

    def mix(self, targetBuf):
        MixTrackBufferList(targetBuf,self.bufferList)

    def mixDown(self,filename,chn=-1):
        targetBuf=TrackBuffer(chn)
        self.mix(targetBuf)
        WriteTrackBufferToWav(targetBuf, filename)
        
    def meteor(self,chn=-1):
        targetBuf=TrackBuffer(chn)
        self.mix(targetBuf)
        meteor = Meteor(self.eventList)
        MeteorPlay(meteor, targetBuf)

    def saveToFile(self,filename):
        meteor = Meteor(self.eventList)
        meteor.save_to_file(filename)





