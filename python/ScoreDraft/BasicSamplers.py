import os
from cffi import FFI

ffi = FFI()
ffi.cdef("""
void* SampleCreate(unsigned origin_sample_rate, unsigned chn, void* ptr_f32_buf, float max_v);
void* InstrumentSampleCreate(unsigned origin_sample_rate, unsigned chn, void* ptr_f32_buf, float max_v, float origin_freq);
void SampleDestroy(void *ptr);
void PercussionGenerate(void* ptr_wavbuf, void* ptr_sample, float fduration);
void InstrumentSingleGenerate(void* ptr_wavbuf, void* ptr_sample, float freq, float fduration);
void InstrumentMultiGenerate(void* ptr_wavbuf, void* ptr_sample_lst, float freq, float fduration);
""")

if os.name == 'nt':
    fn_shared_lib = 'BasicSamplers.dll'
elif os.name == "posix":
    fn_shared_lib = 'libBasicSamplers.so'

path_shared_lib = os.path.dirname(__file__)+"/"+fn_shared_lib
Native = ffi.dlopen(path_shared_lib)

from .ScoreDraftCore import F32Buf

class Sample:
    def __init__(self, origin_sample_rate, channel_num, f32buf, max_value = -1.0):
        self.m_buf = f32buf
        self.m_cptr = Native.SampleCreate(origin_sample_rate, channel_num, f32buf.m_cptr, max_value)
        
    def __del__(self):
        Native.SampleDestroy(self.m_cptr)
        
class InstrumentSample(Sample):
    def __init__(self, origin_sample_rate, channel_num, f32buf, max_value = -1.0, origin_freq = -1.0):
        self.m_buf = f32buf
        self.m_cptr = Native.InstrumentSampleCreate(origin_sample_rate, channel_num, f32buf.m_cptr, max_value, origin_freq)
        
import wave

def loadSample(filename, is_instrument):
    wavS16=bytes()
    nChn = 1
    nFrames = 0
    framerate = 44100
    origin_freq = -1.0
    with wave.open(filename, mode='rb') as wavFile:
        nFrames =wavFile.getnframes() 
        nChn = wavFile.getnchannels()
        wavS16=wavFile.readframes(nFrames)
        framerate = wavFile.getframerate()
        
    f32buf = F32Buf.from_s16(wavS16)
        
    if is_instrument:
        freq_fn = filename[0:len(filename)-4]+".freq"
        if os.path.isfile(freq_fn):
            with open(freq_fn, "r") as f:
                origin_freq= float(f.readline())
        return InstrumentSample(framerate, nChn, f32buf, origin_freq = origin_freq)
    
    else:
        return Sample(framerate, nChn, f32buf)
        

Samples_Percussion = {}

def GetSample_Percussion(fn):
    if not (fn in Samples_Percussion):
        Samples_Percussion[fn] = loadSample(fn, False)
    return Samples_Percussion[fn]
    

Samples_Single = {}

def GetSample_Single(fn):
    if not (fn in Samples_Single):
        Samples_Single[fn] = loadSample(fn, True)
    return Samples_Single[fn]

Samples_Multi = {}

def GetSamples_Multi(path):
    if not (path in Samples_Multi):
        samples = []
        if os.path.isdir(path):
            for item in os.listdir(path):
                fn = path+'/'+item
                if os.path.isfile(fn) and item.endswith(".wav"):
                    samples+=[loadSample(fn, True)]
        Samples_Multi[path] = samples
    return Samples_Multi[path]
    
from .ScoreDraftCore import ObjArray, WavBuffer
from .Instrument import Instrument
from .Percussion import Percussion
from .Catalog import Catalog
Catalog['Engines'] += ['InstrumentSampler_Single - Instrument']
Catalog['Engines'] += ['InstrumentSampler_Multi - Instrument']
Catalog['Engines'] += ['PercussionSampler - Percussion']


class EnginePercussionSampler:
    def __init__(self, sample):
        self.sample=sample      
    def tune(self, cmd):
        pass
    def generateWave(self, fduration, sampleRate):
        wav = F32Buf(0)
        wav_buf = WavBuffer(sampleRate, 1, wav)       
        Native.PercussionGenerate(wav_buf.m_cptr, self.sample.m_cptr, fduration)
        return wav_buf

class EngineInstrumentSampler_Single:
    def __init__(self, sample):
        self.sample=sample      
    def tune(self, cmd):
        pass
    def generateWave(self, freq, fduration, sampleRate):
        wav = F32Buf(0)
        wav_buf = WavBuffer(sampleRate, 1, wav)       
        Native.InstrumentSingleGenerate(wav_buf.m_cptr, self.sample.m_cptr, freq, fduration)
        return wav_buf

class EngineInstrumentSampler_Multi:
    def __init__(self, samples):
        self.samples = samples  
        self.sample_lst = ObjArray(self.samples)
    def tune(self, cmd):
        pass
    def generateWave(self, freq, fduration, sampleRate):
        wav = F32Buf(0)
        wav_buf = WavBuffer(sampleRate, 1, wav)
        Native.InstrumentMultiGenerate(wav_buf.m_cptr, self.sample_lst.m_cptr, freq, fduration)        
        return wav_buf

class PercussionSampler(Percussion):
    def __init__(self, wavPath):
        Percussion.__init__(self)
        sample = GetSample_Percussion(wavPath)
        self.engine = EnginePercussionSampler(sample)

class InstrumentSampler_Single(Instrument):
    def __init__(self, wavPath):
        Instrument.__init__(self)
        sample = GetSample_Single(wavPath)
        self.engine = EngineInstrumentSampler_Single(sample)

class InstrumentSampler_Multi(Instrument):
    def __init__(self, folderPath):
        Instrument.__init__(self)
        samples = GetSamples_Multi(folderPath)
        self.engine = EngineInstrumentSampler_Multi(samples)



