import os
from cffi import FFI

ffi = FFI()
ffi.cdef("""
void GeneratePureSin(void* ptr_wavbuf, float freq, float fduration);
void GenerateSquare(void* ptr_wavbuf, float freq, float fduration);
void GenerateTriangle(void* ptr_wavbuf, float freq, float fduration);
void GenerateSawtooth(void* ptr_wavbuf, float freq, float fduration);
void GenerateNaivePiano(void* ptr_wavbuf, float freq, float fduration);
void GenerateBottleBlow(void* ptr_wavbuf, float freq, float fduration);
""")

if os.name == 'nt':
    fn_shared_lib = 'SimpleInstruments.dll'
elif os.name == "posix":
    fn_shared_lib = 'libSimpleInstruments.so'

path_shared_lib = os.path.dirname(__file__)+"/"+fn_shared_lib
Native = ffi.dlopen(path_shared_lib)

from .Instrument import Instrument
from .ScoreDraftCore import F32Buf
from .ScoreDraftCore import WavBuffer

class Engine:
    def __init__(self, generator):
        self.generator= generator
    def tune(self, cmd):
        pass
    def generateWave(self, freq, fduration, sampleRate):
        wav = F32Buf(0)
        wav_buf = WavBuffer(sampleRate, 1, wav)
        self.generator(wav_buf.m_cptr, freq, fduration)
        return wav_buf

def EnginePureSin():
    return Engine(Native.GeneratePureSin)
    
def EngineSquare():
    return Engine(Native.GenerateSquare)
    
def EngineTriangle():
    return Engine(Native.GenerateTriangle)
    
def EngineSawtooth():
    return Engine(Native.GenerateSawtooth)
    
def EngineNaivePiano():
    return Engine(Native.GenerateNaivePiano)
    
def EngineBottleBlow():
    return Engine(Native.GenerateBottleBlow)

class PureSin(Instrument):
    def __init__(self):
        Instrument.__init__(self)
        self.engine = EnginePureSin()

class Square(Instrument):
    def __init__(self):
        Instrument.__init__(self)
        self.engine = EngineSquare()

class Triangle(Instrument):
    def __init__(self):
        Instrument.__init__(self)
        self.engine = EngineTriangle()

class Sawtooth(Instrument):
    def __init__(self):
        Instrument.__init__(self)
        self.engine = EngineSawtooth()

class NaivePiano(Instrument):
    def __init__(self):
        Instrument.__init__(self)
        self.engine = EngineNaivePiano()

class BottleBlow(Instrument):
    def __init__(self):
        Instrument.__init__(self)
        self.engine = EngineBottleBlow()

