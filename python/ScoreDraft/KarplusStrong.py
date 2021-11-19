import os
from cffi import FFI

ffi = FFI()
ffi.cdef("""
void KarplusStrongGenerate(void* ptr_wavbuf, float freq, float fduration, float cut_freq, float loop_gain, float sustain_gain);
""")

if os.name == 'nt':
    fn_shared_lib = 'KarplusStrong.dll'
elif os.name == "posix":
    fn_shared_lib = 'libKarplusStrong.so'

path_shared_lib = os.path.dirname(__file__)+"/"+fn_shared_lib
Native = ffi.dlopen(path_shared_lib)

from .Instrument import Instrument
from .ScoreDraftCore import F32Buf
from .ScoreDraftCore import WavBuffer

from .Catalog import Catalog
Catalog['Engines'] += ['KarplusStrongInstrument - Instrument']


class EngineKarplusStrong:
    def __init__(self):
        self.cut_freq=10000.0
        self.loop_gain=0.99
        self.sustain_gain=0.8
    def tune(self, cmd):
        pass
    def generateWave(self, freq, fduration, sampleRate):
        wav = F32Buf(0)
        wav_buf = WavBuffer(sampleRate, 1, wav)
        Native.KarplusStrongGenerate(wav_buf.m_cptr, freq,fduration, self.cut_freq, self.loop_gain, self.sustain_gain)
        return wav_buf

class KarplusStrongInstrument(Instrument):
    def __init__(self):
        Instrument.__init__(self)
        self.engine = EngineKarplusStrong()
    def setCutFrequency(self, cut_freq):
        # This is the cut-frequency of the feedback filter for pitch 261.626Hz
        self.engine.cut_freq = cut_freq
    def setLoopGain(self, loop_gain):
        self.engine.loop_gain = loop_gain
    def setSustainGain(self, sustain_gain):
        self.engine.sustain_gain = sustain_gain


