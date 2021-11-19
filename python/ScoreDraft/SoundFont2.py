import os
from cffi import FFI

ffi = FFI()
ffi.cdef("""
// SF2Bank
void* SF2BankCreate(const char* filename);
void SF2BankDestroy(void* ptr);
unsigned long long SF2BankGetNumberPresets(void* ptr);
const char* SF2BankGetPresetName(void* ptr, int i);
int SF2BankGetPresetBankNum(void* ptr, int i);
int SF2BankGetPresetNumber(void* ptr, int i);

// SF2Tone
void* SF2ToneCreate(void* ptr_bank, unsigned preset_index);
void SF2ToneDestroy(void* ptr);

// SF2Synth
void SF2SynthNote(void* ptr_wavbuf, void* ptr_tone, float key, float vel, unsigned numSamples, unsigned outputmode, float global_gain_db);
""")

if os.name == 'nt':
    fn_shared_lib = 'SoundFont2.dll'
    fs_encoding = "mbcs"
elif os.name == "posix":
    fn_shared_lib = 'libSoundFont2.so'
    fs_encoding = "utf-8"

path_shared_lib = os.path.dirname(__file__)+"/"+fn_shared_lib
Native = ffi.dlopen(path_shared_lib)

class SF2Bank:
    def __init__(self, filename):
        self.m_cptr = Native.SF2BankCreate(filename.encode(fs_encoding))
    
    def __del__(self):
        Native.SF2BankDestroy(self.m_cptr)
        
    def num_presets(self):
        return Native.SF2BankGetNumberPresets(self.m_cptr)
        
    def get_preset_info(self, i):
        info = {}
        info['presetName'] = ffi.string(Native.SF2BankGetPresetName(self.m_cptr, i)).decode('utf-8')
        info['bank'] = Native.SF2BankGetPresetBankNum(self.m_cptr, i)
        info['preset'] = Native.SF2BankGetPresetNumber(self.m_cptr, i)
        return info
        
SF2Banks={}

def GetSF2Bank(filename):
    if not (filename in SF2Banks):
        SF2Banks[filename] = SF2Bank(filename)
    return SF2Banks[filename]
    
def ListPresets(filename):
    sf2 = GetSF2Bank(filename)
    num_presets = sf2.num_presets()
    for i in range(num_presets):
        preset = sf2.get_preset_info(i)
        print ('%d : %s bank=%d number=%d' % (i, preset['presetName'], preset['bank'], preset['preset']))
        
import math
from .ScoreDraftCore import F32Buf, WavBuffer
from .Instrument import Instrument
from .Catalog import Catalog
Catalog['Engines'] += ['SF2Instrument - Instrument']
        
class EngineSoundFont2:
    def __init__(self, bank, preset_index):
        self.m_bank = bank
        self.m_preset_index = preset_index
        self.m_cptr = Native.SF2ToneCreate(bank.m_cptr, preset_index)
        self.global_gain_db = 0.0
        self.vel = 1.0
    
    def __del__(self):
        Native.SF2ToneDestroy(self.m_cptr)

    def isGMDrum(self):
        preset_info = self.m_bank.get_preset_info(self.m_preset_index)
        return preset_info['bank'] == 128

    def tune(self, cmd):
        cmd_split= cmd.split(' ')
        cmd_len=len(cmd_split)
        if cmd_len>=1:
            if cmd_len>1 and cmd_split[0]=='velocity':
                self.vel = float(cmd_split[1])
                return True
        return False

    def generateWave(self, freq, fduration, sampleRate):
        key = math.log(freq / 261.626)/math.log(2)*12.0+60.0
        num_samples = int(fduration * sampleRate * 0.001+0.5)
        wav = F32Buf(0)
        wav_buf = WavBuffer(sampleRate, 1, wav)  
        Native.SF2SynthNote(wav_buf.m_cptr, self.m_cptr, key, self.vel, num_samples, 0, self.global_gain_db)        
        return wav_buf 

class SF2Instrument(Instrument):
    def __init__(self, fn, preset_index):
        Instrument.__init__(self)
        sf2 = GetSF2Bank(fn)
        self.engine= EngineSoundFont2(sf2, preset_index)
        
    def isGMDrum(self):
        return self.engine.isGMDrum()


