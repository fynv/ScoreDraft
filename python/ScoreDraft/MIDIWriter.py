import os
from cffi import FFI
from .ScoreDraftCore import ObjArray

ffi = FFI()
ffi.cdef("""
void* NoteCreate(float freq_rel, int duration);
void NoteDestroy(void* ptr);
void* WriteToMidi(void* ptr_seq_list, unsigned tempo, float refFreq, const char* fileName);
""")

if os.name == 'nt':
    fn_shared_lib = 'MIDIWriter.dll'
    fs_encoding = "mbcs"
elif os.name == "posix":
    fn_shared_lib = 'libMIDIWriter.so'
    fs_encoding = "utf-8"

path_shared_lib = os.path.dirname(__file__)+"/"+fn_shared_lib
Native = ffi.dlopen(path_shared_lib)

class Note:
    def __init__(self, freq_rel, duration):
        self.m_cptr = Native.NoteCreate(freq_rel, duration)
        
    def __del__(self):
        Native.NoteDestroy(self.m_cptr)        
        
def WriteNoteSequencesToMidi(seqList, tempo, refFreq, fileName):
    '''
    Write a list of note sequences to a MIDI file.
    seqList -- a list of note sequences.
    tempo -- an integer indicating tempo in beats/minute.
    refFreq -- a float indicating reference frequency in Hz.
    fileName -- a string.
    '''    
    obj_seq_list = ObjArray([ObjArray([Note(note[0], note[1]) for note in seq]) for seq in seqList])
    Native.WriteToMidi(obj_seq_list.m_cptr, tempo, refFreq, fileName.encode(fs_encoding))
