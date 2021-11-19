import os
from cffi import FFI

ffi = FFI()
ffi.cdef("""
unsigned HaveCUDA();
void* FrqDataPointCreate(double freq, double dyn);
void FrqDataPointDestroy(void* ptr);
void* FrqDataCreate();
void FrqDataDestroy(void* ptr);
void FrqDataSet(void* ptr, int interval, double key, void* ptr_data_points);
void FrqDataDetect(void* ptr, void* ptr_f32_buf, int interval);
void* SourceMapCtrlPntCreate(float srcPos, float dstPos, int isVowel);
void SourceMapCtrlPntDestroy(void* ptr);
void* PieceCreate(void* ptr_f32buf, void* ptr_frq_data, void* ptr_src_map);
void PieceDestroy(void* ptr);
void* GeneralCtrlPntCreate(float value, float dstPos);
void GeneralCtrlPntDestroy(void* ptr);
void* SentenceDescriptorCreate(void* ptr_pieces, void* ptr_piece_map, void* ptr_freq_map, void* ptr_volume_map);
void SentenceDescriptorDestroy(void* ptr);
void GenerateSentence(void* ptr_wavbuf, void* ptr_sentence);
void GenerateSentenceCUDA(void* ptr_wavbuf, void* ptr_sentence);
""")

if os.name == 'nt':
    fn_shared_lib = 'VoiceSampler.dll'
elif os.name == "posix":
    fn_shared_lib = 'libVoiceSampler.so'

path_shared_lib = os.path.dirname(__file__)+"/"+fn_shared_lib
Native = ffi.dlopen(path_shared_lib)

from .ScoreDraftCore import ObjArray, F32Buf, WavBuffer

def HaveCUDA():
    return Native.HaveCUDA() != 0
    
class FrqDataPoint:
    def __init__(self, freq, dyn):
        self.m_cptr = Native.FrqDataPointCreate(freq, dyn)
        
    def __del__(self):
        Native.FrqDataPointDestroy(self.m_cptr)
        
class FrqData:
    def __init__(self):
        self.m_cptr = Native.FrqDataCreate()
        
    def __del__(self):
        Native.FrqDataDestroy(self.m_cptr)
        
    def set(self, interval, key, lst_data_points):
        obj_arr = ObjArray(lst_data_points)
        Native.FrqDataSet(self.m_cptr, interval, key, obj_arr.m_cptr)
        
    def detect(self, f32_buf, interval):
        Native.FrqDataDetect(self.m_cptr, f32_buf.m_cptr, interval)

class SourceMapCtrlPnt:
    def __init__(self, srcPos, dstPos, isVowel):
        self.m_cptr = Native.SourceMapCtrlPntCreate(srcPos, dstPos, isVowel)
        
    def __del__(self):
        Native.SourceMapCtrlPntDestroy(self.m_cptr)
        
class Piece:
    def __init__(self, f32buf, frq_data, src_map):
        self.m_buf = f32buf
        obj_arr = ObjArray(src_map)
        self.m_cptr = Native.PieceCreate(f32buf.m_cptr, frq_data.m_cptr, obj_arr.m_cptr)
    
    def __del__(self):
        Native.PieceDestroy(self.m_cptr)
        
class GeneralCtrlPnt:
    def __init__(self, value, dstPos):
        self.m_cptr = Native.GeneralCtrlPntCreate(value, dstPos)
        
    def __del__(self):
        Native.GeneralCtrlPntDestroy(self.m_cptr)
        
class SentenceDescriptor:
    def __init__(self, pieces, piece_map, freq_map, volume_map):
        self.m_pieces = pieces
        obj_pieces = ObjArray(pieces)
        obj_piece_map = ObjArray(piece_map)
        obj_freq_map = ObjArray(freq_map)
        obj_volume_map = ObjArray(volume_map)
        self.m_cptr = Native.SentenceDescriptorCreate(obj_pieces.m_cptr, obj_piece_map.m_cptr, obj_freq_map.m_cptr, obj_volume_map.m_cptr)
    
    def __del__(self):
        Native.SentenceDescriptorDestroy(self.m_cptr)
        
def CreateSentenceDescriptor(desc_dictionary):
    lst_pieces = desc_dictionary["pieces"]
    pieces = []
    for obj_piece in lst_pieces:
        obj_src = obj_piece["src"]
        wav = obj_src["wav"]
        frq_data = obj_src["frq"]
        
        lst_map = obj_piece["map"]
        src_map = []
        num_ctrlpnts = len(lst_map)
        for j in range(num_ctrlpnts):
            tuple_ctrlpnt = lst_map[j]
            src_pos = tuple_ctrlpnt[0]
            dst_pos = tuple_ctrlpnt[1]
            if j < num_ctrlpnts - 1:
                isVowel = tuple_ctrlpnt[2]
            else:
                isVowel = 0
            src_map += [SourceMapCtrlPnt(src_pos, dst_pos, isVowel)]
            
        pieces += [Piece(wav, frq_data, src_map)]
        
    lst_piece_map = desc_dictionary["piece_map"]
    piece_map = [GeneralCtrlPnt(tuple_ctrlpnt[0], tuple_ctrlpnt[1]) for tuple_ctrlpnt in lst_piece_map]
    
    lst_freq_map = desc_dictionary["freq_map"]
    freq_map = [GeneralCtrlPnt(tuple_ctrlpnt[0], tuple_ctrlpnt[1]) for tuple_ctrlpnt in lst_freq_map]
    
    lst_volume_map = desc_dictionary["volume_map"]
    volume_map = [GeneralCtrlPnt(tuple_ctrlpnt[0], tuple_ctrlpnt[1]) for tuple_ctrlpnt in lst_volume_map]
    
    return SentenceDescriptor(pieces, piece_map, freq_map, volume_map)    
    
def GenerateSentence(desc_dictionary):
    sentence_desc = CreateSentenceDescriptor(desc_dictionary)
    wav = F32Buf(0)
    wav_buf = WavBuffer(44100.0, 1, wav)       
    Native.GenerateSentence(wav_buf.m_cptr, sentence_desc.m_cptr)
    return wav_buf
    
def GenerateSentenceCUDA(desc_dictionary):
    sentence_desc = CreateSentenceDescriptor(desc_dictionary)
    wav = F32Buf(0)
    wav_buf = WavBuffer(44100.0, 1, wav)       
    Native.GenerateSentenceCUDA(wav_buf.m_cptr, sentence_desc.m_cptr)
    return wav_buf

