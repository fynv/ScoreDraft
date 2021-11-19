import os
import threading
from cffi import FFI

ffi = FFI()
ffi.cdef("""
void* PCMPlayerCreate(double sample_rate, unsigned ui);
void PCMPlayerDestroy(void* ptr);
void PlayTrack(void* ptr, void* ptr_track);
float GetRemainingTime(void* ptr);
void MainLoop(void* ptr);
""")

if os.name == 'nt':
    fn_shared_lib = 'PCMPlayer.dll'
elif os.name == "posix":
    fn_shared_lib = 'libPCMPlayer.so'

path_shared_lib = os.path.dirname(__file__)+"/"+fn_shared_lib
Native = ffi.dlopen(path_shared_lib)

class PCMPlayer:
    def __init__(self, sample_rate = 44100.0, ui = False):
        self.m_cptr = Native.PCMPlayerCreate(sample_rate, ui)
        
    def __del__(self):
        Native.PCMPlayerDestroy(self.m_cptr)
        
    def play_track(self, track):
        Native.PlayTrack(self.m_cptr, track.m_cptr)
        
    def remaining_time(self):
        return Native.GetRemainingTime(self.m_cptr)
        
    def main_loop(self):
        Native.MainLoop(self.m_cptr)

class AsyncUIPCMPlayer:
    def __init__(self, sample_rate = 44100.0):
        self.player = None
        self.player_ready = threading.Event()
        
    def _ui_thread(self):
        self.player = PCMPlayer(ui = True)   
        self.player_ready.set() 
        self.player.main_loop()
        self.player = None
        
    def play_track(self, track):
        if self.player is None:
            threading.Thread(target = self._ui_thread).start()
            self.player_ready.wait()
            self.player_ready.clear()
        self.player.play_track(track)
    
    def remaining_time(self):
        if self.player is None:
            return 0.0
        return self.player.remaining_time()
