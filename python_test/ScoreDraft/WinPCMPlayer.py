from . import PyWinPCMPlayer

def PlayTrackBuffer(buf):
	'''
	Using Win32 API to playback a track-buffer.
	buf -- an instance of TrackBuffer.
	Note that this function is a async call. Please keep the main thread busy or do a sleep to let the playback continue.
	'''
	PyWinPCMPlayer.PlayTrackBuffer(buf.m_cptr)

def PlayGetRemainingTime():
	'''
	Monitoring how much time in seconds is remaining in current play-back.
	'''
	return PyWinPCMPlayer.PyWinPCMPlayer()
