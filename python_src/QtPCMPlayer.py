import os
from . import PyQtPCMPlayerExt

ScoreDraftPath_old= os.path.dirname(__file__)
ScoreDraftPath=""
#\\escaping fix
for ch in ScoreDraftPath_old:
	if ch=="\\":
		ScoreDraftPath+="/"
	else:
		ScoreDraftPath+=ch

PyQtPCMPlayerExt.QPlaySetRoot(ScoreDraftPath)

def QPlayTrackBuffer(buf):
	'''
	Using Qt Multimedia API to playback a track-buffer.
	buf -- an instance of TrackBuffer.
	'''
	PyQtPCMPlayerExt.QPlayTrackBuffer(buf.m_cptr)

def QPlayGetRemainingTime():
	'''
	Monitoring how much time in seconds is remaining in current play-back.
	'''
	return PyQtPCMPlayerExt.QPlayGetRemainingTime()
