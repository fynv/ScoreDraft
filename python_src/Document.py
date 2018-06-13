from .TrackBuffer import TrackBuffer
from .TrackBuffer import MixTrackBufferList
from .TrackBuffer import WriteTrackBufferToWav

from .Instrument import Instrument
from .Percussion import Percussion
from .Singer import Singer

class Document:
	'''
	An utility class to simplify user-side coding.
	The class maintains a list of track-buffers and some shared states (tempo and reference-frequency)
	'''
	def __init__ (self):
		self.bufferList=[]
		self.tempo=80
		self.refFreq=261.626

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
		'''
		The created track-buffer will be added to the track-buffer list of the document
		The index of the new track-buffer is returned
		'''
		buf=TrackBuffer(chn)
		self.bufferList.append(buf)
		return len(self.bufferList)-1

	def setTrackVolume(self, bufferIndex, volume):
		self.bufferList[bufferIndex].setVolume(volume)

	def setTrackPan(self, bufferIndex, pan):
		self.bufferList[bufferIndex].setPan(pan)

	def playNoteSeq(self, seq, instrument, bufferIndex=-1):
		'''
		Play a note sequence in the context of a document.
		instrument -- An instance of Instrument
		When bufferIndex==-1, a new track-buffer will be returned. Otherwise, an existing track-buffer will 
		be used and result is appended.
		The index of the target track-buffer is returned.
		'''
		if bufferIndex==-1:
			bufferIndex= self.newBuf()		
		buf=self.bufferList[bufferIndex]
		instrument.play(buf, seq, self.tempo, self.refFreq)
		return bufferIndex	

	def playBeatSeq(self, seq, percList, bufferIndex=-1):
		'''
		Play a beat sequence in the context of a document.
		When bufferIndex==-1, a new track-buffer will be returned. Otherwise, an existing track-buffer will 
		be used and result is appended.
		The index of the target track-buffer is returned.
		'''
		if bufferIndex==-1:
			bufferIndex= self.newBuf()		
		buf=self.bufferList[bufferIndex]			
		Percussion.play(percList, buf, seq, self.tempo)
		return bufferIndex

	def sing(self, seq, singer, bufferIndex=-1):
		'''
		Sing a sequence in the context of a document.
		When bufferIndex==-1, a new track-buffer will be returned. Otherwise, an existing track-buffer will 
		be used and result is appended.
		The index of the target track-buffer is returned.
		'''
		if bufferIndex==-1:
			bufferIndex= self.newBuf()		
		buf=self.bufferList[bufferIndex]
		singer.sing( buf, seq, self.tempo, self.refFreq)
		return bufferIndex

	def trackToWav(self, bufferIndex, filename):
		WriteTrackBufferToWav(self.bufferList[bufferIndex], filename)

	def mix(self, targetBuf):
		'''
		Mix the track-buffers in the document to a target buffer.
		targetBuf -- An instance of TrackBuffer
		'''
		MixTrackBufferList(targetBuf,self.bufferList)

	def mixDown(self,filename,chn=-1):
		'''
		Mix the track-buffers in the document to a temporary buffer and write to a .wav file.
		filename -- a string
		'''
		targetBuf=TrackBuffer(chn)
		self.mix(targetBuf)
		WriteTrackBufferToWav(targetBuf, filename)

