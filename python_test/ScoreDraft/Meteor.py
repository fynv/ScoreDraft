from .TrackBuffer import TrackBuffer
from .TrackBuffer import MixTrackBufferList
from .TrackBuffer import WriteTrackBufferToWav

from .Instrument import Instrument
from .Percussion import Percussion
from .Singer import Singer

try:
	from .Extensions import MeteorInitVisualizer
	from .Extensions import MeteorDelVisualizer
	from .Extensions import MeteorProcessNoteSeq
	from .Extensions import MeteorProcessBeatSeq
	from .Extensions import MeteorProcessSingingSeq
	from .Extensions import MeteorPlay
	from .Extensions import MeteorSaveToFile

	class Document:
		def __init__ (self):
			self.visualizerId=MeteorInitVisualizer()
			self.bufferList=[]
			self.tempo=80
			self.refFreq=261.626
		def __del__(self):
			MeteorDelVisualizer(self.visualizerId)

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
			buf=TrackBuffer(chn)
			self.bufferList.append(buf)
			return len(self.bufferList)-1

		def setTrackVolume(self, bufferIndex, volume):
			self.bufferList[bufferIndex].setVolume(volume)

		def setTrackPan(self, bufferIndex, pan):
			self.bufferList[bufferIndex].setPan(pan)

		def playNoteSeq(self, seq, instrument, bufferIndex=-1):
			if bufferIndex==-1:
				bufferIndex= self.newBuf()		
			buf=self.bufferList[bufferIndex]
			MeteorProcessNoteSeq(self.visualizerId, instrument, buf, seq, self.tempo, self.refFreq)
			instrument.play(buf, seq, self.tempo, self.refFreq)
			return bufferIndex	

		def playBeatSeq(self, seq, percList, bufferIndex=-1):
			if bufferIndex==-1:
				bufferIndex= self.newBuf()		
			buf=self.bufferList[bufferIndex]	
			MeteorProcessBeatSeq(self.visualizerId, percList, buf, seq, self.tempo)	
			Percussion.play(percList, buf, seq, self.tempo)
			return bufferIndex

		def sing(self, seq, singer, bufferIndex=-1):
			if bufferIndex==-1:
				bufferIndex= self.newBuf()		
			buf=self.bufferList[bufferIndex]
			MeteorProcessSingingSeq(self.visualizerId, singer, buf, seq, self.tempo, self.refFreq)
			singer.sing( buf, seq, self.tempo, self.refFreq)
			return bufferIndex

		def trackToWav(self, bufferIndex, filename):
			WriteTrackBufferToWav(self.bufferList[bufferIndex], filename)

		def mix(self, targetBuf):
			MixTrackBufferList(targetBuf,self.bufferList)

		def mixDown(self,filename,chn=-1):
			targetBuf=TrackBuffer(chn)
			self.mix(targetBuf)
			WriteTrackBufferToWav(targetBuf, filename)

		def meteor(self,chn=-1):
			targetBuf=TrackBuffer(chn)
			self.mix(targetBuf)
			MeteorPlay(self.visualizerId, targetBuf)

		def saveToFile(self, filename):
			MeteorSaveToFile(self.visualizerId, filename)
except ImportError:
	pass
	
	