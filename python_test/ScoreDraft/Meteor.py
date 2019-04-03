from .TrackBuffer import TrackBuffer
from .TrackBuffer import MixTrackBufferList
from .TrackBuffer import WriteTrackBufferToWav

from .Instrument import Instrument
from .Percussion import Percussion
from .Singer import Singer

try:
	from .PyMeteor import MeteorInitVisualizer, MeteorDelVisualizer
	from .PyMeteor import MeteorProcessNoteSeq
	from .PyMeteor import MeteorProcessBeatSeq
	from .PyMeteor import MeteorProcessSingingSeq
	from .PyMeteor import MeteorPlay
	from .PyMeteor import MeteorSaveToFile

	class Visualizer:
		def __init__ (self):
			self.m_cptr = MeteorInitVisualizer()
			self.m_last_id = 0
			self.m_id_map = {}

		def __del__ (self):
			MeteorDelVisualizer(self.m_cptr)

		def ProcessNoteSeq(self, instrument, buf, seq, tempo, refFreq):
			if not instrument in self.m_id_map:
				self.m_id_map[instrument]=self.m_last_id
				self.m_last_id+=1
			inst_id = self.m_id_map[instrument]
			MeteorProcessNoteSeq(self.m_cptr, inst_id, instrument.isGMDrum(), buf.m_cptr, seq, tempo, refFreq)

		def ProcessBeatSeq(self, percList, buf, seq, tempo):
			percIds = []
			for perc in percList:
				if not perc in self.m_id_map:
					self.m_id_map[perc]=self.m_last_id
					self.m_last_id+=1
				percIds += [ self.m_id_map[perc] ]
			MeteorProcessBeatSeq(self.m_cptr, percIds, buf.m_cptr, seq, tempo)

		def ProcessSingingSeq(self, singer, buf, seq, tempo, refFreq):
			if not singer in self.m_id_map:
				self.m_id_map[singer]=self.m_last_id
				self.m_last_id+=1
			singer_id = self.m_id_map[singer]
			MeteorProcessSingingSeq(self.m_cptr, singer_id,  buf.m_cptr, seq, tempo, refFreq)

		def Play(self, buf):
			MeteorPlay(self.m_cptr, buf.m_cptr)

		def SaveToFile(self, filename):
			MeteorSaveToFile(self.m_cptr, filename)

	class Document:
		def __init__ (self):
			self.visualizer = Visualizer()
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
			self.visualizer.ProcessNoteSeq( instrument, buf, seq, self.tempo, self.refFreq)
			instrument.play(buf, seq, self.tempo, self.refFreq)
			return bufferIndex	

		def playBeatSeq(self, seq, percList, bufferIndex=-1):
			if bufferIndex==-1:
				bufferIndex= self.newBuf()		
			buf=self.bufferList[bufferIndex]	
			self.visualizer.ProcessBeatSeq( percList, buf, seq, self.tempo)	
			Percussion.play(percList, buf, seq, self.tempo)
			return bufferIndex

		def sing(self, seq, singer, bufferIndex=-1):
			if bufferIndex==-1:
				bufferIndex= self.newBuf()		
			buf=self.bufferList[bufferIndex]
			self.visualizer.ProcessSingingSeq( singer, buf, seq, self.tempo, self.refFreq)
			singer.sing( buf, seq, self.tempo, self.refFreq)
			return bufferIndex

		def trackToWav(self, bufferIndex, filename):
			WriteTrackBufferToWav(self.bufferList[bufferIndex], filename)

		def mix(self, targetBuf):
			MixTrackBufferList(targetBuf,self.bufferList)

		def mixDown(self, filename, chn=-1):
			targetBuf=TrackBuffer(chn)
			self.mix(targetBuf)
			WriteTrackBufferToWav(targetBuf, filename)

		def meteor(self,chn=-1):
			targetBuf=TrackBuffer(chn)
			self.mix(targetBuf)
			self.visualizer.Play(targetBuf)

		def saveToFile(self, filename):
			self.visualizer.SaveToFile(filename)

except ImportError:
	pass
	
	