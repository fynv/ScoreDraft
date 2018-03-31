import ScoreDraft

class Document:
	def __init__ (self):
		self.visualizerId=ScoreDraft.MeteorInitVisualizer()
		self.bufferList=[]
		self.tempo=80
		self.refFreq=261.626
	def __del__(self):
		ScoreDraft.MeteorDelVisualizer(self.visualizerId)

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
		buf=ScoreDraft.TrackBuffer(chn)
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
		ScoreDraft.MeteorProcessNoteSeq(self.visualizerId, instrument, buf.getCursor()/44100.0, seq, self.tempo, self.refFreq)
		instrument.play(buf, seq, self.tempo, self.refFreq)
		return bufferIndex	

	def playBeatSeq(self, seq, percList, bufferIndex=-1):
		if bufferIndex==-1:
			bufferIndex= self.newBuf()		
		buf=self.bufferList[bufferIndex]	
		ScoreDraft.MeteorProcessBeatSeq(self.visualizerId, percList, buf.getCursor()/44100.0, seq, self.tempo)	
		ScoreDraft.Percussion.play(percList, buf, seq, self.tempo)
		return bufferIndex

	def sing(self, seq, singer, bufferIndex=-1):
		if bufferIndex==-1:
			bufferIndex= self.newBuf()		
		buf=self.bufferList[bufferIndex]
		ScoreDraft.MeteorProcessSingingSeq(self.visualizerId, singer, buf.getCursor()/44100.0, seq, self.tempo, self.refFreq)
		singer.sing( buf, seq, self.tempo, self.refFreq)
		return bufferIndex

	def trackToWav(self, bufferIndex, filename):
		ScoreDraft.WriteTrackBufferToWav(self.bufferList[bufferIndex], filename)

	def mix(self, targetBuf):
		ScoreDraft.MixTrackBufferList(targetBuf,self.bufferList)

	def mixDown(self,filename,chn=-1):
		targetBuf=ScoreDraft.TrackBuffer(chn)
		self.mix(targetBuf)
		ScoreDraft.WriteTrackBufferToWav(targetBuf, filename)

	def meteor(self,chn=-1):
		targetBuf=ScoreDraft.TrackBuffer(chn)
		self.mix(targetBuf)
		ScoreDraft.MeteorPlay(self.visualizerId, targetBuf)
