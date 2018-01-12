import os 
import PyScoreDraft

class TrackBuffer:
	def __init__ (self):
		self.id= PyScoreDraft.InitTrackBuffer()
	def __del__(self):
		PyScoreDraft.DelTrackBuffer(self.id)

class Instrument:
	def __init__ (self, clsId):
		self.id=PyScoreDraft.InitInstrument(clsId)
	def __del__ (self):
		PyScoreDraft.DelInstrument(self.id)
	def play(self, buf, seq, volume=1.0, tempo=80, refFreq=264.0):
		PyScoreDraft.InstrumentPlay(buf.id, self.id, seq, volume, tempo, refFreq)
	def tune(self, cmd):
		PyScoreDraft.InstrumentTune(self.id, cmd)

class Percussion:
	def __init__(self, clsId):
		self.id=PyScoreDraft.InitPercussion(clsId)
	def __del__ (self):
		PyScoreDraft.DelPercussion(self.id)		
	def tune(self, cmd):
		PyScoreDraft.PercussionTune(self.id, cmd)

def PercussionPlay(buf, seq, percList, volume=1.0, tempo=80):
		percIdList= [perc.id for perc in percList]
		PyScoreDraft.PercussionPlay(buf.id, percIdList, seq, volume, tempo)

class Singer:
	def __init__(self, clsId):
		self.id=PyScoreDraft.InitSinger(clsId)
	def __del__ (self):
		PyScoreDraft.DelSinger(self.id)		
	def sing(self, buf, seq, singer, volume=1.0, tempo=80, refFreq=264.0):
		PyScoreDraft.Sing(buf.id, self.id, seq, volume, tempo, refFreq)
	def tune(self, cmd):
		PyScoreDraft.SingerTune(self.id, cmd)

def MixTrackBufferList (targetbuf, bufferList):
	bufIdList=[buf.id for buf in bufferList] 
	PyScoreDraft.MixTrackBufferList(targetbuf.id, bufIdList)

def WriteTrackBufferToWav(buf, filename):
	PyScoreDraft.WriteTrackBufferToWav(buf.id, filename)

# generate dynamic code
g_generated_code_and_summary=PyScoreDraft.GenerateCode()
exec(g_generated_code_and_summary[0])

def PrintGeneratedCode():
	print (g_generated_code_and_summary[0])

def PrintGeneratedCodeSummary():
	print (g_generated_code_and_summary[1])

class Document:
	def __init__ (self):
		self.bufferList=[]
		self.tempo=80
		self.refFreq=264.0

	def getBuffer(self, i):
		return self.bufferList[i]

	def getTempo(self):
		return self.tempo

	def setTempo(self,tempo):
		self.tempo=tempo

	def getReferenceFrequency(self):
		return self.refFreq

	def setReferenceFreqeuncy(self,refFreq):
		self.refFreq=refFreq

	def newBuf(self):
		buf=TrackBuffer()
		self.bufferList.append(buf)
		return len(self.bufferList)-1

	def playNoteSeq(self, seq, instrument, volume=1.0, bufferIndex=-1):
		if bufferIndex==-1:
			bufferIndex= self.newBuf()		
		buf=self.bufferList[bufferIndex]
		instrument.play(buf, seq, volume, self.tempo, self.refFreq)
		return bufferIndex	

	def playBeatSeq(self, seq, percList, volume=1.0, bufferIndex=-1):
		if bufferIndex==-1:
			bufferIndex= self.newBuf()		
		buf=self.bufferList[bufferIndex]			
		PercussionPlay(buf, seq, percList, volume, self.tempo)
		return bufferIndex

	def sing(self, seq, singer, volume=1.0, bufferIndex=-1):
		if bufferIndex==-1:
			bufferIndex= self.newBuf()		
		buf=self.bufferList[bufferIndex]
		singer.sing( buf, seq, singer, volume, self.tempo, self.refFreq)
		return bufferIndex

	def trackToWav(self, bufferIndex, filename):
		WriteTrackBufferToWav(self.bufferList[bufferIndex], filename)

	def mix(self, targetBuf):
		MixTrackBufferList(targetBuf,self.bufferList)

	def mixDown(self,filename):
		targetBuf=TrackBuffer()
		self.mix(targetBuf)
		WriteTrackBufferToWav(targetBuf, filename)

