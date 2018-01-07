import os 
import PyScoreDraft
from PyScoreDraft import WriteNoteSequencesToMidi

class Instrument:
	def __init__ (self, clsId):
		self.inst_id=PyScoreDraft.InitInstrument(clsId)
	def tune(self, cmd):
		PyScoreDraft.InstrumentTune(self.inst_id, cmd)

class Percussion:
	def __init__(self, clsId):
		self.perc_id=PyScoreDraft.InitPercussion(clsId)
	def tune(self, cmd):
		PyScoreDraft.PercussionTune(self.perc_id, cmd)

class Singer:
	def __init__(self, clsId):
		self.singer_id=PyScoreDraft.InitSinger(clsId)
	def tune(self, cmd):
		PyScoreDraft.SingerTune(self.perc_id, cmd)

g_instList=PyScoreDraft.ListInstruments();
for i in range(len(g_instList)):
	funcDef="""
def """+g_instList[i]+"""():
	return Instrument("""+str(i)+""")"""
	exec(funcDef)

g_percList=PyScoreDraft.ListPercussions();
for i in range(len(g_percList)):
	funcDef="""
def """+g_percList[i]+"""():
	return Percussion("""+str(i)+""")"""
	exec(funcDef)

g_singerList=PyScoreDraft.ListSingers();
for i in range(len(g_singerList)):
	funcDef="""
def """+g_singerList[i]+"""():
	return Singer("""+str(i)+""")"""
	exec(funcDef)


class Document:
	def __init__ (self):
		self.bufferList=[]
		self.tempo=80
		self.refFreq=264.0

	def playNoteSeq(self, seq, instrument, volume=1.0):
		bufferId=PyScoreDraft.InstrumentPlay(instrument.inst_id, seq, volume, self.tempo, self.refFreq)
		self.bufferList.append(bufferId)

	def playBeatSeq(self, seq, percList, volume=1.0):
		percIdList= [perc.perc_id for perc in percList]
		bufferId= PyScoreDraft.PercussionPlay(percIdList, seq, volume, self.tempo)
		self.bufferList.append(bufferId)

	def sing(self, seq, singer, volume=1.0):
		bufferId = PyScoreDraft.Sing(singer.singer_id, seq, volume, self.tempo, self.refFreq)
		self.bufferList.append(bufferId)

	def trackToWav(self,trackIndex, filename):
		PyScoreDraft.WriteTrackBufferToWav(self.bufferList[trackIndex], filename)

	def mixDown(self,filename):
		bufferId=PyScoreDraft.MixTrackBufferList(self.bufferList)
		PyScoreDraft.WriteTrackBufferToWav(bufferId, filename)
