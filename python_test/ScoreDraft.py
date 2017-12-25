import os 
import PyScoreDraft

class Instrument:
	def __init__ (self, creator):
		self.inst_id=creator()
	def play(self, seqId, volume, tempo, refFreq):
		return PyScoreDraft.InstrumentPlay(self.inst_id, seqId, volume, tempo, refFreq)
	def tune(self, nob, value):
		PyScoreDraft.InstrumentTune(self.inst_id, nob, value)

class Percussion:
	def __init__(self, creator):
		self.perc_id=creator()
	def tune(self, nob, value):
		PyScoreDraft.PercussionTune(self.inst_id, nob, value)

class PercussionList:
	def __init__(self):
		self.percList_id=PyScoreDraft.InitPercussionList()
	def add(self, perc):
		PyScoreDraft.AddPercussionToList(self.percList_id, perc.perc_id)
	def play(self, seqId, volume, tempo):
		return PyScoreDraft.PercussionPlay(self.percList_id, seqId, volume, tempo)

def PureSin():
	return Instrument(PyScoreDraft.InitPureSin)

def Square():
	return Instrument(PyScoreDraft.InitSquare)

def SawTooth():
	return Instrument(PyScoreDraft.InitSawTooth)

def Triangle():
	return Instrument(PyScoreDraft.InitTriangle)

def NaivePiano():
	return Instrument(PyScoreDraft.InitNaivePiano)

def BottleBlow():
	return Instrument(PyScoreDraft.InitBottleBlow)

def TestPerc():
	return Percussion(PyScoreDraft.InitTestPerc)

class Document:
	def __init__ (self):
		self.bufferListId=PyScoreDraft.InitTrackBufferList()
		self.tempo=80
		self.refFreq=264.0

	def playNoteSeq(self, seq, instrument, volume=1.0):
		seqId= PyScoreDraft.InitNoteSequence()
		for note in seq:
			PyScoreDraft.AddNoteToSequence (seqId, note[0], note[1])
		bufferId=instrument.play(seqId, volume, self.tempo, self.refFreq)
		PyScoreDraft.AddTrackBufferToList(self.bufferListId, bufferId)

	def playBeatSeq(self, seq, percList, volume=1.0):
		seqId= PyScoreDraft.InitBeatSequence()
		for beat in seq:
			PyScoreDraft.AddBeatToSequence(seqId, beat[0], beat[1])
		bufferId=percList.play(seqId,volume,self.tempo)
		PyScoreDraft.AddTrackBufferToList(self.bufferListId, bufferId)

	def mixDown(self,filename):
		bufferId=PyScoreDraft.MixTrackBufferList(self.bufferListId)
		PyScoreDraft.WriteTrackBufferToWav(bufferId, filename)
