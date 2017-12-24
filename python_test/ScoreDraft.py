import os 
import PyScoreDraft

class Instrument:
	def __init__ (self, creator):
		self.inst_id=creator()
	def play(self, seqId, volume, tempo, refFreq):
		return PyScoreDraft.InstrumentPlay(self.inst_id, seqId, volume, tempo, refFreq)
	def tune(self, nob, value):
		PyScoreDraft.InstrumentTune(self.inst_id, nob, value)

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

class Document:
	def __init__ (self):
		self.bufferListId=PyScoreDraft.InitTrackBufferList()
		self.tempo=80
		self.refFreq=264.0

	def playSeq(self, parsedSeq, instrument, volume=1.0):
		seqId= PyScoreDraft.InitNoteSequence()
		for note in parsedSeq:
			PyScoreDraft.AddNoteToSequence (seqId, note[0], note[1])
		bufferId=instrument.play(seqId, volume, self.tempo, self.refFreq)
		PyScoreDraft.AddTrackBufferToList(self.bufferListId, bufferId)

	def mixDown(self,filename):
		bufferId=PyScoreDraft.MixTrackBufferList(self.bufferListId)
		PyScoreDraft.WriteTrackBufferToWav(bufferId, filename)
