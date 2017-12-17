import os 
import PyScoreDraft


PureSin=PyScoreDraft.PureSinPlay
Sawtooth=PyScoreDraft.SawtoothPlay
NaivePiano=PyScoreDraft.NaivePianoPlay
BottleBlow=PyScoreDraft.BottleBlowPlay

class Document:
	def __init__ (self):
		self.bufferListId=PyScoreDraft.InitTrackBufferList()
		self.tempo=80
		self.refFreq=264.0

	def playSeq(self, parsedSeq, volume=1.0, instrument=Sawtooth):
		seqId= PyScoreDraft.InitNoteSequence()
		for note in parsedSeq:
			PyScoreDraft.AddNoteToSequence (seqId, note[0], note[1])
		bufferId=instrument(seqId, volume, self.tempo, self.refFreq)
		PyScoreDraft.AddTrackBufferToList(self.bufferListId, bufferId)

	def mixDown(self,filename):
		bufferId=PyScoreDraft.MixTrackBufferList(self.bufferListId)
		PyScoreDraft.WriteTrackBufferToWav(bufferId, filename)
