import os 
import PyScoreDraft

class Freqs:
	def __init__ (self):
		self.freqs=[ 1.0, 2.0**(1.0/12.0) , 2.0**(2.0/12.0), 2.0**(3.0/12.0), 2.0**(4.0/12.0), 2.0**(5.0/12.0), 2.0**(6.0/12.0), 2.0**(7.0/12.0), 2.0**(8.0/12.0), 2.0**(9.0/12.0), 2.0**(10.0/12.0), 2.0**(11.0/12.0), -1.0]
	def set_do(self, freq):
		self.freqs[0]=freq
	def set_re(self, freq):
		self.freqs[2]=freq
	def set_mi(self, freq):
		self.freqs[4]=freq
	def set_fa(self, freq):
		self.freqs[5]=freq
	def set_so(self, freq):
		self.freqs[7]=freq
	def set_la(self, freq):
		self.freqs[9]=freq
	def set_ti(self, freq):
		self.freqs[11]=freq

def parseSeq(original, freqTab=Freqs()):
	return [ (freqTab.freqs[elem[0]]*(2.0**(elem[1]-5.0)),elem[2]) for elem in original]

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

	def parseAndPlaySeq(self, seq, volume=1.0, instrument=Sawtooth, freqTab=Freqs()):
		seqId= PyScoreDraft.InitNoteSequence()
		for note in seq:
			PyScoreDraft.AddNoteToSequence (seqId, freqTab.freqs[note[0]]*(2.0**(note[1]-5.0)), note[2])
		bufferId=instrument(seqId, volume, self.tempo, self.refFreq)
		PyScoreDraft.AddTrackBufferToList(self.bufferListId, bufferId)

	def mixDown(self,filename):
		bufferId=PyScoreDraft.MixTrackBufferList(self.bufferListId)
		PyScoreDraft.WriteTrackBufferToWav(bufferId, filename)



if __name__=="__main__":
	seqId= PyScoreDraft.InitNoteSequence()
	PyScoreDraft.AddNoteToSequence (seqId, 1.0, 48)
	PyScoreDraft.AddNoteToSequence (seqId, 9.0/8.0, 48)
	PyScoreDraft.AddNoteToSequence (seqId, 5.0/4.0, 48)
	PyScoreDraft.AddNoteToSequence (seqId, 4.0/3.0, 48)
	PyScoreDraft.AddNoteToSequence (seqId, 3.0/2.0, 48)
	PyScoreDraft.AddNoteToSequence (seqId, 5.0/3.0, 48)
	PyScoreDraft.AddNoteToSequence (seqId, 15.0/8.0, 48)
	PyScoreDraft.AddNoteToSequence (seqId, 2.0, 48)

	bufferId=Sawtooth(seqId, 1.0, 80, 264.0)
	PyScoreDraft.WriteTrackBufferToWav(bufferId, 'test.wav')

	os.system("pause") 

