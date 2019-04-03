from . import PyMIDIWriter

def WriteNoteSequencesToMidi(seqList, tempo, refFreq, fileName):
	'''
	Write a list of note sequences to a MIDI file.
	seqList -- a list of note sequences.
	tempo -- an integer indicating tempo in beats/minute.
	refFreq -- a float indicating reference frequency in Hz.
	fileName -- a string.
	'''
	PyMIDIWriter.WriteToMidi(seqList, tempo, refFreq, fileName)
