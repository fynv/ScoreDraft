from . import PyScoreDraft

class Instrument:
	'''
	Structure to define an instrument object 
	An instrument object can be used to play note-sequences to a track-buffer object
	'''
	def __del__ (self):
		PyScoreDraft.DelInstrument(self.id)

	def tune(self, cmd):
		'''
		Sending a tuning command to an instrument.
		cmd -- A string to be parsed by the instrument.
		       Different instrument can support different sets of tuning commands.
		       A command common to all instruments is "volume", the value range is [0.0, 1.0] example: 
		       inst.tune("volume 0.5")
		       Another command common to all instruments is "pan",  the value range is [-1.0, 1.0] example: 
		       inst.tune("pan -0.5")
		'''
		PyScoreDraft.InstrumentTune(self.id, cmd)

	def play(self, buf, seq, tempo=80, refFreq=261.626):
		'''
		buf -- An instance of TrackBuffer, the result of play will be appended to the buffer.
		seq -- A list of notes [note1, note2, ... ]. Each of the notes is a tuple (freq, duration)
		       freq: a floating point frequency multiplier relative to a reference frequency "refFreq".
		             The physical frequency of the generated note is freq*refFreq in Hz.
		       duration: an integer that defines the duration of the note. 1 beat = 48, 1/2 beat =24 etc.

		       When "freq" is negative and "duration" is positive, a silent time period is defined, which 
		       occupies the "duration" but does not make any sound.
		       When "freq" is negative and "duration" is also negative, a "backspace" is generated. The 
		       cursor of note-sequence play will be brought backwards by "duration". In this way, the new 
		       notes can be overlapped with old ones, so that harmonies and chords can be generated.

		       Utility functions are provided in ScoreDraftNotes.py to simplify the computation of "freq".
		       For example "do(5,48)" will return a tuple with "freq" set to 1.0, which is "do" at octave "5".

		       Tuning commands can also be mixed in the list to tune the instrument on the fly, example:
		        [do(5,48),re(5,48), "volume 2.0", mi(5,48)... ]
		tempo -- an integer defining the tempo of play in beats/minute.
		refFreq  --  a floating point defining the reference-frequency in Hz.

		'''
		PyScoreDraft.InstrumentPlay(buf.id, self.id, seq, tempo, refFreq)

	def setNoteVolume(self,volume):
		PyScoreDraft.InstrumentSetNoteVolume(self.id, volume)

	def setNotePan(self,pan):
		PyScoreDraft.InstrumentSetNotePan(self.id, pan)
		
