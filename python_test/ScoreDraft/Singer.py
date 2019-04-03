from . import PyScoreDraft

class Singer:
	'''
	Structure to define an singer object 
	An singer object can be used to sing singing-sequences to a track-buffer object
	'''

	def tune(self, cmd):
		'''
		Sending a tuning command to an singer.
		cmd -- A string to be parsed by the singer.
		       Different singers can support different sets of tuning commands.
		       A command common to all singers is "volume",  the value range is [0.0, 1.0], example: 
		       singer.tune("volume 0.5")
		       Another command common to all singers is "pan",  the value range is [-1.0, 1.0] example: 
		       singer.tune("pan -0.5")
		       Another command common to all singers is "default_lyric", example: 
		       singer.tune("default_lyric la")
		       This will make the singer to sing "la" when an empty lyric "" is recieved
		'''
		PyScoreDraft.SingerTune(self.m_cptr, cmd)

	def sing(self, buf, seq, tempo=80, refFreq=261.626):
		'''
		buf -- An instance of TrackBuffer, the result of play will be appended to the buffer.
		seq -- A list of singing-segments [seg1, seg2, ... ]. Each of the seg is a tuple 
					(lyric1, note1, note2, lyric2, note3, ...)
		       lyrics: they are strings, telling the singer what lyric to sing
		       notes: they are the same kind of notes used in instrument play. 

		       In many cases, there is only 1 note following a lyric.
		       When there are more than 1 notes follwoing a lyric, all these notes will split the duration of that lyric.
		       All lyrics and notes in the same tuple are intended to be sung continuously.
			   However, when there are silence notes/backapsaces, the singing-segment will be broken into multiple 
			   segments to sing.
		      
		       Raw notes can be mixed with singing-segments in the list. They will be sung using the default lyric.
		       Vice-versa, if you pass a singing-sequence to an instrument, the notes contained in the sequence will
		       get played, and lyrics ignored. We aim to provide maximum compatibility between the two.
		       Tuning commands can also be mixed in the list to tune the singer on the fly, example:
		        [("ha",do(5,48),re(5,48)), ("la",mi(5,48)), "volume 2.0", "default_lyric ba", fa(5,48)... ]

		       seq can also contain rapping segments like:
		       (lyric1, duration1, freq_start1, freq_end1, lyric2, duration2, freq_start2, freq_end2...)
		       freq_starts and freq_ends are used to define the tones syllables.
		       They are relative frequencies. The physical frequencies will be freq_starts*refFreq and freq_ends*refFreq

		tempo -- an integer defining the tempo of singing in beats/minute.
		refFreq  --  a floating point defining the reference-frequency in Hz.
		'''
		PyScoreDraft.Sing(buf.m_cptr, self.m_cptr, seq, tempo, refFreq)

	def setDefaultLyric(self, lyric):
		PyScoreDraft.SingerSetDefaultLyric(self.m_cptr, lyric)

	def setNoteVolume(self,volume):
		PyScoreDraft.SingerSetNoteVolume(self.m_cptr, volume)

	def setNotePan(self,pan):
		PyScoreDraft.SingerSetNotePan(self.m_cptr, pan)
		