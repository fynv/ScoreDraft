from . import PyScoreDraft

def ObjectToId(obj):
	'''
	Utility only used intenally. User don't use it.
	'''
	if type(obj) is list:
		return [ObjectToId(sub_obj) for sub_obj in obj]
	else:
		return obj.id


class Percussion:
	'''
	Structure to define an percussion object 
	An percussion object can be used to play beat-sequences to a track-buffer object
	'''
	def __del__ (self):
		PyScoreDraft.DelPercussion(self.id)	

	def tune(self, cmd):
		'''
		Sending a tuning command to an percussion.
		cmd -- A string to be parsed by the percussion.
		       Different percussions can support different sets of tuning commands.
		       A command common to all percussions is "volume", the value range is [0.0, 1.0], example: 
		       perc.tune("volume 0.5")
		       Another command common to all percussions is "pan",  the value range is [-1.0, 1.0] example: 
		       perc.tune("pan -0.5")
		'''
		PyScoreDraft.PercussionTune(self.id, cmd)

	@staticmethod
	def play(percList, buf, seq, tempo=80):
		'''
		Typically, multiple percussions are used when playing a beat sequence, so here "play()" is defined as static.
		buf -- An instance of TrackBuffer, the result of play will be appended to the buffer.
		seq -- A list of beats [beat1, beat2, ... ]. Each of the beats is a tuple (index, duration)
		       index: a integer. When it is non-negative, it references a percussion in a percussion list, which is used to make the noise.
		       duration: the same as the "duration" of a note.

		       Negative "index" values are used in the same way as negative "freq" values of notes to define silent time periods and backspaces.

		       Tuning commands can also be mixed in the list to tune the percussions on the fly, example:
		         [(0,48), (1,48), (0,"volume 2.0"), (1,"volume 3.0"), (0,48), (1,48)... ]

		       In the beat sequence case, an index need to be provided to choose which persecussion the command is sent to.

		tempo -- an integer defining the tempo of play in beats/minute.		
		'''
		PyScoreDraft.PercussionPlay(buf.id, ObjectToId(percList), seq, tempo)

	def setBeatVolume(self,volume):
		PyScoreDraft.PercussionSetBeatVolume(self.id, volume)

	def setBeatPan(self,pan):
		PyScoreDraft.PercussionSetBeatPan(self.id, pan)
		