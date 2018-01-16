import ScoreDraft

class Arrangement:
	def __init__ (self):
		self.tracks=[]
		self.actions=[]

	def newInstrumentTrack(self, name, instrument):
		self.tracks.append((name, 0, instrument, 1.0))
		return len(self.tracks)-1

	def newPercussionTrack(self, name, percList):
		self.tracks.append((name, 1, percList, 1.0))
		return len(self.tracks)-1

	def newSingerTrack(self, name, singer):
		self.tracks.append((name, 2, singer, 1.0))
		return len(self.tracks)-1

	def setTrackVolume(trackIndex, volume):
		self.tracks[trackIndex][3]=volume

	def addAction(self, action):
		self.actions.append(action)
		return len(self.actions)-1

	def runActions(self, doc):
		trackMap = [ doc.newTrack(track[0], track[1], track[2]) for track in self.tracks]
		for action in self.actions:
			action.run(doc, trackMap)
		for i in range(len(self.tracks)):
			doc.setTrackBufferVolume(trackMap[i], self.tracks[i][3])


# Action definitions
class SetTempo:
	def __init__ (self, tempo):
		self.tempo=tempo
	def run(self, doc, trackMap):
		doc.setTempo(self.tempo)

class SetReferenceFrequency:
	def __init__ (self, refFreq):
		self.refFreq=refFreq
	def run(self, doc, trackMap):
		doc.setReferenceFreqeuncy(self.refFreq)

class Perform:
	def __init__ (self, trackIndex, seg):
		self.trackIndex=trackIndex
		self.seg=seg
	def run(self, doc, trackMap):
		doc.trackPerform(trackMap[self.trackIndex], self.seg)


class Document:
	def __init__ (self):
		self.tracks=[]
		self.tempo=80
		self.refFreq=264.0

	def newTrack(self, name, typeId, performer):
		buf=ScoreDraft.TrackBuffer()
		self.tracks.append((name, typeId, performer, buf))
		return len(self.tracks)-1

	def newInstrumentTrack(self, name, instrument):
		return self.newTrack(name, 0, insrument)

	def newPercussionTrack(self, name, percList):
		return self.newTrack(name, 1, percList)

	def newSingerTrack(self, name, singer):
		return self.newTrack(name, 2, singer)

	def setTempo(self,tempo):
		self.tempo=tempo

	def setReferenceFreqeuncy(self,refFreq):
		self.refFreq=refFreq

	def trackPerform(self, trackIndex, seq):
		track=self.tracks[trackIndex]
		if track[1]==0: #instrument play
			track[2].play(track[3], seq, self.tempo, self.refFreq)
		elif track[1]==1: #percussion play
			ScoreDraft.Percussion.play(track[2], track[3], seq, self.tempo)
		elif track[1]==2: #sing
			track[2].sing( track[3], seq, self.tempo, self.refFreq)

	def setTrackBufferVolume(self, trackIndex, volume):
		self.tracks[trackIndex][3].setVolume(volume)

	def mix(self, targetBuf):		
		bufferList=[track[3] for track in self.tracks]
		ScoreDraft.MixTrackBufferList(targetBuf,bufferList)

	def mixDown(self,filename):
		targetBuf=ScoreDraft.TrackBuffer()
		self.mix(targetBuf)
		ScoreDraft.WriteTrackBufferToWav(targetBuf, filename)


	




