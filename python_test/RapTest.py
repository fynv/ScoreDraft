#!/usr/bin/python3

import ScoreDraft
from ScoreDraftNotes import *
from ScoreDraftRapChinese import *
from tang300 import poems

#durations=[ [48,48,24,24,48, 24,24,24,24,48] ]

#durations=[ [24,36,32,32,36, 24,36,32,32,36] ]

'''
durations=[ [36,60,48,36,60, 36,60,48,36,60] ]

poem=poems[0]
divider= poem[0]*2

assert(divider==10)

seq=[]

for i in range(int(len(poem[1])/divider)):
	line=()
	for j in range(poem[0]):
		line += RapTone(poem[1][i*divider+j][0], poem[1][i*divider+j][1], durations[0][j] )
	seq+=[line, BL(48)]

	line=()
	for j in range(poem[0],divider):
		line += RapTone(poem[1][i*divider+j][0], poem[1][i*divider+j][1], durations[0][j] )
	seq+=[line,BL(48)]
'''


durations=[ [36,60,36,60,48,48,48, 36,60,36,60,48,48,48] ]

poem=poems[50]
divider= poem[0]*2

assert(divider==14)

seq=[]

for i in range(int(len(poem[1])/divider)):
	line=()
	for j in range(poem[0]):
		line+=RapTone(poem[1][i*divider+j][0],poem[1][i*divider+j][1], durations[0][j]) 
	seq+=[line, BL(48)]
	
	line=()
	for j in range(poem[0],divider):
		line+=RapTone(poem[1][i*divider+j][0],poem[1][i*divider+j][1], durations[0][j])
	seq+=[line,BL(48)]


buf=ScoreDraft.TrackBuffer()

GePing= ScoreDraft.GePing_UTAU()
GePing.sing(buf, seq, 120)


#ScoreDraft.QPlayTrackBuffer(buf)

ScoreDraft.WriteTrackBufferToWav(buf, "rap.wav")
