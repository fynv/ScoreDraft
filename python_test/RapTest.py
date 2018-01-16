#!/usr/bin/python3

import ScoreDraft
from ScoreDraftNotes import *
from tang300 import poems

doc=ScoreDraft.Document()
doc.tempo=120

GePing= ScoreDraft.GePing()

#seq = [ ("kan", 4, 48), ("jian", 4, 24), ("de", 4,24), ("kan", 4, 24), ("bu", 2, 24), ("jian", 4, 24), ("de", 4,24) ]
#seq += [ ("shun", 4, 48), ("jian", 1,24), ("de", 4,24), ("yong",3,48), ("heng",2,24), ("de", 4,24)]

#seq = [ ("yi", 2, 48), ("wang",4, 48), ("er",4,24), ("san", 1,24), ("li", 3, 48)]
#seq += [ ("yan", 1, 24), ("cun",1, 24), ("si",4,24), ("wu", 3,24), ("jia", 1, 48), BL(48)]

#seq += [ ("ting", 2, 48), ("tai",2, 48), ("liu",4,24), ("qi", 1,24), ("zuo", 4, 48)]
#seq += [ ("ba", 1, 24), ("jiu",3, 24), ("shi",2,24), ("zhi", 1,24), ("hua", 1, 48), BL(48)]

durations=[ [48,48,24,24,48, 24,24,24,24,48] ]

poem=poems[17]
divider= poem[0]*2

assert(divider==10)

seq=[]

for i in range(int(len(poem[1])/divider)):
	for j in range(10):
		seq+=[(poem[1][i*10+j][0],poem[1][i*divider+j][1], durations[0][j])]
	seq+=[BL(48)]

doc.sing(seq, ScoreDraft.GePing())


doc.mixDown('RapTest.wav')
