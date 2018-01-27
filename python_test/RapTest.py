#!/usr/bin/python3

import ScoreDraft
from ScoreDraftNotes import *
from tang300 import poems
import TsuroVCVConverter
import CVVCChineseConverter

#seq = [ ("kan", 4, 48), ("jian", 4, 24), ("de", 4,24), ("kan", 4, 24), ("bu", 2, 24), ("jian", 4, 24), ("de", 4,24) ]
#seq += [ ("shun", 4, 48), ("jian", 1,24), ("de", 4,24), ("yong",3,48), ("heng",2,24), ("de", 4,24)]

#seq = [ ("yi", 2, 48), ("wang",4, 48), ("er",4,24), ("san", 1,24), ("li", 3, 48)]
#seq += [ ("yan", 1, 24), ("cun",1, 24), ("si",4,24), ("wu", 3,24), ("jia", 1, 48), BL(48)]

#seq += [ ("ting", 2, 48), ("tai",2, 48), ("liu",4,24), ("qi", 1,24), ("zuo", 4, 48)]
#seq += [ ("ba", 1, 24), ("jiu",3, 24), ("shi",2,24), ("zhi", 1,24), ("hua", 1, 48), BL(48)]


#durations=[ [48,48,24,24,48, 24,24,24,24,48] ]

#durations=[ [24,36,32,32,36, 24,36,32,32,36] ]

'''
durations=[ [36,60,48,36,60, 36,60,48,36,60] ]

poem=poems[42]
divider= poem[0]*2

assert(divider==10)

seq=[]

for i in range(int(len(poem[1])/divider)):
	for j in range(poem[0]):
		seq+=[(poem[1][i*divider+j][0],poem[1][i*divider+j][1], durations[0][j])]
	seq+=[BL(48)]
	for j in range(poem[0],divider):
		seq+=[(poem[1][i*divider+j][0],poem[1][i*divider+j][1], durations[0][j])]
	seq+=[BL(48)]


'''

durations=[ [36,60,36,60,48,48,48, 36,60,36,60,48,48,48] ]

poem=poems[50]
divider= poem[0]*2

assert(divider==14)

seq=[]

for i in range(int(len(poem[1])/divider)):
	line=()
	for j in range(poem[0]):
		line+=(poem[1][i*divider+j][0],poem[1][i*divider+j][1], durations[0][j]) 
	seq+=[line, BL(48) ]
	
	line=()
	for j in range(poem[0],divider):
		line+=(poem[1][i*divider+j][0],poem[1][i*divider+j][1], durations[0][j])
	seq+=[line,BL(48)]

buf=ScoreDraft.TrackBuffer()

'''
GePing= ScoreDraft.GePing()
GePing.sing(buf, seq, 120)
'''

GePing= ScoreDraft.GePing_UTAU()

def HeNanToneGenerator(ToneInfoForEachSyllable):
	ret=[]
	for toneInfo in ToneInfoForEachSyllable:
		if toneInfo[0]<=1:
			ret+=[(0.7,1.0)]
		elif toneInfo[0]==2:
			ret+=[(1.0,0.5)]
		elif toneInfo[0]==3:
			ret+=[(1.0,1.0)]
		elif toneInfo[0]==4:
			ret+=[(1.0,0.5)]
	return ret

ScoreDraft.UtauDraftSetRapToneGenerator(GePing, HeNanToneGenerator)

GePing.sing(buf, seq, 120)

'''
WanEr=  ScoreDraft.WanEr_UTAU()
WanEr.tune ("rap_freq 1.5")
ScoreDraft.UtauDraftSetLyricConverter(WanEr, TsuroVCVConverter.TsuroVCVConverter)
WanEr.sing(buf, seq, 120)
'''

'''
Ayaka = ScoreDraft.Ayaka_UTAU()
ScoreDraft.UtauDraftSetLyricConverter(Ayaka, CVVCChineseConverter.CVVCChineseConverter)
Ayaka.tune ("rap_freq 1.5")
Ayaka.sing(buf, seq, 120)
'''

#ScoreDraft.QPlayTrackBuffer(buf)

ScoreDraft.WriteTrackBufferToWav(buf, "rap.wav")
