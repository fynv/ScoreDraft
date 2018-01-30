#!/usr/bin/python3

import ScoreDraft
from ScoreDraftRapChinese import *
import CVVCChineseConverter
import TsuroVCVConverter

GePing= ScoreDraft.GePing_UTAU()
Ayaka = ScoreDraft.Ayaka_UTAU()
WanEr = ScoreDraft.WanEr_UTAU()
ScoreDraft.UtauDraftSetLyricConverter(Ayaka, CVVCChineseConverter.CVVCChineseConverter)
ScoreDraft.UtauDraftSetLyricConverter(WanEr, TsuroVCVConverter.TsuroVCVConverter)

singer= GePing
#singer=Ayaka
#singer=WanEr

#SetRapBaseFreq(2.0)

i=0
with open("tts.txt",'r') as f:
	while True:
		line = f.readline()
		if not line:
			break
		line=line.replace("\n","")
		notes=()
		for word in line.split(' '):
			if len(word)==0:
				continue
			word_=word[0:len(word)-1]
			_tone=word[len(word)-1]
			notes+=RapTone(word_,int(_tone),24)

		if len(notes)==0:
			continue

		buf=ScoreDraft.TrackBuffer()

		singer.sing(buf, [notes], 80)
		ScoreDraft.WriteTrackBufferToWav(buf, "tts_out/"+str(i)+".wav")

		i+=1
