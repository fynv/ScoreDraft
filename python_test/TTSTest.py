#!/usr/bin/python3

import ScoreDraft
from ScoreDraftRapChinese import *

def read_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines

lines = read_txt("tts.txt")	

for i in range(len(lines)):
	line=lines[i]
	line=line.replace("\n","")
	notes=()
	for word in line.split(' '):
		word_=word[0:len(word)-1]
		_tone=word[len(word)-1]
		notes+=RapTone(word_,int(_tone),24)

	buf=ScoreDraft.TrackBuffer()

	GePing= ScoreDraft.GePing_UTAU()
	GePing.sing(buf, [notes], 80)
	ScoreDraft.WriteTrackBufferToWav(buf, "tts_out/"+str(i)+".wav")
