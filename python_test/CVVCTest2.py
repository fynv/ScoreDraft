#!/usr/bin/python3

import ScoreDraft
from ScoreDraftNotes import *

def getCV(CVLyric):
	vowels= ["a","e","i","o","u","v"]
	min_i=len(CVLyric)
	for c in vowels:
		i=CVLyric.find(c)
		if i>-1 and i<min_i:
			min_i=i

	consonant= CVLyric[0:min_i]
	vowel=CVLyric[min_i:len(CVLyric)]

	if CVLyric=="zhi" or CVLyric=="chi" or CVLyric=="shi" or CVLyric=="ri":
		vowel="ir"
	if CVLyric=="zi" or CVLyric=="ci" or CVLyric=="si":
		vowel="i0"
	if CVLyric=="ju" or CVLyric=="qu" or CVLyric=="xu" or CVLyric=="yu":
		vowel="v"

	if vowel=="ia":
		vowel="a"
	if vowel=="iao":
		vowel="ao"
	if vowel=="ian":
		vowel="an"
	if vowel=="iang":
		vowel="ang"
	if vowel=="iong":
		vowel="ong"
	if vowel=="iu":
		vowel="ou"
	if vowel=="ua":
		vowel="a"
	if vowel=="uai":
		vowel="ai"
	if vowel=="uan":
		vowel="an"
	if vowel=="uai":
		vowel="ai"
	if vowel=="ui":
		vowel="ei"
	if vowel=="uang":
		vowel="ang"
	if vowel=="un":
		vowel="en"
	if vowel=="uo":
		vowel="o"
	if (vowel=="ue"):
		vowel="e0"

	if consonant=="j":
		if vowel[0]=="u":
			consonant="jw"
		else:
			consonant="jy"
	if consonant=="y":
		if vowel[0]=="u":
			consonant="v"
		else:
			consonant="y"

	return (consonant,vowel)


def CVVCChineseConverter(LyricForEachSyllable):	
	CV = [getCV(lyric) for lyric in  LyricForEachSyllable]
	ret=[]
	for i in range(len(LyricForEachSyllable)):
		lyric=LyricForEachSyllable[i]
		if i==0:
			lyric='- '+lyric
		elif CV[i][0]=="":
			lyric=CV[i-1][1]+" "+lyric
		if i<len(LyricForEachSyllable)-1 and CV[i+1]!="":
			ret+=[(lyric,0.7, CV[i][1]+" "+CV[i+1][0], 0.3)]
		else:
			ret+=[(lyric,1.0)]
	return ret

#seq=[("ni",2, 48, "hao", 3, 48, "ya", 1,48)]

line= ("zheng", re(5,24), "yue", do(5,48), "li", re(5,24), "cai", mi(5,36), so(5,12), "hua", mi(5,24), la(4,24))
line+=("wu", re(5,24), "you", do(5,48), "hua", re(5,24), "cai", mi(5,24), do(5,12), re(5,12), mi(5,24), BL(24))
seq = [line]

line= ("er", mi(5,12), so(5,12), "yue", la(5,48), "jian", do(6,24), "cai", la(5, 48), "hua", so(5,24), mi(5,24))
line+= ("hua", la(5,24), "you", mi(5,48), "zheng", re(5,12), mi(5,12), "kai", do(5,24), BL(24))
seq += [line]

line= ("er", do(5,12), re(5,12), "yue", mi(5,12), so(5,12), "jian", mi(5,48), "cai", re(5,24), "hua", do(5,24))
line+=("hua", la(4,24), "you", do(5,48), "zheng", la(4,12), do(5,12), "kai", la(4,96), BL(96))
seq += [line]


Ayaka = ScoreDraft.Ayaka_UTAU()
ScoreDraft.UtauDraftSetLyricConverter(Ayaka, CVVCChineseConverter)
Ayaka.tune ("rap_freq 1.5")

doc=ScoreDraft.Document()
doc.setReferenceFreqeuncy(440.0)
doc.sing(seq, Ayaka)
doc.mixDown('cvvc2.wav')

