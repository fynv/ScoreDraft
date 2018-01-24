#!/usr/bin/python3

import ScoreDraft
from ScoreDraftNotes import *

doc=ScoreDraft.Document()

#seq= [ ("- xin", do(6,24), "in nian", do(6,24), "ian hao", do(6,48), "ao ya", so(5,48))]
#seq+= [ ("- xin", mi(6,24), "in nian", mi(6,24), "ian hao", mi(6,48), "ao ya", do(6,48))]

#seq = [ ("- xin", 1, 48, "in nian", 2, 48, "ian hao", 3, 48)]

#seq= [ ("- xin", do(6,48), "in nian", re(6,48), "ian hao", so(5,48))]


seq= [ ("xin", do(6,24), "nian", do(6,24), "hao", do(6,48), "ya", so(5,48))]

WanEr=  ScoreDraft.WanEr_UTAU()
WanEr.tune ("rap_freq 2.0")

def pinyinCVParser(CVLyric):
	vowels= ["a","e","i","o","u","v"]
	min_i=len(CVLyric)
	for c in vowels:
		i=CVLyric.find(c)
		if i>-1 and i<min_i:
			min_i=i
	consonant= CVLyric[0:min_i]
	vowel=CVLyric[min_i:len(CVLyric)]
	if vowel=="i" and (consonant=="zh" or consonant=="ch" or consonant=="sh"):
		vowel="ir"
	if vowel=="i" and (consonant=="z" or consonant=="c" or consonant=="s"):
		vowel="iz"
	if vowel=="u" and (consonant=="j" or consonant=="q" or consonant=="x"):
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
	return (consonant,vowel,CVLyric,"")

ScoreDraft.UtauDraftSetCVParser(WanEr, pinyinCVParser)
WanEr.tune("cv2vcv")

doc.sing(seq, WanEr)
doc.mixDown('vcv.wav')



