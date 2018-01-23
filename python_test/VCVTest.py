#!/usr/bin/python3

import ScoreDraft
from ScoreDraftNotes import *

doc=ScoreDraft.Document()

seq= [ ("- xin", do(6,24), "in nian", do(6,24), "ian hao", do(6,48), "ao ya", so(5,48))]
seq+= [ ("- xin", mi(6,24), "in nian", mi(6,24), "ian hao", mi(6,48), "ao ya", do(6,48))]

#seq = [ ("- xin", 1, 48, "in nian", 2, 48, "ian hao", 3, 48)]

#seq= [ ("- xin", do(6,48), "in nian", re(6,48), "ian hao", so(5,48))]

WanEr=  ScoreDraft.WanEr_UTAU()
WanEr.tune ("rap_freq 2.0")

doc.sing(seq, WanEr)
doc.mixDown('vcv.wav')



