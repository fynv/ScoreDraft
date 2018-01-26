#!/usr/bin/python3

import ScoreDraft
from ScoreDraftNotes import *

doc=ScoreDraft.Document()
doc.setTempo(100)

seq = [ ("- gU", mi(5,18), "U d", mi(5,6), "baI", so(5,18), "aI dZ", so(5,6), "dZoU", so(5,12), la(5,36) )]
seq += [ ("- mi", mi(5,18), "i g", mi(5,6), "gA", so(5,12), "A t", so(5,4), "t@", so(5,6), "@ g", so(5,2), "goU", so(5,12), la(5,36))]
seq += [ ("- mi", so(5,36), "i oU", mi(5,9), "oU m", mi(5,3), "maI", so(5,24), "aI oU", re(5,72)), BL(48)]

seq += [ ("- gU", 1, 18, "U d", 1,6, "baI", 1, 18, "aI dZ", 1,6,  "dZoU", 4, 36 ), BL(12)]
seq += [ ("- mi", 1, 18, "i g", 1,6, "gA", 1, 12,  "A t", 1, 4, "t@", 4, 6, "@ g",1 ,2, "goU",4, 36), BL(12)]


Teto = ScoreDraft.TetoEng_UTAU()
Teto.tune ("rap_freq 1.0")

doc.sing(seq, Teto)
doc.mixDown('cvvc.wav')

