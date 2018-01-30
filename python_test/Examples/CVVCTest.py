#!/usr/bin/python3
import sys
sys.path+=['../']

import ScoreDraft
from ScoreDraftNotes import *

doc=ScoreDraft.Document()
doc.setTempo(100)

seq = [ ("- gU", mi(5,18), "U d", mi(5,6), "baI", so(5,18), "aI dZ", so(5,6), "dZoU", so(5,12), la(5,36) )]
seq += [ ("- mi", mi(5,18), "i g", mi(5,6), "gA", so(5,12), "A t", so(5,4), "t@", so(5,6), "@ g", so(5,2), "goU", so(5,12), la(5,36))]
seq += [ ("- mi", so(5,36), "i oU", mi(5,9), "oU m", mi(5,3), "maI", so(5,24), "aI oU", re(5,72)), BL(48)]

seq += [ ("- gU", 18, 0.75, 1.0, "U d", 6, 1.0, 1.0, "baI", 18, 0.75, 1.0, "aI dZ", 6, 1.0, 1.0,  "dZoU", 36, 1.0, 0.5 ), BL(12)]
seq += [ ("- mi", 18, 0.75, 1.0, "i g", 6, 1.0, 1.0, "gA", 12, 0.75, 1.0, "A t", 4, 1.0, 1.0, "t@", 6, 1.0, 1.0, "@ g", 2, 1.0, 1.0, "goU", 36, 1.0, 0.5), BL(12)]

Teto = ScoreDraft.TetoEng_UTAU()

doc.sing(seq, Teto)
doc.mixDown('cvvc.wav')

