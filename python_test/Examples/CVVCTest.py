#!/usr/bin/python3

import sys
sys.path+=['../']

import ScoreDraft
from ScoreDraftNotes import *
import TTEnglishConverter

doc=ScoreDraft.Document()
doc.setTempo(100)

'''
seq = [ ("- gU", mi(5,18), "U d", mi(5,6), "baI", so(5,18), "aI dZ", so(5,6), "dZoU", so(5,12), la(5,36) )]
seq += [ ("- mi", mi(5,18), "i g", mi(5,6), "gA", so(5,12), "A t", so(5,4), "t@", so(5,6), "@ g", so(5,2), "goU", so(5,12), la(5,36))]
seq += [ ("- mi", so(5,36), "i oU", mi(5,9), "oU m", mi(5,3), "maI", so(5,24), "aI oU", re(5,72)), BL(48)]

seq += [ ("- gU", 18, 0.75, 1.0, "U d", 6, 1.0, 1.0, "baI", 18, 0.75, 1.0, "aI dZ", 6, 1.0, 1.0,  "dZoU", 36, 1.0, 0.5 ), BL(12)]
seq += [ ("- mi", 18, 0.75, 1.0, "i g", 6, 1.0, 1.0, "gA", 12, 0.75, 1.0, "A t", 4, 1.0, 1.0, "t@", 6, 1.0, 1.0, "@ g", 2, 1.0, 1.0, "goU", 36, 1.0, 0.5), BL(12)]
'''

# seq = [ ('- paI', 20, 2.0, 2.0, 'aI T', 4, 2.0, 2.0, 'TA', 20, 1.5, 1.0, 'A n-', 4, 1.0, 1.0)]
#seq = [ ('- sk', 4, 1.0, 1.0, 'kO', 20, 1.0, 0.9, 'O r', 12, 0.9, 0.7, 'dr', 4, 1.0, 1.0, 'r{', 20, 1.0, 0.7, '{ f', 4, 0.7, 0.7, 'f t-', 4, 0.7, 0.6)]



seq = [ ("gUd", mi(5,24), "baI", so(5,24), "dZoU", so(5,12), la(5,36) )]
seq += [ ("mi", mi(5,24), "gAt", so(5,16), "t@", so(5,8), "goU", so(5,12), la(5,36))]
seq += [ ("mi", so(5,36), "oU", mi(5,12), "maI", so(5,24), "oU", re(5,72)), BL(48)]


#seq = [ ('skOr', 36, 1.0, 0.7, 'dr{ft', 36, 1.0, 0.6)]

#seq = [ ('paI', 24, 2.0, 2.0, 'TAn', 24, 1.5, 1.0)]


Teto = ScoreDraft.TetoEng_UTAU()
ScoreDraft.UtauDraftSetLyricConverter(Teto, TTEnglishConverter.TTEnglishConverter)

doc.sing(seq, Teto)
doc.mixDown('cvvc.wav')

