#!/usr/bin/python3
import sys
sys.path+=['../']

import ScoreDraft
from ScoreDraftNotes import *
import CVVCChineseConverter


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
ScoreDraft.UtauDraftSetLyricConverter(Ayaka, CVVCChineseConverter.CVVCChineseConverter)
Ayaka.tune ("rap_freq 1.5")

doc=ScoreDraft.Document()
doc.setReferenceFreqeuncy(440.0)
doc.sing(seq, Ayaka)
doc.mixDown('cvvc2.wav')

