#!/usr/bin/python3

import sys
sys.path+=['../']

import ScoreDraft
from ScoreDraftNotes import *
import XiaYYConverter

'''
line= ("zheng", re(5,24), "yue", do(5,48), "li", re(5,24), "cai", mi(5,36), so(5,12), "hua", mi(5,24), la(4,24))
line+=("wu", re(5,24), "you", do(5,48), "hua", re(5,24), "cai", mi(5,24), do(5,12), re(5,12), mi(5,24), BL(24))
seq = [line]

line= ("er", mi(5,12), so(5,12), "yue", la(5,48), "jian", do(6,24), "cai", la(5, 48), "hua", so(5,24), mi(5,24))
line+= ("hua", la(5,24), "you", mi(5,48), "zheng", re(5,12), mi(5,12), "kai", do(5,24), BL(24))
seq += [line]

line= ("er", do(5,12), re(5,12), "yue", mi(5,12), so(5,12), "jian", mi(5,48), "cai", re(5,24), "hua", do(5,24))
line+=("hua", la(4,24), "you", do(5,48), "zheng", la(4,12), do(5,12), "kai", la(4,96), BL(96))
seq += [line]
'''

seq=[ ("ming", so(5,16), "tian", so(5,16), "ni", so(5,16), "shi", so(5,16), "fou", mi(5,16), "hui", fa(5,16), "xiang", so(5,48), "qi", ti(5,32)), BL(16)]
seq+=[ ("zuo", la(5,16), "tian", la(5,16), "ni", la(5,16), "xie", la(5,16), "de", fa(5,16), "ri", la(5,16), "ji", so(5,72)), BL(24)]
XiaYY = ScoreDraft.XiaYY_UTAU()
ScoreDraft.UtauDraftSetLyricConverter(XiaYY, XiaYYConverter.XiaYYConverter)

doc=ScoreDraft.Document()
doc.setTempo(60)
#doc.setReferenceFrequency(440.0)
doc.sing(seq, XiaYY)
doc.mixDown('XiaYY.wav')

