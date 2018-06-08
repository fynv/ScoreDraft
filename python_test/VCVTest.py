#!/usr/bin/python3

import ScoreDraft
from ScoreDraft.Notes import *

doc=ScoreDraft.Document()

#seq= [ ("- xin", do(6,24), "in nian", do(6,24), "ian hao", do(6,48), "ao ya", so(5,48))]
#seq+= [ ("- xin", mi(6,24), "in nian", mi(6,24), "ian hao", mi(6,48), "ao ya", do(6,48))]

#seq= [ ("xin", do(6,24), "nian", do(6,24), "hao", do(6,48), "ya", so(5,48))]
#seq+= [ ("xin", mi(6,24), "nian", mi(6,24), "hao", mi(6,48), "ya", do(6,48))]


line= ("zheng", re(5,24), "yue", do(5,48), "li", re(5,24), "cai", mi(5,36), so(5,12), "hua", mi(5,24), la(4,24))
line+=("wu", re(5,24), "you", do(5,48), "hua", re(5,24), "cai", mi(5,24), do(5,12), re(5,12), mi(5,24), BL(24))
seq = [line]

line= ("er", mi(5,12), so(5,12), "yue", la(5,48), "jian", do(6,24), "cai", la(5, 48), "hua", so(5,24), mi(5,24))
line+= ("hua", la(5,24), "you", mi(5,48), "zheng", re(5,12), mi(5,12), "kai", do(5,24), BL(24))
seq += [line]

line= ("er", do(5,12), re(5,12), "yue", mi(5,12), so(5,12), "jian", mi(5,48), "cai", re(5,24), "hua", do(5,24))
line+=("hua", la(4,24), "you", do(5,48), "zheng", la(4,12), do(5,12), "kai", la(4,96), BL(96))
seq += [line]

doc.setReferenceFrequency(440.0)

# link to voicebank: https://pan.baidu.com/s/1i4VkcDZ
WanEr=  ScoreDraft.WanEr_UTAU()
WanEr.setLyricConverter(ScoreDraft.TsuroVCVConverter)

doc.sing(seq, WanEr)
doc.mixDown('vcv.wav')



