#!/usr/bin/python3

import sys
sys.path+=['../']

import ScoreDraft
from ScoreDraftNotes import *
import VCCVEnglishConverter

doc=ScoreDraft.Document()
doc.setTempo(100)


#seq = [ ('la', do(5,48), 'la', do(5,48), 'la', so(5,48), 'la', so(5,48), 'la',la(5,48), 'la',la(5,48), 'la',so(5,96))]

#seq = [ ("g6d", mi(5,24), "bI", so(5,24), "zhO", so(5,12), la(5,36) )]
#seq += [ ("mi", mi(5,24), "gat", so(5,16), "t6", so(5,8), "gO", so(5,12), la(5,36))]
#seq += [ ("mi", so(5,36), "yO", mi(5,12), "mI", so(5,24), "O", re(5,72)), BL(48)]

#seq = [ ('sk0r', 36, 1.0, 0.7, 'dr@ft', 36, 1.0, 0.6)]

#seq = [ ('pI', 24, 2.0, 2.0, 'dhan', 24, 1.5, 1.0)]

seq = [ ("ma", mi(5,24), "ma", re(5,24), mi(5,48)), BL(24)]
seq +=[ ("do",mi(5,24),"yO", so(5,24), "ri", la(5,24), "mem", mi(5,12),re(5,12), "b3", re(5,72)), BL(24)]
seq +=[ ("dhx",do(5,12), re(5,12), "Od", mi(5,24), "str0l", so(5,24), "h@t", so(5,72)), BL(24)]
seq +=[ ("yO", mi(5,12),ti(5,36)),BL(12),("gAv", la(5,24), "t6",so(5,12), "mE", mi(5,96))]

# link to voicebank: https://docs.google.com/uc?id=0B8k4SxZamGmvbXc2NEExTG5rUnM&export=download
singer = ScoreDraft.Yami_UTAU()
ScoreDraft.UtauDraftSetLyricConverter(singer, VCCVEnglishConverter.VCCVEnglishConverter)
singer.tune("constvc 80.0")

doc.sing(seq, singer)
doc.mixDown('vccv.wav')

