#!/usr/bin/python3

import sys
sys.path+=['../']

import ScoreDraft
from ScoreDraftNotes import *

doc=ScoreDraft.Document()

seq= [ ("jin f1", do(5,24), ti(4,24), do(5,24)), ("ke f1", re(5,24)), ("la f1", mi(5,24)) ]

# link to voicebank: http://pan.baidu.com/s/1mi6N6oK
doc.sing(seq, ScoreDraft.jklex_UTAU())
doc.mixDown('KeLa.wav')
