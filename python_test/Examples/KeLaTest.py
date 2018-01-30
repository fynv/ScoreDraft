#!/usr/bin/python3
import sys
sys.path+=['../']

import ScoreDraft
from ScoreDraftNotes import *

doc=ScoreDraft.Document()

seq= [ ("jin_f1", do(5,24), ti(4,24), do(5,24)), ("ke_f1", re(5,24)), ("la_f1", mi(5,24)) ]

doc.sing(seq, ScoreDraft.KeLaTest())
doc.mixDown('KeLa.wav')
