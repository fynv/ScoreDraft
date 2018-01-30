#!/usr/bin/python3

import sys
sys.path+=['../']

import ScoreDraft
from ScoreDraftNotes import *
import JPVCVConverter

doc=ScoreDraft.Document()
doc.setTempo(120)

seq= [ ('あ', mi(5,48), 'り', so(5,48), 'が', ti(5,48), 'とぅ', do(6,144), ti(5,144), so(5,144))]
#doc.sing(seq, ScoreDraft.uta_UTAU())

Ayaka= ScoreDraft.Ayaka2_UTAU()
ScoreDraft.UtauDraftSetLyricConverter(Ayaka, JPVCVConverter.JPVCVConverter)
doc.sing(seq, Ayaka)


doc.mixDown('uta.wav')
