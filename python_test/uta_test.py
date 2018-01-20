#!/usr/bin/python3

import ScoreDraft
from ScoreDraftNotes import *

doc=ScoreDraft.Document()
doc.setTempo(120)

seq= [ ('あ', mi(5,48)), ('り', so(5,48)), ('が', ti(5,48)), ('ど', do(6,144), ti(5,144), so(5,144))]

doc.sing(seq, ScoreDraft.uta())

doc.mixDown('uta.wav')
