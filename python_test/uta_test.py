#!/usr/bin/python3

import ScoreDraft
from ScoreDraft.Notes import *

doc=ScoreDraft.Document()
doc.setTempo(120)

seq= [ ('あ', mi(5,48), 'り', so(5,48), 'が', ti(5,48), 'とぅ', do(6,144), ti(5,144), so(5,144))]

# voicebank: please copy the "voice/uta" folder from UTAU
doc.sing(seq, ScoreDraft.uta_UTAU())


'''
Ayaka= ScoreDraft.Ayaka2_UTAU()
Ayaka.setLyricConverter(ScoreDraft.JPVCVConverter)
doc.sing(seq, Ayaka)
'''

doc.mixDown('uta.wav')
