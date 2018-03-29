#!/usr/bin/python3

import sys
sys.path+=['../']

import ScoreDraft
from ScoreDraftNotes import *
import VCCVEnglishConverter

doc=ScoreDraft.Document()
doc.setTempo(100)


seq = [ ('twin', do(5,48), 'k9l', do(5,48), 'twin', so(5,48), 'k9l', so(5,48), 'li',la(5,48), 't9l',la(5,48), 'sta',so(5,96))]

Yami = ScoreDraft.Yami_UTAU()
ScoreDraft.UtauDraftSetLyricConverter(Yami, VCCVEnglishConverter.VCCVEnglishConverter)

doc.sing(seq, Yami)
doc.mixDown('vccv.wav')

