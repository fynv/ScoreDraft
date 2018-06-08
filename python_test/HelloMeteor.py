#!/usr/bin/python3

import ScoreDraft
from ScoreDraft.Notes import *

doc=ScoreDraft.MeteorDocument()

seq=[do(),do(),so(),so(),la(),la(),so(5,96)]

doc.playNoteSeq(seq, ScoreDraft.NaivePiano())
doc.meteor()
doc.mixDown('HelloMeteor.wav')
