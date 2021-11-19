#!/usr/bin/python3

import ScoreDraft
from ScoreDraft.Notes import *

doc=ScoreDraft.Document()

seq=[do(),do(),so(),so(),la(),la(),so(5,96)]

doc.playNoteSeq(seq, ScoreDraft.KarplusStrongInstrument())
#doc.mixDown('Hello.wav')

targetBuf=ScoreDraft.TrackBuffer(-1)
doc.mix(targetBuf)

ScoreDraft.PlayTrackBuffer(targetBuf)