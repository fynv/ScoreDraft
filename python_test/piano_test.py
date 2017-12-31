#!/usr/bin/python3

import os 
import ScoreDraft
from ScoreDraftNotes import *

Piano = ScoreDraft.Piano()

doc=ScoreDraft.Document()
doc.refFreq=318.0
doc.tempo=80

seq1 = [la(4,48), do(5,48), re(5,36), do(5,36), re(5,24)]
seq2 = [la(3,96), BK(96), do(4,96), BK(96), mi(4,96), re(4,96), BK(96), fa(4,96), BK(96), la(4,96)]

seq1 = seq1+[re(5,48), so(5,24), fa(5,24), mi(5,12), re(5,24), mi(5,56)]
seq2 = seq2+[so(3,96), BK(96), ti(3,96), BK(96), re(4,96), do(4,96), BK(96), mi(4,96), BK(96), so(4,96)]

seq1 = seq1+[mi(5,48), so(5,48), la(5,36), re(5,36), do(5,24)]
seq2 = seq2+[la(3,96), BK(96), do(4,96), BK(96), mi(4,96), re(4,96), BK(96), fa(4,96), BK(96), la(4,96)]

seq1 = seq1+[so(5,48), mi(5,24), so(5,24), so(5,36), BK(36), mi(5,36), la(5,56), BK(56), fa(5,56)]
seq2 = seq2+[mi(4,96), BK(96), so(4,96), BK(96), ti(4,96), fa(4,96), BK(96), la(4,96), BK(96), do(5,96)];

doc.playNoteSeq(seq1, Piano, 1.0)
doc.playNoteSeq(seq2, Piano, 1.0)
doc.mixDown('piano_test.wav')
