#!/usr/bin/python3

import os 
import ScoreDraft
from ScoreDraftNotes import *

def soS(octave=5, duration=48):
	return (Freqs[8]*(2.0**(octave-5.0)), duration)

def set_soS(freq):
	Freqs[8]=freq

Cello=ScoreDraft.Cello()
String = ScoreDraft.String()
VoiceLah= ScoreDraft.Lah()

doc=ScoreDraft.Document()
doc.refFreq=264.0
doc.tempo=75

seq1 = [BL(192), la(5,48), do(6,24), ti(5,12), la(5,12), ti(5,24), mi(6,24), mi(5,48)]
seq2 = [mi(4,96), BK(96), la(4,96), mi(4,96), BK(96), la(4,96),  mi(4,96), BK(96), la(4,96), mi(4,96), BK(96), soS(4,96)]
seq3 = [la(2,24), mi(3,24), do(4,24), mi(3,24),la(2,24), mi(3,24), do(4,24), mi(3,24), la(2,24), mi(3,24), do(4,24), mi(3,24),soS(2,24), mi(3,24), ti(3,24), mi(3,24)]

seq1 = seq1 + [la(5,48), do(6,24), ti(5,12), la(5,12), soS(5,72), mi(5,24)]
seq2 = seq2 + [mi(4,96), BK(96), la(4,96),  mi(4,96), BK(96), soS(4,96)]
seq3 = seq3 + [la(2,24), mi(3,24), do(4,24), mi(3,24),soS(2,24), mi(3,24), ti(3,24), mi(3,24)]

seq1 = seq1 + [la(5,48), do(6,24), ti(5,12), la(5,12), ti(5,24), mi(6,24), mi(5,24), mi(5,24)]
seq2 = seq2 + [mi(4,96), BK(96), la(4,96),  mi(4,96), BK(96), soS(4,96)]
seq3 = seq3 + [la(2,24), mi(3,24), do(4,24), mi(3,24),soS(2,24), mi(3,24), ti(3,24), mi(3,24)]

seq1 = seq1 + [fa(5,12), so(5,12), la(5,24), do(6,24), la(5,24), ti(5,72), mi(5,24)]
seq2 = seq2 + [fa(4,96), BK(96), la(4,96),  mi(4,96), BK(96), soS(4,96)]
seq3 = seq3 + [la(2,24), fa(3,24), re(4,24), fa(3,24), soS(2,24), mi(3,24), ti(3,24), mi(3,24)]

seq1 = seq1 + [la(5,48), do(6,24), ti(5,12), la(5,12), ti(5,24), mi(6,24), mi(5,24), mi(5,24)]
seq2 = seq2 + [mi(4,96), BK(96), la(4,96),  mi(4,96), BK(96), soS(4,96)]
seq3 = seq3 + [la(2,24), mi(3,24), do(4,24), mi(3,24),soS(2,24), mi(3,24), ti(3,24), mi(3,24)]

seq1 = seq1 + [la(5,48), do(6,24), ti(5,12), la(5,12), soS(5,72), mi(5,24)]
seq2 = seq2 + [mi(4,96), BK(96), la(4,96),  mi(4,96), BK(96), soS(4,96)]
seq3 = seq3 + [la(2,24), mi(3,24), do(4,24), mi(3,24),soS(2,24), mi(3,24), ti(3,24), mi(3,24)]

seq1 = seq1 + [la(5,48), do(6,24), ti(5,12), la(5,12), ti(5,24), mi(6,24), mi(5,24), mi(5,24)]
seq2 = seq2 + [mi(4,96), BK(96), la(4,96),  mi(4,96), BK(96), soS(4,96)]
seq3 = seq3 + [la(2,24), mi(3,24), do(4,24), mi(3,24),soS(2,24), mi(3,24), ti(3,24), mi(3,24)]

seq1 = seq1 + [fa(5,12), so(5,12), la(5,24), do(6,24), ti(5,24), la(5,96)]
seq2 = seq2 + [fa(4,96), BK(96), la(4,96),  mi(4,96), BK(96), la(4,96)]
seq3 = seq3 + [la(2,24), mi(3,24), do(4,24), mi(3,24), la(2,24), mi(3,24), do(4,24), mi(3,24)]

seq1 = seq1 + [mi(6,24), mi(6,24), mi(6,24), mi(6,24), fa(6,12), mi(6,12), mi(6,12), re(6,12), re(6,48)]
seq2 = seq2 + [mi(4,96), BK(96), la(4,96),  re(4,96), BK(96), soS(4,96)]
seq3 = seq3 + [la(2,24), mi(3,24), do(4,24), mi(3,24), soS(2,24), mi(3,24), ti(3,24), mi(3,24)]

seq1 = seq1 + [re(6,24), re(6,24), re(6,24), re(6,24), mi(6,12), re(6,12), re(6,12), do(6,12), do(6,48)]
seq2 = seq2 + [fa(4,96), BK(96), la(4,96), mi(4,96), BK(96), la(4,96)]
seq3 = seq3 + [la(2,24), fa(3,24), re(4,24), fa(3,24), la(2,24), mi(3,24), do(3,24), mi(3,24)]

seq1 = seq1 + [do(6,24), do(6,24), do(6,24), ti(5,12), do(6,12), re(6,24), la(5,24), re(6,48)]
seq2 = seq2 + [fa(4,96), BK(96), la(4,96),  fa(4,96), BK(96), la(4,96)]
seq3 = seq3 + [la(2,24), fa(3,24), do(4,24), fa(3,24), la(2,24), fa(3,24), re(4,24), fa(3,24)]

seq1 = seq1 + [do(6,24), ti(5,12), la(5,12), so(5,96), BL(24), mi(5,24)]
seq2 = seq2 + [re(4,96), BK(96), so(4,96),re(4,96), BK(96), so(4,96)]
seq3 = seq3 + [so(2,24), re(3,24), ti(4,24), re(3,24), so(2,96), BK(96), re(3,96)]

seq1 = seq1 + [ti(5,192)]
seq2 = seq2 + [soS(4,96), BK(96), ti(4,96), ti(3,96), BK(96), mi(4,96)]
seq3 = seq3 + [mi(4,12), ti(3,12), soS(3,12), mi(3,12), soS(3,12), mi(3,12), ti(2,12), soS(2,12), mi(2,96)]

doc.playNoteSeq(seq1, VoiceLah, 1.0)
doc.playNoteSeq(seq2, String, 0.5)
doc.playNoteSeq(seq3, Cello, 0.5)
doc.mixDown('NaushikaRequiem.wav')
