import os 
import ScoreDraft
from ScoreDraftNotes import *

def soS(octave=5, duration=48):
	return (Freqs[8]*(2.0**(octave-5.0)), duration)

doc=ScoreDraft.Document()
doc.refFreq=264.0 *1.25
doc.tempo=120

seq1 = [do(6,72), ti(5,24), la(5,24), so(5,72)]
seq2 = [la(3,192), BK(144), mi(4,48), so(4,48), do(5,48)]

seq1 = seq1 + [fa(5,96), BL(24), so(5,24), la(5,24), do(6,24)]
seq2 = seq2 + [re(3,192), BK(144), la(3,48), do(4,48),fa(4,48)]

seq1 = seq1 + [ti(5,72), la(5,24), so(5,24), fa(5,72)]
seq2 = seq2 + [so(3,192), BK(144), re(4,48), fa(4,48), ti(4,48)]

seq1 = seq1 + [mi(5,144), BL(48)]
seq2 = seq2 + [do(3,192), BK(144), so(3,48), do(4,48), mi(4,48)]

seq1 = seq1 + [la(5,72), so(5,24), fa(5,24), mi(5,72)]
seq2 = seq2 + [fa(3,192), BK(144), do(4,48), mi(4,48), la(4,48)]

seq1 = seq1 + [re(5,72), mi(5,24), fa(5,24), la(5,72)]
seq2 = seq2 + [re(3,192), BK(144), la(3,48), do(4,48), fa(4,48)]

seq1 = seq1 + [soS(5,72), fa(5,24), mi(5,24), re(5,72)]
seq2 = seq2 + [mi(3,192), BK(144), ti(3,48), re(4,48), soS(4,48)]

seq1 = seq1 + [do(5,144), BL(48)]
seq2 = seq2 + [la(3,192), BK(144), mi(4,48), so(4,48), do(5,48)]

seq1 = seq1 + [re(5,24), la(5,72), la(5,96)]
seq2 = seq2 + [re(3,192), BK(144), la(3,48), do(4,48), fa(4,48)]	

seq1 = seq1 + [BL(96), do(6,24), ti(5,72)]
seq2 = seq2 + [so(3,192), BK(144), re(4,48), fa(4,48), ti(4,48)]	

seq1 = seq1 + [so(5,144), BL(48)]
seq2 = seq2 + [mi(3,192), BK(144), ti(3,48), re(4,48), so(4,48)]

seq1 = seq1 + [la(4,24), fa(5,72), fa(5,96)]
seq2 = seq2 + [re(3,192), BK(144), la(3,48), do(4,48), fa(4,48)]

seq1 = seq1 + [BL(96), la(5,72), so(5,24)]
seq2 = seq2 + [so(3,192), BK(144), re(4,48), fa(4,48), ti(4,48)]	

seq1 = seq1 + [fa(5,24), mi(5,120), BL(48)]
seq2 = seq2 + [do(3,192), BK(144), so(3,48), do(4,48), mi(4,48)]

seq1 = seq1 +  [do(6,72), ti(5,24), la(5,24), so(5,72)]
seq2 = seq2 +  [la(3,192), BK(144), mi(4,48), so(4,48), do(5,48)]

seq1 = seq1 + [fa(5,96), BL(24), so(5,24), la(5,24), do(6,24)]
seq2 = seq2 + [re(3,192), BK(144), la(3,48), do(4,48),fa(4,48)]

seq1 = seq1 + [ti(5,72), la(5,24), so(5,24), fa(5,72)]
seq2 = seq2 + [so(3,192), BK(144), re(4,48), fa(4,48), ti(4,48)]

seq1 = seq1 + [mi(5,144), BL(48)]
seq2 = seq2 + [do(3,192), BK(144), so(3,48), do(4,48), mi(4,48)]

seq1 = seq1 + [la(5,72), so(5,24), fa(5,24), mi(5,72)]
seq2 = seq2 + [fa(3,192), BK(144), do(4,48), mi(4,48), la(4,48)]

seq1 = seq1 + [re(5,72), mi(5,24), fa(5,24), la(5,72)]
seq2 = seq2 + [re(3,192), BK(144), la(3,48), do(4,48), fa(4,48)]

seq1 = seq1 + [soS(5,72), la(5,24), ti(5,24), ti(5,72)]
seq2 = seq2 + [mi(3,192), BK(144), ti(3,48), re(4,48), soS(4,48)]

seq1 = seq1 + [do(6,24), ti(5,24), la(5,96), BL(48)]
seq2 = seq2 + [la(3,192), BK(144), mi(4,48), so(4,48), do(5,48)]

seq1 = seq1 + [la(5,24), so(5,72), la(5,24), so(5,24), fa(5,48)]
seq2 = seq2 + [re(3,192), BK(144), la(3,48), do(4,48), fa(4,48)]

seq1 = seq1 + [BL(96), do(6,24), ti(5,72)]
seq2 = seq2 + [so(3,192), BK(144), re(4,48), fa(4,48), ti(4,48)]	

seq1 = seq1 + [mi(6,144), BL(48)]
seq2 = seq2 + [fa(3,192), BK(144), do(4,48), mi(4,48), la(4,48)]

seq1 = seq1 + [mi(6,24), do(6,72), do(6,96)]
seq2 = seq2 + [re(3,192), BK(144), la(3,48), do(4,48), fa(4,48)]

seq1 = seq1 + [BL(96), ti(5,24), re(6,72)]
seq2 = seq2 + [so(3,192), BK(144), re(4,48), fa(4,48), ti(4,48)]	

seq1 = seq1 + [do(6,192)]
seq2 = seq2 + [do(3,192), BK(180), so(3,180), BK(168), do(4,168), BK(156), mi(4,156), BK(144), so(4,144), BK(132), do(5,132) ]	

instrument=ScoreDraft.NaivePiano()

doc.playNoteSeq(seq1, instrument, 1.0)
doc.playNoteSeq(seq2, instrument, 1.0)
doc.mixDown('FlyMeToTheMoon_eq.wav')

os.system("pause") 
