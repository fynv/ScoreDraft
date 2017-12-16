import os 
import ScoreDraft
from ScoreDraftNotes import *

def soS(octave=5, duration=48):
	return (8, octave, duration)

def set_soS(freqTab, freq):
	freqTab.freqs[8]=freq

doc=ScoreDraft.Document()
doc.refFreq=264.0 *1.25
doc.tempo=120

freqTab=ScoreDraft.Freqs()
freqTab.set_re(10.0/9.0)
freqTab.set_mi(5.0/4.0)
freqTab.set_fa(4.0/3.0)
freqTab.set_so(3.0/2.0)
freqTab.set_la(5.0/3.0)
freqTab.set_ti(15.0/8.0)
set_soS(freqTab, 25.0/16.0)

seq1 = ScoreDraft.parseSeq([do(6,72), ti(5,24), la(5,24), so(5,72)], freqTab)
seq2 = ScoreDraft.parseSeq([la(3,192), BK(144), mi(4,48), so(4,48), do(5,48)], freqTab)

seq1 = seq1 + ScoreDraft.parseSeq([fa(5,96), BL(24), so(5,24), la(5,24), do(6,24)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([re(3,192), BK(144), la(3,48), do(4,48),fa(4,48)], freqTab)

freqTab.set_re(9.0/8.0)
freqTab.set_fa(21.0/16.0)

seq1 = seq1 + ScoreDraft.parseSeq([ti(5,72), la(5,24), so(5,24), fa(5,72)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([so(3,192), BK(144), re(4,48), fa(4,48), ti(4,48)], freqTab)

freqTab.set_fa(4.0/3.0)

seq1 = seq1 + ScoreDraft.parseSeq([mi(5,144), BL(48)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([do(3,192), BK(144), so(3,48), do(4,48), mi(4,48)],freqTab)

freqTab.set_re(10.0/9.0)

seq1 = seq1 + ScoreDraft.parseSeq([la(5,72), so(5,24), fa(5,24), mi(5,72)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([fa(3,192), BK(144), do(4,48), mi(4,48), la(4,48)],freqTab)

seq1 = seq1 + ScoreDraft.parseSeq([re(5,72), mi(5,24), fa(5,24), la(5,72)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([re(3,192), BK(144), la(3,48), do(4,48), fa(4,48)],freqTab)

freqTab.set_re(35.0/32.0)
seq1 = seq1 + ScoreDraft.parseSeq([soS(5,72), fa(5,24), mi(5,24), re(5,72)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([mi(3,192), BK(144), ti(3,48), re(4,48), soS(4,48)],freqTab)

freqTab.set_re(10.0/9.0)
seq1 = seq1 + ScoreDraft.parseSeq([do(5,144), BL(48)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([la(3,192), BK(144), mi(4,48), so(4,48), do(5,48)],freqTab)

seq1 = seq1 + ScoreDraft.parseSeq([re(5,24), la(5,72), la(5,96)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([re(3,192), BK(144), la(3,48), do(4,48), fa(4,48)],freqTab)	

freqTab.set_re(9.0/8.0)
freqTab.set_fa(21.0/16.0)

seq1 = seq1 + ScoreDraft.parseSeq([BL(96), do(6,24), ti(5,72)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([so(3,192), BK(144), re(4,48), fa(4,48), ti(4,48)],freqTab)	

freqTab.set_fa(4.0/3.0)
seq1 = seq1 + ScoreDraft.parseSeq([so(5,144), BL(48)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([mi(3,192), BK(144), ti(3,48), re(4,48), so(4,48)],freqTab)

freqTab.set_re(10.0/9.0)
seq1 = seq1 + ScoreDraft.parseSeq([la(4,24), fa(5,72), fa(5,96)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([re(3,192), BK(144), la(3,48), do(4,48), fa(4,48)],freqTab)

freqTab.set_re(9.0/8.0)
freqTab.set_fa(21.0/16.0)
freqTab.set_la(27.0/16.0)

seq1 = seq1 + ScoreDraft.parseSeq([BL(96), la(5,72), so(5,24)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([so(3,192), BK(144), re(4,48), fa(4,48), ti(4,48)],freqTab)	

seq1 = seq1 + ScoreDraft.parseSeq([fa(5,24), mi(5,120), BL(48)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([do(3,192), BK(144), so(3,48), do(4,48), mi(4,48)],freqTab)

freqTab.set_re(10.0/9.0)
freqTab.set_fa(4.0/3.0)
freqTab.set_la(5.0/3.0)

seq1 = seq1 +  ScoreDraft.parseSeq([do(6,72), ti(5,24), la(5,24), so(5,72)], freqTab)
seq2 = seq2 +  ScoreDraft.parseSeq([la(3,192), BK(144), mi(4,48), so(4,48), do(5,48)], freqTab)

seq1 = seq1 + ScoreDraft.parseSeq([fa(5,96), BL(24), so(5,24), la(5,24), do(6,24)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([re(3,192), BK(144), la(3,48), do(4,48),fa(4,48)], freqTab)

freqTab.set_re(9.0/8.0)
freqTab.set_fa(21.0/16.0)

seq1 = seq1 + ScoreDraft.parseSeq([ti(5,72), la(5,24), so(5,24), fa(5,72)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([so(3,192), BK(144), re(4,48), fa(4,48), ti(4,48)], freqTab)

freqTab.set_fa(4.0/3.0)

seq1 = seq1 + ScoreDraft.parseSeq([mi(5,144), BL(48)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([do(3,192), BK(144), so(3,48), do(4,48), mi(4,48)],freqTab)

freqTab.set_re(10.0/9.0)

seq1 = seq1 + ScoreDraft.parseSeq([la(5,72), so(5,24), fa(5,24), mi(5,72)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([fa(3,192), BK(144), do(4,48), mi(4,48), la(4,48)],freqTab)

seq1 = seq1 + ScoreDraft.parseSeq([re(5,72), mi(5,24), fa(5,24), la(5,72)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([re(3,192), BK(144), la(3,48), do(4,48), fa(4,48)],freqTab)

freqTab.set_re(35.0/32.0)
seq1 = seq1 + ScoreDraft.parseSeq([soS(5,72), la(5,24), ti(5,24), ti(5,72)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([mi(3,192), BK(144), ti(3,48), re(4,48), soS(4,48)],freqTab)

freqTab.set_re(10.0/9.0)
seq1 = seq1 + ScoreDraft.parseSeq([do(6,24), ti(5,24), la(5,96), BL(48)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([la(3,192), BK(144), mi(4,48), so(4,48), do(5,48)],freqTab)

seq1 = seq1 + ScoreDraft.parseSeq([la(5,24), so(5,72), la(5,24), so(5,24), fa(5,48)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([re(3,192), BK(144), la(3,48), do(4,48), fa(4,48)],freqTab)

freqTab.set_re(9.0/8.0)
freqTab.set_fa(21.0/16.0)

seq1 = seq1 + ScoreDraft.parseSeq([BL(96), do(6,24), ti(5,72)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([so(3,192), BK(144), re(4,48), fa(4,48), ti(4,48)],freqTab)	

freqTab.set_re(10.0/9.0)
freqTab.set_fa(4.0/3.0)
seq1 = seq1 + ScoreDraft.parseSeq([mi(6,144), BL(48)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([fa(3,192), BK(144), do(4,48), mi(4,48), la(4,48)],freqTab)

seq1 = seq1 + ScoreDraft.parseSeq([mi(6,24), do(6,72), do(6,96)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([re(3,192), BK(144), la(3,48), do(4,48), fa(4,48)], freqTab)

freqTab.set_re(9.0/8.0)
freqTab.set_fa(21.0/16.0)

seq1 = seq1 + ScoreDraft.parseSeq([BL(96), ti(5,24), re(6,72)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([so(3,192), BK(144), re(4,48), fa(4,48), ti(4,48)],freqTab)	

seq1 = seq1 + ScoreDraft.parseSeq([do(6,192)], freqTab)
seq2 = seq2 + ScoreDraft.parseSeq([do(3,192), BK(180), so(3,180), BK(168), do(4,168), BK(156), mi(4,156), BK(144), so(4,144), BK(132), do(5,132) ],freqTab)	

doc.playSeq(seq1,1.0, ScoreDraft.BottleBlow)
doc.playSeq(seq2,0.5, ScoreDraft.NaivePiano)
doc.mixDown('FlyMeToTheMoon_just.wav')

os.system("pause") 
