import os 
import ScoreDraft

def dong(duration=48):
	return (0,duration)

def pia(duration=48):
	return (1,duration)

def Bl(duration=48):
	return (-1, duration)

def Bk(duration=48):
	return (-1, -duration)

doc=ScoreDraft.Document()

seq = [dong(), pia(), dong(), pia(), dong(), pia(),dong(), pia(),dong(), pia(),dong(), pia()]

test_dong=ScoreDraft.TestPerc()
clap=ScoreDraft.clap()

doc.playBeatSeq(seq, [test_dong, clap], 1.0)
doc.mixDown('test_perc.wav')

os.system("pause") 