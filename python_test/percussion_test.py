import os 
import ScoreDraft

def dong(duration=48):
	return (0,duration)

def Bl(duration=48):
	return (-1, duration)

def Bk(duration=48):
	return (-1, -duration)

doc=ScoreDraft.Document()

seq = [dong(), Bl(), dong(), Bl(), dong(), Bl(),dong(), Bl(),dong(), Bl(),dong(), Bl()]

perc=ScoreDraft.TestPerc()

doc.playBeatSeq(seq, [perc], 1.0)
doc.mixDown('test_perc.wav')

os.system("pause") 