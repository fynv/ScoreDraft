import os 
import ScoreDraft

test_dong=ScoreDraft.TestPerc()
clap=ScoreDraft.clap()

perc_list= [test_dong, clap]

def dong(duration=48):
	return (0,duration)

def pia(duration=48):
	return (1,duration)

def Bl(duration=48):
	return (-1, duration)

def Bk(duration=48):
	return (-1, -duration)

doc=ScoreDraft.Document()

seq = [dong(), pia(24), dong(24), dong(), pia(24), dong(24), dong(), pia(24), dong(24), (0,"volume 1.5"), (1,"volume 2.0"), dong(), pia(24), dong(24), dong(), pia(24), dong(24), dong(), pia(24), dong(24)]


doc.playBeatSeq(seq, perc_list, 1.0)
doc.mixDown('test_perc.wav')

os.system("pause") 