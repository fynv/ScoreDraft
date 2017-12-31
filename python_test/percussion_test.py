#!/usr/bin/python3

import os 
import ScoreDraft

BassDrum=ScoreDraft.BassDrum()
Snare=ScoreDraft.Snare()

perc_list= [BassDrum, Snare]

def dong(duration=48):
	return (0,duration)

def ca(duration=48):
	return (1,duration)

def Bl(duration=48):
	return (-1, duration)

def Bk(duration=48):
	return (-1, -duration)

doc=ScoreDraft.Document()
doc.tempo=120

seq = [dong(), ca(24), dong(24), dong(), ca(), dong(), ca(24), dong(24), dong(), ca()]


doc.playBeatSeq(seq, perc_list, 1.0)
doc.mixDown('test_perc.wav')
