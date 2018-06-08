#!/usr/bin/python3

import time
from random import random
import ScoreDraft
from ScoreDraft.Notes import *

def calcPi():
    q, r, t, k, n, l = 1, 0, 1, 1, 3, 3
    while True:
        if 4*q+r-t < n*t:
            yield n
            nr = 10*(r-n*t)
            n  = ((10*(3*q+r))//t)-10*n
            q  *= 10
            r  = nr
        else:
            nr = (2*q+r)*l
            nn = (q*(7*k)+2+(r*l))//(t*l)
            q  *= k
            t  *= l
            l  += 2
            k += 1
            n  = nn
            r  = nr
 
def dong(duration=48):
	return (0,duration)

def ca(duration=48):
	return (1,duration)

def chi(duration=48):
	return (2,duration)

def Bl(duration=48):
	return (-1, duration)

def Bk(duration=48):
	return (-1, -duration)

beats_repeat = [dong(), ca(), dong(), ca(), Bk(192), chi(24), chi(24), chi(24), chi(24), chi(24), chi(24), chi(16), chi(16), chi(16)]

beats0 = beats_repeat + beats_repeat
beats =[]

beats += beats_repeat
guitar_notes = [do(4,192), BK(186), mi(4,186), BK(180), so(4,180), BK(174), do(5,174)]

beats += beats_repeat
guitar_notes += [so(3,192), BK(186), ti(3,186), BK(180), re(4,180), BK(174), so(4,174)]

beats += beats_repeat
guitar_notes += [la(3,192), BK(186), do(4,186), BK(180), mi(4,180), BK(174), la(4,174)]

beats += beats_repeat
guitar_notes += [mi(3,192), BK(186), ti(4,186), BK(180), mi(4,180), BK(174), so(4,174)]

beats += beats_repeat
guitar_notes += [fa(3,192), BK(186), do(5,186), BK(180), fa(4,180), BK(174), la(4,174)]

beats += beats_repeat
guitar_notes += [do(4,192), BK(186), mi(4,186), BK(180), so(4,180), BK(174), do(5,174)]

beats += beats_repeat
guitar_notes += [fa(3,192), BK(186), do(5,186), BK(180), fa(4,180), BK(174), la(4,174)]

beats += beats_repeat
guitar_notes += [so(3,192), BK(186), ti(3,186), BK(180), re(4,180), BK(174), so(4,174)]

durations = [24, 24, 24, 24, 24, 24, 16, 16, 16]

def freqCalc(octave, i):
    return Freqs[i]*(2.0**(octave-5.0))

chords = [
[ freqCalc(4, 7), freqCalc(5, 0), freqCalc(5, 4), freqCalc(5, 7), freqCalc(6, 0) ],
[ freqCalc(4, 7), freqCalc(4, 11), freqCalc(5, 2), freqCalc(5, 7), freqCalc(5, 11) ],
[ freqCalc(4, 9), freqCalc(5, 0), freqCalc(5, 4), freqCalc(5, 9), freqCalc(6, 0) ],
[ freqCalc(4, 7), freqCalc(4, 11), freqCalc(5, 4), freqCalc(5, 7), freqCalc(5, 11) ],
[ freqCalc(4, 9), freqCalc(5, 0), freqCalc(5, 5), freqCalc(5, 9), freqCalc(6, 0)],
[ freqCalc(4, 7), freqCalc(5, 0), freqCalc(5, 4), freqCalc(5, 7), freqCalc(6, 0) ],
[ freqCalc(4, 9), freqCalc(5, 0), freqCalc(5, 5), freqCalc(5, 9), freqCalc(6, 0)],
[ freqCalc(4, 7), freqCalc(4, 11), freqCalc(5, 2), freqCalc(5, 7), freqCalc(5, 11) ]
]

def chooseFreq(chord, lastFreq):
    weights= []
    for i in range(len(chord)):
        f = chord[i]
        if f== lastFreq:
            weights+=[0.0]
        else:
            weights+=[1.0/( (f-lastFreq)*(f-lastFreq))]
            
    div = sum(weights)
    weights = [w/div for w in weights]

    r=random()
    acc=0.0
    for i in range(len(weights)):
        acc+= weights[i]
        if r<=acc:
         return chord[i]  

# resources

singer=ScoreDraft.GePing_UTAU()
guitar=ScoreDraft.CleanGuitar()

BassDrum=ScoreDraft.BassDrum()
Snare=ScoreDraft.Snare()
ClosedHitHat = ScoreDraft.ClosedHitHat()

perc_list= [BassDrum, Snare, ClosedHitHat]

# Mix0
track_mix=ScoreDraft.TrackBuffer()
track_drum=ScoreDraft.TrackBuffer()
ScoreDraft.Percussion.play(perc_list, track_drum, beats0, 120)

ScoreDraft.MixTrackBufferList(track_mix,[track_drum]);

#Lyric Map
lyric_map = ['ling', 'yi', 'er', 'san', 'si', 'wu','liu', 'qi', 'ba', 'jiu']

# Mix 1
track_sing=ScoreDraft.TrackBuffer()
track_guitar= ScoreDraft.TrackBuffer()
track_drum=ScoreDraft.TrackBuffer()

pi_calc= calcPi()

freq= freqCalc(5, 4)
singingSeq = []

line = (lyric_map[next(pi_calc)], (freq, durations[0]), 'dian', (freq, durations[1]))

j0=2

for i in range(8):
    for j in range(j0, len(durations)):
        line += (lyric_map[next(pi_calc)], (freq, durations[j]))
    singingSeq+=[line]
    if i<7:
        j0=0
        freq=chooseFreq(chords[i+1], freq)
        line=()

singer.sing(track_sing,singingSeq, 120)
guitar.play(track_guitar, guitar_notes, 120)
ScoreDraft.Percussion.play(perc_list, track_drum, beats, 120)

ScoreDraft.MixTrackBufferList(track_mix,[track_sing, track_guitar, track_drum]);
ScoreDraft.QPlayTrackBuffer(track_mix)

# Mix 2~n
while 1:
    while (ScoreDraft.QPlayGetRemainingTime()>5.0):
        time.sleep(2.0)
    if (ScoreDraft.QPlayGetRemainingTime()<0.0):
        break
    track_mix=ScoreDraft.TrackBuffer()
    track_sing=ScoreDraft.TrackBuffer()
    singingSeq = []
    for i in range(8):
        freq=chooseFreq(chords[i], freq)
        line=()
        for j in range(len(durations) ):
            line += (lyric_map[next(pi_calc)], (freq, durations[j]))
        singingSeq+=[line]
            
    singer.sing(track_sing,singingSeq, 120)
    ScoreDraft.MixTrackBufferList(track_mix,[track_sing, track_guitar, track_drum]);
    ScoreDraft.QPlayTrackBuffer(track_mix)
