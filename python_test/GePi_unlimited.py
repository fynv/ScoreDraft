#!/usr/bin/python3

import time
import ScoreDraft
from ScoreDraftNotes import *

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

voiceNotes = [mi(5,24),mi(5,24),mi(5,24),mi(5,24),mi(5,24),mi(5,24),mi(5,16),mi(5,16),mi(5,16)]
beats += beats_repeat
guitar_notes = [do(4,192), BK(186), mi(4,186), BK(180), so(4,180), BK(174), do(5,174)]

voiceNotes += [so(5,24),so(5,24),so(5,24),so(5,24),so(5,24),so(5,24),so(5,16),so(5,16),so(5,16)]
beats += beats_repeat
guitar_notes += [so(3,192), BK(186), ti(3,186), BK(180), re(4,180), BK(174), so(4,174)]

voiceNotes += [do(5,24),do(5,24),do(5,24),do(5,24),do(5,24),do(5,24),do(5,16),do(5,16),do(5,16)]
beats += beats_repeat
guitar_notes += [la(3,192), BK(186), do(4,186), BK(180), mi(4,180), BK(174), la(4,174)]

voiceNotes += [mi(5,24),mi(5,24),mi(5,24),mi(5,24),mi(5,24),mi(5,24),mi(5,16),mi(5,16),mi(5,16)]
beats += beats_repeat
guitar_notes += [mi(3,192), BK(186), ti(4,186), BK(180), mi(4,180), BK(174), so(4,174)]

voiceNotes += [la(4,24),la(4,24),la(4,24),la(4,24),la(4,24),la(4,24),la(4,16),la(4,16),la(4,16)]
beats += beats_repeat
guitar_notes += [fa(3,192), BK(186), do(5,186), BK(180), fa(4,180), BK(174), la(4,174)]

voiceNotes += [so(4,24),so(4,24),so(4,24),so(4,24),so(4,24),so(4,24),so(4,16),so(4,16),so(4,16)]
beats += beats_repeat
guitar_notes += [do(4,192), BK(186), mi(4,186), BK(180), so(4,180), BK(174), do(5,174)]

voiceNotes += [la(4,24),la(4,24),la(4,24),la(4,24),la(4,24),la(4,24),la(4,16),la(4,16),la(4,16)]
beats += beats_repeat
guitar_notes += [fa(3,192), BK(186), do(5,186), BK(180), fa(4,180), BK(174), la(4,174)]

voiceNotes += [re(5,24),re(5,24),re(5,24),re(5,24),re(5,24),re(5,24),re(5,16),re(5,16),re(5,16)]
beats += beats_repeat
guitar_notes += [so(3,192), BK(186), ti(3,186), BK(180), re(4,180), BK(174), so(4,174)]

# resources

singer=ScoreDraft.GePing()
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

singingSeq = [ (lyric_map[next(pi_calc)],voiceNotes[0]), ('dian', voiceNotes[1])]

for i in range(2, len(voiceNotes) ):
	singingSeq += [(lyric_map[next(pi_calc)], voiceNotes[i])]

singer.sing(track_sing,singingSeq, 120)
guitar.play(track_guitar, guitar_notes, 120)
ScoreDraft.Percussion.play(perc_list, track_drum, beats, 120)

ScoreDraft.MixTrackBufferList(track_mix,[track_sing, track_guitar, track_drum]);
ScoreDraft.PlayTrackBuffer(track_mix)

# Mix 2~n
while 1:
    while (ScoreDraft.PlayGetRemainingTime()>5.0):
        time.sleep(2.0)
    track_mix=ScoreDraft.TrackBuffer()
    track_sing=ScoreDraft.TrackBuffer()
    singingSeq = []
    for j in range(len(voiceNotes) ):
        singingSeq += [(lyric_map[next(pi_calc)], voiceNotes[j])]
    singer.sing(track_sing,singingSeq, 120)
    ScoreDraft.MixTrackBufferList(track_mix,[track_sing, track_guitar, track_drum]);
    ScoreDraft.PlayTrackBuffer(track_mix)
