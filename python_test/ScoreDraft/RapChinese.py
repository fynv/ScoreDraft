baseFreq=1.0

def SetRapBaseFreq(freq):
	global baseFreq   
	baseFreq=freq

def CRap(lyric, tone, duration=48):
	if tone <= 1:
		return (lyric, duration, baseFreq, baseFreq)
	elif tone == 2:
		return (lyric, duration, baseFreq*0.7, baseFreq)
	elif tone == 3:
		return (lyric, duration, baseFreq*0.5, baseFreq*0.75)
	elif tone == 4:
		return (lyric, duration, baseFreq, baseFreq*0.5)
	else:
		return (lyric, duration, baseFreq*0.75, baseFreq*0.55)
