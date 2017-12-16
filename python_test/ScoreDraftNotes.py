
def do(octave=5, duration=48):
	return (0, octave, duration)

def re(octave=5, duration=48):
	return (2, octave, duration)

def mi(octave=5, duration=48):
	return (4, octave, duration)

def fa(octave=5, duration=48):
	return (5, octave, duration)

def so(octave=5, duration=48):
	return (7, octave, duration)

def la(octave=5, duration=48):
	return (9, octave, duration)

def ti(octave=5, duration=48):
	return (11, octave, duration)

def BL(duration=48):
	return (-1, 5, duration)

def BK(duration=48):
	return (-1, 5, -duration)
