Freqs=[2.0**(v/12.0) for v in range(12)]

def note(octave, freq, duration):
	return (freq*(2.0**(octave-5.0)), duration)

def do(octave=5, duration=48):
	return note(octave,Freqs[0],duration)

def set_do(freq):
	Freqs[0]=freq

def re(octave=5, duration=48):
	return note(octave,Freqs[2],duration)

def set_re(freq):
	Freqs[2]=freq

def mi(octave=5, duration=48):
	return note(octave,Freqs[4],duration)

def set_mi(freq):
	Freqs[4]=freq

def fa(octave=5, duration=48):
	return note(octave,Freqs[5],duration)

def set_fa(freq):
	Freqs[5]=freq

def so(octave=5, duration=48):
	return note(octave,Freqs[7],duration)

def set_so(freq):
	Freqs[7]=freq

def la(octave=5, duration=48):
	return note(octave,Freqs[9],duration)

def set_la(freq):
	Freqs[9]=freq

def ti(octave=5, duration=48):
	return note(octave,Freqs[11],duration)

def set_ti(freq):
	Freqs[11]=freq


def BL(duration=48):
	return (-1.0, duration)

def BK(duration=48):
	return (-1.0, -duration)
