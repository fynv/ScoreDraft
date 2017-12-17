
Freqs=[ 1.0, 2.0**(1.0/12.0) , 2.0**(2.0/12.0), 2.0**(3.0/12.0), 2.0**(4.0/12.0), 2.0**(5.0/12.0), 2.0**(6.0/12.0), 2.0**(7.0/12.0), 2.0**(8.0/12.0), 2.0**(9.0/12.0), 2.0**(10.0/12.0), 2.0**(11.0/12.0)]


def do(octave=5, duration=48):
	return (Freqs[0]*(2.0**(octave-5.0)), duration)

def set_do(freq):
	Freqs[0]=freq

def re(octave=5, duration=48):
	return (Freqs[2]*(2.0**(octave-5.0)), duration)

def set_re(freq):
	Freqs[2]=freq

def mi(octave=5, duration=48):
	return (Freqs[4]*(2.0**(octave-5.0)), duration)

def set_mi(freq):
	Freqs[4]=freq

def fa(octave=5, duration=48):
	return (Freqs[5]*(2.0**(octave-5.0)), duration)

def set_fa(freq):
	Freqs[5]=freq

def so(octave=5, duration=48):
	return (Freqs[7]*(2.0**(octave-5.0)), duration)

def set_so(freq):
	Freqs[7]=freq

def la(octave=5, duration=48):
	return (Freqs[9]*(2.0**(octave-5.0)), duration)

def set_la(freq):
	Freqs[9]=freq

def ti(octave=5, duration=48):
	return (Freqs[11]*(2.0**(octave-5.0)), duration)

def set_ti(freq):
	Freqs[11]=freq

def BL(duration=48):
	return (-1.0, duration)

def BK(duration=48):
	return (-1.0, -duration)
