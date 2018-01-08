#include "FrequencyDetection.h"
#include "fft.h"
#include <memory.h>
#include <stdio.h>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

float fetchFrequency(unsigned length, float *samples, unsigned sampleRate)
{
	unsigned len = 1;
	unsigned l = 0;
	while (len < length*2)
	{
		l++;
		len <<= 1;
	}

	DComp* fftData = new DComp[len];
	memset(fftData, 0, sizeof(DComp)*len);

	for (unsigned i = 0; i<length; i++)
	{
		fftData[i].Re = (double)samples[i];
		fftData[i].Im = 0.0;
	}
	fft(fftData, l);

	// self-correlation
	for (unsigned i = 0; i<len; i++)
	{
		DComp c = fftData[i];
		fftData[i].Re = c.Re*c.Re + c.Im*c.Im;
		fftData[i].Im = 0.0;
	}

	ifft(fftData, l);
	if (fftData[0].Re<0.2)	return -1.0f;
	
	unsigned maxi = (unsigned)(-1);
	
	double lastV = fftData[0].Re;
	double maxV = 0.0f;
	bool ascending = false;

	for (unsigned i = sampleRate / 500; i < min(sampleRate / 40, len / 2); i++)
	{
		double v = fftData[i].Re;
		if (!ascending)
		{
			if (v > lastV) ascending = true;
		}
		else
		{
			if (v < lastV)
			{
				if (fftData[i - 1].Re>maxV)
				{
					maxV = fftData[i - 1].Re;
					maxi = i - 1;
				}
				ascending = false;
			}
		}
		lastV = v;
	}

	float freq;

	if (maxi != (unsigned)(-1) && maxV>0.4f* fftData[0].Re)
	{
		freq = (float)sampleRate / (float)maxi;
	}
	else
	{
		freq = 0.0f;
	}

	delete[] fftData;

	return freq;
}
