#include "FrequencyDetection.h"
#include "fft.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

float fetchFrequency(unsigned length, float *samples, unsigned sampleRate)
{
	unsigned len = 1;
	unsigned l = -1;
	while (len <= length)
	{
		l++;
		len <<= 1;
	}
	len = 1 << l;

	DComp* fftData = new DComp[len];

	for (unsigned i = 0; i<len; i++)
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

	double thresh = fftData[0].Re*0.7;
	double lastV = fftData[0].Re;
	bool ascending = false;
	unsigned maxi = 0;
	for (unsigned i = sampleRate / 2000; i < min(sampleRate / 30, len); i++)
	{
		double v = fftData[i].Re;
		if (v > thresh)
		{
			if (!ascending)
			{
				if (v > lastV) ascending = true;
			}
			else
			{
				if (v < lastV)
				{
					maxi = i - 1;
					break;
				}
			}
			lastV = v;
		}
	}

	float freq = (float)sampleRate / (float)maxi;
	delete[] fftData;

	return freq;
}
