#include "FrequencyDetection.h"
#include "fft.h"
#include <memory.h>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

float fetchFrequency(unsigned length, float *samples, unsigned sampleRate, bool& success)
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

	if (fftData[0].Re<0.05)
	{
		success = false;
		return 0.0f;
	}

	unsigned maxi = (unsigned)(-1);
	double threshRate = 0.8;

	while (maxi == (unsigned)(-1) && threshRate>0.4)
	{
		double thresh = fftData[0].Re*threshRate;
		double lastV = fftData[0].Re;
		bool ascending = false;

		for (unsigned i = sampleRate / 500; i < min(sampleRate / 60, len / 2); i++)
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
		threshRate -= 0.15;
	}

	delete[] fftData;

	float freq;

	if (maxi != (unsigned)(-1))
	{
		freq = (float)sampleRate / (float)maxi;
		success = true;
	}
	else
	{
		freq = 0.0f;
		success = false;
	}

	return freq;
}
