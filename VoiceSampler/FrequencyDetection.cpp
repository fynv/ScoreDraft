#include "FrequencyDetection.h"
#include "fft.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

void fetchFrequency(unsigned length, float *samples, unsigned sampleRate, float& freq, float& dyn)
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
	
	dyn = (float)fftData[0].Re*700.0f;
	freq = 55.0f;

	if (fftData[0].Re > 0.01)
	{
		unsigned maxi = (unsigned)(-1);

		double lastV = fftData[0].Re;
		double maxV = 0.0f;
		bool ascending = false;

		for (unsigned i = sampleRate / 600; i < min(sampleRate / 55, len / 2); i++)
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

		if (maxi != (unsigned)(-1) && maxV > 0.3f* fftData[0].Re)
		{
			freq = (float)sampleRate / (float)maxi;
		}
	}

	delete[] fftData;

}
