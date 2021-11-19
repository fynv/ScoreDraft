#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define SCOREDRAFT_API __declspec(dllexport)
#else
#define SCOREDRAFT_API 
#endif

extern "C"
{
	SCOREDRAFT_API void KarplusStrongGenerate(void* ptr_wavbuf, float freq, float fduration, float cut_freq, float loop_gain, float sustain_gain);
}

#include <cmath>
#include <cstdlib>
#include <memory.h>
#include <WavBuffer.h>
#include <vector>
#include "fft.h"

inline float rand01()
{
	float f = (float)rand() / (float)RAND_MAX;
	if (f < 0.0000001f) f = 0.0000001f;
	if (f > 0.9999999f) f = 0.9999999f;
	return f;
}

inline void GeneratePinkNoise(float period, std::vector<float>& buf)
{
	unsigned uLen = (unsigned)ceilf(period);
	unsigned l = 0;
	unsigned fftLen = 1;
	while (fftLen < uLen)
	{
		fftLen <<= 1;
		l++;
	}

	std::vector<DComp> fftData(fftLen);
	memset(&fftData[0], 0, sizeof(DComp)*fftLen);

	for (unsigned i = 1; i < (unsigned)(period) / 2; i++)
	{
		float amplitude = (float)fftLen / sqrtf((float)i);
		float phase = rand01()*(float)(2.0*PI);
		fftData[i].Re = (double)(amplitude*cosf(phase));
		fftData[i].Im = (double)(amplitude*sinf(phase));

		fftData[fftLen - i].Re = fftData[i].Re;
		fftData[fftLen - i].Im = -fftData[i].Im;
	}

	ifft(&fftData[0], l);

	unsigned pnLen = (unsigned)ceilf(period*2.0f);	
	buf.resize(pnLen);

	float rate = (float)fftLen / period;
	for (unsigned i = 0; i < pnLen; i++)
	{
		int ipos1 = (int)ceilf(((float)i - 0.5f)*rate);
		if (ipos1 < 0) ipos1 = 0;
		int ipos2 = (int)floorf(((float)i + 0.5f)*rate);
		int count = ipos2 - ipos1 + 1;

		float sum = 0.0f;
		for (int ipos = ipos1; ipos <= ipos2; ipos++)
		{
			int _ipos = ipos;
			while (_ipos >= fftLen) _ipos -= fftLen;
			sum += (float)fftData[_ipos].Re;
		}
		buf[i] = sum / (float)count;
	}	
}

void KarplusStrongGenerate(void* ptr_wavbuf, float freq, float fduration, float cut_freq, float loop_gain, float sustain_gain)
{
	WavBuffer* wavbuf = (WavBuffer*)ptr_wavbuf;
	float sampleRate = wavbuf->m_sampleRate;

	float sustain_periods = logf(0.01f) / logf(sustain_gain);
	float fNumOfSamples = fduration * sampleRate*0.001f;

	float period = sampleRate / freq;
	std::vector<float> pinkNoise;
	GeneratePinkNoise(period, pinkNoise);

	float sustainLen = sustain_periods * period;
	size_t totalLen = (size_t)ceilf(fNumOfSamples + sustainLen);
	wavbuf->Allocate(1, totalLen);	

	cut_freq = cut_freq / 261.626f* freq;
	float a = (float)(1.0 - exp(-2.0*PI* cut_freq / sampleRate));

	unsigned pos = 0;
	while (pos < totalLen)
	{
		float value = 0.0f;
		if ((float)pos < period*2.0f)
			value += pinkNoise[pos] * 0.5f*(cosf(((float)pos - period) / period * PI) + 1.0f);

		if ((float)pos >= period)
		{
			float gain = (float)pos < fNumOfSamples ? loop_gain : sustain_gain;

			float refPos = (float)pos - period;

			int refPos1 = (int)refPos;
			int refPos2 = refPos1 + 1;
			float frac = refPos - (float)refPos1;

			// linear interpolation
			float ref = wavbuf->m_data[refPos1] * (1.0f - frac) + wavbuf->m_data[refPos2] * frac;

			value += gain * a*ref + (1.0f - a)*wavbuf->m_data[pos - 1];
		}


		wavbuf->m_data[pos] = value;
		pos++;
	}
}

