#include <math.h>
#include <memory.h>
#include "Sample.h"
#include "InstrumentMultiSampler.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

static void s_generateNoteWave(const InstrumentSample& sample, float* outBuf, unsigned outBufLen, float sampleFreq, float k)
{
	unsigned chn = sample.m_chn;
	float origin_SampleFreq = sample.m_origin_freq / (float)sample.m_origin_sample_rate;
	unsigned maxSample = (unsigned)((float)sample.m_wav_length*origin_SampleFreq / sampleFreq);

	float mult = 1.0f / sample.m_max_v;

	bool interpolation = sampleFreq <= origin_SampleFreq;

	for (unsigned j = 0; j < min(outBufLen, maxSample); j++)
	{
		float wave[2];
		if (interpolation)
		{
			float pos = (float)j *sampleFreq / origin_SampleFreq;
			int ipos1 = (int)pos;
			float frac = pos - (float)ipos1;
			int ipos2 = ipos1 + 1;
			if (ipos2 >= (int)sample.m_wav_length) ipos2 = (int)sample.m_wav_length - 1;

			// linear interpolation
			//wave = m_wav_samples[ipos1] * (1.0f - frac) + m_wav_samples[ipos2] * frac;

			// cubic interpolation
			int ipos0 = ipos1 - 1;
			if (ipos0 < 0) ipos0 = 0;

			int ipos3 = ipos1 + 2;
			if (ipos3 >= (int)sample.m_wav_length) ipos3 = (int)sample.m_wav_length - 1;

			for (unsigned c = 0; c < chn; c++)
			{
				float p0 = sample.m_wav_samples[ipos0*chn + c];
				float p1 = sample.m_wav_samples[ipos1*chn + c];
				float p2 = sample.m_wav_samples[ipos2*chn + c];
				float p3 = sample.m_wav_samples[ipos3*chn + c];

				wave[c] = (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*powf(frac, 3.0f) +
					(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
					(-0.5f*p0 + 0.5f*p2)*frac + p1;
			}

		}
		else
		{
			int ipos1 = (int)ceilf(((float)j - 0.5f)*sampleFreq / origin_SampleFreq);
			int ipos2 = (int)floorf(((float)j + 0.5f)*sampleFreq / origin_SampleFreq);
			if (ipos1 < 0) ipos1 = 0;
			if (ipos2 >= (int)sample.m_wav_length) ipos2 = (int)sample.m_wav_length - 1;
			int count = ipos2 - ipos1 + 1;

			for (unsigned c = 0; c < chn; c++)
			{
				float sum = 0.0f;
				for (int ipos = ipos1; ipos <= ipos2; ipos++)
				{
					sum += sample.m_wav_samples[ipos*chn + c];
				}
				wave[c] = sum / (float)count;
			}
		}

		for (unsigned c = 0; c < chn; c++)
		{
			outBuf[j*chn + c] += k* wave[c] * mult;
		}
	}

}

void InstrumentMultiSample(const std::vector<InstrumentSample>& samples, float* outBuf, unsigned outBufLen, float sampleFreq)
{
	if (samples.size() < 1) return;
	unsigned chn = samples[0].m_chn;

	bool useSingle = false;
	unsigned I;

	{
		const InstrumentSample& wav = samples[0];
		float origin_SampleFreq = wav.m_origin_freq / (float)wav.m_origin_sample_rate;

		if (sampleFreq <= origin_SampleFreq)
		{
			I = 0;
			useSingle = true;
		}
	}

	if (!useSingle)
	{
		const InstrumentSample& wav = samples[samples.size() - 1];
		float origin_SampleFreq = wav.m_origin_freq / (float)wav.m_origin_sample_rate;

		if (sampleFreq >= origin_SampleFreq)
		{
			I = (unsigned)(samples.size() - 1);
			useSingle = true;
		}
	}

	if (!useSingle)
	{
		for (size_t i = 0; i < samples.size() - 1; i++)
		{
			const InstrumentSample& wav = samples[i + 1];
			float origin_SampleFreq = wav.m_origin_freq / (float)wav.m_origin_sample_rate;

			if (sampleFreq == origin_SampleFreq)
			{
				I = (unsigned)(i + 1);
				useSingle = true;
				break;
			}
			else if (sampleFreq < origin_SampleFreq)
			{
				I = (unsigned)i;
				break;
			}
		}
	}
	memset(outBuf, 0, sizeof(float)*outBufLen*chn);
	if (useSingle)
	{
		s_generateNoteWave(samples[I], outBuf, outBufLen, sampleFreq, 1.0f);
	}
	else
	{
		const InstrumentSample& wav1 = samples[I];
		const InstrumentSample& wav2 = samples[I + 1];
		float origin_SampleFreq1 = wav1.m_origin_freq / (float)wav1.m_origin_sample_rate;
		float origin_SampleFreq2 = wav2.m_origin_freq / (float)wav2.m_origin_sample_rate;

		float k2 = logf(sampleFreq / origin_SampleFreq1) / logf(origin_SampleFreq2 / origin_SampleFreq1);
		float k1 = 1.0f - k2;

		s_generateNoteWave(wav1, outBuf, outBufLen, sampleFreq, k1);
		s_generateNoteWave(wav2, outBuf, outBufLen, sampleFreq, k2);
	}

	for (unsigned j = 0; j < outBufLen; j++)
	{
		float x2 = (float)j / (float)outBufLen;
		float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);

		for (unsigned c = 0; c < chn; c++)
			outBuf[j*chn + c] = amplitude*outBuf[j*chn + c];
	}
}
