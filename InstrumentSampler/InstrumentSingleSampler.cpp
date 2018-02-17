#include "InstrumentSingleSampler.h"

#include <string.h>
#include <math.h>
#include "InstrumentSample.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


InstrumentSingleSampler::InstrumentSingleSampler()
{
	m_sample = nullptr;
}

InstrumentSingleSampler::~InstrumentSingleSampler()
{

}


void InstrumentSingleSampler::GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf)
{
	if (!m_sample) return;

	float origin_SampleFreq = m_sample->m_origin_freq / (float)m_sample->m_origin_sample_rate;
	unsigned maxSample = (unsigned)((float)m_sample->m_wav_length*origin_SampleFreq / sampleFreq);

	noteBuf->m_sampleNum = min((unsigned)ceilf(fNumOfSamples), maxSample);
	noteBuf->Allocate();

	float mult = 1.0f / m_sample->m_max_v;

	bool interpolation = sampleFreq <= origin_SampleFreq;

	for (unsigned j = 0; j < noteBuf->m_sampleNum; j++)
	{
		float x2 = (float)j / fNumOfSamples;
		float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);

		float wave;
		if (interpolation)
		{
			float pos = (float)j *sampleFreq / origin_SampleFreq;
			int ipos1 = (int)pos;
			float frac = pos - (float)ipos1;
			int ipos2 = ipos1 + 1;
			if (ipos2 >= (int)m_sample->m_wav_length) ipos2 = (int)m_sample->m_wav_length - 1;

			// linear interpolation
			//wave = m_wav_samples[ipos1] * (1.0f - frac) + m_wav_samples[ipos2] * frac;

			// cubic interpolation
			int ipos0 = ipos1 - 1;
			if (ipos0 < 0) ipos0 = 0;

			int ipos3 = ipos1 + 2;
			if (ipos3 >= (int)m_sample->m_wav_length) ipos3 = (int)m_sample->m_wav_length - 1;

			float p0 = m_sample->m_wav_samples[ipos0];
			float p1 = m_sample->m_wav_samples[ipos1];
			float p2 = m_sample->m_wav_samples[ipos2];
			float p3 = m_sample->m_wav_samples[ipos3];

			wave = (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*powf(frac, 3.0f) +
				(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
				(-0.5f*p0 + 0.5f*p2)*frac + p1;
		}
		else
		{
			int ipos1 = (int)ceilf(((float)j - 0.5f)*sampleFreq / origin_SampleFreq);
			int ipos2 = (int)floorf(((float)j + 0.5f)*sampleFreq / origin_SampleFreq);
			if (ipos1 < 0) ipos1 = 0;
			if (ipos2 >= (int)m_sample->m_wav_length) ipos2 = (int)m_sample->m_wav_length - 1;
			int count = ipos2 - ipos1 + 1;
			float sum = 0.0f;
			for (int ipos = ipos1; ipos <= ipos2; ipos++)
			{
				sum += m_sample->m_wav_samples[ipos];
			}
			wave = sum / (float)count;
		}

		noteBuf->m_data[j] = amplitude*wave*mult;
	}
}
