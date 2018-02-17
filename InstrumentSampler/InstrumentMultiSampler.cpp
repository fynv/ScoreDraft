#include "InstrumentMultiSampler.h"

#include <string.h>
#include <math.h>
#include <ReadWav.h>
#include <float.h>

#include <stdlib.h>

#include "FrequencyDetection.h"

#include "fft.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


InstrumentMultiSampler::InstrumentMultiSampler()
{
	m_SampleWavList = nullptr;
}

InstrumentMultiSampler::~InstrumentMultiSampler()
{

}

void InstrumentMultiSampler::_generateNoteWave(unsigned index, float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf)
{
	if (m_SampleWavList == nullptr) return;

	InstrumentSample_deferred wav = (*m_SampleWavList)[index];

	float origin_SampleFreq = wav->m_origin_freq / (float)wav->m_origin_sample_rate;
	unsigned maxSample = (unsigned)((float)wav->m_wav_length*origin_SampleFreq / sampleFreq);

	noteBuf->m_sampleNum = min((unsigned)ceilf(fNumOfSamples), maxSample);
	noteBuf->Allocate();

	float mult = 1.0f / wav->m_max_v;

	bool interpolation = sampleFreq <= origin_SampleFreq;

	for (unsigned j = 0; j < noteBuf->m_sampleNum; j++)
	{
		float wave;
		if (interpolation)
		{
			float pos = (float)j *sampleFreq / origin_SampleFreq;
			int ipos1 = (int)pos;
			float frac = pos - (float)ipos1;
			int ipos2 = ipos1 + 1;
			if (ipos2 >= (int)wav->m_wav_length) ipos2 = (int)wav->m_wav_length - 1;

			// linear interpolation
			//wave = m_wav_samples[ipos1] * (1.0f - frac) + m_wav_samples[ipos2] * frac;

			// cubic interpolation
			int ipos0 = ipos1 - 1;
			if (ipos0 < 0) ipos0 = 0;

			int ipos3 = ipos1 + 2;
			if (ipos3 >= (int)wav->m_wav_length) ipos3 = (int)wav->m_wav_length - 1;

			float p0 = wav->m_wav_samples[ipos0];
			float p1 = wav->m_wav_samples[ipos1];
			float p2 = wav->m_wav_samples[ipos2];
			float p3 = wav->m_wav_samples[ipos3];

			wave = (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*powf(frac, 3.0f) +
				(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
				(-0.5f*p0 + 0.5f*p2)*frac + p1;
		}
		else
		{
			int ipos1 = (int)ceilf(((float)j - 0.5f)*sampleFreq / origin_SampleFreq);
			int ipos2 = (int)floorf(((float)j + 0.5f)*sampleFreq / origin_SampleFreq);
			if (ipos1 < 0) ipos1 = 0;
			if (ipos2 >= (int)wav->m_wav_length) ipos2 = (int)wav->m_wav_length - 1;
			int count = ipos2 - ipos1 + 1;
			float sum = 0.0f;
			for (int ipos = ipos1; ipos <= ipos2; ipos++)
			{
				sum += wav->m_wav_samples[ipos];
			}
			wave = sum / (float)count;
		}

		noteBuf->m_data[j] = wave*mult;
	}
}

void InstrumentMultiSampler::_interpolateBuffers(const float* src1, const float* src2, float* dst, unsigned length, float freq1, float freq2, float freq)
{
	float k2 = logf(freq / freq1) / logf(freq2 / freq1);
	float k1 = 1.0f - k2;

	for (unsigned j = 0; j <length; j++)
	{
		dst[j] = k1*src1[j] + k2*src2[j];
	}	
}


void InstrumentMultiSampler::GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf)
{
	if (m_SampleWavList == nullptr) return;

	std::vector<InstrumentSample_deferred>& sampleList = *m_SampleWavList;
	if (sampleList.size() < 1) return;

	bool useSingle = false;
	unsigned I;

	{
		InstrumentSample_deferred wav = sampleList[0];
		float origin_SampleFreq = wav->m_origin_freq / (float)wav->m_origin_sample_rate;

		if (sampleFreq <= origin_SampleFreq)
		{
			I = 0;
			useSingle = true;
		}
	}

	if (!useSingle)
	{
		InstrumentSample_deferred wav = sampleList[sampleList.size() - 1];
		float origin_SampleFreq = wav->m_origin_freq / (float)wav->m_origin_sample_rate;

		if (sampleFreq >= origin_SampleFreq)
		{
			I = (unsigned)(sampleList.size() - 1);
			useSingle = true;
		}
	}

	if (!useSingle)
	{
		for (size_t i = 0; i < sampleList.size() - 1; i++)
		{
			InstrumentSample_deferred wav = sampleList[i + 1];
			float origin_SampleFreq = wav->m_origin_freq / (float)wav->m_origin_sample_rate;

			if (sampleFreq == origin_SampleFreq)
			{
				I = (unsigned)(i+1);
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

	
	InstrumentSample_deferred wav1 = sampleList[I];
	InstrumentSample_deferred wav2 = sampleList[I + 1];
	float origin_SampleFreq1 = wav1->m_origin_freq / (float)wav1->m_origin_sample_rate;
	float origin_SampleFreq2 = wav2->m_origin_freq / (float)wav2->m_origin_sample_rate;

	if (useSingle)
	{
		NoteBuffer tmpBuffer;
		_generateNoteWave(I, fNumOfSamples, sampleFreq, &tmpBuffer);

		noteBuf->m_sampleNum = tmpBuffer.m_sampleNum;
		noteBuf->Allocate();

		for (unsigned j = 0; j < noteBuf->m_sampleNum; j++)
		{
			float x2 = (float)j / (float)noteBuf->m_sampleNum;
			float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);

			noteBuf->m_data[j] = amplitude*tmpBuffer.m_data[j];
		}
	}
	else
	{
		NoteBuffer tmpBuffer1;
		_generateNoteWave(I, fNumOfSamples, sampleFreq, &tmpBuffer1);

		NoteBuffer tmpBuffer2;
		_generateNoteWave(I+1, fNumOfSamples, sampleFreq, &tmpBuffer2);

		unsigned minLength = min(tmpBuffer1.m_sampleNum, tmpBuffer2.m_sampleNum);

		noteBuf->m_sampleNum = minLength;
		noteBuf->Allocate();

		_interpolateBuffers(tmpBuffer1.m_data, tmpBuffer2.m_data, noteBuf->m_data, minLength, origin_SampleFreq1, origin_SampleFreq2, sampleFreq);

		for (unsigned j = 0; j < noteBuf->m_sampleNum; j++)
		{
			float x2 = (float)j / (float)noteBuf->m_sampleNum;
			float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);

			noteBuf->m_data[j] *= amplitude;
		}
	}

}

