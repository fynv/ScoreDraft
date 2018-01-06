#include "InstrumentSingleSampler.h"

#include <string.h>
#include <math.h>
#include <ReadWav.h>

#include "FrequencyDetection.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


InstrumentSingleSampler::InstrumentSingleSampler()
{
	m_wav_length = 0;
	m_wav_samples = nullptr;
}

InstrumentSingleSampler::~InstrumentSingleSampler()
{
	delete[] m_wav_samples;
}

bool InstrumentSingleSampler::LoadWav(const char* name)
{
	char filename[1024];
	sprintf(filename, "InstrumentSamples/%s.wav", name);

	delete[] m_wav_samples;
	m_wav_length = 0;
	m_wav_samples = nullptr;

	ReadWav reader;
	reader.OpenFile(filename);
	if (!reader.ReadHeader(m_origin_sample_rate, m_wav_length)) return false;

	m_wav_samples = new float[m_wav_length];
	if (!reader.ReadSamples(m_wav_samples, m_wav_length, m_max_v))
	{
		delete[] m_wav_samples;
		return false;
	}

	_fetchOriginFreq(name);

	return true;
}

void InstrumentSingleSampler::_fetchOriginFreq(const char* name)
{
	char filename[1024];
	sprintf(filename, "InstrumentSamples/%s.freq", name);

	FILE *fp = fopen(filename, "r");
	if (fp)
	{
		fscanf(fp, "%f", &m_origin_freq);
		fclose(fp);
	}
	else
	{
		m_origin_freq = fetchFrequency(m_wav_length, m_wav_samples, m_origin_sample_rate);
		printf("Detected frequency of %s.wav = %fHz\n", name, m_origin_freq);
		fp = fopen(filename, "w");
		fprintf(fp, "%f\n", m_origin_freq);
		fclose(fp);
	}
}

void InstrumentSingleSampler::GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf)
{
	float origin_SampleFreq = m_origin_freq / (float)m_origin_sample_rate;
	unsigned maxSample = (unsigned)((float)m_wav_length*origin_SampleFreq / sampleFreq);

	noteBuf->m_sampleNum = min((unsigned)ceilf(fNumOfSamples), maxSample);
	noteBuf->Allocate();

	float mult = m_noteVolume / m_max_v;

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
			if (ipos2 >= (int)m_wav_length) ipos2 = (int)m_wav_length - 1;

			// linear interpolation
			//wave = m_wav_samples[ipos1] * (1.0f - frac) + m_wav_samples[ipos2] * frac;

			// cubic interpolation
			int ipos0 = ipos1 - 1;
			if (ipos0 < 0) ipos0 = 0;

			int ipos3 = ipos1 + 2;
			if (ipos3 >= (int)m_wav_length) ipos3 = (int)m_wav_length - 1;

			float p0 = m_wav_samples[ipos0];
			float p1 = m_wav_samples[ipos1];
			float p2 = m_wav_samples[ipos2];
			float p3 = m_wav_samples[ipos3];

			wave = (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*powf(frac, 3.0f) +
				(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
				(-0.5f*p0 + 0.5f*p2)*frac + p1;
		}
		else
		{
			int ipos1 = (int)ceilf(((float)j - 0.5f)*sampleFreq / origin_SampleFreq);
			int ipos2 = (int)floorf(((float)j + 0.5f)*sampleFreq / origin_SampleFreq);
			if (ipos1 < 0) ipos1 = 0;
			if (ipos2 >= (int)m_wav_length) ipos2 = (int)m_wav_length - 1;
			int count = ipos2 - ipos1 + 1;
			float sum = 0.0f;
			for (int ipos = ipos1; ipos <= ipos2; ipos++)
			{
				sum += m_wav_samples[ipos];
			}
			wave = sum / (float)count;
		}

		noteBuf->m_data[j] = amplitude*wave*mult;
	}
}
