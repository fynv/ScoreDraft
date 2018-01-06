#include "InstrumentMultiSampler.h"

#include <string.h>
#include <math.h>
#include <ReadWav.h>
#include <float.h>

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

}

InstrumentMultiSampler::~InstrumentMultiSampler()
{
	for (size_t i = 0; i < m_SampleWavList.size();i++)
		delete[] m_SampleWavList[i].m_wav_samples;
}

bool InstrumentMultiSampler::LoadWav(const char* instrument_name, const char* filename)
{
	char wavFilename[1024];
	sprintf(wavFilename, "InstrumentSamples/%s/%s.wav", instrument_name, filename);

	SampleWav wav;

	ReadWav reader;
	reader.OpenFile(wavFilename);
	if (!reader.ReadHeader(wav.m_origin_sample_rate, wav.m_wav_length)) return false;

	wav.m_wav_samples = new float[wav.m_wav_length];
	if (!reader.ReadSamples(wav.m_wav_samples, wav.m_wav_length, wav.m_max_v))
	{
		delete[] wav.m_wav_samples;
		return false;
	}
	wav._fetchOriginFreq(instrument_name, filename);

	m_SampleWavList.push_back(wav);

	return true;
}


void InstrumentMultiSampler::SampleWav::_fetchOriginFreq(const char* instrument_name, const char* filename)
{
	char freqFilename[1024];
	sprintf(freqFilename, "InstrumentSamples/%s/%s.freq", instrument_name, filename);

	FILE *fp = fopen(freqFilename, "r");
	if (fp)
	{
		fscanf(fp, "%f", &m_origin_freq);
		fclose(fp);
	}
	else
	{
		m_origin_freq = fetchFrequency(m_wav_length, m_wav_samples, m_origin_sample_rate);
		printf("Detected frequency of %s.wav = %fHz\n", filename, m_origin_freq);
		fp = fopen(freqFilename, "w");
		fprintf(fp, "%f\n", m_origin_freq);
		fclose(fp);
	}
}

void InstrumentMultiSampler::_generateNoteWave(unsigned index, float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf)
{
	SampleWav& wav = m_SampleWavList[index];

	float origin_SampleFreq = wav.m_origin_freq / (float)wav.m_origin_sample_rate;
	unsigned maxSample = (unsigned)((float)wav.m_wav_length*origin_SampleFreq / sampleFreq);

	noteBuf->m_sampleNum = min((unsigned)ceilf(fNumOfSamples), maxSample);
	noteBuf->Allocate();

	float mult = m_noteVolume / wav.m_max_v;

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
			if (ipos2 >= (int)wav.m_wav_length) ipos2 = (int)wav.m_wav_length - 1;

			// linear interpolation
			//wave = m_wav_samples[ipos1] * (1.0f - frac) + m_wav_samples[ipos2] * frac;

			// cubic interpolation
			int ipos0 = ipos1 - 1;
			if (ipos0 < 0) ipos0 = 0;

			int ipos3 = ipos1 + 2;
			if (ipos3 >= (int)wav.m_wav_length) ipos3 = (int)wav.m_wav_length - 1;

			float p0 = wav.m_wav_samples[ipos0];
			float p1 = wav.m_wav_samples[ipos1];
			float p2 = wav.m_wav_samples[ipos2];
			float p3 = wav.m_wav_samples[ipos3];

			wave = (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*powf(frac, 3.0f) +
				(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
				(-0.5f*p0 + 0.5f*p2)*frac + p1;
		}
		else
		{
			int ipos1 = (int)ceilf(((float)j - 0.5f)*sampleFreq / origin_SampleFreq);
			int ipos2 = (int)floorf(((float)j + 0.5f)*sampleFreq / origin_SampleFreq);
			if (ipos1 < 0) ipos1 = 0;
			if (ipos2 >= (int)wav.m_wav_length) ipos2 = (int)wav.m_wav_length - 1;
			int count = ipos2 - ipos1 + 1;
			float sum = 0.0f;
			for (int ipos = ipos1; ipos <= ipos2; ipos++)
			{
				sum += wav.m_wav_samples[ipos];
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
	if (m_SampleWavList.size() < 1) return;

	unsigned nearestI = 0;
	unsigned nearestI2 = 0;
	float nearestDiff = FLT_MAX;
	float nearestDiff2 = FLT_MAX;

	for (size_t i = 0; i < m_SampleWavList.size(); i++)
	{
		SampleWav& wav = m_SampleWavList[i];
		float origin_SampleFreq = wav.m_origin_freq / (float)wav.m_origin_sample_rate;
		float diff = fabsf(logf(sampleFreq) - logf(origin_SampleFreq));

		if (diff < nearestDiff)
		{
			nearestDiff2 = nearestDiff;
			nearestI2 = nearestI;

			nearestDiff = diff;
			nearestI = (unsigned)i;
		}
		else if (diff < nearestDiff2)
		{
			nearestDiff2 = diff;
			nearestI2 = (unsigned)i;
		}
	}

	SampleWav& wav1 = m_SampleWavList[nearestI];
	SampleWav& wav2 = m_SampleWavList[nearestI2];
	float origin_SampleFreq1 = wav1.m_origin_freq / (float)wav1.m_origin_sample_rate;
	float origin_SampleFreq2 = wav2.m_origin_freq / (float)wav2.m_origin_sample_rate;

	if (((origin_SampleFreq1 <= sampleFreq) && (origin_SampleFreq2 <= sampleFreq)) ||
		((origin_SampleFreq1 >= sampleFreq) && (origin_SampleFreq2 >= sampleFreq)))  /// Single Sample

	{
		NoteBuffer tmpBuffer;
		_generateNoteWave(nearestI, fNumOfSamples, sampleFreq, &tmpBuffer);

		noteBuf->m_sampleNum = tmpBuffer.m_sampleNum;
		noteBuf->Allocate();

		for (unsigned j = 0; j < noteBuf->m_sampleNum; j++)
		{
			float x2 = (float)j / fNumOfSamples;
			float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);

			noteBuf->m_data[j] = amplitude*tmpBuffer.m_data[j];
		}
	}
	else
	{
		NoteBuffer tmpBuffer1;
		_generateNoteWave(nearestI, fNumOfSamples, sampleFreq, &tmpBuffer1);

		NoteBuffer tmpBuffer2;
		_generateNoteWave(nearestI2, fNumOfSamples, sampleFreq, &tmpBuffer2);

		unsigned minLength = min(tmpBuffer1.m_sampleNum, tmpBuffer2.m_sampleNum);

		noteBuf->m_sampleNum = minLength;
		noteBuf->Allocate();

		_interpolateBuffers(tmpBuffer1.m_data, tmpBuffer2.m_data, noteBuf->m_data, minLength, origin_SampleFreq1, origin_SampleFreq2, sampleFreq);

		for (unsigned j = 0; j < noteBuf->m_sampleNum; j++)
		{
			float x2 = (float)j / fNumOfSamples;
			float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);

			noteBuf->m_data[j] *= amplitude;
		}


	}
}

