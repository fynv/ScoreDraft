#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define SCOREDRAFT_API __declspec(dllexport)
#else
#define SCOREDRAFT_API 
#endif

extern "C"
{
	SCOREDRAFT_API void* SampleCreate(unsigned origin_sample_rate, unsigned chn, void* ptr_f32_buf, float max_v);
	SCOREDRAFT_API void* InstrumentSampleCreate(unsigned origin_sample_rate, unsigned chn, void* ptr_f32_buf, float max_v, float origin_freq);
	SCOREDRAFT_API void SampleDestroy(void *ptr);
	SCOREDRAFT_API void PercussionGenerate(void* ptr_wavbuf, void* ptr_sample, float fduration);
	SCOREDRAFT_API void InstrumentSingleGenerate(void* ptr_wavbuf, void* ptr_sample, float freq, float fduration);
	SCOREDRAFT_API void InstrumentMultiGenerate(void* ptr_wavbuf, void* ptr_sample_lst, float freq, float fduration);
}

#include <algorithm>
#include <utils.h>
#include "Sample.h"
#include "FrequencyDetection.h"
#include "PercussionSampler.h"
#include "InstrumentSingleSampler.h"
#include "InstrumentMultiSampler.h"

void CreateSample(Sample* sample, unsigned origin_sample_rate, unsigned chn, void* ptr_f32_buf, float max_v)
{
	F32Buf* buf = (F32Buf*)ptr_f32_buf;	
	if (max_v <= 0.0f)
	{
		max_v = 0.0f;
		for (size_t i = 0; i < buf->size(); i++)
		{
			float v = fabsf((*buf)[i]);
			if (v > max_v) max_v = v;
		}
	}

	sample->m_wav_length = (unsigned)buf->size()/chn;
	sample->m_chn = chn;
	sample->m_wav_samples = buf->data();
	sample->m_max_v = max_v;
	sample->m_origin_sample_rate = origin_sample_rate;		
}

void* SampleCreate(unsigned origin_sample_rate, unsigned chn, void* ptr_f32_buf, float max_v)
{
	Sample* sample = new Sample;
	CreateSample(sample, origin_sample_rate, chn, ptr_f32_buf, max_v);
	return sample;
}


static float s_DetectBaseFreq(const Sample& sample)
{
	float* localMono = nullptr;
	float* pSamples = nullptr;

	if (sample.m_chn == 1)
	{
		pSamples = sample.m_wav_samples;
	}
	else if (sample.m_chn == 2)
	{
		localMono = new float[sample.m_wav_length];
		pSamples = localMono;
		for (unsigned i = 0; i < sample.m_wav_length; i++)
		{
			localMono[i] = 0.5f*(sample.m_wav_samples[i * 2] + sample.m_wav_samples[i * 2 + 1]);
		}
	}
	Buffer buf;
	buf.m_size = sample.m_wav_length;
	buf.m_data = pSamples;
	float baseFreq = fetchFrequency(buf, sample.m_origin_sample_rate);
	delete[] localMono;

	return baseFreq;
}


void* InstrumentSampleCreate(unsigned origin_sample_rate, unsigned chn, void* ptr_f32_buf, float max_v, float origin_freq)
{
	InstrumentSample* sample = new InstrumentSample;
	CreateSample(sample, origin_sample_rate, chn, ptr_f32_buf, max_v);
	if (origin_freq <= 0.0f)
	{
		sample->m_origin_freq = s_DetectBaseFreq(*sample);
	}
	else
	{
		sample->m_origin_freq = origin_freq;
	}
	return sample;
}

void SampleDestroy(void *ptr)
{
	delete (Sample*)ptr;
}

void PercussionGenerate(void* ptr_wavbuf, void* ptr_sample, float fduration)
{
	WavBuffer* wavbuf = (WavBuffer*)ptr_wavbuf;
	Sample* sample = (Sample*)ptr_sample;

	float sampleRate = wavbuf->m_sampleRate;

	float fNumOfSamples = fduration * sampleRate*0.001f;
	size_t len = (size_t)ceilf(fNumOfSamples);
	
	wavbuf->Allocate(sample->m_chn, len);

	PercussionSample(*sample, wavbuf->m_data, (unsigned)len, sampleRate / (float)sample->m_origin_sample_rate);
}

void InstrumentSingleGenerate(void* ptr_wavbuf, void* ptr_sample, float freq, float fduration)
{
	WavBuffer* wavbuf = (WavBuffer*)ptr_wavbuf;
	InstrumentSample* sample = (InstrumentSample*)ptr_sample;

	float sampleRate = wavbuf->m_sampleRate;

	float fNumOfSamples = fduration * sampleRate*0.001f;
	size_t len = (size_t)ceilf(fNumOfSamples);
	float sampleFreq = freq / sampleRate;

	wavbuf->Allocate(sample->m_chn, len);

	InstrumentSingleSample(*sample, wavbuf->m_data, (unsigned)len, sampleFreq);
}

static int compareSampleWav(const void* a, const void* b)
{
	InstrumentSample& wavA = *((InstrumentSample*)a);
	InstrumentSample& wavB = *((InstrumentSample*)b);

	float origin_SampleFreqA = wavA.m_origin_freq / (float)wavA.m_origin_sample_rate;
	float origin_SampleFreqB = wavB.m_origin_freq / (float)wavB.m_origin_sample_rate;

	return origin_SampleFreqA > origin_SampleFreqB ? 1 : -1;
}

void InstrumentMultiGenerate(void* ptr_wavbuf, void* ptr_sample_lst, float freq, float fduration)
{
	WavBuffer* wavbuf = (WavBuffer*)ptr_wavbuf;
	PtrArray* sampleList = (PtrArray*)ptr_sample_lst;	

	unsigned chn = 0;
	std::vector<InstrumentSample> samples;
	for (size_t i = 0; i < sampleList->size(); i++)
	{		
		InstrumentSample* sample = (InstrumentSample*)(*sampleList)[i];	

		if (i == 0)
			chn = sample->m_chn;
		else
			if (chn != sample->m_chn)
			{
				printf("All samples does not have the same number of channels\n");
				return;
			}

		samples.push_back(*sample);
	}
	std::qsort(samples.data(), samples.size(), sizeof(InstrumentSample), compareSampleWav);

	float sampleRate = wavbuf->m_sampleRate;

	float fNumOfSamples = fduration * sampleRate*0.001f;
	size_t len = (size_t)ceilf(fNumOfSamples);
	float sampleFreq = freq / sampleRate;

	wavbuf->Allocate(chn, len);

	InstrumentMultiSample(samples, wavbuf->m_data, (unsigned)len, sampleFreq);
}
