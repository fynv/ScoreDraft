#pragma once

struct Sample
{
	Sample() {}
	virtual ~Sample() {}

	unsigned m_wav_length;
	unsigned m_chn;
	float *m_wav_samples;
	float m_max_v;
	unsigned m_origin_sample_rate;
};

struct InstrumentSample : public Sample
{
	float m_origin_freq;
};


