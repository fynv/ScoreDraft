#ifndef _InstrumentSingleSampler_h
#define _InstrumentSingleSampler_h

#include "PyScoreDraft.h"

class InstrumentSingleSampler : public Instrument
{
public:
	InstrumentSingleSampler();
	~InstrumentSingleSampler();

	bool LoadWav(const char* name);

private:
	void _fetchOriginFreq(const char* name);
	virtual void GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf);

	unsigned m_wav_length;
	float *m_wav_samples;
	float m_max_v;
	float m_origin_freq;
	unsigned m_origin_sample_rate;

};


#endif 

