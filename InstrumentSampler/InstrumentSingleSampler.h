#ifndef _InstrumentSingleSampler_h
#define _InstrumentSingleSampler_h

#include "PyScoreDraft.h"

class InstrumentSample;

class InstrumentSingleSampler : public Instrument
{
public:
	InstrumentSingleSampler();
	~InstrumentSingleSampler();

	void SetSample(InstrumentSample* sample)
	{
		m_sample = sample;
	}

protected:
	virtual void GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf);

private:
	InstrumentSample *m_sample;
};


#endif 

