#ifndef _InstrumentSingleSampler_h
#define _InstrumentSingleSampler_h

#include "TrackBuffer.h"
#include "Instrument.h"

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

