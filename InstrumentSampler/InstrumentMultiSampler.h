#ifndef _InstrumentMultiSampler_h
#define _InstrumentMultiSampler_h

#include "PyScoreDraft.h"
#include <vector>

#include "InstrumentSample.h"

#include <Deferred.h>

typedef Deferred<InstrumentSample> InstrumentSample_deferred;

class InstrumentMultiSampler : public Instrument
{
public:
	InstrumentMultiSampler();
	~InstrumentMultiSampler();

	void SetSampleList(std::vector<InstrumentSample_deferred>* sampleWavList)
	{
		m_SampleWavList = sampleWavList;
	}

private:
	void _generateNoteWave(unsigned index, float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf);

	void _interpolateBuffers(const float* src1, const float* src2, float* dst, unsigned length, float freq1, float freq2, float freq);

	virtual void GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf);


	std::vector<InstrumentSample_deferred>* m_SampleWavList;

};


#endif
