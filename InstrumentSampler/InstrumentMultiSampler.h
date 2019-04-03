#ifndef _InstrumentMultiSampler_h
#define _InstrumentMultiSampler_h

#include <vector>

#include "TrackBuffer.h"
#include "Instrument.h"
#include "InstrumentSample.h"

#include <Deferred.h>

typedef Deferred<InstrumentSample> InstrumentSample_deferred;

class InstrumentMultiSampler : public Instrument
{
public:
	InstrumentMultiSampler();
	virtual ~InstrumentMultiSampler();

	void SetSampleList(std::vector<InstrumentSample_deferred>* sampleWavList)
	{
		m_chn = (*sampleWavList)[0]->m_chn;
		for (unsigned i = 1; i < (unsigned)sampleWavList->size(); i++)
			if ((*sampleWavList)[i]->m_chn != m_chn) return;
		m_SampleWavList = sampleWavList;
	}

protected:
	virtual void GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf);

private:
	void _generateNoteWave(unsigned index, float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf);
	void _interpolateBuffers(const float* src1, const float* src2, float* dst, unsigned length, float freq1, float freq2, float freq);

	unsigned m_chn;
	std::vector<InstrumentSample_deferred>* m_SampleWavList;

};


#endif
