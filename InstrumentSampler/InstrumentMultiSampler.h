#ifndef _InstrumentMultiSampler_h
#define _InstrumentMultiSampler_h

#include "PyScoreDraft.h"
#include <vector>

class InstrumentMultiSampler : public Instrument
{
public:
	InstrumentMultiSampler();
	~InstrumentMultiSampler();

	bool LoadWav(const char* instrument_name, const char* filename);

private:
	void _generateNoteWave(unsigned index, float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf);

	void _interpolateBuffers(const float* src1, const float* src2, float* dst, unsigned length, float freq1, float freq2, float freq);

	virtual void GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf);

	struct SampleWav
	{
		SampleWav()
		{
			m_wav_length = 0;
			m_wav_samples = nullptr;
		}
		void _fetchOriginFreq(const char* instrument_name, const char* filename);

		unsigned m_wav_length;
		float *m_wav_samples;
		float m_max_v;
		float m_origin_freq;
		unsigned m_origin_sample_rate;
	};

	std::vector<SampleWav> m_SampleWavList;

	bool m_sorted;

	static int compareSampleWav(const void* a, const void* b);
	void _sort();

};


#endif
