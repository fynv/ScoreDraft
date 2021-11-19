#pragma once 

#include <cstdio>
#include <vector>
#include "WavBuffer.h"


class SCOREDRAFTCORE_API TrackBuffer
{
public:
	TrackBuffer(unsigned rate = 44100, unsigned chn = 1);
	~TrackBuffer();

	unsigned Rate() const { return m_rate; }
	void SetRate(unsigned rate) { m_rate = rate; }

	unsigned NumberOfChannels() const { return m_chn; }

	unsigned NumberOfSamples()
	{
		return m_length;
	}
	unsigned AlignPos()
	{
		return m_alignPos;
	}

	float Volume() const { return m_volume; }
	float AbsoluteVolume()
	{
		float maxValue = MaxValue();
		return maxValue > 0.0f ? m_volume / maxValue : 1.0f;
	}
	void SetVolume(float vol) { m_volume = vol; }

	float Pan() const { return m_pan; }
	void SetPan(float pan) { m_pan = pan; }

	float GetCursor();
	void SetCursor(float fpos);
	void MoveCursor(float delta);

	void SeekToCursor();
	void WriteBlend(const WavBuffer& wavBuf);

	void Sample(unsigned index, float* sample);
	float MaxValue();

	void GetSamples(unsigned startIndex, unsigned length, float* buffer);

	bool CombineTracks(unsigned num, TrackBuffer** tracks);

	unsigned GetLocalBufferSize();

private:
	FILE *m_fp;

	unsigned m_rate;
	unsigned m_chn;

	float m_volume;
	float m_pan;

	float *m_localBuffer;
	unsigned m_localBufferPos;

	unsigned m_length;
	unsigned m_alignPos;

	float m_cursor;

	inline float _ms2sample(float ms)
	{
		return ms * 0.001f*m_rate;
	}

	void _writeSamples(unsigned count, const float* samples, unsigned alignPos);
	void _seek(unsigned upos);
};


