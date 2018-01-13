#ifndef _scoredraft_TrackBuffer_h
#define _scoredraft_TrackBuffer_h

#include "stdio.h"
#include "Deferred.h"

class TrackBuffer;
typedef Deferred<TrackBuffer> TrackBuffer_deferred;

class TrackBuffer
{
public:
	TrackBuffer(unsigned rate=44100);
	~TrackBuffer();

	unsigned Rate() const { return m_rate; }
	void SetRate(unsigned rate) { m_rate = rate; }
	float Volume() const  { return m_volume; }
	float AbsoluteVolume() 
	{
		float maxValue = MaxValue();
		return maxValue>0.0f? m_volume / maxValue: 1.0f;
	}
	void SetVolume(float vol) {	m_volume = vol;	}

	/// block read-write
	float GetCursor();
	void SetCursor(float fpos);
	void MoveCursor(float delta);

	void WriteBlend(unsigned count, const float* samples, float cursorDelta, unsigned alignPos=0);

	// sample read
	unsigned NumberOfSamples()
	{
		return m_length;
	}
	unsigned AlignPos()
	{
		return m_alignPos;
	}

	float Sample(unsigned index);
	float MaxValue();

	// buffer read
	void GetSamples(unsigned startIndex, unsigned length, float* buffer);
		
	static bool CombineTracks(TrackBuffer& sumbuffer, unsigned num, TrackBuffer_deferred* tracks);
	unsigned GetLocalBufferSize();

private:
	unsigned m_rate;
	FILE *m_fp;

	float m_volume;

	float *m_localBuffer;
	unsigned m_localBufferPos;

	unsigned m_length;
	unsigned m_alignPos;

	float m_cursor;

	void _writeSamples(unsigned count, const float* samples, unsigned alignPos);
	void _seek(unsigned upos);
};


#endif 