#ifndef _scoredraft_TrackBuffer_h
#define _scoredraft_TrackBuffer_h

#include "stdio.h"
#include "Deferred.h"

class NoteBuffer
{
public:
	NoteBuffer();
	~NoteBuffer();

	float m_sampleRate;
	unsigned m_channelNum;
	unsigned m_sampleNum;
	float* m_data;

	float m_cursorDelta;
	unsigned m_alignPos;
	float m_volume;

	void Allocate();
};

class TrackBuffer;
class TrackBuffer_deferred : public Deferred<TrackBuffer>
{
public:
	TrackBuffer_deferred();
	TrackBuffer_deferred(const TrackBuffer_deferred & in);
	TrackBuffer_deferred(unsigned rate, unsigned chn=1);
};

class TrackBuffer
{
public:
	TrackBuffer(unsigned rate = 44100, unsigned chn = 1);
	~TrackBuffer();

	unsigned Rate() const { return m_rate; }
	void SetRate(unsigned rate) { m_rate = rate; }

	unsigned NumberOfChannels() const { return m_chn; }	

	float Volume() const  { return m_volume; }
	float AbsoluteVolume()
	{
		float maxValue = MaxValue();
		return maxValue>0.0f ? m_volume / maxValue : 1.0f;
	}
	void SetVolume(float vol) { m_volume = vol; }

	float GetCursor();
	void SetCursor(float fpos);
	void MoveCursor(float delta);

	void SeekToCursor();

	void WriteBlend(const NoteBuffer& noteBuf);

	unsigned NumberOfSamples()
	{
		return m_length;
	}
	unsigned AlignPos()
	{
		return m_alignPos;
	}

	void Sample(unsigned index, float* sample);
	float MaxValue();

	void GetSamples(unsigned startIndex, unsigned length, float* buffer);

	bool CombineTracks(unsigned num, TrackBuffer_deferred* tracks);
	unsigned GetLocalBufferSize();

private:
	FILE *m_fp;

	unsigned m_rate;
	unsigned m_chn;

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