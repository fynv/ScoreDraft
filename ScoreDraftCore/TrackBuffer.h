#ifndef _scoredraft_TrackBuffer_h
#define _scoredraft_TrackBuffer_h

#include "stdio.h"
#include <utility>
#include <vector>

inline void CalcPan(float pan, float& l, float& r)
{
	if (pan == 0.0f) return;
	else if (pan < 0.0f)
	{
		pan = -pan;
		float ll = l;
		float rl = r*pan;
		float rr = r*(1.0f - pan);
		l = ll + rl;
		r = rr;
	}
	else
	{
		float ll = l*(1.0f - pan);
		float lr = l*pan;
		float rr = r;
		l = ll;
		r = lr + rr;
	}
}

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
	float m_pan;

	void Allocate();
};

typedef std::vector<std::pair<int, float>> TempoMap;
inline float GetTempoMap(const TempoMap& tMap, int beat48)
{
	for (unsigned i = 1; i < tMap.size(); i++)
	{
		if (beat48 < tMap[i].first || i == tMap.size()-1)
			return  ((float)beat48 - (float)tMap[i - 1].first) / ((float)tMap[i].first - (float)tMap[i - 1].first) * (tMap[i].second - (float)tMap[i - 1].second) + (float)tMap[i - 1].second;			
	}
	return 0.0f;
}

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

	float Pan() const { return m_pan; }
	void SetPan(float pan) { m_pan = pan; }

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

	void _writeSamples(unsigned count, const float* samples, unsigned alignPos);
	void _seek(unsigned upos);
};

#endif 