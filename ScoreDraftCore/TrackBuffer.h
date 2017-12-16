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

	unsigned Rate() const;
	float Volume() const;
	void SetVolume(float vol);

	/// block read-write
	void SeekSample(long offset, int origin);
	long Tell();
	void WriteSamples(unsigned count, const float* samples);	
	void ReadSamples(unsigned count, float* samples);

	void WriteBlend(unsigned count, const float* samples);	

	// sample read
	unsigned NumberOfSamples();
	float Sample(int index);
	float MaxValue();
		
	static bool CombineTracks(TrackBuffer& sumbuffer, unsigned num, TrackBuffer_deferred* tracks);

private:
	unsigned m_rate;
	FILE *m_fp;

	float m_volume;

	float *m_localBuffer;
	unsigned m_curPos;
};


#endif 