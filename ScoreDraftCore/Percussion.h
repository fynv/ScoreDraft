#ifndef _scoredraft_Percussion_h
#define _scoredraft_Percussion_h

#include<string>
#include "Deferred.h"

class BeatBuffer
{
public:
	BeatBuffer();
	~BeatBuffer();

	unsigned m_sampleNum;
	float* m_data;
	void Allocate();
};

class BeatTable;
class TrackBuffer;
class BeatSequence;
class Percussion;
typedef Deferred<Percussion> Percussion_deferred;

class Percussion
{
public:
	Percussion();
	~Percussion();

	void PlayBeat(TrackBuffer& buffer, int duration, unsigned tempo = 80);
	static void PlaySilence(TrackBuffer& buffer, int duration, unsigned tempo = 80);
	static void PlayBackspace(TrackBuffer& buffer, int duration, unsigned tempo = 80);
	static void PlayBeats(TrackBuffer& buffer, Percussion_deferred* percussionList, const BeatSequence& seq, unsigned tempo = 80);

	virtual void Tune(std::string nob, float value) {}


protected:
	static void Silence(unsigned numOfSamples, BeatBuffer* noteBuf);
	virtual void GenerateBeatWave(unsigned numOfSamples, BeatBuffer* beatBuf, float BufferSampleRate);

	// acceleration
	bool m_accelerate;
	BeatTable* m_BeatTable;
};


#endif
