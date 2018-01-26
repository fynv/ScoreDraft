#ifndef _scoredraft_Percussion_h
#define _scoredraft_Percussion_h

#include<string>
#include "Deferred.h"

class BeatBuffer
{
public:
	BeatBuffer();
	~BeatBuffer();

	float m_sampleRate;
	unsigned m_sampleNum;
	unsigned m_alignPos;
	float* m_data;
	void Allocate();
};

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

	virtual bool Tune(const char* cmd);


protected:
	static void Silence(unsigned numOfSamples, BeatBuffer* noteBuf);
	virtual void GenerateBeatWave(float fNumOfSamples, BeatBuffer* beatBuf);

	float m_beatVolume;
};


#endif
