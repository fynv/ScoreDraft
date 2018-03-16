#ifndef _scoredraft_Percussion_h
#define _scoredraft_Percussion_h

#include<string>
#include "Deferred.h"

class NoteBuffer;
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

	virtual bool Tune(const char* cmd);


protected:
	static void Silence(unsigned numOfSamples, NoteBuffer* noteBuf);
	virtual void GenerateBeatWave(float fNumOfSamples, NoteBuffer* beatBuf);

	float m_beatVolume;
	float m_beatPan;
};


#endif
