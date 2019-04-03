#ifndef _scoredraft_Percussion_h
#define _scoredraft_Percussion_h

#include <vector>
#include <string>

#include<string>

class NoteBuffer;
class TrackBuffer;
class BeatSequence;
class Percussion;

typedef std::vector<std::pair<int, float>> TempoMap;

class Percussion
{
public:
	Percussion();
	virtual ~Percussion();

	void PlayBeat(TrackBuffer& buffer, int duration, unsigned tempo = 80);
	void PlayBeat(TrackBuffer& buffer, int duration, const TempoMap& tempoMap, int tempoMapOffset = 0, float RefFreq = 261.626f);
	static void PlaySilence(TrackBuffer& buffer, int duration, unsigned tempo = 80);
	static void PlaySilence(TrackBuffer& buffer, int duration, const TempoMap& tempoMap, int tempoMapOffset = 0);
	static void PlayBackspace(TrackBuffer& buffer, int duration, unsigned tempo = 80);
	static void PlayBackspace(TrackBuffer& buffer, int duration, const TempoMap& tempoMap, int tempoMapOffset = 0);

	virtual bool Tune(const char* cmd);

	void SetBeatVolume(float volume) { m_beatVolume = volume; }
	void SetBeatPan(float pan) { m_beatPan = pan; }


protected:
	static void Silence(unsigned numOfSamples, NoteBuffer* noteBuf);
	virtual void GenerateBeatWave(float fNumOfSamples, NoteBuffer* beatBuf);

	float m_beatVolume;
	float m_beatPan;
};


#endif
