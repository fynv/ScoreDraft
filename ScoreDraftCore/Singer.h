#ifndef _scoredraft_Singer_h
#define _scoredraft_Singer_h

#include <vector>
#include <string>

class VoiceBuffer
{
public:
	VoiceBuffer();
	~VoiceBuffer();

	unsigned m_sampleNum;
	float* m_data;
	void Allocate();
};

class TrackBuffer;
class VoicePiece;
class VoiceSequence;

struct SingerNoteParams
{
	float fNumOfSamples;
	float sampleFreq; // periods/sample or 1/samplesPerPeriod
};

class Singer
{
public:
	Singer();
	~Singer();

	void SingPiece(TrackBuffer& buffer, const VoicePiece& piece, unsigned tempo = 80, float RefFreq = 261.626f);
	void SingSequence(TrackBuffer& buffer, const VoiceSequence& seq, unsigned tempo = 80, float RefFreq = 261.626f);

	virtual bool Tune(const char* cmd);

protected:
	void Silence(unsigned numOfSamples, VoiceBuffer* noteBuf);
	virtual void GenerateWave(const char* lyric, std::vector<SingerNoteParams> notes, VoiceBuffer* noteBuf);

	float m_noteVolume;
	std::string m_defaultLyric;
};

#endif
