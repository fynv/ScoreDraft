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
class SingingPiece;
class SingingSequence;
class RapPiece;
class RapSequence;

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

	void SingPiece(TrackBuffer& buffer, const SingingPiece& piece, unsigned tempo = 80, float RefFreq = 261.626f);
	void SingSequence(TrackBuffer& buffer, const SingingSequence& seq, unsigned tempo = 80, float RefFreq = 261.626f);

	void RapAPiece(TrackBuffer& buffer, const RapPiece& piece, unsigned tempo = 80, float RefFreq = 261.626f);
	void RapASequence(TrackBuffer& buffer, const RapSequence& seq, unsigned tempo = 80, float RefFreq = 261.626f);

	std::string GetLyricCharset()
	{
		return m_lyric_charset;
	}

	virtual bool Tune(const char* cmd);

protected:
	void Silence(unsigned numOfSamples, VoiceBuffer* noteBuf);
	virtual void GenerateWave(const char* lyric, std::vector<SingerNoteParams> notes, VoiceBuffer* noteBuf);
	virtual void GenerateWave_Rap(const char* lyric, float fNumOfSamples, float baseSampleFreq, int tone, VoiceBuffer* noteBuf);

	float m_noteVolume;
	std::string m_defaultLyric;
	std::string m_lyric_charset;

	float m_freq_rel_rap;
};

#endif
