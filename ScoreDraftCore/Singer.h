#ifndef _scoredraft_Singer_h
#define _scoredraft_Singer_h

#include <vector>
#include <string>
#include <Deferred.h>

class VoiceBuffer
{
public:
	VoiceBuffer();
	~VoiceBuffer();

	float m_sampleRate;
	unsigned m_sampleNum;
	unsigned m_alignPos;
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

struct SingingPieceInternal
{
	std::string lyric;
	std::vector<SingerNoteParams> notes;
};

typedef Deferred<SingingPieceInternal> SingingPieceInternal_Deferred;
typedef std::vector<SingingPieceInternal_Deferred> SingingPieceInternalList;

struct RapPieceInternal
{
	std::string lyric;
	float fNumOfSamples;
	float baseSampleFreq;
	int tone;
};

typedef Deferred<RapPieceInternal> RapPieceInternal_Deferred;
typedef std::vector<RapPieceInternal_Deferred> RapPieceInternalList;

class Singer
{
public:
	Singer();
	~Singer();

	void SingPiece(TrackBuffer& buffer, const SingingPiece& piece, unsigned tempo = 80, float RefFreq = 261.626f);
	void SingSequence(TrackBuffer& buffer, const SingingSequence& seq, unsigned tempo = 80, float RefFreq = 261.626f);

	void RapAPiece(TrackBuffer& buffer, const RapPiece& piece, unsigned tempo = 80, float RefFreq = 261.626f);
	void RapASequence(TrackBuffer& buffer, const RapSequence& seq, unsigned tempo = 80, float RefFreq = 261.626f);

	void SingConsecutivePieces(TrackBuffer& buffer, const SingingSequence& pieces, unsigned tempo = 80, float RefFreq = 261.626f);
	void RapConsecutivePieces(TrackBuffer& buffer, const RapSequence& pieces, unsigned tempo = 80, float RefFreq = 261.626f);

	std::string GetLyricCharset()
	{
		return m_lyric_charset;
	}

	virtual bool Tune(const char* cmd);

protected:
	void Silence(unsigned numOfSamples, VoiceBuffer* noteBuf);
	virtual void GenerateWave(SingingPieceInternal piece, VoiceBuffer* noteBuf);
	virtual void GenerateWave_Rap(RapPieceInternal piece, VoiceBuffer* noteBuf);

	virtual void GenerateWave_SingConsecutive(SingingPieceInternalList pieceList, VoiceBuffer* noteBuf);
	virtual void GenerateWave_RapConsecutive(RapPieceInternalList pieceList, VoiceBuffer* noteBuf);

	float m_noteVolume;
	std::string m_defaultLyric;
	std::string m_lyric_charset;

	float m_freq_rel_rap;
};

#endif
