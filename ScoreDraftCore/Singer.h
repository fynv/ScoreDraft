#ifndef _scoredraft_Singer_h
#define _scoredraft_Singer_h

#include <vector>
#include <string>
#include <Deferred.h>

class NoteBuffer;
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

	bool isVowel;
};

typedef Deferred<SingingPieceInternal> SingingPieceInternal_Deferred;
typedef std::vector<SingingPieceInternal_Deferred> SingingPieceInternalList;

struct RapPieceInternal
{
	std::string lyric;
	float fNumOfSamples;
	float sampleFreq1;
	float sampleFreq2;

	bool isVowel;
};

typedef Deferred<RapPieceInternal> RapPieceInternal_Deferred;
typedef std::vector<RapPieceInternal_Deferred> RapPieceInternalList;

class Singer
{
public:
	Singer();
	virtual ~Singer();

	void SingPiece(TrackBuffer& buffer, const SingingPiece& piece, unsigned tempo = 80, float RefFreq = 261.626f);
	void RapAPiece(TrackBuffer& buffer, const RapPiece& piece, unsigned tempo = 80, float RefFreq = 261.626f);

	void SingConsecutivePieces(TrackBuffer& buffer, const SingingSequence& pieces, unsigned tempo = 80, float RefFreq = 261.626f);
	void RapConsecutivePieces(TrackBuffer& buffer, const RapSequence& pieces, unsigned tempo = 80, float RefFreq = 261.626f);

	std::string GetLyricCharset()
	{
		return m_lyric_charset;
	}

	virtual bool Tune(const char* cmd);

protected:
	void Silence(unsigned numOfSamples, NoteBuffer* noteBuf);
	virtual void GenerateWave(SingingPieceInternal piece, NoteBuffer* noteBuf);
	virtual void GenerateWave_Rap(RapPieceInternal piece, NoteBuffer* noteBuf);

	virtual void GenerateWave_SingConsecutive(SingingPieceInternalList pieceList, NoteBuffer* noteBuf);
	virtual void GenerateWave_RapConsecutive(RapPieceInternalList pieceList, NoteBuffer* noteBuf);

	float m_noteVolume;
	float m_notePan;

	std::string m_defaultLyric;
	std::string m_lyric_charset;

};

#endif
