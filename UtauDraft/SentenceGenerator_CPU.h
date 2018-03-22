#ifndef _SentenceGenerator_CPU_h
#define _SentenceGenerator_CPU_h

#include "UtauDraft.h"

class SentenceGenerator_CPU : public SentenceGenerator
{
public:
	virtual void GenerateSentence(const UtauSourceFetcher& srcFetcher, unsigned numPieces, const std::string* lyrics, const unsigned* isVowel, const unsigned* lengths, const float *freqAllMap, NoteBuffer* noteBuf);

protected:
	virtual void GeneratePiece(bool isVowel, unsigned uSumLen, const float* freqMap, float& phase, Buffer& dstBuf, bool firstNote, bool hasNextNote, const SourceInfo& srcInfo, const SourceInfo& srcInfo_next, const SourceDerivedInfo& srcDerInfo) = 0;

private:
	void _generatePiece(const UtauSourceFetcher& srcFetcher, const char* lyric, const char* lyric_next, unsigned uSumLen, const float* freqMap, NoteBuffer* noteBuf, unsigned noteBufPos, float& phase, bool firstNote, bool isVowel);
};

#endif
