#ifndef _SentenceGenerator_PSOLA_h
#define _SentenceGenerator_PSOLA_h

#include "SentenceGenerator_CPU.h"

class SentenceGenerator_PSOLA : public SentenceGenerator_CPU
{
protected:
	virtual void GeneratePiece(bool isVowel, unsigned uSumLen, const float* freqMap, float& phase, Buffer& dstBuf, bool firstNote, bool hasNextNote, const SourceInfo& srcInfo, const SourceInfo& srcInfo_next, const SourceDerivedInfo& srcDerInfo);


};

#endif
