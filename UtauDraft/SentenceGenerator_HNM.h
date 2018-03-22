#ifndef _SentenceGenerator_HNM_h
#define _SentenceGenerator_HNM_h

#include "SentenceGenerator_CPU.h"

class SentenceGenerator_HNM : public SentenceGenerator_CPU
{
protected:
	virtual void GeneratePiece(bool isVowel, unsigned uSumLen, const float* freqMap, float& phase, Buffer& dstBuf, bool firstNote, bool hasNextNote, const SourceInfo& srcInfo, const SourceInfo& srcInfo_next, const SourceDerivedInfo& srcDerInfo);


};

#endif
