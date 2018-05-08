#ifndef _SentenceGenerator_CUDA_h
#define _SentenceGenerator_CUDA_h

#include "UtauDraft.h"

class SentenceGenerator_CUDA : public SentenceGenerator
{
public:
	virtual void GenerateSentence(const UtauSourceFetcher& srcFetcher, unsigned numPieces, const std::string* lyrics, const unsigned* isVowel, const float* weights, const unsigned* lengths, const float *freqAllMap, NoteBuffer* noteBuf);
};

#endif
