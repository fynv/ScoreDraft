#include "SentenceGenerator_CUDA.h"

void SentenceGenerator_CUDA::GenerateSentence(const UtauSourceFetcher& srcFetcher, unsigned numPieces, const std::string* lyrics, const unsigned* isVowel, const unsigned* lengths, const float *freqAllMap, NoteBuffer* noteBuf)
{
	printf("here!\n");
}
