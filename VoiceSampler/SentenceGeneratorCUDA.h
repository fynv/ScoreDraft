#ifndef __SentenceGeneratorCUDA_h
#define __SentenceGeneratorCUDA_h

struct SentenceDescriptor;
void GenerateSentenceCUDA(const SentenceDescriptor* desc, float* outBuf, unsigned outBufLen);


#endif

