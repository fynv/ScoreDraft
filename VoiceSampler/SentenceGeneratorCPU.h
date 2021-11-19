#ifndef __SentenceGeneratorCPU_h
#define __SentenceGeneratorCPU_h

struct SentenceDescriptor;
void GenerateSentenceCPU(const SentenceDescriptor* desc, float* outBuf, unsigned outBufLen);


#endif

