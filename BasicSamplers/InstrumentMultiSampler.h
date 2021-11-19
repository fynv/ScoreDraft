#ifndef _InstrumentMultiSampler_h
#define _InstrumentMultiSampler_h

#include <vector>

struct InstrumentSample;
void InstrumentMultiSample(const std::vector<InstrumentSample>& samples, float* outBuf, unsigned outBufLen, float sampleFreq);

#endif
