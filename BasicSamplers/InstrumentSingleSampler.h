#ifndef _InstrumentSingleSampler_h
#define _InstrumentSingleSampler_h

struct InstrumentSample;
void InstrumentSingleSample(const InstrumentSample& sample, float* outBuf, unsigned outBufLen, float sampleFreq);

#endif
