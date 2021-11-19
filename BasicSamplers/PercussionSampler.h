#ifndef _PercussionSampler_h
#define _PercussionSampler_h

struct Sample;
void PercussionSample(const Sample& sample, float* outBuf, unsigned outBufLen, float sampleRatio);

#endif
