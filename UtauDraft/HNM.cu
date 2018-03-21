#include <cuda_runtime.h>
#include <cmath>
#include <VoiceUtil.cuh>


#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

struct SrcSampleInfo
{
	unsigned srcPos;
	float srcSampleFreq;
	float logicalPos;
};

__shared__ unsigned char sbuf[];

__global__
void g_GetMaxVoiced(unsigned srcLen, float* d_srcBuf, SrcSampleInfo* d_SampleLocs, unsigned fixedBeginId, unsigned *d_maxVoiced, unsigned BufSize)
{
	unsigned paramId = blockIdx.x + fixedBeginId;
	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	SrcSampleInfo& posInfo = d_SampleLocs[paramId];

	float fhalfWinlen = 3.0f / posInfo.srcSampleFreq;
	unsigned u_halfWidth = (unsigned)ceilf(fhalfWinlen);
	unsigned uSpecLen = (unsigned)ceilf(fhalfWinlen*0.5f);

	float *s_buf1 = (float*)sbuf;
	float *s_buf2 = (float*)sbuf + u_halfWidth * 2;

	/*unsigned fftLen = 1;
	while (fftLen < u_halfWidth)
	{
		fftLen <<= 1;
	}

	if (u_halfWidth * 2 + 2 * fftLen>BufSize)
	{
		printf("%u %u %u\n", u_halfWidth, fftLen, BufSize);
	}*/

	d_captureFromBuf(srcLen, d_srcBuf, posInfo.srcPos, fhalfWinlen, u_halfWidth, s_buf1);
	d_CreateAmpSpectrumFromWindow(fhalfWinlen, u_halfWidth, s_buf1, s_buf2, uSpecLen);

	unsigned& maxVoiced = *((unsigned*)sbuf + BufSize - 1);
	maxVoiced = 0;

	__syncthreads();

	for (unsigned i = 6 + 3 * workerId; i + 4 < uSpecLen; i += 3 * numWorker)
	{
		unsigned count = 0;
		for (int j = -3; j <= 3; j += 3)
		{
			float absv0 = s_buf2[(int)i + j];
			float absv1 = s_buf2[(int)i + j - 1];
			float absv2 = s_buf2[(int)i + j + 1];

			float rate = absv0 / (absv0 + absv1 + absv2);

			if (rate > 0.7f)
			{
				count++;
			}
		}
		if (count > 1)
		{
			atomicMax(&maxVoiced, i / 3);
		}
	}

	__syncthreads();

	if (workerId==0)
		d_maxVoiced[blockIdx.x] = maxVoiced;
}


void h_GetMaxVoiced(unsigned srcLen, float* d_srcBuf, SrcSampleInfo* d_SampleLocs, unsigned fixedBeginId, unsigned fixedEndId, unsigned *d_maxVoiced, unsigned BufSize)
{
	static const unsigned groupSize = 256;
	unsigned sharedBufSize = (unsigned)sizeof(float)* BufSize;
	g_GetMaxVoiced << < fixedEndId - fixedBeginId, groupSize, sharedBufSize >> > (srcLen, d_srcBuf, d_SampleLocs, fixedBeginId, d_maxVoiced, BufSize);

}


__global__
void g_AnalyzeInput(unsigned srcLen, float* d_srcBuf, SrcSampleInfo* d_SampleLocs, unsigned halfWinLen, 
unsigned specLen, float* d_HarmWindows, float* d_NoiseSpecs, unsigned fixedBeginId, unsigned fixedEndId, unsigned *d_maxVoiced, unsigned BufSize)
{
	unsigned paramId = blockIdx.x;
	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	SrcSampleInfo& posInfo = d_SampleLocs[paramId];

	unsigned maxVoiced = (unsigned)(-1);
	if (paramId >= fixedBeginId && paramId < fixedEndId)
		maxVoiced = d_maxVoiced[paramId - fixedBeginId];

	float srcHalfWinWidth = 1.0f / posInfo.srcSampleFreq;
	unsigned u_srchalfWidth = (unsigned)ceilf(srcHalfWinWidth);
	unsigned uSpecLen = (unsigned)ceilf(srcHalfWinWidth*0.5f);

	float *s_buf1 = (float*)sbuf; // capture
	float *s_buf2 = (float*)sbuf + u_srchalfWidth * 2; // Amplitude spectrum

	d_captureFromBuf(srcLen, d_srcBuf, posInfo.srcPos, srcHalfWinWidth, u_srchalfWidth, s_buf1);
	d_CreateAmpSpectrumFromWindow(srcHalfWinWidth, u_srchalfWidth, s_buf1, s_buf2, uSpecLen);

	if (maxVoiced!=(unsigned)(-1))
	{
		for (unsigned i = maxVoiced + 1 + workerId; i < uSpecLen; i += numWorker)
		{
			float amplitude = s_buf2[i];
			s_buf2[i] = 0.0f;
			if (i < specLen)
			{
				d_NoiseSpecs[i + paramId*specLen] = amplitude;
			}
		}
		for (unsigned i = uSpecLen + workerId; i < specLen; i += numWorker)
			d_NoiseSpecs[i + paramId*specLen] = 0.0f;
	}
	else
	{
		for (unsigned i = workerId; i < specLen; i += numWorker)
			d_NoiseSpecs[i + paramId*specLen] = 0.0f;
	}

	__syncthreads();

	d_CreateSymmetricWindowFromAmpSpec(s_buf2, uSpecLen, srcHalfWinWidth, u_srchalfWidth, s_buf2);

	for (unsigned i = workerId; i < u_srchalfWidth; i += numWorker)
	{
		if (i < halfWinLen)
		{
			d_HarmWindows[i + paramId*halfWinLen] = s_buf2[i];
		}
	}
	for (unsigned i = u_srchalfWidth + workerId; i < halfWinLen; i += numWorker)
	{
		d_HarmWindows[i + paramId*halfWinLen] = 0.0f;
	}
}

void h_AnalyzeInput(unsigned srcLen, float* d_srcBuf, unsigned srcNumSampleLoc, SrcSampleInfo* d_SampleLocs, unsigned halfWinLen, 
	unsigned specLen, float* d_HarmWindows, float* d_NoiseSpecs, unsigned fixedBeginId, unsigned fixedEndId, unsigned *d_maxVoiced, unsigned BufSize)
{
	static const unsigned groupSize = 256;
	unsigned sharedBufSize = (unsigned)sizeof(float)* BufSize;
	g_AnalyzeInput << < srcNumSampleLoc, groupSize, sharedBufSize >> > (srcLen, d_srcBuf, d_SampleLocs, halfWinLen, 
		specLen, d_HarmWindows, d_NoiseSpecs, fixedBeginId, fixedEndId, d_maxVoiced, BufSize);
}


__global__
void g_Synthesis(unsigned srcNumSampleLoc, SrcSampleInfo* d_SampleLocs, unsigned halfWinLen, unsigned specLen, float* d_HarmWindows, float* d_NoiseSpecs,
float sumLen, float *d_destBuf1, float *d_destBuf2, unsigned* d_paramId0s, float destHalfWinLen, float* d_randPhase)
{
	unsigned dstParamId = blockIdx.x;
	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	unsigned paramId0 = d_paramId0s[dstParamId];
	unsigned paramId1 = paramId0 + 1;
	if (paramId1 >= srcNumSampleLoc) paramId1 = paramId0;

	SrcSampleInfo& posInfo0 = d_SampleLocs[paramId0];
	SrcSampleInfo& posInfo1 = d_SampleLocs[paramId1];

	unsigned uSumLen = (unsigned)ceilf(sumLen);
	
	float fDestWinCenter = (float)dstParamId*destHalfWinLen;
	float fParamPos = fDestWinCenter / sumLen;

	float k;
	if (fParamPos >= posInfo1.logicalPos) k = 1.0f;
	else if (fParamPos <= posInfo0.logicalPos) k = 0.0f;
	else
	{
		k = (fParamPos - posInfo0.logicalPos) / (posInfo1.logicalPos - posInfo0.logicalPos);
	}

	float srcHalfWinWidth0 = 1.0f / posInfo0.srcSampleFreq;
	float srcHalfWinWidth1 = 1.0f / posInfo1.srcSampleFreq;
	unsigned u_destHalfWinLen = (unsigned)ceilf(destHalfWinLen);

	float* noiseBuf = (float*)sbuf; // 4xfft len
	for (unsigned i = workerId; i < u_destHalfWinLen; i += numWorker)
		noiseBuf[i] = 0.0f;
	__syncthreads();
	AmpSpec_Scale(srcHalfWinWidth0, d_NoiseSpecs + specLen*paramId0, destHalfWinLen, noiseBuf, 1.0f - k);
	AmpSpec_Scale(srcHalfWinWidth1, d_NoiseSpecs + specLen*paramId1, destHalfWinLen, noiseBuf, k);

	unsigned uSpecLen = (unsigned)ceilf(destHalfWinLen*0.5f);

	// apply random phases
	float* d_p_rand = d_randPhase + dstParamId*uSpecLen;
	d_CreateNoiseWindowFromAmpSpec(noiseBuf, d_p_rand, uSpecLen, destHalfWinLen, u_destHalfWinLen, noiseBuf);

	Win_WriteToBuf(uSumLen, dstParamId % 2 == 0 ? d_destBuf1 : d_destBuf2, (unsigned)fDestWinCenter, destHalfWinLen, noiseBuf);	

	float* harmBuf = (float*)sbuf;
	for (unsigned i = workerId; i < u_destHalfWinLen; i += numWorker)
		harmBuf[i] = 0.0f;
	__syncthreads();

	SymWin_Repitch_FormantPreserved(srcHalfWinWidth0, d_HarmWindows + halfWinLen*paramId0, destHalfWinLen, harmBuf, 1.0f-k);
	SymWin_Repitch_FormantPreserved(srcHalfWinWidth1, d_HarmWindows + halfWinLen*paramId1, destHalfWinLen, harmBuf, k);

	SymWin_WriteToBuf(uSumLen, dstParamId % 2 == 0 ? d_destBuf1 : d_destBuf2, (unsigned)fDestWinCenter, destHalfWinLen, harmBuf);

}

void h_Synthesis(unsigned srcNumSampleLoc, SrcSampleInfo* d_SampleLocs, unsigned halfWinLen, unsigned specLen, float* d_HarmWindows, float* d_NoiseSpecs,
	float sumLen, float *d_destBuf1, float *d_destBuf2, unsigned count_dst_windows, unsigned* d_paramId0s, float destHalfWinLen, float* d_randPhase, unsigned BufSize)
{
	static const unsigned groupSize = 256;
	unsigned sharedBufSize = (unsigned)sizeof(float)* BufSize;
	g_Synthesis << < count_dst_windows, groupSize, sharedBufSize >> > (srcNumSampleLoc, d_SampleLocs, halfWinLen, specLen, d_HarmWindows, d_NoiseSpecs,
		sumLen, d_destBuf1, d_destBuf2, d_paramId0s, destHalfWinLen, d_randPhase);

}

__global__
void g_Merge2Bufs(unsigned uSumLen, float *d_destBuf1, float *d_destBuf2)
{
	unsigned pos = threadIdx.x + blockIdx.x*blockDim.x;
	if (pos < uSumLen)
		d_destBuf1[pos] += d_destBuf2[pos];
}

void h_Merge2Bufs(unsigned uSumLen, float *d_destBuf1, float *d_destBuf2)
{
	static const unsigned groupSize = 256;
	g_Merge2Bufs << < (uSumLen - 1) / groupSize + 1, groupSize >> >(uSumLen, d_destBuf1, d_destBuf2);
}
