#include <cuda_runtime.h>
#include <cmath>
#include <VoiceUtil.cuh>

#include <stdio.h>

unsigned calcGroupSize(unsigned workPerGroup)
{
	unsigned s = 1;
	while (s < workPerGroup && s<256)
		s <<= 1;
	return s;
}

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

template <class T>
struct VectorView
{
	unsigned count;
	T* d_data;
};

struct SrcSampleInfo
{
	unsigned srcPos;
	float srcSampleFreq;
	float dstPos;
	int isVowel;
};

struct Job
{
	unsigned pieceId;
	unsigned jobOfPiece;
};

struct DstPieceInfo
{
	float minSampleFreq;
	unsigned uSumLen;
	float tempLen;
	unsigned uTempLen;
	float fTmpWinCenter0;
};

struct SynthJobInfo
{
	unsigned pieceId;
	unsigned jobOfPiece;
	unsigned srcPieceId0;
	unsigned srcPieceId1;
	float k_srcPiece;
	unsigned paramId00;
	unsigned paramId10;
	float k0;
	float k1;
	float destHalfWinLen;
};

struct CUDATempBuffer
{
	unsigned count;
	float *d_data;
};

extern __shared__ unsigned char sbuf[];

__global__
void g_GetMaxVoiced(VectorView<VectorView<float>> cuSrcBufs, VectorView<VectorView<SrcSampleInfo>> pieceInfoList,
VectorView<VectorView<unsigned>> cuMaxVoicedLists, VectorView<Job> jobMap, unsigned BufSize)
{
	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	const Job& job = jobMap.d_data[blockIdx.x];
	unsigned pieceId = job.pieceId;
	const VectorView<SrcSampleInfo>& pieceInfo = pieceInfoList.d_data[pieceId];
	unsigned paramId = job.jobOfPiece;

	SrcSampleInfo& posInfo = pieceInfo.d_data[paramId];

	float fhalfWinlen = 3.0f / posInfo.srcSampleFreq;
	unsigned u_halfWidth = (unsigned)ceilf(fhalfWinlen);
	unsigned uSpecLen = (unsigned)ceilf(fhalfWinlen*0.5f);

	unsigned fftLen = 1;
	while (fftLen < u_halfWidth)
	{
		fftLen <<= 1;
	}

	bool skip = u_halfWidth * 2 + fftLen * 2> BufSize;
	float *s_buf1 = (float*)sbuf;
	float *s_buf2 = (float*)sbuf + u_halfWidth * 2;

	if (!skip)
	{
		const VectorView<float>& srcBuf = cuSrcBufs.d_data[pieceId];

		d_captureFromBuf(srcBuf.count, srcBuf.d_data, posInfo.srcPos, fhalfWinlen, u_halfWidth, s_buf1);
		d_CreateAmpSpectrumFromWindow(fhalfWinlen, u_halfWidth, s_buf1, s_buf2, uSpecLen);
	}

	unsigned& maxVoiced = *((unsigned*)sbuf + BufSize - 1);
	if (workerId == 0)
		maxVoiced = 0;

	__syncthreads();

	if (!skip)
	{

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
				atomicMax(&maxVoiced, i / 3 + 1);
			}
		}

		__syncthreads();
	}

	VectorView<unsigned>& d_maxVoicedList = cuMaxVoicedLists.d_data[pieceId];

	if (workerId==0)
		d_maxVoicedList.d_data[job.jobOfPiece] = maxVoiced;
}

void h_GetMaxVoiced(VectorView<VectorView<float>> cuSrcBufs, VectorView<VectorView<SrcSampleInfo>> pieceInfoList,
	VectorView<VectorView<unsigned>> cuMaxVoicedLists, VectorView<Job> jobMap, unsigned BufSize)
{
	if (BufSize > 12000) BufSize = 12000;
	unsigned groupSize = calcGroupSize(BufSize/4);
	unsigned sharedBufSize = (unsigned)sizeof(float)* BufSize;
	g_GetMaxVoiced << < jobMap.count, groupSize, sharedBufSize >> > (cuSrcBufs, pieceInfoList, cuMaxVoicedLists, jobMap, BufSize);

}

__global__
void g_AnalyzeInput(VectorView<VectorView<float>> cuSrcBufs, VectorView<VectorView<SrcSampleInfo>> pieceInfoList, unsigned halfWinLen,
unsigned specLen, VectorView<VectorView<float>> cuHarmWindows, VectorView<VectorView<float>> cuNoiseSpecs,
VectorView<VectorView<unsigned>> cuMaxVoicedLists, VectorView<Job> jobMap, unsigned BufSize)
{
	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	const Job& job = jobMap.d_data[blockIdx.x];
	unsigned pieceId = job.pieceId;
	const VectorView<SrcSampleInfo>& pieceInfo = pieceInfoList.d_data[pieceId];
	unsigned paramId = job.jobOfPiece;

	SrcSampleInfo& posInfo = pieceInfo.d_data[paramId];

	unsigned *d_maxVoiced = cuMaxVoicedLists.d_data[pieceId].d_data;
	float* d_HarmWindows = cuHarmWindows.d_data[pieceId].d_data;
	float* d_NoiseSpecs = cuNoiseSpecs.d_data[pieceId].d_data;

	unsigned maxVoiced = (unsigned)(-1);
	if (posInfo.isVowel<2)
		maxVoiced = d_maxVoiced[paramId];
	
	float srcHalfWinWidth = 1.0f / posInfo.srcSampleFreq;
	unsigned u_srchalfWidth = (unsigned)ceilf(srcHalfWinWidth);
	unsigned uSpecLen = (unsigned)ceilf(srcHalfWinWidth*0.5f);

	float *s_buf1 = (float*)sbuf; // capture
	float *s_buf2 = (float*)sbuf + u_srchalfWidth * 2; // Amplitude spectrum

	const VectorView<float>& srcBuf = cuSrcBufs.d_data[pieceId];

	d_captureFromBuf(srcBuf.count, srcBuf.d_data, posInfo.srcPos, srcHalfWinWidth, u_srchalfWidth, s_buf1);
	d_CreateAmpSpectrumFromWindow(srcHalfWinWidth, u_srchalfWidth, s_buf1, s_buf2, uSpecLen);

	for (unsigned i = workerId; i < specLen; i += numWorker)
	{
		float amplitude = 0.0f;
		if (posInfo.isVowel<2 && i>maxVoiced && i < uSpecLen)
		{
			amplitude = s_buf2[i];
			s_buf2[i] = 0.0f;
		}
		d_NoiseSpecs[i + paramId*specLen] = amplitude;
	}

	__syncthreads();

	d_CreateSymmetricWindowFromAmpSpec(s_buf2, uSpecLen, srcHalfWinWidth, u_srchalfWidth, s_buf2);

	for (unsigned i = workerId; i < halfWinLen; i += numWorker)
	{
		float v = 0.0f;
		if (i < u_srchalfWidth)
		{
			v = s_buf2[i];
		}
		d_HarmWindows[i + paramId*halfWinLen] = v;
	}
}

void h_AnalyzeInput(VectorView<VectorView<float>> cuSrcBufs, VectorView<VectorView<SrcSampleInfo>> pieceInfoList, unsigned halfWinLen,
	unsigned specLen, VectorView<VectorView<float>> cuHarmWindows, VectorView<VectorView<float>> cuNoiseSpecs,
	VectorView<VectorView<unsigned>> cuMaxVoicedLists, VectorView<Job> jobMap, unsigned BufSize)
{
	unsigned groupSize = calcGroupSize(BufSize / 4);
	unsigned sharedBufSize = (unsigned)sizeof(float)* BufSize;
	g_AnalyzeInput << < jobMap.count, groupSize, sharedBufSize >> > (cuSrcBufs, pieceInfoList,
		halfWinLen, specLen, cuHarmWindows, cuNoiseSpecs, cuMaxVoicedLists, jobMap, BufSize);
}

__global__
void g_Synthesis(VectorView<VectorView<SrcSampleInfo>> cuSrcPieceInfos, unsigned halfWinLen, unsigned specLen,
VectorView<VectorView<float>> cuHarmWindows, VectorView<VectorView<float>> cuNoiseSpecs,
VectorView<DstPieceInfo> cuDstPieceInfos, VectorView<CUDATempBuffer> cuTmpBufs1, VectorView<CUDATempBuffer> cuTmpBufs2,
VectorView<float> cuRandPhase, VectorView<SynthJobInfo> cuSynthJobs)
{
	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	const SynthJobInfo& job = cuSynthJobs.d_data[blockIdx.x];
	unsigned dstPieceId = job.pieceId;
	unsigned srcPieceId0 = job.srcPieceId0;
	unsigned srcPieceId1 = job.srcPieceId1;
	unsigned dstParamId = job.jobOfPiece;
	const VectorView<SrcSampleInfo>& srcPieceInfo0 = cuSrcPieceInfos.d_data[srcPieceId0];
	const VectorView<SrcSampleInfo>& srcPieceInfo1 = cuSrcPieceInfos.d_data[srcPieceId1];
	const DstPieceInfo& dstPieceInfo = cuDstPieceInfos.d_data[dstPieceId];

	CUDATempBuffer& cuTmpBuf = dstParamId % 2 == 0 ? cuTmpBufs1.d_data[dstPieceId] : cuTmpBufs2.d_data[dstPieceId];

	float tempHalfWinLen = 1.0f / dstPieceInfo.minSampleFreq;
	unsigned u_tempHalfWinLen = (unsigned)ceilf(tempHalfWinLen);
	float fTmpWinCenter = dstPieceInfo.fTmpWinCenter0 + dstParamId* tempHalfWinLen;

	float destHalfWinLen = job.destHalfWinLen;
	unsigned u_destHalfWinLen = (unsigned)ceilf(destHalfWinLen);
	unsigned uSpecLen = (unsigned)ceilf(destHalfWinLen*0.5f);
	unsigned uRandPhaseInterval = (unsigned)ceilf(tempHalfWinLen*0.5f);

	unsigned paramId00, paramId01, paramId10, paramId11;
	float srcHalfWinWidth00, srcHalfWinWidth01, srcHalfWinWidth10, srcHalfWinWidth11;

	float* noiseBuf = (float*)sbuf; // max(2 * fftLen, 2 * u_tempHalfWinLen) +2*fftlen
	for (unsigned i = workerId; i < u_tempHalfWinLen; i += numWorker)
		noiseBuf[i] = 0.0f;
	__syncthreads();

	bool haveNoise = false;
	if (job.k_srcPiece < 1.0f)
	{
		paramId00 = job.paramId00;
		paramId01 = paramId00 + 1;
		if (paramId01 >= srcPieceInfo0.count) paramId01 = paramId00;

		SrcSampleInfo& posInfo0 = srcPieceInfo0.d_data[paramId00];
		SrcSampleInfo& posInfo1 = srcPieceInfo0.d_data[paramId01];
		srcHalfWinWidth00 = 1.0f / posInfo0.srcSampleFreq;
		srcHalfWinWidth01 = 1.0f / posInfo1.srcSampleFreq;

		float* d_NoiseSpecs = cuNoiseSpecs.d_data[srcPieceId0].d_data;

		float k = job.k0;
		if (k < 1.0f && posInfo0.isVowel<2)
		{
			haveNoise = true;
			AmpSpec_Scale(srcHalfWinWidth00, d_NoiseSpecs + specLen*paramId00, destHalfWinLen, noiseBuf, (1.0f - k)*(1.0f - job.k_srcPiece));
		}

		if (k > 0.0f && posInfo1.isVowel<2)
		{
			haveNoise = true;
			AmpSpec_Scale(srcHalfWinWidth01, d_NoiseSpecs + specLen*paramId01, destHalfWinLen, noiseBuf, k*(1.0f - job.k_srcPiece));
		}
	}

	if (job.k_srcPiece > 0.0f)
	{
		paramId10= job.paramId10;
		paramId11 = paramId10 + 1;
		if (paramId11 >= srcPieceInfo1.count) paramId11 = paramId10;

		SrcSampleInfo& posInfo0 = srcPieceInfo1.d_data[paramId10];
		SrcSampleInfo& posInfo1 = srcPieceInfo1.d_data[paramId11];
		srcHalfWinWidth10 = 1.0f / posInfo0.srcSampleFreq;
		srcHalfWinWidth11 = 1.0f / posInfo1.srcSampleFreq;

		float* d_NoiseSpecs = cuNoiseSpecs.d_data[srcPieceId1].d_data;

		float k = job.k1;
		if (k < 1.0f && posInfo0.isVowel<2)
		{
			haveNoise = true;
			AmpSpec_Scale(srcHalfWinWidth10, d_NoiseSpecs + specLen*paramId10, destHalfWinLen, noiseBuf, (1.0f - k)*job.k_srcPiece);
		}

		if (k > 0.0f && posInfo1.isVowel<2)
		{
			haveNoise = true;
			AmpSpec_Scale(srcHalfWinWidth11, d_NoiseSpecs + specLen*paramId11, destHalfWinLen, noiseBuf, k*job.k_srcPiece);
		}
	}

	if (haveNoise)
	{
		// apply random phases
		float* d_p_rand = cuRandPhase.d_data + dstParamId*uRandPhaseInterval;
		d_CreateNoiseWindowFromAmpSpec(noiseBuf, d_p_rand, uSpecLen, destHalfWinLen, u_destHalfWinLen, noiseBuf, tempHalfWinLen);
		Win_WriteToBuf(cuTmpBuf.count, cuTmpBuf.d_data, fTmpWinCenter, tempHalfWinLen, noiseBuf);
	}

	float* harmBuf = (float*)sbuf;
	for (unsigned i = workerId; i < tempHalfWinLen; i += numWorker)
		harmBuf[i] = 0.0f;
	__syncthreads();

	if (job.k_srcPiece < 1.0f)
	{
		float *d_HarmWindows = cuHarmWindows.d_data[srcPieceId0].d_data;

		float k = job.k0;
		if (k < 1.0f)
			SymWin_Repitch_FormantPreserved(srcHalfWinWidth00, d_HarmWindows + halfWinLen*paramId00, destHalfWinLen, harmBuf, (1.0f - k)*(1.0f - job.k_srcPiece));
		if (k > 0.0f)
			SymWin_Repitch_FormantPreserved(srcHalfWinWidth01, d_HarmWindows + halfWinLen*paramId01, destHalfWinLen, harmBuf, k*(1.0f - job.k_srcPiece));
	}

	if (job.k_srcPiece > 0.0f)
	{
		float *d_HarmWindows = cuHarmWindows.d_data[srcPieceId1].d_data;

		float k = job.k1;
		if (k < 1.0f)
			SymWin_Repitch_FormantPreserved(srcHalfWinWidth10, d_HarmWindows + halfWinLen*paramId10, destHalfWinLen, harmBuf, (1.0f - k)*job.k_srcPiece);
		if (k > 0.0f)
			SymWin_Repitch_FormantPreserved(srcHalfWinWidth11, d_HarmWindows + halfWinLen*paramId11, destHalfWinLen, harmBuf, k*job.k_srcPiece);
	}

	float* scaled_harmBuf = (float*)sbuf + u_destHalfWinLen;
	d_ScaleSymWindow(destHalfWinLen, u_destHalfWinLen, harmBuf, scaled_harmBuf, tempHalfWinLen);

	SymWin_WriteToBuf(cuTmpBuf.count, cuTmpBuf.d_data, fTmpWinCenter, tempHalfWinLen, scaled_harmBuf);
}


void h_Synthesis(VectorView<VectorView<SrcSampleInfo>> cuSrcPieceInfos, unsigned halfWinLen, unsigned specLen,
	VectorView<VectorView<float>> cuHarmWindows, VectorView<VectorView<float>> cuNoiseSpecs,
	VectorView<DstPieceInfo> cuDstPieceInfos, VectorView<CUDATempBuffer> cuTmpBufs1, VectorView<CUDATempBuffer> cuTmpBufs2,
	VectorView<float> cuRandPhase, VectorView<SynthJobInfo> cuSynthJobs, unsigned BufSize)
{
	unsigned groupSize = calcGroupSize(BufSize / 4);
	unsigned sharedBufSize = (unsigned)sizeof(float)* BufSize;

	g_Synthesis << < cuSynthJobs.count, groupSize, sharedBufSize >> > (cuSrcPieceInfos, halfWinLen, specLen,
		cuHarmWindows, cuNoiseSpecs, cuDstPieceInfos, cuTmpBufs1, cuTmpBufs2, cuRandPhase, cuSynthJobs);
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



