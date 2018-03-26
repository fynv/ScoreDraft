#include <cuda_runtime.h>
#include <cmath>
#include <VoiceUtil.cuh>

#include <vector>

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
class CUDAVector
{
public:
	unsigned count;
	T* d_data;
};

template <class T_GPU, class T_CPU>
class CUDAImagedVector : public CUDAVector<T_GPU>{};

template <class T>
class CUDALevel2Vector : public CUDAImagedVector<CUDAVector<T>, std::vector<T>>{};

class CUDASrcBuf : public CUDAVector<float> {};

struct SourceInfo;
typedef CUDAImagedVector<CUDASrcBuf, SourceInfo> CUDASrcBufList;

struct SrcSampleInfo
{
	unsigned srcPos;
	float srcSampleFreq;
	float logicalPos;
};

struct CUDASrcPieceInfo
{
	CUDAVector<SrcSampleInfo> SampleLocations;
	CUDAVector<SrcSampleInfo> SampleLocations_next;
	unsigned fixedBeginId;
	unsigned fixedEndId;
	unsigned fixedBeginId_next;
	unsigned fixedEndId_next;
};

struct SrcPieceInfo;
typedef CUDAImagedVector<CUDASrcPieceInfo, SrcPieceInfo> CUDASrcPieceInfoList;

struct Job
{
	unsigned pieceId;
	unsigned isNext;
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
	float k1;
	float k1_next;
	float k2;
	unsigned paramId0;
	unsigned paramId0_next;
	float destHalfWinLen;
};

struct CUDATempBuffer
{
	unsigned count;
	float *d_data;
};

__shared__ unsigned char sbuf[];

__global__
void g_GetMaxVoiced(CUDASrcBufList cuSrcBufs, CUDASrcPieceInfoList pieceInfoList,
CUDALevel2Vector<unsigned> cuMaxVoicedLists, CUDALevel2Vector<unsigned> cuMaxVoicedLists_next, CUDAVector<Job> jobMap, unsigned BufSize)
{
	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	const Job& job = jobMap.d_data[blockIdx.x];
	unsigned pieceId = job.pieceId;
	const CUDASrcPieceInfo& pieceInfo = pieceInfoList.d_data[pieceId];
	bool isNext = job.isNext != 0;
	unsigned paramId = job.jobOfPiece + (isNext ? pieceInfo.fixedBeginId_next : pieceInfo.fixedBeginId);

	SrcSampleInfo& posInfo = isNext ? pieceInfo.SampleLocations_next.d_data[paramId] : pieceInfo.SampleLocations.d_data[paramId];

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
		const CUDASrcBuf& srcBuf = cuSrcBufs.d_data[isNext ? pieceId + 1 : pieceId];

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

	CUDAVector<unsigned>& d_maxVoicedList= isNext ? cuMaxVoicedLists_next.d_data[pieceId] : cuMaxVoicedLists.d_data[pieceId];

	if (workerId==0)
		d_maxVoicedList.d_data[job.jobOfPiece] = maxVoiced;
}

void h_GetMaxVoiced(CUDASrcBufList cuSrcBufs, CUDASrcPieceInfoList pieceInfoList,
	CUDALevel2Vector<unsigned> cuMaxVoicedLists, CUDALevel2Vector<unsigned> cuMaxVoicedLists_next, CUDAVector<Job> jobMap, unsigned BufSize)
{
	if (BufSize > 12000) BufSize = 12000;
	unsigned groupSize = calcGroupSize(BufSize/4);
	unsigned sharedBufSize = (unsigned)sizeof(float)* BufSize;
	g_GetMaxVoiced << < jobMap.count, groupSize, sharedBufSize >> > (cuSrcBufs, pieceInfoList, cuMaxVoicedLists, cuMaxVoicedLists_next, jobMap, BufSize);

}

__global__
void g_AnalyzeInput(CUDASrcBufList cuSrcBufs, CUDASrcPieceInfoList pieceInfoList, unsigned halfWinLen,
unsigned specLen, CUDALevel2Vector<float> cuHarmWindows, CUDALevel2Vector<float> cuNoiseSpecs,
CUDALevel2Vector<float> cuHarmWindows_next, CUDALevel2Vector<float> cuNoiseSpecs_next,
CUDALevel2Vector<unsigned> cuMaxVoicedLists, CUDALevel2Vector<unsigned> cuMaxVoicedLists_next, CUDAVector<Job> jobMap, unsigned BufSize)
{
	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	const Job& job = jobMap.d_data[blockIdx.x];
	unsigned pieceId = job.pieceId;
	const CUDASrcPieceInfo& pieceInfo = pieceInfoList.d_data[pieceId];
	bool isNext = job.isNext != 0;
	unsigned paramId = job.jobOfPiece;

	SrcSampleInfo& posInfo = isNext ? pieceInfo.SampleLocations_next.d_data[paramId] : pieceInfo.SampleLocations.d_data[paramId];

	unsigned fixedBeginId = isNext ? pieceInfo.fixedBeginId_next : pieceInfo.fixedBeginId;
	unsigned fixedEndId = isNext ? pieceInfo.fixedEndId_next : pieceInfo.fixedEndId;
	unsigned *d_maxVoiced = isNext ? cuMaxVoicedLists_next.d_data[pieceId].d_data : cuMaxVoicedLists.d_data[pieceId].d_data;
	float* d_HarmWindows = isNext ? cuHarmWindows_next.d_data[pieceId].d_data : cuHarmWindows.d_data[pieceId].d_data;
	float* d_NoiseSpecs = isNext ? cuNoiseSpecs_next.d_data[pieceId].d_data : cuNoiseSpecs.d_data[pieceId].d_data;

	unsigned maxVoiced = (unsigned)(-1);
	if (fixedBeginId != (unsigned)(-1) && paramId >= fixedBeginId && paramId < fixedEndId)
		maxVoiced = d_maxVoiced[paramId - fixedBeginId];
	
	float srcHalfWinWidth = 1.0f / posInfo.srcSampleFreq;
	unsigned u_srchalfWidth = (unsigned)ceilf(srcHalfWinWidth);
	unsigned uSpecLen = (unsigned)ceilf(srcHalfWinWidth*0.5f);

	float *s_buf1 = (float*)sbuf; // capture
	float *s_buf2 = (float*)sbuf + u_srchalfWidth * 2; // Amplitude spectrum

	const CUDASrcBuf& srcBuf = cuSrcBufs.d_data[isNext ? pieceId + 1 : pieceId];

	d_captureFromBuf(srcBuf.count, srcBuf.d_data, posInfo.srcPos, srcHalfWinWidth, u_srchalfWidth, s_buf1);
	d_CreateAmpSpectrumFromWindow(srcHalfWinWidth, u_srchalfWidth, s_buf1, s_buf2, uSpecLen);

	for (unsigned i = workerId; i < specLen; i += numWorker)
	{
		float amplitude = 0.0f;
		if (maxVoiced != (unsigned)(-1) && i>maxVoiced && i < uSpecLen)
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

void h_AnalyzeInput(CUDASrcBufList cuSrcBufs, CUDASrcPieceInfoList pieceInfoList, unsigned halfWinLen,
	unsigned specLen, CUDALevel2Vector<float> cuHarmWindows, CUDALevel2Vector<float> cuNoiseSpecs,
	CUDALevel2Vector<float> cuHarmWindows_next, CUDALevel2Vector<float> cuNoiseSpecs_next,
	CUDALevel2Vector<unsigned> cuMaxVoicedLists, CUDALevel2Vector<unsigned> cuMaxVoicedLists_next, CUDAVector<Job> jobMap, unsigned BufSize)
{
	unsigned groupSize = calcGroupSize(BufSize / 4);
	unsigned sharedBufSize = (unsigned)sizeof(float)* BufSize;
	g_AnalyzeInput << < jobMap.count, groupSize, sharedBufSize >> > (cuSrcBufs, pieceInfoList,
		halfWinLen, specLen, cuHarmWindows, cuNoiseSpecs, cuHarmWindows_next, cuNoiseSpecs_next,
		cuMaxVoicedLists, cuMaxVoicedLists_next, jobMap, BufSize);
}

__global__
void g_Synthesis(CUDASrcPieceInfoList cuSrcPieceInfos, unsigned halfWinLen, unsigned specLen, 
CUDALevel2Vector<float> cuHarmWindows, CUDALevel2Vector<float> cuNoiseSpecs,
CUDALevel2Vector<float> cuHarmWindows_next, CUDALevel2Vector<float> cuNoiseSpecs_next, 
CUDAVector<DstPieceInfo> cuDstPieceInfos, CUDAVector<CUDATempBuffer> cuTmpBufs1, CUDAVector<CUDATempBuffer> cuTmpBufs2,
CUDAVector<float> cuRandPhase, CUDAVector<SynthJobInfo> cuSynthJobs)
{
	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	const SynthJobInfo& job = cuSynthJobs.d_data[blockIdx.x];
	unsigned pieceId = job.pieceId;
	unsigned dstParamId = job.jobOfPiece;
	const CUDASrcPieceInfo& srcPieceInfo = cuSrcPieceInfos.d_data[pieceId];
	const DstPieceInfo& dstPieceInfo = cuDstPieceInfos.d_data[pieceId];

	CUDATempBuffer& cuTmpBuf = dstParamId % 2 == 0 ? cuTmpBufs1.d_data[pieceId] : cuTmpBufs2.d_data[pieceId];

	float tempHalfWinLen = 1.0f / dstPieceInfo.minSampleFreq;
	unsigned u_tempHalfWinLen = (unsigned)ceilf(tempHalfWinLen);
	float fTmpWinCenter = dstPieceInfo.fTmpWinCenter0 + dstParamId* tempHalfWinLen;

	float destHalfWinLen = job.destHalfWinLen;
	unsigned u_destHalfWinLen = (unsigned)ceilf(destHalfWinLen);
	unsigned uSpecLen = (unsigned)ceilf(destHalfWinLen*0.5f);
	unsigned uRandPhaseInterval = (unsigned)ceilf(tempHalfWinLen*0.5f);

	unsigned paramId0, paramId1, paramId0_next, paramId1_next;
	float srcHalfWinWidth0, srcHalfWinWidth1, srcHalfWinWidth0_next, srcHalfWinWidth1_next;

	float* noiseBuf = (float*)sbuf; // max(2 * fftLen, 2 * u_tempHalfWinLen) +2*fftlen
	for (unsigned i = workerId; i < u_tempHalfWinLen; i += numWorker)
		noiseBuf[i] = 0.0f;
	__syncthreads();

	bool haveNoise = false;
	if (job.k2 < 1.0f)
	{
		paramId0 = job.paramId0;
		paramId1 = paramId0 + 1;
		if (paramId1 >= srcPieceInfo.SampleLocations.count) paramId1 = paramId0;

		SrcSampleInfo& posInfo0 = srcPieceInfo.SampleLocations.d_data[paramId0];
		SrcSampleInfo& posInfo1 = srcPieceInfo.SampleLocations.d_data[paramId1];
		srcHalfWinWidth0 = 1.0f / posInfo0.srcSampleFreq;
		srcHalfWinWidth1 = 1.0f / posInfo1.srcSampleFreq;

		float* d_NoiseSpecs = cuNoiseSpecs.d_data[pieceId].d_data;

		float k = job.k1;
		if (k < 1.0f && paramId0 >= srcPieceInfo.fixedBeginId && paramId0 < srcPieceInfo.fixedEndId)
		{
			haveNoise = true;
			AmpSpec_Scale(srcHalfWinWidth0, d_NoiseSpecs + specLen*paramId0, destHalfWinLen, noiseBuf, (1.0f - k)*(1.0f - job.k2));
		}

		if (k > 0.0f && paramId1 >= srcPieceInfo.fixedBeginId && paramId1 < srcPieceInfo.fixedEndId)
		{
			haveNoise = true;
			AmpSpec_Scale(srcHalfWinWidth1, d_NoiseSpecs + specLen*paramId1, destHalfWinLen, noiseBuf, k*(1.0f - job.k2));
		}
	}

	if (job.k2 > 0.0f)
	{
		paramId0_next= job.paramId0_next;
		paramId1_next = paramId0_next + 1;
		if (paramId1_next >= srcPieceInfo.SampleLocations_next.count) paramId1_next = paramId0_next;

		SrcSampleInfo& posInfo0 = srcPieceInfo.SampleLocations_next.d_data[paramId0_next];
		SrcSampleInfo& posInfo1 = srcPieceInfo.SampleLocations_next.d_data[paramId1_next];
		srcHalfWinWidth0_next = 1.0f / posInfo0.srcSampleFreq;
		srcHalfWinWidth1_next = 1.0f / posInfo1.srcSampleFreq;

		float* d_NoiseSpecs = cuNoiseSpecs_next.d_data[pieceId].d_data;

		float k = job.k1_next;
		if (k < 1.0f && paramId0_next >= srcPieceInfo.fixedBeginId_next && paramId0_next < srcPieceInfo.fixedEndId_next)
		{
			haveNoise = true;
			AmpSpec_Scale(srcHalfWinWidth0_next, d_NoiseSpecs + specLen*paramId0_next, destHalfWinLen, noiseBuf, (1.0f - k)*job.k2);
		}

		if (k > 0.0f && paramId1_next >= srcPieceInfo.fixedBeginId_next && paramId1_next < srcPieceInfo.fixedEndId_next)
		{
			haveNoise = true;
			AmpSpec_Scale(srcHalfWinWidth1_next, d_NoiseSpecs + specLen*paramId1_next, destHalfWinLen, noiseBuf, k*job.k2);
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

	if (job.k2 < 1.0f)
	{
		float *d_HarmWindows = cuHarmWindows.d_data[pieceId].d_data;

		float k = job.k1;
		if (k < 1.0f)
			SymWin_Repitch_FormantPreserved(srcHalfWinWidth0, d_HarmWindows + halfWinLen*paramId0, destHalfWinLen, harmBuf, (1.0f - k)*(1.0f - job.k2));
		if (k > 0.0f)
			SymWin_Repitch_FormantPreserved(srcHalfWinWidth1, d_HarmWindows + halfWinLen*paramId1, destHalfWinLen, harmBuf, k*(1.0f - job.k2));
	}

	if (job.k2 > 0.0f)
	{
		float *d_HarmWindows = cuHarmWindows_next.d_data[pieceId].d_data;

		float k = job.k1_next;
		if (k < 1.0f)
			SymWin_Repitch_FormantPreserved(srcHalfWinWidth0_next, d_HarmWindows + halfWinLen*paramId0_next, destHalfWinLen, harmBuf, (1.0f - k)*job.k2);
		if (k > 0.0f)
			SymWin_Repitch_FormantPreserved(srcHalfWinWidth1_next, d_HarmWindows + halfWinLen*paramId1_next, destHalfWinLen, harmBuf, k*job.k2);
	}

	float* scaled_harmBuf = (float*)sbuf + u_destHalfWinLen;
	d_ScaleSymWindow(destHalfWinLen, u_destHalfWinLen, harmBuf, scaled_harmBuf, tempHalfWinLen);

	SymWin_WriteToBuf(cuTmpBuf.count, cuTmpBuf.d_data, fTmpWinCenter, tempHalfWinLen, scaled_harmBuf);
}


void h_Synthesis(CUDASrcPieceInfoList cuSrcPieceInfos, unsigned halfWinLen, unsigned specLen,
	CUDALevel2Vector<float> cuHarmWindows, CUDALevel2Vector<float> cuNoiseSpecs,
	CUDALevel2Vector<float> cuHarmWindows_next, CUDALevel2Vector<float> cuNoiseSpecs_next,
	CUDAVector<DstPieceInfo> cuDstPieceInfos, CUDAVector<CUDATempBuffer> cuTmpBufs1, CUDAVector<CUDATempBuffer> cuTmpBufs2,
	CUDAVector<float> cuRandPhase, CUDAVector<SynthJobInfo> cuSynthJobs, unsigned BufSize)
{
	unsigned groupSize = calcGroupSize(BufSize / 4);
	unsigned sharedBufSize = (unsigned)sizeof(float)* BufSize;

	g_Synthesis << < cuSynthJobs.count, groupSize, sharedBufSize >> > (cuSrcPieceInfos, halfWinLen, specLen,
		cuHarmWindows, cuNoiseSpecs, cuHarmWindows_next, cuNoiseSpecs_next, cuDstPieceInfos, cuTmpBufs1, cuTmpBufs2,
		cuRandPhase, cuSynthJobs);
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


