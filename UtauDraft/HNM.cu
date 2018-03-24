#include <cuda_runtime.h>
#include <cmath>
#include <VoiceUtil.cuh>

#include <vector>

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

	float *s_buf1 = (float*)sbuf;
	float *s_buf2 = (float*)sbuf + u_halfWidth * 2;

	const CUDASrcBuf& srcBuf = cuSrcBufs.d_data[isNext ? pieceId + 1 : pieceId];

	d_captureFromBuf(srcBuf.count, srcBuf.d_data, posInfo.srcPos, fhalfWinlen, u_halfWidth, s_buf1);
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

	CUDAVector<unsigned>& d_maxVoicedList= isNext ? cuMaxVoicedLists_next.d_data[pieceId] : cuMaxVoicedLists.d_data[pieceId];

	if (workerId==0)
		d_maxVoicedList.d_data[job.jobOfPiece] = maxVoiced;
}

void h_GetMaxVoiced(CUDASrcBufList cuSrcBufs, CUDASrcPieceInfoList pieceInfoList,
	CUDALevel2Vector<unsigned> cuMaxVoicedLists, CUDALevel2Vector<unsigned> cuMaxVoicedLists_next, CUDAVector<Job> jobMap, unsigned BufSize)
{
	static const unsigned groupSize = 256;
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
	static const unsigned groupSize = 256;
	unsigned sharedBufSize = (unsigned)sizeof(float)* BufSize;
	g_AnalyzeInput << < jobMap.count, groupSize, sharedBufSize >> > (cuSrcBufs, pieceInfoList,
		halfWinLen, specLen, cuHarmWindows, cuNoiseSpecs, cuHarmWindows_next, cuNoiseSpecs_next,
		cuMaxVoicedLists, cuMaxVoicedLists_next, jobMap, BufSize);
}
