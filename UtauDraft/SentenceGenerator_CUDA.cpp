#include <cuda_runtime.h>
#include "SentenceGenerator_CUDA.h"

struct SrcSampleInfo
{
	unsigned srcPos;
	float srcSampleFreq;
	float logicalPos;
};

struct SrcPieceInfo
{
	std::vector<SrcSampleInfo> SampleLocations;
	std::vector<SrcSampleInfo> SampleLocations_next;
	unsigned fixedBeginId;
	unsigned fixedEndId;
	unsigned fixedBeginId_next;
	unsigned fixedEndId_next;
};

struct CUDASrcPieceInfo
{
	SrcSampleInfo* d_SampleLocations;
	SrcSampleInfo* d_SampleLocations_next;
	unsigned fixedBeginId;
	unsigned fixedEndId;
	unsigned fixedBeginId_next;
	unsigned fixedEndId_next;
};

struct CUDASrcBuf
{
	unsigned bufSize;
	float* d_buf;
};

template <class T>
class CUDAList
{
	unsigned count;
	T* d_data;

	CUDAList()
	{
		d_data = nullptr;
	}

	~CUDAList()
	{
		if (d_data)
			cudaFree(d_data);
	}

	void Allocate(unsigned count)
	{
		this->count = count;
		if (d_data)
			cudaFree(d_data);
		cudaMalloc(&d_data, sizeof(T)*count);
	}

	void Fill(T* cpuData)
	{
		cudaMemcpy(d_data, cpuData, sizeof(T)*count);
	}

	void AllocateFill(unsigned count, T* cpuData)
	{
		Allocate(count);
		Fill(cpuData);
	}
};

void SentenceGenerator_CUDA::GenerateSentence(const UtauSourceFetcher& srcFetcher, unsigned numPieces, const std::string* lyrics, const unsigned* isVowel_list, const unsigned* lengths, const float *freqAllMap, NoteBuffer* noteBuf)
{
	std::vector<SourceInfo> srcInfos;
	srcInfos.resize(numPieces);

	for (unsigned i = 0; i < numPieces; i++)
		if (!srcFetcher.FetchSourceInfo(lyrics[i].data(), srcInfos[i])) return;	

	std::vector<CUDASrcBuf> cuSourceBufs;
	cuSourceBufs.resize(numPieces);

	for (unsigned i = 0; i < numPieces; i++)
	{
		cuSourceBufs[i].bufSize = (unsigned)srcInfos[i].source.m_data.size();
		cudaMalloc(&cuSourceBufs[i].d_buf, cuSourceBufs[i].bufSize*sizeof(float));
		cudaMemcpy(cuSourceBufs[i].d_buf, srcInfos[i].source.m_data.data(), cuSourceBufs[i].bufSize*sizeof(float), cudaMemcpyHostToDevice);
	}

	CUDASrcBuf* d_cuSourceBufs;
	cudaMalloc(&d_cuSourceBufs, sizeof(CUDASrcBuf)*numPieces);
	cudaMemcpy(d_cuSourceBufs, cuSourceBufs.data(), sizeof(CUDASrcBuf)*numPieces, cudaMemcpyHostToDevice);

	SourceInfo _dummyNext;

	std::vector<SrcPieceInfo> SrcPieceInfos;
	SrcPieceInfos.resize(numPieces);

	std::vector<CUDASrcPieceInfo> cuSrcPieceInfos;
	cuSrcPieceInfos.resize(numPieces);

	float max_srcHalfWinWidth = 0.0f;
	float max_freqDetectHalfWinWidth = 0.0f;

	for (unsigned i = 0; i < numPieces; i++)
	{

		bool firstNote = (i == 0);
		bool hasNextNote = (i < numPieces - 1);

		SourceDerivedInfo srcDerInfo;			 
		srcDerInfo.DeriveInfo(firstNote, hasNextNote, lengths[i], srcInfos[i], hasNextNote ? srcInfos[i + 1] : _dummyNext);

		SrcPieceInfo& srcPieceInfo = SrcPieceInfos[i];

		bool _isVowel = isVowel_list[i] != 0;

		// current note info
		{
			SourceInfo& srcInfo = srcInfos[i];

			float fPeriodCount = 0.0f;

			float fStartPos = firstNote ? srcDerInfo.overlap_pos : srcDerInfo.preutter_pos;
			float logicalPos = 0.0f;

			if (fStartPos < 0.0f)
			{
				logicalPos += (-fStartPos)*(firstNote ? srcDerInfo.headerWeight : srcDerInfo.fixed_Weight);
				fStartPos = 0.0f;
			}
			unsigned startPos = (unsigned)fStartPos;

			unsigned& fixedBeginId = srcPieceInfo.fixedBeginId;
			fixedBeginId = (unsigned)(-1);
			unsigned& fixedEndId = srcPieceInfo.fixedEndId;
			fixedEndId = (unsigned)(-1);
			std::vector<SrcSampleInfo>& SampleLocations = srcPieceInfo.SampleLocations;

			for (unsigned srcPos = startPos; srcPos < srcInfo.source.m_data.size(); srcPos++)
			{
				float srcSampleFreq;
				float srcFreqPos = (srcInfo.srcbegin + (float)srcPos) / (float)srcInfo.frq.m_window_interval;
				unsigned uSrcFreqPos = (unsigned)srcFreqPos;
				float fracSrcFreqPos = srcFreqPos - (float)uSrcFreqPos;

				float freq1 = (float)srcInfo.frq[uSrcFreqPos].freq;
				if (freq1 <= 55.0f) freq1 = (float)srcInfo.frq.m_key_freq;

				float freq2 = (float)srcInfo.frq[uSrcFreqPos + 1].freq;
				if (freq2 <= 55.0f) freq2 = (float)srcInfo.frq.m_key_freq;

				float sampleFreq1 = freq1 / (float)srcInfo.source.m_sampleRate;
				float sampleFreq2 = freq2 / (float)srcInfo.source.m_sampleRate;

				srcSampleFreq = sampleFreq1*(1.0f - fracSrcFreqPos) + sampleFreq2*fracSrcFreqPos;

				unsigned paramId = (unsigned)fPeriodCount;
				if (paramId >= SampleLocations.size())
				{
					bool isVowel = _isVowel && ((float)srcPos >= srcDerInfo.fixed_end || (float)srcPos < srcDerInfo.overlap_pos);
					if (!isVowel && fixedBeginId == (unsigned)(-1))
						fixedBeginId = paramId;
					if (isVowel && fixedBeginId != (unsigned)(-1) && fixedEndId == (unsigned)(-1))
						fixedEndId = paramId;

					SrcSampleInfo sl;
					sl.srcSampleFreq = srcSampleFreq;
					sl.srcPos = srcPos;
					sl.logicalPos = logicalPos;

					SampleLocations.push_back(sl);

					float srcHalfWinWidth = 1.0f / srcSampleFreq;
					if (max_srcHalfWinWidth < srcHalfWinWidth)
						max_srcHalfWinWidth = srcHalfWinWidth;

					if (!isVowel)
					{
						float halfWinlen = 3.0f / srcSampleFreq;
						if (halfWinlen > max_freqDetectHalfWinWidth)
							max_freqDetectHalfWinWidth = halfWinlen;
					}
				}

				fPeriodCount += srcSampleFreq;

				if (firstNote && (float)srcPos < srcDerInfo.preutter_pos)
				{
					logicalPos += srcDerInfo.headerWeight;
				}
				else if ((float)srcPos < srcDerInfo.fixed_end)
				{
					logicalPos += srcDerInfo.fixed_Weight;
				}
				else
				{
					logicalPos += srcDerInfo.vowel_Weight;
				}
			}
		}

		// next note info
		if (hasNextNote)
		{
			SourceInfo& srcInfo = srcInfos[i+1];

			float fPeriodCount = 0.0f;
			float logicalPos = 1.0f - srcDerInfo.preutter_pos_next*srcDerInfo.fixed_Weight;

			unsigned& fixedBeginId = srcPieceInfo.fixedBeginId_next;
			fixedBeginId = (unsigned)(-1);
			unsigned& fixedEndId = srcPieceInfo.fixedEndId_next;
			fixedEndId = (unsigned)(-1);
			std::vector<SrcSampleInfo>& SampleLocations = srcPieceInfo.SampleLocations_next;

			for (unsigned srcPos = 0; (float)srcPos < srcDerInfo.preutter_pos_next; srcPos++)
			{
				float srcSampleFreq;
				float srcFreqPos = (srcInfo.srcbegin + (float)srcPos) / (float)srcInfo.frq.m_window_interval;
				unsigned uSrcFreqPos = (unsigned)srcFreqPos;
				float fracSrcFreqPos = srcFreqPos - (float)uSrcFreqPos;

				float freq1 = (float)srcInfo.frq[uSrcFreqPos].freq;
				if (freq1 <= 55.0f) freq1 = (float)srcInfo.frq.m_key_freq;

				float freq2 = (float)srcInfo.frq[uSrcFreqPos + 1].freq;
				if (freq2 <= 55.0f) freq2 = (float)srcInfo.frq.m_key_freq;

				float sampleFreq1 = freq1 / (float)srcInfo.source.m_sampleRate;
				float sampleFreq2 = freq2 / (float)srcInfo.source.m_sampleRate;

				srcSampleFreq = sampleFreq1*(1.0f - fracSrcFreqPos) + sampleFreq2*fracSrcFreqPos;
				unsigned paramId = (unsigned)fPeriodCount;
				if (paramId >= SampleLocations.size())
				{
					bool isVowel = _isVowel && (float)srcPos < srcDerInfo.overlap_pos_next;
					if (!isVowel && fixedBeginId == (unsigned)(-1))
						fixedBeginId = paramId;
					if (isVowel && fixedBeginId != (unsigned)(-1) && fixedEndId == (unsigned)(-1))
						fixedEndId = paramId;

					SrcSampleInfo sl;
					sl.srcSampleFreq = srcSampleFreq;
					sl.srcPos = srcPos;
					sl.logicalPos = logicalPos;

					SampleLocations.push_back(sl);

					float srcHalfWinWidth = 1.0f / srcSampleFreq;
					if (max_srcHalfWinWidth < srcHalfWinWidth)
						max_srcHalfWinWidth = srcHalfWinWidth;

					if (!isVowel)
					{
						float halfWinlen = 3.0f / srcSampleFreq;
						if (halfWinlen > max_freqDetectHalfWinWidth)
							max_freqDetectHalfWinWidth = halfWinlen;
					}
				}
				fPeriodCount += srcSampleFreq;
				logicalPos += srcDerInfo.fixed_Weight;
			}
		}

		cuSrcPieceInfos[i].fixedBeginId = srcPieceInfo.fixedBeginId;
		cuSrcPieceInfos[i].fixedEndId = srcPieceInfo.fixedEndId;
		cuSrcPieceInfos[i].fixedBeginId_next = srcPieceInfo.fixedBeginId_next;
		cuSrcPieceInfos[i].fixedEndId_next = srcPieceInfo.fixedEndId_next;

		cudaMalloc(&cuSrcPieceInfos[i].d_SampleLocations, srcPieceInfo.SampleLocations.size()*sizeof(SrcSampleInfo));
		cudaMemcpy(cuSrcPieceInfos[i].d_SampleLocations, srcPieceInfo.SampleLocations.data(), srcPieceInfo.SampleLocations.size()*sizeof(SrcSampleInfo), cudaMemcpyHostToDevice);

		cudaMalloc(&cuSrcPieceInfos[i].d_SampleLocations_next, srcPieceInfo.SampleLocations_next.size()*sizeof(SrcSampleInfo));
		cudaMemcpy(cuSrcPieceInfos[i].d_SampleLocations_next, srcPieceInfo.SampleLocations_next.data(), srcPieceInfo.SampleLocations_next.size()*sizeof(SrcSampleInfo), cudaMemcpyHostToDevice);

	}

	CUDASrcPieceInfo *d_cuSrcPieceInfos;
	cudaMalloc(&d_cuSrcPieceInfos, sizeof(CUDASrcPieceInfo)*cuSrcPieceInfos.size());
	cudaMemcpy(d_cuSrcPieceInfos, cuSrcPieceInfos.data(), sizeof(CUDASrcPieceInfo)*cuSrcPieceInfos.size(), cudaMemcpyHostToDevice);

	
	cudaFree(d_cuSrcPieceInfos);
	for (unsigned i = 0; i < numPieces; i++)
	{
		cudaFree(cuSrcPieceInfos[i].d_SampleLocations);
		cudaFree(cuSrcPieceInfos[i].d_SampleLocations_next);
	}

	cudaFree(d_cuSourceBufs);	
	for (unsigned i = 0; i < numPieces; i++)
	{
		cudaFree(cuSourceBufs[i].d_buf);
	}


}
