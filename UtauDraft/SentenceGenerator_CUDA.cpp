#include <cuda_runtime.h>
#include "SentenceGenerator_CUDA.h"

template <class T>
class CUDAList
{
public:
	unsigned count;
	T* d_data;

	CUDAList()
	{
		count = 0;
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
		if (count > 0)
			cudaMalloc(&d_data, sizeof(T)*count);
		else
			d_data = nullptr;

	}

	void Fill(const T* cpuData)
	{
		if (count>0)
			cudaMemcpy(d_data, cpuData, sizeof(T)*count, cudaMemcpyHostToDevice);
	}

	void AllocateFill(unsigned count, const T* cpuData)
	{
		Allocate(count);
		if (count>0)
			Fill(cpuData);
	}

	void AllocateFill(const std::vector<T>& cpuData)
	{
		Allocate((unsigned)cpuData.size());
		if (count>0)
			Fill(cpuData.data());
	}
};

template <class T>
class ImagedCUDAList
{
public:
	std::vector<T> cpuList;
	CUDAList<T> gpuList;

	void SyncCPUToGPU()
	{
		gpuList.AllocateFill(cpuList);
	}
};

typedef CUDAList<float> CUDASrcBuf;

class CUDASrcBufList : public ImagedCUDAList<CUDASrcBuf>
{
public:
	void AllocateFill(const std::vector<SourceInfo>& sourceInfoList)
	{
		cpuList.resize(sourceInfoList.size());
		for (unsigned i = 0; i < (unsigned)sourceInfoList.size(); i++)
		{
			cpuList[i].AllocateFill(sourceInfoList[i].source.m_data);
		}

		SyncCPUToGPU();
	}

};


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
	CUDAList<SrcSampleInfo> SampleLocations;
	CUDAList<SrcSampleInfo> SampleLocations_next;
	unsigned fixedBeginId;
	unsigned fixedEndId;
	unsigned fixedBeginId_next;
	unsigned fixedEndId_next;
};

class CUDASrcPieceInfoList : public ImagedCUDAList<CUDASrcPieceInfo>
{
public:
	void AllocateFill(const std::vector<SrcPieceInfo>& srcPieceList)
	{
		cpuList.resize(srcPieceList.size());
		for (unsigned i = 0; i < (unsigned)srcPieceList.size(); i++)
		{
			cpuList[i].fixedBeginId = srcPieceList[i].fixedBeginId;
			cpuList[i].fixedBeginId_next = srcPieceList[i].fixedBeginId_next;
			cpuList[i].fixedEndId = srcPieceList[i].fixedEndId;
			cpuList[i].fixedEndId_next = srcPieceList[i].fixedEndId_next;
			cpuList[i].SampleLocations.AllocateFill(srcPieceList[i].SampleLocations);
			cpuList[i].SampleLocations_next.AllocateFill(srcPieceList[i].SampleLocations_next);
		}
		SyncCPUToGPU();
	}
};

void SentenceGenerator_CUDA::GenerateSentence(const UtauSourceFetcher& srcFetcher, unsigned numPieces, const std::string* lyrics, const unsigned* isVowel_list, const unsigned* lengths, const float *freqAllMap, NoteBuffer* noteBuf)
{
	std::vector<SourceInfo> srcInfos;
	srcInfos.resize(numPieces);

	for (unsigned i = 0; i < numPieces; i++)
		if (!srcFetcher.FetchSourceInfo(lyrics[i].data(), srcInfos[i])) return;	

	CUDASrcBufList cuSourceBufs;
	cuSourceBufs.AllocateFill(srcInfos);

	SourceInfo _dummyNext;

	std::vector<SrcPieceInfo> SrcPieceInfos;
	SrcPieceInfos.resize(numPieces);

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
	}
	
	CUDASrcPieceInfoList cuSrcPieceInfos;
	cuSrcPieceInfos.AllocateFill(SrcPieceInfos);	

}
