#include <cuda_runtime.h>
#include "SentenceGenerator_CUDA.h"

template <class T>
class CUDAVector
{
public:
	CUDAVector()
	{
		count = 0;
		d_data = nullptr;
	}

	~CUDAVector()
	{
		Free();
	}

	unsigned Count() const
	{
		return count;
	}

	T* Pointer()
	{
		return d_data;
	}

	const T* ConstPointer() const
	{
		return d_data;
	}

	operator T*()
	{
		return d_data;
	}

	operator const T*()
	{
		return d_data;
	}

	void Free()
	{
		if (d_data != nullptr)
		{
			cudaFree(d_data);
			d_data = nullptr;
		}
		count = 0;
	}

	void Allocate(unsigned count)
	{
		Free();
		this->count = count;
		if (count>0)
			cudaMalloc(&d_data, sizeof(T)*count);
	}

	const CUDAVector& operator = (const std::vector<T>& cpuVec)
	{
		Free();
		Allocate((unsigned)cpuVec.size());
		if (count > 0)
		{
			cudaMemcpy(d_data, cpuVec.data(), sizeof(T)*count, cudaMemcpyHostToDevice);
		}

		return *this;
	}

	void ToCPU(std::vector<T>& cpuVec) const
	{
		cpuVec.resize(count);
		cudaMemcpy(cpuVec.data(), d_data, sizeof(T)*count, cudaMemcpyDeviceToHost);
	}

private:
	unsigned count;
	T* d_data;

};

template <class T_GPU, class T_CPU>
class CUDAImagedVector
{
public:
	CUDAImagedVector()
	{

	}
	~CUDAImagedVector()
	{
		Free();
	}

	unsigned Count() const
	{
		return m_vec.Count();
	}

	T_GPU* Pointer()
	{
		return m_vec.Pointer();
	}

	const T_GPU* ConstPointer() const
	{
		return m_vec.ConstPointer();
	}

	operator T_GPU*()
	{
		return m_vec.Pointer();
	}

	operator const T_GPU*() const
	{
		return m_vec.ConstPointer();
	}

	void Free()
	{
		if (m_vec.Count() > 0)
		{
			T_GPU* temp = (T_GPU*) ::operator new (m_vec.Count()*sizeof(T_GPU));
			cudaMemcpy(temp, m_vec.ConstPointer(), sizeof(T_GPU)*m_vec.Count(), cudaMemcpyDeviceToHost);
			delete[] temp;
		}
		m_vec.Free();
	}

	const CUDAImagedVector& operator = (const std::vector<T_CPU>& cpuVecs)
	{
		Free();
		m_vec.Allocate((unsigned)cpuVecs.size());
		if (m_vec.Count() > 0)
		{
			T_GPU* temp = new T_GPU[cpuVecs.size()];
			for (unsigned i = 0; i < (unsigned)cpuVecs.size(); i++)
				temp[i] = cpuVecs[i];
			cudaMemcpy(m_vec.Pointer(), temp, sizeof(T_GPU)*m_vec.Count(), cudaMemcpyHostToDevice);
			printf("7\n");
			::operator delete[](temp);
			printf("8\n");
		}
		return *this;
	}

	void ToCPU(std::vector<T_CPU>& cpuVecs) const
	{
		cpuVecs.resize(m_vec.count);
		if (m_vec.Count() > 0)
		{
			T_GPU* temp = (T_GPU*) ::operator new (m_vec.Count()*sizeof(T_GPU));
			cudaMemcpy(temp, m_vec.ConstPointer(), sizeof(T_GPU)*m_vec.Count(), cudaMemcpyDeviceToHost);
			for (unsigned i = 0; i < (unsigned)cpuVecs.size(); i++)
				temp[i].ToCPU(cpuVecs[i]);
			::operator delete[](temp);
		}
	}

protected:
	CUDAVector<T_GPU> m_vec;
};

template <class T>
class CUDALevel2Vector : public CUDAImagedVector<CUDAVector<T>, std::vector<T>>
{
public:
	void Allocate(const std::vector<unsigned>& counts)
	{
		Free();
		m_vec.Allocate((unsigned)counts.size());
		if (m_vec.Count() > 0)
		{
			CUDAVector<T>* temp = new CUDAVector<T>[counts.size()];
			for (unsigned i = 0; i < (unsigned)counts.size(); i++)
				temp[i].Allocate(counts[i]);
			cudaMemcpy(m_vec.Pointer(), temp, sizeof(CUDAVector<T>)*m_vec.Count(), cudaMemcpyHostToDevice);
			operator delete[](temp);
		}
	}

};


class CUDASrcBuf : public CUDAVector<float>
{
public:
	const CUDASrcBuf&  operator = (const SourceInfo& cpuVec)
	{
		CUDAVector<float>::operator=(cpuVec.source.m_data);
		return *this;
	}

	void ToCPU(SourceInfo& cpuVec) const
	{
		CUDAVector<float>::ToCPU(cpuVec.source.m_data);
	}
};

typedef CUDAImagedVector<CUDASrcBuf, SourceInfo> CUDASrcBufList;

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
	CUDAVector<SrcSampleInfo> SampleLocations;
	CUDAVector<SrcSampleInfo> SampleLocations_next;
	unsigned fixedBeginId;
	unsigned fixedEndId;
	unsigned fixedBeginId_next;
	unsigned fixedEndId_next;


	const CUDASrcPieceInfo& operator = (const SrcPieceInfo& cpuVec)
	{
		SampleLocations = cpuVec.SampleLocations;
		SampleLocations_next = cpuVec.SampleLocations_next;
		fixedBeginId = cpuVec.fixedBeginId;
		fixedEndId = cpuVec.fixedEndId;
		fixedBeginId_next = cpuVec.fixedBeginId_next;
		fixedEndId_next = cpuVec.fixedEndId_next;

		return *this;
	}

	void ToCPU(SrcPieceInfo& cpuVec) const
	{
		SampleLocations.ToCPU(cpuVec.SampleLocations);
		SampleLocations_next.ToCPU(cpuVec.SampleLocations_next);
		cpuVec.fixedBeginId = fixedBeginId;
		cpuVec.fixedEndId = fixedEndId;
		cpuVec.fixedBeginId_next = fixedBeginId_next;
		cpuVec.fixedEndId_next = fixedEndId_next;
	}

};

typedef CUDAImagedVector<CUDASrcPieceInfo, SrcPieceInfo> CUDASrcPieceInfoList;

struct Job
{
	unsigned pieceId;
	unsigned isNext;
	unsigned jobOfPiece;
};

void h_GetMaxVoiced(CUDASrcBufList cuSrcBufs, CUDASrcPieceInfoList pieceInfoList,
	CUDALevel2Vector<unsigned> cuMaxVoicedLists, CUDALevel2Vector<unsigned> cuMaxVoicedLists_next, CUDAVector<Job> jobMap, unsigned BufSize);

void SentenceGenerator_CUDA::GenerateSentence(const UtauSourceFetcher& srcFetcher, unsigned numPieces, const std::string* lyrics, const unsigned* isVowel_list, const unsigned* lengths, const float *freqAllMap, NoteBuffer* noteBuf)
{
	std::vector<SourceInfo> srcInfos;
	srcInfos.resize(numPieces);

	for (unsigned i = 0; i < numPieces; i++)
		if (!srcFetcher.FetchSourceInfo(lyrics[i].data(), srcInfos[i])) return;	

	CUDASrcBufList cuSourceBufs;
	printf("d\n");
	cuSourceBufs=srcInfos;
	printf("e\n");

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
	cuSrcPieceInfos = SrcPieceInfos;

	CUDALevel2Vector<unsigned> cuMaxVoicedLists;
	std::vector<unsigned> countMaxVoiceds;
	countMaxVoiceds.resize(numPieces);

	CUDALevel2Vector<unsigned> cuMaxVoicedLists_next;
	std::vector<unsigned> countMaxVoiceds_next;
	countMaxVoiceds_next.resize(numPieces-1);

	for (unsigned i = 0; i < numPieces; i++)
	{
		countMaxVoiceds[i] = SrcPieceInfos[i].fixedEndId - SrcPieceInfos[i].fixedBeginId;
		if (i<numPieces-1)
			countMaxVoiceds_next[i] = SrcPieceInfos[i].fixedEndId_next - SrcPieceInfos[i].fixedBeginId_next;
	}
	cuMaxVoicedLists.Allocate(countMaxVoiceds);
	cuMaxVoicedLists_next.Allocate(countMaxVoiceds_next);

	std::vector<Job> jobMap;
	for (unsigned i = 0; i < numPieces; i++)
	{
		for (unsigned j = 0; j < countMaxVoiceds[i]; j++)
		{
			Job job;
			job.isNext = false;
			job.pieceId = i;
			job.jobOfPiece = j;
			jobMap.push_back(job);
		}

		if (i<numPieces - 1)
			for (unsigned j = 0; j < countMaxVoiceds_next[i]; j++)
			{
				Job job;
				job.isNext = true;
				job.pieceId = i;
				job.jobOfPiece = j;
				jobMap.push_back(job);
			}
	}
	CUDAVector<Job> cuJobMap;
	cuJobMap=jobMap;

	unsigned cuHalfWinLen = (unsigned)ceilf(max_srcHalfWinWidth);
	unsigned cuSpecLen = (unsigned)ceilf(max_srcHalfWinWidth*0.5f);

	unsigned fftLen = 1;
	while (fftLen < max_freqDetectHalfWinWidth)
		fftLen <<= 1;
	unsigned BufSize = (unsigned)ceilf(max_freqDetectHalfWinWidth) * 2 + fftLen * 2;
	printf("BufSize: %u\n", BufSize);

	h_GetMaxVoiced(cuSourceBufs, cuSrcPieceInfos, cuMaxVoicedLists, cuMaxVoicedLists_next, cuJobMap, BufSize);

}
