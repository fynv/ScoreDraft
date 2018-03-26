#include <cuda_runtime.h>
#include "SentenceGenerator_CUDA.h"
#include "TrackBuffer.h"
#include <assert.h>

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

	void Update(const std::vector<T>& cpuVec)
	{
		assert(count == (unsigned)cpuVec.size());
		if (count > 0)
		{
			cudaMemcpy(d_data, cpuVec.data(), sizeof(T)*count, cudaMemcpyHostToDevice);
		}
	}

	void MakeLeaky()
	{
		count = 0;
		d_data = nullptr;
	}

private:
	unsigned count;
	T* d_data;

};

template <class T_GPU, class T_CPU>
class CUDAImagedVector
{
protected:
	class Leaky_T_GPU : public T_GPU
	{
	public:
		Leaky_T_GPU(){}
		~Leaky_T_GPU()
		{
			MakeLeaky();
		}
		using T_GPU::operator =;
	};
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
			std::vector<T_GPU> temp;
			m_vec.ToCPU(temp);
		}
		m_vec.Free();
	}

	const CUDAImagedVector& operator = (const std::vector<T_CPU>& cpuVecs)
	{
		Free();
		m_vec.Allocate((unsigned)cpuVecs.size());
		if (m_vec.Count() > 0)
		{
			std::vector<Leaky_T_GPU> temp;
			temp.resize(cpuVecs.size());
			for (unsigned i = 0; i < (unsigned)cpuVecs.size(); i++)
				temp[i] = cpuVecs[i];
			cudaMemcpy(m_vec.Pointer(), temp.data(), sizeof(T_GPU)*m_vec.Count(), cudaMemcpyHostToDevice);
		}
		return *this;
	}

	void ToCPU(std::vector<T_CPU>& cpuVecs) const
	{
		cpuVecs.resize(m_vec.Count());
		if (m_vec.Count() > 0)
		{
			std::vector<Leaky_T_GPU> temp;
			temp.resize(m_vec.Count());
			cudaMemcpy(temp.data(), m_vec.ConstPointer(), sizeof(T_GPU)*m_vec.Count(), cudaMemcpyDeviceToHost);
			for (unsigned i = 0; i < (unsigned)cpuVecs.size(); i++)
				temp[i].ToCPU(cpuVecs[i]);
		}
	}

	void Update(const std::vector<T_CPU>& cpuVecs)
	{
		assert(m_vec.Count() == (unsigned)cpuVecs.size());
		if (m_vec.Count() > 0)
		{
			std::vector<Leaky_T_GPU> temp;
			temp.resize(m_vec.Count());
			cudaMemcpy(temp.data(), m_vec.ConstPointer(), sizeof(T_GPU)*m_vec.Count(), cudaMemcpyDeviceToHost);
			for (unsigned i = 0; i < (unsigned)cpuVecs.size(); i++)
				temp[i].Update(cpuVecs[i]);
			cudaMemcpy(m_vec.Pointer(), temp.data(), sizeof(T_GPU)*m_vec.Count(), cudaMemcpyHostToDevice);
		}
	}

	void MakeLeaky()
	{
		m_vec.MakeLeaky();
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
			std::vector<Leaky_T_GPU> temp;
			temp.resize(counts.size());
			for (unsigned i = 0; i < (unsigned)counts.size(); i++)
			{
				temp[i].Allocate(counts[i]);
			}
			cudaMemcpy(m_vec.Pointer(), temp.data(), sizeof(CUDAVector<T>)*m_vec.Count(), cudaMemcpyHostToDevice);
		}
	}
	using CUDAImagedVector<CUDAVector<T>, std::vector<T>>::operator =;
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

	void Update(const SourceInfo& cpuVec)
	{
		CUDAVector<float>::Update(cpuVec.source.m_data);
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

	void Update(const SrcPieceInfo& cpuVec)
	{
		SampleLocations.Update(cpuVec.SampleLocations);
		SampleLocations_next.Update(cpuVec.SampleLocations_next);
		fixedBeginId = cpuVec.fixedBeginId;
		fixedEndId = cpuVec.fixedEndId;
		fixedBeginId_next = cpuVec.fixedBeginId_next;
		fixedEndId_next = cpuVec.fixedEndId_next;
	}

	void MakeLeaky()
	{
		SampleLocations.MakeLeaky();
		SampleLocations_next.MakeLeaky();
	}

};

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

void h_GetMaxVoiced(CUDASrcBufList cuSrcBufs, CUDASrcPieceInfoList pieceInfoList,
	CUDALevel2Vector<unsigned> cuMaxVoicedLists, CUDALevel2Vector<unsigned> cuMaxVoicedLists_next, CUDAVector<Job> jobMap, unsigned BufSize);

void h_AnalyzeInput(CUDASrcBufList cuSrcBufs, CUDASrcPieceInfoList pieceInfoList, unsigned halfWinLen,
	unsigned specLen, CUDALevel2Vector<float> cuHarmWindows, CUDALevel2Vector<float> cuNoiseSpecs,
	CUDALevel2Vector<float> cuHarmWindows_next, CUDALevel2Vector<float> cuNoiseSpecs_next,
	CUDALevel2Vector<unsigned> cuMaxVoicedLists, CUDALevel2Vector<unsigned> cuMaxVoicedLists_next, CUDAVector<Job> jobMap, unsigned BufSize);

void h_Synthesis(CUDASrcPieceInfoList cuSrcPieceInfos, unsigned halfWinLen, unsigned specLen,
	CUDALevel2Vector<float> cuHarmWindows, CUDALevel2Vector<float> cuNoiseSpecs,
	CUDALevel2Vector<float> cuHarmWindows_next, CUDALevel2Vector<float> cuNoiseSpecs_next,
	CUDAVector<DstPieceInfo> cuDstPieceInfos, CUDAVector<CUDATempBuffer> cuTmpBufs1, CUDAVector<CUDATempBuffer> cuTmpBufs2,
	CUDAVector<float> cuRandPhase, CUDAVector<SynthJobInfo> cuSynthJobs, unsigned BufSize);

void h_Merge2Bufs(unsigned uSumLen, float *d_destBuf1, float *d_destBuf2);

void SentenceGenerator_CUDA::GenerateSentence(const UtauSourceFetcher& srcFetcher, unsigned numPieces, const std::string* lyrics, const unsigned* isVowel_list, const unsigned* lengths, const float *freqAllMap, NoteBuffer* noteBuf)
{
	std::vector<SourceInfo> srcInfos;
	srcInfos.resize(numPieces);

	for (unsigned i = 0; i < numPieces; i++)
		if (!srcFetcher.FetchSourceInfo(lyrics[i].data(), srcInfos[i])) return;	

	CUDASrcBufList cuSourceBufs;
	cuSourceBufs=srcInfos;

	SourceInfo _dummyNext;

	std::vector<SrcPieceInfo> SrcPieceInfos;
	SrcPieceInfos.resize(numPieces);

	std::vector<SourceDerivedInfo> SrcDerInfos;
	SrcDerInfos.resize(numPieces);

	float max_srcHalfWinWidth = 0.0f;
	float max_freqDetectHalfWinWidth = 0.0f;


	for (unsigned i = 0; i < numPieces; i++)
	{

		bool firstNote = (i == 0);
		bool hasNextNote = (i < numPieces - 1);

		SourceDerivedInfo& srcDerInfo = SrcDerInfos[i];
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
			if (fixedBeginId != (unsigned)(-1) && fixedEndId == (unsigned)(-1))
				fixedEndId = (unsigned)SampleLocations.size();
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
			if (fixedBeginId != (unsigned)(-1) && fixedEndId == (unsigned)(-1))
				fixedEndId = (unsigned)SampleLocations.size();
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
		if (SrcPieceInfos[i].fixedBeginId != -1)
			countMaxVoiceds[i] = SrcPieceInfos[i].fixedEndId - SrcPieceInfos[i].fixedBeginId;
		else
			countMaxVoiceds[i] = 0;

		if (i < numPieces - 1)
		{
			if (SrcPieceInfos[i].fixedBeginId_next != -1)
				countMaxVoiceds_next[i] = SrcPieceInfos[i].fixedEndId_next - SrcPieceInfos[i].fixedBeginId_next;
			else
				countMaxVoiceds_next[i] = 0;
		}
	}
	cuMaxVoicedLists.Allocate(countMaxVoiceds);
	cuMaxVoicedLists_next.Allocate(countMaxVoiceds_next);

	std::vector<Job> jobMap;
	for (unsigned i = 0; i < numPieces; i++)
	{
		for (unsigned j = 0; j < countMaxVoiceds[i]; j++)
		{
			Job job;
			job.isNext = 0;
			job.pieceId = i;
			job.jobOfPiece = j;
			jobMap.push_back(job);
		}

		if (i<numPieces - 1)
			for (unsigned j = 0; j < countMaxVoiceds_next[i]; j++)
			{
				Job job;
				job.isNext = 1;
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
	while ((float)fftLen < max_freqDetectHalfWinWidth)
		fftLen <<= 1;
	unsigned BufSize = (unsigned)ceilf(max_freqDetectHalfWinWidth) * 2 + fftLen * 2;
	//printf("BufSize: %u\n", BufSize);

	h_GetMaxVoiced(cuSourceBufs, cuSrcPieceInfos, cuMaxVoicedLists, cuMaxVoicedLists_next, cuJobMap, BufSize);

	std::vector<std::vector<unsigned>> h_maxVoicedLists;
	cuMaxVoicedLists.ToCPU(h_maxVoicedLists);

	/*std::vector<std::vector<unsigned>> h_maxVoicedLists_next;
	cuMaxVoicedLists_next.ToCPU(h_maxVoicedLists_next);

	FILE *fp = fopen("dump_gpu.txt","a");
	for (unsigned i = 0; i < h_maxVoicedLists.size(); i++)
	{
		std::vector<unsigned>& sublist = h_maxVoicedLists[i];
		for (unsigned j = 0; j < sublist.size(); j++)
		{
			fprintf(fp, "%u ", sublist[j]);
		}
		fprintf(fp, "\n");
		if (i < h_maxVoicedLists_next.size())
		{
			std::vector<unsigned>& sublist = h_maxVoicedLists_next[i];
			for (unsigned j = 0; j < sublist.size(); j++)
			{
				fprintf(fp, "%u ", sublist[j]);
			}
			fprintf(fp, "\n");
		}
	}

	fclose(fp);*/
	
	for (unsigned i = 0; i < h_maxVoicedLists.size(); i++)
	{
		std::vector<unsigned>& sublist = h_maxVoicedLists[i];
		SrcPieceInfo& srcPieceInfo = SrcPieceInfos[i];
		SourceDerivedInfo& srcDerInfo = SrcDerInfos[i];

		unsigned lastmaxVoiced = 0;
		for (unsigned j = 0; j < sublist.size(); j++)
		{
			unsigned srcPos = srcPieceInfo.SampleLocations[j + srcPieceInfo.fixedBeginId].srcPos;
			if ((float)srcPos >= srcDerInfo.preutter_pos && sublist[j] < lastmaxVoiced)
			{
				sublist[j] = lastmaxVoiced;
			}
			lastmaxVoiced = sublist[j];
		}
	}
	cuMaxVoicedLists.Update(h_maxVoicedLists);

	std::vector<unsigned> cuTotalHalfWinLen;
	cuTotalHalfWinLen.resize(numPieces);
	std::vector<unsigned> cuTotalSpecLen;
	cuTotalSpecLen.resize(numPieces);

	std::vector<unsigned> cuTotalHalfWinLen_next;
	cuTotalHalfWinLen_next.resize(numPieces-1);
	std::vector<unsigned> cuTotalSpecLen_next;
	cuTotalSpecLen_next.resize(numPieces-1);

	for (unsigned i = 0; i < numPieces; i++)
	{
		cuTotalHalfWinLen[i] = cuHalfWinLen*(unsigned)SrcPieceInfos[i].SampleLocations.size();
		cuTotalSpecLen[i] = cuSpecLen*(unsigned)SrcPieceInfos[i].SampleLocations.size();

		if (i < numPieces - 1)
		{
			cuTotalHalfWinLen_next[i] = cuHalfWinLen*(unsigned)SrcPieceInfos[i].SampleLocations_next.size();
			cuTotalSpecLen_next[i] = cuSpecLen*(unsigned)SrcPieceInfos[i].SampleLocations_next.size();
		}
	}

	CUDALevel2Vector<float> cuHarmWindows;
	cuHarmWindows.Allocate(cuTotalHalfWinLen);
	CUDALevel2Vector<float> cuNoiseSpecs;
	cuNoiseSpecs.Allocate(cuTotalSpecLen);

	CUDALevel2Vector<float> cuHarmWindows_next;
	cuHarmWindows_next.Allocate(cuTotalHalfWinLen_next);
	CUDALevel2Vector<float> cuNoiseSpecs_next;
	cuNoiseSpecs_next.Allocate(cuTotalSpecLen_next);

	fftLen = 1;
	while (fftLen < cuHalfWinLen)
		fftLen <<= 1;

	BufSize = cuHalfWinLen * 2 + fftLen * 2;

	jobMap.clear();
	for (unsigned i = 0; i < numPieces; i++)
	{
		for (unsigned j = 0; j < (unsigned)SrcPieceInfos[i].SampleLocations.size(); j++)
		{
			Job job;
			job.isNext = 0;
			job.pieceId = i;
			job.jobOfPiece = j;
			jobMap.push_back(job);
		}

		if (i<numPieces - 1)
			for (unsigned j = 0; j < (unsigned)SrcPieceInfos[i].SampleLocations_next.size(); j++)
			{
				Job job;
				job.isNext = 1;
				job.pieceId = i;
				job.jobOfPiece = j;
				jobMap.push_back(job);
			}
	}
	cuJobMap = jobMap;

	h_AnalyzeInput(cuSourceBufs, cuSrcPieceInfos, cuHalfWinLen, cuSpecLen, cuHarmWindows, cuNoiseSpecs, cuHarmWindows_next,
		cuNoiseSpecs_next, cuMaxVoicedLists, cuMaxVoicedLists_next, cuJobMap, BufSize);

	/*std::vector<std::vector<float>> HarmWindows;
	cuHarmWindows.ToCPU(HarmWindows);
	std::vector<std::vector<float>> HarmWindows_next;
	cuHarmWindows_next.ToCPU(HarmWindows_next);

	for (unsigned i = 0; i < numPieces; i++)
	{
		char filename[100];
		sprintf(filename, "dump\\%d_a.raw", i);
		FILE *fp = fopen(filename, "wb");
		fwrite(HarmWindows[i].data(), sizeof(float), HarmWindows[i].size(), fp);
		fclose(fp);
	}

	for (unsigned i = 0; i < numPieces-1; i++)
	{
		char filename[100];
		sprintf(filename, "dump\\%d_b.raw", i);
		FILE *fp = fopen(filename, "wb");
		fwrite(HarmWindows_next[i].data(), sizeof(float), HarmWindows_next[i].size(), fp);
		fclose(fp);
	}
	*/

	std::vector<const float*> freqMaps;
	freqMaps.resize(numPieces);
	std::vector<std::vector<float>> stretchingMaps;
	stretchingMaps.resize(numPieces);
	std::vector<DstPieceInfo> DstPieceInfos;
	DstPieceInfos.resize(numPieces);

	unsigned noteBufPos = 0;
	unsigned sumTmpBufLen = 0;
	for (unsigned i = 0; i < numPieces; i++)
	{
		DstPieceInfo& dstPieceInfo = DstPieceInfos[i];
		dstPieceInfo.uSumLen = lengths[i];
		freqMaps[i] = freqAllMap + noteBufPos;
		dstPieceInfo.minSampleFreq = FLT_MAX;
		for (unsigned pos = 0; pos < dstPieceInfo.uSumLen; pos++)
		{
			float sampleFreq = freqMaps[i][pos];
			if (sampleFreq < dstPieceInfo.minSampleFreq) dstPieceInfo.minSampleFreq = sampleFreq;
		}
		stretchingMaps[i].resize(dstPieceInfo.uSumLen);

		float pos_tmpBuf = 0.0f;
		for (unsigned pos = 0; pos < dstPieceInfo.uSumLen; pos++)
		{
			float sampleFreq;
			sampleFreq = freqMaps[i][pos];

			float speed = sampleFreq / dstPieceInfo.minSampleFreq;
			pos_tmpBuf += speed;
			stretchingMaps[i][pos] = pos_tmpBuf;
		}
		dstPieceInfo.tempLen = stretchingMaps[i][dstPieceInfo.uSumLen - 1];
		dstPieceInfo.uTempLen = (unsigned)ceilf(dstPieceInfo.tempLen);

		sumTmpBufLen += dstPieceInfo.uTempLen;

		noteBufPos += dstPieceInfo.uSumLen;
	}

	std::vector<SynthJobInfo> SynthJobs;

	float phase = 0.0f;
	unsigned maxRandPhaseLen = 0;
	float maxtempHalfWinLen = 0.0f;
	for (unsigned i = 0; i < numPieces; i++)
	{
		DstPieceInfo& dstPieceInfo = DstPieceInfos[i];
		float tempHalfWinLen = 1.0f / dstPieceInfo.minSampleFreq;
		if (tempHalfWinLen > maxtempHalfWinLen)
			maxtempHalfWinLen = tempHalfWinLen;

		unsigned paramId0 = 0;
		unsigned paramId0_next = 0;
		unsigned pos_final = 0;
		
		while (phase > -1.0f) phase -= 1.0f;

		SourceDerivedInfo& srcDerInfo = SrcDerInfos[i];

		float transitionEnd = 1.0f - (srcDerInfo.preutter_pos_next - srcDerInfo.overlap_pos_next)*srcDerInfo.fixed_Weight;
		float transitionStart = transitionEnd* (1.0f - _transition);

		float tempLen = dstPieceInfo.tempLen;
		unsigned uSumLen = dstPieceInfo.uSumLen;

		const float* freqMap = freqMaps[i];
		std::vector<float>& stretchingMap = stretchingMaps[i];

		SrcPieceInfo& srcPieceInfo = SrcPieceInfos[i];
		std::vector<SrcSampleInfo>& SampleLocations = srcPieceInfo.SampleLocations;
		std::vector<SrcSampleInfo>& SampleLocations_next = srcPieceInfo.SampleLocations_next;

		bool hasNextNote = (i < numPieces - 1);
		if (hasNextNote)
		{
			if (SampleLocations_next.size() == 0)
			{
				hasNextNote = false;
			}
		}

		float fTmpWinCenter = phase*tempHalfWinLen;
		dstPieceInfo.fTmpWinCenter0 = fTmpWinCenter;
		unsigned jobOfPiece = 0;

		while (fTmpWinCenter - tempHalfWinLen <= tempLen)
		{
			SynthJobInfo synthJob;
			synthJob.pieceId = i;
			synthJob.jobOfPiece = jobOfPiece;

			while (fTmpWinCenter > stretchingMap[pos_final] && pos_final < uSumLen - 1) pos_final++;
			float fParamPos = (float)pos_final / float(uSumLen);
			bool in_transition = hasNextNote && _transition > 0.0f && _transition < 1.0f && fParamPos >= transitionStart;

			float destSampleFreq;
			destSampleFreq = freqMap[pos_final];
			float destHalfWinLen = powf(2.0f, _gender) / destSampleFreq;
			synthJob.destHalfWinLen = destHalfWinLen;

			unsigned paramId1 = paramId0 + 1;
			while (paramId1 < SampleLocations.size() && SampleLocations[paramId1].logicalPos < fParamPos)
			{
				paramId0++;
				paramId1 = paramId0 + 1;
			}
			if (paramId1 == SampleLocations.size()) paramId1 = paramId0;
			synthJob.paramId0 = paramId0;
			synthJob.paramId0_next = 0;

			unsigned paramId1_next = paramId0_next + 1;
			if (in_transition)
			{
				while (paramId1_next < SampleLocations_next.size() && SampleLocations_next[paramId1_next].logicalPos < fParamPos)
				{
					paramId0_next++;
					paramId1_next = paramId0_next + 1;
				}
				if (paramId1_next == SampleLocations_next.size()) paramId1_next = paramId0_next;

				synthJob.paramId0_next = paramId0_next;
			}

			SrcSampleInfo& sl0 = SampleLocations[paramId0];
			SrcSampleInfo& sl1 = SampleLocations[paramId1];

			float k;
			if (fParamPos >= sl1.logicalPos) k = 1.0f;
			else if (fParamPos <= sl0.logicalPos) k = 0.0f;
			else
			{
				k = (fParamPos - sl0.logicalPos) / (sl1.logicalPos - sl0.logicalPos);
			}
			synthJob.k1 = k;
			synthJob.k1_next = 0.0f;
			synthJob.k2 = 0.0f;

			if (in_transition)
			{
				SrcSampleInfo& sl0_next = SampleLocations_next[paramId0_next];
				SrcSampleInfo& sl1_next = SampleLocations_next[paramId1_next];
				float k;
				if (fParamPos >= sl1_next.logicalPos) k = 1.0f;
				else if (fParamPos <= sl0_next.logicalPos) k = 0.0f;
				else
				{
					k = (fParamPos - sl0_next.logicalPos) / (sl1_next.logicalPos - sl0_next.logicalPos);
				}
				synthJob.k1_next = k;
				float x = (fParamPos - transitionEnd) / (transitionEnd*_transition);
				if (x > 0.0f)
					synthJob.k2 = 1.0f;
				else
					synthJob.k2 = 0.5f*(cosf(x*(float)PI) + 1.0f);
			}

			SynthJobs.push_back(synthJob);

			jobOfPiece++;
			fTmpWinCenter += tempHalfWinLen;
		}

		unsigned uSpecLen = (unsigned)ceilf(tempHalfWinLen*0.5f);
		unsigned randPhaseLen = uSpecLen*jobOfPiece;
		if (randPhaseLen > maxRandPhaseLen)
			maxRandPhaseLen = randPhaseLen;

		phase = (fTmpWinCenter - tempLen) / tempHalfWinLen;
	}

	CUDAVector<DstPieceInfo> cuDstPieceInfos;
	cuDstPieceInfos = DstPieceInfos;

	CUDAVector<SynthJobInfo> cuSynthJobs;
	cuSynthJobs = SynthJobs;

	CUDAVector<float> cuSumTmpBuf1;
	CUDAVector<float> cuSumTmpBuf2;
	cuSumTmpBuf1.Allocate(sumTmpBufLen);
	cuSumTmpBuf2.Allocate(sumTmpBufLen);

	cudaMemset(cuSumTmpBuf1, 0, sizeof(float)*sumTmpBufLen);
	cudaMemset(cuSumTmpBuf2, 0, sizeof(float)*sumTmpBufLen);

	std::vector<CUDATempBuffer> tmpBufs1;
	tmpBufs1.resize(numPieces);
	std::vector<CUDATempBuffer> tmpBufs2;
	tmpBufs2.resize(numPieces);

	float *pTmpBuf1 = cuSumTmpBuf1.Pointer();
	float *pTmpBuf2 = cuSumTmpBuf2.Pointer();
	for (unsigned i = 0; i < numPieces; i++)
	{
		unsigned count = DstPieceInfos[i].uTempLen;
		tmpBufs1[i].count = count;
		tmpBufs1[i].d_data = pTmpBuf1;
		tmpBufs2[i].count = count;
		tmpBufs2[i].d_data = pTmpBuf2;
		pTmpBuf1 += count;
		pTmpBuf2 += count;
	}

	CUDAVector<CUDATempBuffer> cuTmpBufs1;
	cuTmpBufs1 = tmpBufs1;
	CUDAVector<CUDATempBuffer> cuTmpBufs2;
	cuTmpBufs2 = tmpBufs2;

	std::vector<float> randPhase;
	randPhase.resize(maxRandPhaseLen);

	for (unsigned i = 0; i < maxRandPhaseLen; i++)
		randPhase[i] = rand01();

	CUDAVector<float> cuRandPhase;
	cuRandPhase = randPhase;

	fftLen = 1;
	while ((float)fftLen < maxtempHalfWinLen)
		fftLen <<= 1;

	BufSize = fftLen * 4;
	//printf("BufSize: %u\n", BufSize);

	h_Synthesis(cuSrcPieceInfos, cuHalfWinLen, cuSpecLen, cuHarmWindows, cuNoiseSpecs, cuHarmWindows_next, cuNoiseSpecs_next,
		cuDstPieceInfos, cuTmpBufs1, cuTmpBufs2, cuRandPhase, cuSynthJobs, BufSize);

	h_Merge2Bufs(sumTmpBufLen, cuSumTmpBuf1, cuSumTmpBuf2);

	std::vector<float> sumTmpBuf;
	cuSumTmpBuf1.ToCPU(sumTmpBuf);
	/*FILE *fp = fopen("dmp.raw", "wb");
	fwrite(sumTmpBuf.data(), sizeof(float), sumTmpBuf.size(), fp);
	fclose(fp);
	*/

	float* pTmpBuf = &sumTmpBuf[0];
	float* pDstBuf = noteBuf->m_data;
	for (unsigned i = 0; i < numPieces; i++)
	{
		unsigned uSumLen = DstPieceInfos[i].uSumLen;
		float *stretchingMap = &stretchingMaps[i][0];
		const float *freqMap = freqMaps[i];
		float minSampleFreq = DstPieceInfos[i].minSampleFreq;
		unsigned uTempLen = DstPieceInfos[i].uTempLen;

		for(unsigned pos = 0; pos < uSumLen; pos++)
		{
			float pos_tmpBuf = stretchingMap[pos];
			float sampleFreq;
			sampleFreq = freqMap[pos];

			float speed = sampleFreq / minSampleFreq;

			int ipos1 = (int)ceilf(pos_tmpBuf - speed*0.5f);
			int ipos2 = (int)floorf(pos_tmpBuf + speed*0.5f);

			float sum = 0.0f;
			for (int ipos = ipos1; ipos <= ipos2; ipos++)
			{
				if (ipos >= 0 && ipos < uTempLen)
					sum += pTmpBuf[ipos];
			}
			float value = sum / (float)(ipos2 - ipos1 + 1);
			pDstBuf[pos] = value;
		}
		pTmpBuf += uTempLen;
		pDstBuf += uSumLen;
	}

	

}
