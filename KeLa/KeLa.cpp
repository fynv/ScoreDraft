#include "PyScoreDraft.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <dlfcn.h>
#endif

#include <string.h>
#include <cmath>
#include <ReadWav.h>
#include "FrequencyDetection.h"
#include <float.h>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#include "fft.h"
#include "VoiceUtil.h"
using namespace VoiceUtil;

struct SymmetricWindowWithPosition
{
	SymmetricWindow win;
	float center;
};


void DetectFreqs(const Buffer& buf, std::vector<float>& frequencies, unsigned step, float& ave_freq)
{
	unsigned halfWinLen = 1024;
	float* temp = new float[halfWinLen * 2];

	for (unsigned center = 0; center < buf.m_data.size(); center += step)
	{
		Window win;
		win.CreateFromBuffer(buf, (float)center, (float)halfWinLen);

		for (int i = -(int)halfWinLen; i < (int)halfWinLen; i++)
			temp[i + halfWinLen] = win.GetSample(i);

		float freq = fetchFrequency(halfWinLen * 2, temp, buf.m_sampleRate);

		frequencies.push_back(freq);
	}

	delete[] temp;

	struct Range
	{
		unsigned begin;
		unsigned end;

		Range()
		{
			begin = (unsigned)(-1);
			end = (unsigned)(-1);
		}
		unsigned Length()
		{
			return end - begin;
		}
	};

	Range BestRange;
	Range CurRange;

	for (size_t i = 0; i < frequencies.size(); i++)
	{
		if (frequencies[i] > 0)
		{
			if (CurRange.begin == (unsigned)(-1))
			{
				CurRange.begin = (unsigned)i;
			}
			else
			{
				CurRange.end = (unsigned)i;
				if (CurRange.Length() > BestRange.Length()) BestRange = CurRange;
			}
		}
		else
		{
			CurRange.begin = (unsigned)(-1);
		}
	}

	ave_freq = 0.0f;
	float count = 0.0f;
	unsigned sampleBegin = (unsigned) (-1);
	size_t sampleEnd = (unsigned)(-1);
	for (size_t i = 0; i < frequencies.size(); i++)
	{
		if ((i<BestRange.begin || i>BestRange.end) && frequencies[i] > 0.0f)
		{
			frequencies[i] = 0.0f;
		}
		if (frequencies[i] >= 0.0f)
		{
			if (sampleBegin == (unsigned)(-1))
				sampleBegin = (unsigned)i;
			sampleEnd = (unsigned)i+1;
		}		
		if (frequencies[i] > 0.0f)
		{
			ave_freq += frequencies[i];
			count += 1.0f;
		}
	}
	ave_freq /= count;

	for (unsigned i = sampleBegin; i < sampleEnd; i++)
		if (frequencies[i] < 0.0f) frequencies[i] = 0.0f;

}

class KeLa : public Singer
{
public:
	void SetName(const char* name)
	{
		m_name = name;
	}
	virtual void GenerateWave(const char* lyric, std::vector<SingerNoteParams> notes, VoiceBuffer* noteBuf)
	{
		if (notes.size() < 1) return;

		float minSampleFreq = FLT_MAX;
		float sumLen = 0.0f;
		for (size_t i = 0; i < notes.size(); i++)
		{
			float sampleFreq = notes[i].sampleFreq;
			if (sampleFreq < minSampleFreq) minSampleFreq = sampleFreq;
			sumLen += notes[i].fNumOfSamples;
		}
		unsigned uSumLen = (unsigned)ceilf(sumLen);		
		float *freqMap = new float[uSumLen];

		unsigned pos = 0;
		float targetPos = 0.0f;
		float sampleFreq;
		for (size_t i = 0; i < notes.size(); i++)
		{
			sampleFreq = notes[i].sampleFreq;
			targetPos += notes[i].fNumOfSamples;
		
			for (; (float)pos < targetPos; pos++)
			{
				freqMap[pos] = sampleFreq;
			}
		}
		for (; pos < uSumLen; pos++)
		{
			freqMap[pos] = sampleFreq;
		}

		/// Make frequency tweakings here

		/// calculate finalBuffer->tmpBuffer map
		float* stretchingMap = new float[uSumLen];
		
		float pos_tmpBuf = 0.0f;
		for (pos = 0; pos < uSumLen; pos++)
		{
			sampleFreq = freqMap[pos];
			float speed = sampleFreq / minSampleFreq;
			pos_tmpBuf += speed;
			stretchingMap[pos] = pos_tmpBuf;
		}

		_generateWave(lyric, sumLen, minSampleFreq, freqMap, stretchingMap, noteBuf);

		delete[] stretchingMap;
		delete[] freqMap;	

	}
private:
	void _generateWave(const char* lyric, float sumLen, float minSampleFreq, float* freqMap, float* stretchingMap, VoiceBuffer* noteBuf)
	{
		unsigned uSumLen = (unsigned)ceilf(sumLen);

		char path[1024];
		sprintf(path, "KeLaSamples/%s/%s.wav", m_name.data(), lyric);

		Buffer source;
		float maxv;
		if (!ReadWavToBuffer(path, source, maxv)) return;

		unsigned freq_step = 256;
		std::vector<float> frequencies;
		float ave_freq;

		sprintf(path, "KeLaSamples/%s/%s.freq", m_name.data(), lyric);
		FILE* fp = fopen(path, "r");
		if (fp)
		{
			ave_freq = 0.0f;
			float count = 0.0f;
			while (!feof(fp))
			{
				float f;
				if (fscanf(fp, "%f", &f))
				{
					frequencies.push_back(f);
				}
				else break;

				if (f > 0.0f)
				{
					ave_freq += f;
					count += 1.0f;
				}
			}
			fclose(fp);

			ave_freq /= count;
		}
		else
		{
			DetectFreqs(source, frequencies, freq_step, ave_freq);
			fp = fopen(path, "w");
			for (size_t i = 0; i < frequencies.size(); i++)
			{
				fprintf(fp, "%f\n", frequencies[i]);
			}
			fclose(fp);
		}

		unsigned unvoicedBegin = (unsigned)(-1);
		unsigned voicedBegin = (unsigned)(-1);
		unsigned voicedEnd = (unsigned)(-1);
		unsigned unvoicedEnd = (unsigned)(-1);

		unsigned voicedBegin_id = (unsigned)(-1);
		unsigned voicedEnd_id = (unsigned)(-1);

		float firstFreq;
		float lastFreq;

		for (size_t i = 0; i < frequencies.size(); i++)
		{
			if (frequencies[i] >= 0.0f && unvoicedBegin == (unsigned)(-1))
				unvoicedBegin = i>0 ? (unsigned)i*freq_step - (freq_step / 2) : 0;
			if (frequencies[i] > 0.0f && voicedBegin == (unsigned)(-1))
			{
				voicedBegin_id = (unsigned)i;
				voicedBegin = i > 0 ? (unsigned)i*freq_step - (freq_step / 2) : 0;
				firstFreq = frequencies[i];
			}

			if (frequencies[i] <= 0.0f && voicedBegin != (unsigned)(-1) && voicedEnd == (unsigned)(-1))
			{
				voicedEnd_id = (unsigned)i;
				voicedEnd = (unsigned)i*freq_step - (freq_step / 2);
				lastFreq = frequencies[i - 1];
			}

			if (frequencies[i] < 0.0f && voicedEnd != (unsigned)(-1) && unvoicedEnd == (unsigned)(-1))
			{
				unvoicedEnd = (unsigned)i*freq_step - (freq_step / 2);
				break;
			}
		}
		if (voicedEnd == (unsigned)(-1))
		{
			voicedEnd = (unsigned)source.m_data.size();
			lastFreq = frequencies[frequencies.size() - 1];
		}
		if (unvoicedEnd == (unsigned)(-1))
		{
			unvoicedEnd = (unsigned)source.m_data.size();
		}

		std::vector<SymmetricWindow> windows;
		float fPeriodCount = 0.0f;
		for (unsigned srcPos = voicedBegin; srcPos < voicedEnd; srcPos++)
		{
			float srcFreqPos = (float)srcPos / (float)freq_step;
			unsigned uSrcFreqPos = (unsigned)srcFreqPos;
			float fracSrcFreqPos = srcFreqPos - (float)uSrcFreqPos;

			float srcFreq;
			if (uSrcFreqPos >= frequencies.size() - 1)
			{
				if (frequencies.size() - 1 >= voicedEnd_id) srcFreq = lastFreq;
				else srcFreq = frequencies[frequencies.size() - 1];
			}
			else
			{
				float freq1;
				if (uSrcFreqPos < voicedBegin_id) freq1 = firstFreq;
				else if (uSrcFreqPos >= voicedEnd_id) freq1 = lastFreq;
				else freq1 = frequencies[uSrcFreqPos];

				float freq2;
				if (uSrcFreqPos + 1 < voicedBegin_id) freq2 = firstFreq;
				else if (uSrcFreqPos + 1 >= voicedEnd_id) freq2 = lastFreq;
				else freq2 = frequencies[uSrcFreqPos + 1];

				srcFreq = freq1*(1.0f - fracSrcFreqPos) + freq2*fracSrcFreqPos;
			}

			unsigned winId = (unsigned)fPeriodCount;
			if (winId >= windows.size())
			{
				float srcHalfWinWidth = (float)source.m_sampleRate / srcFreq;
				Window srcWin;
				srcWin.CreateFromBuffer(source, (float)srcPos, srcHalfWinWidth);

				SymmetricWindow symWin;
				symWin.CreateFromAsymmetricWindow(srcWin);

				windows.push_back(symWin);
			}

			fPeriodCount += srcFreq / (float)source.m_sampleRate;

		}

		float tempLen = stretchingMap[uSumLen - 1];
		unsigned uTempLen = (unsigned)ceilf(tempLen);
		Buffer tempBuf;
		tempBuf.m_sampleRate = source.m_sampleRate;
		tempBuf.m_data.resize(uTempLen);
		tempBuf.SetZero();

		float rateUnvoiced0 = minSampleFreq * (float)source.m_sampleRate / firstFreq;
		float lenUnvoiced0 = (float)(voicedBegin - unvoicedBegin) / rateUnvoiced0;

		float rateUnvoiced1 = minSampleFreq * (float)source.m_sampleRate / lastFreq;
		float lenUnvoiced1 = (float)(unvoicedEnd - voicedEnd) / rateUnvoiced1;

		if (lenUnvoiced0 > tempLen)
		{
			lenUnvoiced0 = tempLen;
			lenUnvoiced1 = 0.0f;
		}

		if (lenUnvoiced0 + lenUnvoiced1 > tempLen)
			lenUnvoiced1 = tempLen - lenUnvoiced0;

		float lenVoiced = tempLen - lenUnvoiced0 - lenUnvoiced1;

		float tempHalfWinLen = 1.0f / minSampleFreq;

		// unvoiced 0
		float srcPos = (float)unvoicedBegin;
		bool interpolation = rateUnvoiced0 < 1.0f;
		for (unsigned uTmpPos = 0; (float)uTmpPos < lenUnvoiced0 && uTmpPos < uTempLen; uTmpPos++, srcPos += rateUnvoiced0)
		{
			float tmpValue;
			if (interpolation)
			{
				int ipos1 = (int)floorf(srcPos);
				float frac = srcPos - (float)ipos1;
				int ipos2 = ipos1 + 1;
				int ipos0 = ipos1 - 1;
				int ipos3 = ipos1 + 2;

				float p0 = source.GetSample(ipos0);
				float p1 = source.GetSample(ipos1);
				float p2 = source.GetSample(ipos2);
				float p3 = source.GetSample(ipos3);

				tmpValue = (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*powf(frac, 3.0f) +
					(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
					(-0.5f*p0 + 0.5f*p2)*frac + p1;
			}
			else
			{
				int ipos1 = (int)ceilf(srcPos - rateUnvoiced0*0.5f);
				int ipos2 = (int)floorf(srcPos + rateUnvoiced0*0.5f);

				float sum = 0.0f;
				for (int ipos = ipos1; ipos <= ipos2; ipos++)
				{
					sum += source.GetSample(ipos);
				}
				tmpValue = sum / (float)(ipos2 - ipos1 + 1);

			}

			float ampl = 1.0f;
			if (uTmpPos>lenUnvoiced0 - tempHalfWinLen)
			{
				float progress = (uTmpPos - (lenUnvoiced0 - tempHalfWinLen)) / tempHalfWinLen;
				ampl = (cosf(progress * (float)PI) + 1.0f)*0.5f;
			}

			tempBuf.m_data[uTmpPos] = tmpValue*ampl;
		}

		// voiced
		float fTmpWinCenter;
		unsigned pos_final = 0;

		for (fTmpWinCenter = lenUnvoiced0; fTmpWinCenter < lenUnvoiced0 + lenVoiced; fTmpWinCenter += tempHalfWinLen)
		{
			float fWinId = (float)windows.size()* (fTmpWinCenter - lenUnvoiced0) / lenVoiced;
			unsigned winId0 = min((unsigned)fWinId, (unsigned)windows.size() - 1);
			unsigned winId1 = min(winId0 + 1, (unsigned)windows.size() - 1);
			float k = fWinId - (float)winId0;

			while (fTmpWinCenter > stretchingMap[pos_final]) pos_final++;

			float destSampleFreq = freqMap[pos_final];
			float destHalfWinLen = 1.0f / destSampleFreq;

			SymmetricWindow& win0 = windows[winId0];
			SymmetricWindow shiftedWin0;

			SymmetricWindow& win1 = windows[winId1];
			SymmetricWindow shiftedWin1;

			SymmetricWindow l_win;
			SymmetricWindow* destWin = &l_win;

			shiftedWin0.Repitch_FormantPreserved(win0, destHalfWinLen);

			if (winId0 == winId1)
			{
				destWin = &shiftedWin0;
			}
			else
			{
				shiftedWin1.Repitch_FormantPreserved(win1, destHalfWinLen);
				l_win.m_halfWidth = destHalfWinLen;
				unsigned u_halfWidth = (unsigned)ceilf(destHalfWinLen);
				l_win.m_data.resize(u_halfWidth);

				for (unsigned i = 0; i < destHalfWinLen; i++)
					l_win.m_data[i] = (1.0f - k) *shiftedWin0.m_data[i] + k* shiftedWin1.m_data[i];
			}

			SymmetricWindow l_win2;
			SymmetricWindow *winToMerge = &l_win2;
			if (destHalfWinLen == tempHalfWinLen)
			{
				winToMerge = destWin;
			}
			else
			{
				l_win2.Scale(*destWin, tempHalfWinLen);
			}
			winToMerge->MergeToBuffer(tempBuf, fTmpWinCenter);

		}

		// unvoiced 1
		float unvoicedBegin1 = fTmpWinCenter - tempHalfWinLen;
		srcPos = (float)voicedEnd;
		for (unsigned uTmpPos = (unsigned)ceilf(unvoicedBegin1); uTmpPos < uTempLen; uTmpPos++, srcPos += rateUnvoiced1)
		{
			float tmpValue;
			if (interpolation)
			{
				int ipos1 = (int)floorf(srcPos);
				float frac = srcPos - (float)ipos1;
				int ipos2 = ipos1 + 1;
				int ipos0 = ipos1 - 1;
				int ipos3 = ipos1 + 2;

				float p0 = source.GetSample(ipos0);
				float p1 = source.GetSample(ipos1);
				float p2 = source.GetSample(ipos2);
				float p3 = source.GetSample(ipos3);

				tmpValue = (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*powf(frac, 3.0f) +
					(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
					(-0.5f*p0 + 0.5f*p2)*frac + p1;
			}
			else
			{
				int ipos1 = (int)ceilf(srcPos - rateUnvoiced0*0.5f);
				int ipos2 = (int)floorf(srcPos + rateUnvoiced0*0.5f);

				float sum = 0.0f;
				for (int ipos = ipos1; ipos <= ipos2; ipos++)
				{
					sum += source.GetSample(ipos);
				}
				tmpValue = sum / (float)(ipos2 - ipos1 + 1);

			}

			float ampl = 1.0f;
			if (uTmpPos < unvoicedBegin1 + tempHalfWinLen)
			{
				float progress = (uTmpPos - (unvoicedBegin1 + tempHalfWinLen)) / tempHalfWinLen;
				ampl = (cosf(progress * (float)PI) + 1.0f)*0.5f;
			}
			tempBuf.m_data[uTmpPos] += tmpValue*ampl;
		}

		// post processing
		noteBuf->m_sampleNum = uSumLen;
		noteBuf->Allocate();

		float multFac = m_noteVolume / maxv;

		for (unsigned pos = 0; pos < uSumLen; pos++)
		{
			float pos_tmpBuf = stretchingMap[pos];
			float sampleFreq = freqMap[pos];
			float speed = sampleFreq / minSampleFreq;

			int ipos1 = (int)ceilf(pos_tmpBuf - speed*0.5f);
			int ipos2 = (int)floorf(pos_tmpBuf + speed*0.5f);

			float sum = 0.0f;
			for (int ipos = ipos1; ipos <= ipos2; ipos++)
			{
				sum += tempBuf.GetSample(ipos);
			}
			float value = sum / (float)(ipos2 - ipos1 + 1);

			float x2 = (float)pos / sumLen;
			float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);

			noteBuf->m_data[pos] = amplitude*value*multFac;
		}
	}

	std::string m_name;
};


class KeLaFactory : public InstrumentFactory
{
public:
	KeLaFactory()
	{
#ifdef _WIN32
		WIN32_FIND_DATAA ffd;
		HANDLE hFind = INVALID_HANDLE_VALUE;

		hFind = FindFirstFileA("KeLaSamples\\*", &ffd);
		if (INVALID_HANDLE_VALUE == hFind) return;

		do
		{
			if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY && strcmp(ffd.cFileName, ".") != 0 && strcmp(ffd.cFileName, "..") != 0)
			{
				m_SingerList.push_back(ffd.cFileName);
			}

		} while (FindNextFile(hFind, &ffd) != 0);

#else
		DIR *dir;
		struct dirent *entry;

		if (dir = opendir("KeLaSamples"))
		{
			while ((entry = readdir(dir)) != NULL)
			{
				if (entry->d_type == DT_DIR)
				{
					if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
						m_SingerList.push_back(entry->d_name);				
				}
			}
		}
#endif
	}

	virtual void GetSingerList(std::vector<std::string>& list)
	{
		list = m_SingerList;
	}

	virtual void InitiateSinger(unsigned clsInd, Singer_deferred& singer)
	{
		singer = Singer_deferred::Instance<KeLa>();
		singer.DownCast<KeLa>()->SetName(m_SingerList[clsInd].data());
	}

private:
	std::vector<std::string> m_SingerList;
};


PY_SCOREDRAFT_EXTENSION_INTERFACE GetFactory()
{
	static KeLaFactory fac;
	return &fac;
}

