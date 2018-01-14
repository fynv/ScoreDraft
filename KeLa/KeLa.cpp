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


void DetectFreqs(const Buffer& buf, std::vector<float>& frequencies, unsigned step)
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
	}

	for (unsigned i = sampleBegin; i < sampleEnd; i++)
		if (frequencies[i] < 0.0f) frequencies[i] = 0.0f;

}

class KeLa : public Singer
{
public:
	KeLa()
	{
		m_transition = 0.1f;
	}
	void SetName(const char* name)
	{
		m_name = name;
#ifdef _WIN32
		WIN32_FIND_DATAA ffd;
		HANDLE hFind = INVALID_HANDLE_VALUE;

		char searchPath[1024];
		sprintf(searchPath, "KeLaSamples\\%s\\*.wav", m_name.data());

		hFind = FindFirstFileA(searchPath, &ffd);
		if (INVALID_HANDLE_VALUE != hFind)
		{
			do
			{
				if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;

				char name[1024];
				memcpy(name, ffd.cFileName, strlen(ffd.cFileName) - 4);
				name[strlen(ffd.cFileName) - 4] = 0;
				m_defaultLyric = name;
				break;

			} while (FindNextFile(hFind, &ffd) != 0);
		}
#else
		DIR *dir;
		struct dirent *entry;

		char dirPath[1024];
		sprintf(dirPath, "KeLaSamples/%s", m_name.data());

		if (dir = opendir(dirPath))
		{
			while ((entry = readdir(dir)) != NULL)
			{
				if (entry->d_type != DT_DIR)
				{
					const char* ext = entry->d_name + strlen(entry->d_name) - 4;
					if (strcmp(ext, ".wav") == 0)
					{
						char name[1024];
						memcpy(name, entry->d_name, strlen(entry->d_name) - 4);
						name[strlen(entry->d_name) - 4] = 0;
						m_defaultLyric = name;
						break;
					}
				}

			}

		}
#endif

	}
	virtual void GenerateWave(const char* lyric, std::vector<SingerNoteParams> notes, VoiceBuffer* noteBuf)
	{
		if (notes.size() < 1) return;

		float sumLen = 0.0f;
		for (size_t i = 0; i < notes.size(); i++)
			sumLen += notes[i].fNumOfSamples;
		
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

		/// Transition
		if (m_transition > 0.0f && m_transition<1.0f)
		{
			targetPos = 0.0f;
			for (size_t i = 0; i < notes.size() - 1; i++)
			{
				float sampleFreq0 = notes[i].sampleFreq;
				float sampleFreq1 = notes[i + 1].sampleFreq;
				targetPos += notes[i].fNumOfSamples;

				float transStart = targetPos - m_transition*notes[i].fNumOfSamples;
				for (unsigned pos = (unsigned)ceilf(transStart); pos <= (unsigned)floorf(targetPos); pos++)
				{
					float k = (cosf(((float)pos - targetPos) / (targetPos - transStart)   * (float)PI) + 1.0f)*0.5f;
					freqMap[pos] = (1.0f - k)* sampleFreq0 + k*sampleFreq1;
				}

			}
		}

		/// Viberation
		/*for (pos = 0; pos < uSumLen; pos++)
		{
			float vib = 1.0f - 0.02f*cosf(2.0f*PI* (float)pos*10.0f / 44100.0f);
			freqMap[pos] *= vib;
		}*/

		_generateWave(lyric, sumLen, freqMap, noteBuf);

		delete[] freqMap;	

	}
private:
	void _generateWave(const char* lyric, float sumLen, float* freqMap, VoiceBuffer* noteBuf)
	{
		unsigned uSumLen = (unsigned)ceilf(sumLen);

		/// calculate finalBuffer->tmpBuffer map
		float minSampleFreq = FLT_MAX;
		for (unsigned pos = 0; pos < uSumLen; pos++)
		{
			float sampleFreq = freqMap[pos];
			if (sampleFreq < minSampleFreq) minSampleFreq = sampleFreq;
		}

		float* stretchingMap = new float[uSumLen];

		float pos_tmpBuf = 0.0f;
		for (unsigned pos = 0; pos < uSumLen; pos++)
		{
			float sampleFreq = freqMap[pos];
			float speed = sampleFreq / minSampleFreq;
			pos_tmpBuf += speed;
			stretchingMap[pos] = pos_tmpBuf;
		}

		char path[1024];
		sprintf(path, "KeLaSamples/%s/%s.wav", m_name.data(), lyric);

		Buffer source;
		float maxv;
		if (!ReadWavToBuffer(path, source, maxv)) return;

		unsigned freq_step = 256;
		std::vector<float> frequencies;
		
		sprintf(path, "KeLaSamples/%s/%s.freq", m_name.data(), lyric);
		FILE* fp = fopen(path, "r");
		if (fp)
		{
			while (!feof(fp))
			{
				float f;
				if (fscanf(fp, "%f", &f))
				{
					frequencies.push_back(f);
				}
				else break;
			}
			fclose(fp);
		}
		else
		{
			DetectFreqs(source, frequencies, freq_step);
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
			voicedEnd_id = (unsigned)frequencies.size();
			voicedEnd = (unsigned)source.m_data.size();
			lastFreq = frequencies[voicedEnd_id - 1];
		}
		if (unvoicedEnd == (unsigned)(-1))
		{
			unvoicedEnd = (unsigned)source.m_data.size();
		}

		unsigned voicedLen = voicedEnd - voicedBegin;
		unsigned totalLen = unvoicedEnd - unvoicedBegin;
		unsigned unvoicedLen = totalLen - voicedLen;

		float voicedWeight;
		float unvoicedWeight;

		float k = 1.0f;

		float voiced_portion = (float)voicedLen / (float)totalLen;
		if (voiced_portion < 0.8f)
		{
			k = ((float)voicedLen / (float)(unvoicedLen))  * (0.2 / 0.8);
		}

		if (sumLen > totalLen)
		{
			float k2 = (float)voicedLen / (float)(sumLen - unvoicedLen);
			if (k2 < k) k = k2;
		}

		voicedWeight = 1.0f / (k* unvoicedLen + voicedLen);
		unvoicedWeight = k* voicedWeight;


		class SymmetricWindowWithPosition : public SymmetricWindow
		{
		public:
			float m_pos;
		};

		std::vector<SymmetricWindowWithPosition> windows;
		float fPeriodCount = 0.0f;
		float logicalPos = 0.0f;
		for (unsigned srcPos = unvoicedBegin; srcPos < unvoicedEnd; srcPos++)
		{
			float srcFreqPos = (float)srcPos / (float)freq_step;
			unsigned uSrcFreqPos = (unsigned)srcFreqPos;
			float fracSrcFreqPos = srcFreqPos - (float)uSrcFreqPos;

			float sampleFreq1;
			if (uSrcFreqPos < voicedBegin_id) sampleFreq1 = freqMap[0];
			else if (uSrcFreqPos >= voicedEnd_id) sampleFreq1 = freqMap[uSumLen - 1];
			else sampleFreq1 = frequencies[uSrcFreqPos] / (float)source.m_sampleRate;

			float sampleFreq2;
			if (uSrcFreqPos + 1 < voicedBegin_id) sampleFreq2 = freqMap[0];
			else if (uSrcFreqPos + 1 >= voicedEnd_id) sampleFreq2 = freqMap[uSumLen - 1];
			else sampleFreq2 = frequencies[uSrcFreqPos + 1] / (float)source.m_sampleRate;

			float srcSampleFreq = sampleFreq1*(1.0f - fracSrcFreqPos) + sampleFreq2*fracSrcFreqPos;

			unsigned winId = (unsigned)fPeriodCount;
			if (winId >= windows.size())
			{
				float srcHalfWinWidth = 1.0f / srcSampleFreq;
				Window srcWin;
				srcWin.CreateFromBuffer(source, (float)srcPos, srcHalfWinWidth);

				SymmetricWindowWithPosition symWin;
				symWin.CreateFromAsymmetricWindow(srcWin);
				symWin.m_pos = logicalPos;

				windows.push_back(symWin);

			}
			fPeriodCount += srcSampleFreq;
			if (srcPos < voicedBegin || srcPos >= voicedEnd)
			{
				logicalPos += unvoicedWeight;
			}
			else
			{
				logicalPos += voicedWeight;
			}
		}

		float tempLen = stretchingMap[uSumLen - 1];
		unsigned uTempLen = (unsigned)ceilf(tempLen);

		Buffer tempBuf;
		tempBuf.m_sampleRate = source.m_sampleRate;
		tempBuf.m_data.resize(uTempLen);
		tempBuf.SetZero();

		float tempHalfWinLen = 1.0f / minSampleFreq;

		unsigned winId0 = 0;
		unsigned pos_final = 0;
		
		for (float fTmpWinCenter = 0.0f; fTmpWinCenter <= tempLen; fTmpWinCenter += tempHalfWinLen)
		{
			float fWinPos = fTmpWinCenter / tempLen;
			while (winId0<windows.size() - 1 && windows[winId0].m_pos < fWinPos) winId0++;

			unsigned winId1 = winId0 + 1;

			if (winId1 == windows.size()) winId1 = winId0;
			SymmetricWindowWithPosition& win0 = windows[winId0];
			SymmetricWindowWithPosition& win1 = windows[winId1];

			float k;
			if (fTmpWinCenter >= win1.m_pos) k = 1.0f;
			else
			{
				k = (fTmpWinCenter - win0.m_pos) / (win1.m_pos - win0.m_pos);
			}

			while (fTmpWinCenter > stretchingMap[pos_final]) pos_final++;

			float destSampleFreq = freqMap[pos_final];
			float destHalfWinLen = 1.0f / destSampleFreq;

			SymmetricWindow shiftedWin0;
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


		delete[] stretchingMap;

	}

	std::string m_name;

	float m_transition;
};

class KeLaInitializer : public SingerInitializer
{
public:
	std::string m_name;
	std::string GetComment()
	{
		return std::string("\t# A singer based on KeLa engine and samples in the directory ") + m_name + "\n";
	}
	virtual Singer_deferred Init()
	{
		Singer_deferred singer = Singer_deferred::Instance<KeLa>();
		singer.DownCast<KeLa>()->SetName(m_name.data());
		return singer;
	}
};

PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft)
{
	static std::vector<KeLaInitializer> s_initializers;

#ifdef _WIN32
	WIN32_FIND_DATAA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	hFind = FindFirstFileA("KeLaSamples\\*", &ffd);
	if (INVALID_HANDLE_VALUE == hFind) return;

	do
	{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY && strcmp(ffd.cFileName, ".") != 0 && strcmp(ffd.cFileName, "..") != 0)
		{
			KeLaInitializer initializer;
			initializer.m_name = ffd.cFileName;
			s_initializers.push_back(initializer);
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
				{
					KeLaInitializer initializer;
					initializer.m_name = entry->d_name;
					s_initializers.push_back(initializer);
				}	
			}
		}
	}
#endif
	for (unsigned i = 0; i < s_initializers.size(); i++)
		pyScoreDraft->RegisterSingerClass(s_initializers[i].m_name.data(), &s_initializers[i], s_initializers[i].GetComment().data());
}

