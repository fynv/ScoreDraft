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

	for (size_t i = 0; i < frequencies.size(); i++)
	{
		if ((i<BestRange.begin || i>BestRange.end) && frequencies[i] > 0.0f)
		{
			frequencies[i] = 0.0f;
		}
	}

	bool cut = false;
	for (unsigned i = BestRange.begin - 1; i != (unsigned)(-1); i--)
	{
		if (frequencies[i] < 0)
		{
			cut = true;
			continue;
		}
		if (cut &&  frequencies[i]>=0)
		{
			frequencies[i] = -1;
			continue;
		}

		if (frequencies[i] > 0)
		{
			frequencies[i] = 0.0f;
		}
	}

	cut = false;
	for (unsigned i = BestRange.end + 1; i < (unsigned)frequencies.size(); i++)
	{
		if (frequencies[i] < 0)
		{
			cut = true;
			continue;
		}
		if (cut &&  frequencies[i] >= 0)
		{
			frequencies[i] = -1;
			continue;
		}

		if (frequencies[i] > 0)
		{
			frequencies[i] = 0.0f;
		}
	}


}

class KeLa : public Singer
{
public:
	KeLa()
	{
		m_transition = 0.1f;
		m_rap_distortion = 1.0f;
	}
	void SetName(const char* root, const char* name)
	{
		m_root = root;
		m_name = name;

#ifdef _WIN32

		char charsetFn[1024];
		sprintf(charsetFn, "%s/KeLaSamples/%s/charset", m_root.data(), m_name.data());
		FILE* fp_charset = fopen(charsetFn, "r");
		if (fp_charset)
		{
			char charsetName[100];
			fscanf(fp_charset, "%s", charsetName);
			m_lyric_charset = charsetName;
			fclose(fp_charset);
		}
		
		WIN32_FIND_DATAA ffd;
		HANDLE hFind = INVALID_HANDLE_VALUE;

		char searchPath[1024];
		sprintf(searchPath, "%s/KeLaSamples/%s/*.wav", m_root.data(), m_name.data());

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
		sprintf(dirPath, "%s/KeLaSamples/%s", m_root.data(), m_name.data());

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
	virtual bool Tune(const char* cmd)
	{
		if (!Singer::Tune(cmd))
		{
			char command[1024];
			sscanf(cmd, "%s", command);

			if (strcmp(command, "rap_distortion") == 0)
			{
				float value;
				if (sscanf(cmd + strlen("rap_distortion")+1, "%f", &value))
					m_rap_distortion = value;
				return true;
			}
		}
		return false;
	}
	virtual void GenerateWave(SingingPieceInternal piece, NoteBuffer* noteBuf)
	{
		if (piece.notes.size() < 1) return;

		float sumLen = 0.0f;
		for (size_t i = 0; i < piece.notes.size(); i++)
			sumLen += piece.notes[i].fNumOfSamples;
		
		unsigned uSumLen = (unsigned)ceilf(sumLen);		
		float *freqMap = new float[uSumLen];

		unsigned pos = 0;
		float targetPos = 0.0f;
		float sampleFreq;
		for (size_t i = 0; i < piece.notes.size(); i++)
		{
			sampleFreq = piece.notes[i].sampleFreq;
			targetPos += piece.notes[i].fNumOfSamples;
		
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
			for (size_t i = 0; i < piece.notes.size() - 1; i++)
			{
				float sampleFreq0 = piece.notes[i].sampleFreq;
				float sampleFreq1 = piece.notes[i + 1].sampleFreq;
				targetPos += piece.notes[i].fNumOfSamples;

				float transStart = targetPos - m_transition*piece.notes[i].fNumOfSamples;
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

		_generateWave(piece.lyric.data(), sumLen, freqMap, noteBuf);

		delete[] freqMap;	

	}

	virtual void GenerateWave_Rap(RapPieceInternal piece, NoteBuffer* noteBuf)
	{
		float sumLen = piece.fNumOfSamples;
		unsigned uSumLen = (unsigned)ceilf(sumLen);
		float *freqMap = new float[uSumLen];

		for (unsigned i = 0; i < uSumLen; i++)
		{
			float x = (float)i / (float)(uSumLen - 1);
			freqMap[i] = piece.sampleFreq1 + (piece.sampleFreq2 - piece.sampleFreq1)*x;
		}

		_generateWave(piece.lyric.data(), sumLen, freqMap, noteBuf);
		delete[] freqMap;

		/// Distortion 
		if (m_rap_distortion > 1.0f)
		{
			float maxV = 0.0f;
			for (unsigned pos = 0; pos < uSumLen; pos++)
			{
				float v = noteBuf->m_data[pos];
				if (fabsf(v) > maxV) maxV = v;
			}

			for (unsigned pos = 0; pos < uSumLen; pos++)
			{
				float x2 = (float)pos / sumLen;
				float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);

				float v = noteBuf->m_data[pos];
				v *= m_rap_distortion;
				if (v > maxV) v = maxV;
				if (v < -maxV) v = -maxV;
				noteBuf->m_data[pos] = v;
			}
		}
	}

	virtual void GenerateWave_SingConsecutive(SingingPieceInternalList pieceList, NoteBuffer* noteBuf)
	{
		typedef Deferred<NoteBuffer> NoteBuffer_Deferred;
		std::vector <NoteBuffer_Deferred> subBufs;

		float sumAllLen = 0.0f;

		for (unsigned i = 0; i < (unsigned)pieceList.size(); i++)
		{
			SingingPieceInternal& piece = *pieceList[i];
			NoteBuffer_Deferred subBuf;
			subBuf->m_sampleRate = noteBuf->m_sampleRate;
			GenerateWave(piece, subBuf);
			sumAllLen += subBuf->m_sampleNum;
			subBufs.push_back(subBuf);
		}
		unsigned uSumAllLen = (unsigned)ceilf(sumAllLen);
		noteBuf->m_sampleNum = uSumAllLen;
		noteBuf->Allocate();

		float bufPos = 0.0f;
		for (unsigned i = 0; i < (unsigned)subBufs.size(); i++)
		{
			NoteBuffer& subBuf = *subBufs[i];
			unsigned noteStart = (unsigned)ceilf(bufPos);
			unsigned noteEnd = (unsigned)ceilf(bufPos + subBuf.m_sampleNum);
			for (unsigned j = noteStart; j < noteEnd; j++)
			{
				noteBuf->m_data[j] = subBuf.m_data[j - noteStart];
			}
			bufPos += subBuf.m_sampleNum;
		}
		
	}

	virtual void GenerateWave_RapConsecutive(RapPieceInternalList pieceList, NoteBuffer* noteBuf)
	{
		typedef Deferred<NoteBuffer> NoteBuffer_Deferred;
		std::vector <NoteBuffer_Deferred> subBufs;

		float sumAllLen = 0.0f;

		for (unsigned i = 0; i < (unsigned)pieceList.size(); i++)
		{
			RapPieceInternal&  piece = *pieceList[i];
			NoteBuffer_Deferred subBuf;
			subBuf->m_sampleRate = noteBuf->m_sampleRate;
			GenerateWave_Rap(piece, subBuf);
			sumAllLen += subBuf->m_sampleNum;
			subBufs.push_back(subBuf);
		}

		unsigned uSumAllLen = (unsigned)ceilf(sumAllLen);
		noteBuf->m_sampleNum = uSumAllLen;
		noteBuf->Allocate();

		float bufPos = 0.0f;
		for (unsigned i = 0; i < (unsigned)subBufs.size(); i++)
		{
			NoteBuffer& subBuf = *subBufs[i];
			unsigned noteStart = (unsigned)ceilf(bufPos);
			unsigned noteEnd = (unsigned)ceilf(bufPos + subBuf.m_sampleNum);
			for (unsigned j = noteStart; j < noteEnd; j++)
			{
				noteBuf->m_data[j] = subBuf.m_data[j - noteStart];
			}
			bufPos += subBuf.m_sampleNum;
		}

	}

private:
	void _generateWave(const char* lyric, float sumLen, float* freqMap, NoteBuffer* noteBuf)
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
		sprintf(path, "%s/KeLaSamples/%s/%s.wav", m_root.data(), m_name.data(), lyric);

		Buffer source;
		float maxv;
		if (!ReadWavToBuffer(path, source, maxv)) return;

		unsigned freq_step = 256;
		std::vector<float> frequencies;
		
		sprintf(path, "%s/KeLaSamples/%s/%s.freq", m_root.data(), m_name.data(), lyric);
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
				unvoicedBegin = (unsigned)i*freq_step;
			if (frequencies[i] > 0.0f && voicedBegin == (unsigned)(-1))
			{
				voicedBegin_id = (unsigned)i;
				voicedBegin = (unsigned)i*freq_step;
				firstFreq = frequencies[i];
			}

			if (frequencies[i] <= 0.0f && voicedBegin != (unsigned)(-1) && voicedEnd == (unsigned)(-1))
			{
				voicedEnd_id = (unsigned)i;
				voicedEnd = (unsigned)(i-1)*freq_step;
				lastFreq = frequencies[i - 1];
			}

			if (frequencies[i] < 0.0f && voicedEnd != (unsigned)(-1) && unvoicedEnd == (unsigned)(-1))
			{
				unvoicedEnd = (unsigned)(i-1)*freq_step;
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
			k = ((float)voicedLen / (float)unvoicedLen)  * (0.2f / 0.8f);
		}

		if (sumLen > totalLen)
		{
			float k2 = (float)voicedLen / (sumLen - (float)unvoicedLen);
			if (k2 < k) k = k2;
		}

		voicedWeight = 1.0f / (k* (float)unvoicedLen + (float)voicedLen);
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

			unsigned winId1 = winId0 + 1;

			while (winId1 < windows.size() && windows[winId1].m_pos < fWinPos)
			{
				winId0++;
				winId1 = winId0 + 1;
			}			

			if (winId1 == windows.size()) winId1 = winId0;
			SymmetricWindowWithPosition& win0 = windows[winId0];
			SymmetricWindowWithPosition& win1 = windows[winId1];

			float k;
			if (fWinPos >= win1.m_pos) k = 1.0f;
			else
			{
				k = (fWinPos - win0.m_pos) / (win1.m_pos - win0.m_pos);
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

		float multFac = 1.0f / maxv;

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
	std::string m_root;

	float m_transition;
	float m_rap_distortion;
};

class KeLaInitializer : public SingerInitializer
{
public:
	std::string m_name;
	std::string m_root;
	std::string GetComment()
	{
		return std::string("\t# A singer based on KeLa engine and samples in the directory ") + m_name + "\n";
	}
	virtual Singer_deferred Init()
	{
		Singer_deferred singer = Singer_deferred::Instance<KeLa>();
		singer.DownCast<KeLa>()->SetName(m_root.data(), m_name.data());
		return singer;
	}
};

PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft, const char* root)
{
	static std::vector<KeLaInitializer> s_initializers;

#ifdef _WIN32
	WIN32_FIND_DATAA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	char findStr[1024];
	sprintf(findStr, "%s/KeLaSamples/*", root);

	hFind = FindFirstFileA(findStr, &ffd);
	if (INVALID_HANDLE_VALUE == hFind) return;

	do
	{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY && strcmp(ffd.cFileName, ".") != 0 && strcmp(ffd.cFileName, "..") != 0)
		{
			KeLaInitializer initializer;
			initializer.m_name = ffd.cFileName;
			initializer.m_root = root;
			s_initializers.push_back(initializer);
		}

	} while (FindNextFile(hFind, &ffd) != 0);

#else
	DIR *dir;
	struct dirent *entry;
	char searchPath[1024];
	sprintf(searchPath, "%s/KeLaSamples", root);

	if (dir = opendir(searchPath))
	{
		while ((entry = readdir(dir)) != NULL)
		{
			if (entry->d_type == DT_DIR)
			{
				if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
				{
					KeLaInitializer initializer;
					initializer.m_name = entry->d_name;
					initializer.m_root = root;
					s_initializers.push_back(initializer);
				}	
			}
		}
	}
#endif
	for (unsigned i = 0; i < s_initializers.size(); i++)
		pyScoreDraft->RegisterSingerClass(s_initializers[i].m_name.data(), &s_initializers[i], s_initializers[i].GetComment().data());
}

