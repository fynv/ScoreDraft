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

#include "OtoMap.h"
#include "FrqData.h"

struct SymmetricWindowWithPosition
{
	SymmetricWindow win;
	float center;
};

bool ReadWavLocToBuffer(VoiceLocation loc, Buffer& buf, float& begin, float& end)
{
	Buffer whole;
	float maxV;
	if (!ReadWavToBuffer(loc.filename.data(), whole, maxV)) return false;

	begin = loc.offset*(float)whole.m_sampleRate*0.001f;
	if (loc.cutoff > 0.0f)
		end = (float)whole.m_data.size() - loc.cutoff*(float)whole.m_sampleRate*0.001f;
	else
		end = begin - loc.cutoff*(float)whole.m_sampleRate*0.001f;

	unsigned uBegin = (unsigned)floorf(begin);
	unsigned uEnd = (unsigned)floorf(end);
	
	buf.m_sampleRate = whole.m_sampleRate;
	buf.m_data.resize(uEnd - uBegin);

	float acc = 0.0f;
	for (unsigned i = uBegin; i < uEnd; i++)
	{
		acc += whole.m_data[i] * whole.m_data[i];
	}
	acc = sqrtf((float)(uEnd - uBegin) / acc);

	for (unsigned i = uBegin; i < uEnd; i++)
	{
		buf.m_data[i - uBegin] = whole.m_data[i] * acc;
	}
	
	return true;
}

class UtauDraft : public Singer
{
public:
	UtauDraft()
	{
		m_transition = 0.1f;
		m_rap_distortion = 1.0f;
	}

	void SetOtoMap(OtoMap* otoMap)
	{
		m_OtoMap = otoMap;
		m_defaultLyric = m_OtoMap->begin()->first;
	}

	void SetCharset(const char* charset)
	{
		m_lyric_charset = charset;
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
				if (sscanf(cmd + strlen("rap_distortion") + 1, "%f", &value))
					m_rap_distortion = value;
				return true;
			}
		}
		return false;
	}

	virtual void GenerateWave(SingingPieceInternal piece, VoiceBuffer* noteBuf)
	{
		if (piece.notes.size() < 1) return;

		float sumLen = 0.0f;
		for (size_t i = 0; i < piece.notes.size(); i++)
			sumLen += piece.notes[i].fNumOfSamples;

		float firstNoteHead = this->getFirstNoteHeadSamples(piece.lyric.data());
		sumLen += firstNoteHead;

		unsigned uSumLen = (unsigned)ceilf(sumLen);

		noteBuf->m_sampleNum = uSumLen;
		noteBuf->m_alignPos = (unsigned)firstNoteHead;
		noteBuf->Allocate();

		float *freqMap = new float[uSumLen];

		unsigned pos = 0;
		float targetPos = firstNoteHead;
		float sampleFreq;
		for (size_t i = 0; i < piece.notes.size(); i++)
		{
			sampleFreq = piece.notes[i].sampleFreq;
			targetPos += piece.notes[i].fNumOfSamples;

			for (; (float)pos < targetPos && pos<uSumLen; pos++)
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
			targetPos = firstNoteHead;
			for (size_t i = 0; i < piece.notes.size() - 1; i++)
			{
				float sampleFreq0 = piece.notes[i].sampleFreq;
				float sampleFreq1 = piece.notes[i + 1].sampleFreq;
				targetPos += piece.notes[i].fNumOfSamples;

				float transStart =  targetPos - m_transition*piece.notes[i].fNumOfSamples;
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

		float phase = 0.0f;
		_generateWave(piece.lyric.data(), nullptr, uSumLen, freqMap, noteBuf, 0, phase, true);

		delete[] freqMap;
		// Envolope
		for (unsigned pos = 0; pos < uSumLen; pos++)
		{
			float x2 = (float)pos / sumLen;
			float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);
			noteBuf->m_data[pos] *= amplitude;
		}
	}

	virtual void GenerateWave_Rap(RapPieceInternal piece, VoiceBuffer* noteBuf)
	{
		float sumLen = piece.fNumOfSamples;

		float firstNoteHead = this->getFirstNoteHeadSamples(piece.lyric.data());
		sumLen += firstNoteHead;

		unsigned uSumLen = (unsigned)ceilf(sumLen);

		noteBuf->m_sampleNum = uSumLen;
		noteBuf->m_alignPos = (unsigned)firstNoteHead;
		noteBuf->Allocate();

		float *freqMap = new float[uSumLen];

		if (piece.tone <= 1)
		{
			for (unsigned i = 0; i < uSumLen; i++)
			{
				freqMap[i] = piece.baseSampleFreq;
			}
		}
		else if (piece.tone == 2)
		{
			float lowFreq = piece.baseSampleFreq*0.7f;
			for (unsigned i = 0; i < uSumLen; i++)
			{
				float x = (float)i / (float)(uSumLen - 1);
				freqMap[i] = lowFreq + (piece.baseSampleFreq - lowFreq)*x;
			}
		}
		else if (piece.tone == 3)
		{
			float highFreq = piece.baseSampleFreq*0.75f;
			float lowFreq = piece.baseSampleFreq*0.5f;
			for (unsigned i = 0; i < uSumLen; i++)
			{
				float x = (float)i / (float)(uSumLen - 1);
				freqMap[i] = lowFreq + (highFreq - lowFreq)*x;
			}
		}
		else if (piece.tone == 4)
		{
			float lowFreq = piece.baseSampleFreq*0.5f;
			for (unsigned i = 0; i < uSumLen; i++)
			{
				float x = (float)i / (float)(uSumLen - 1);
				freqMap[i] = piece.baseSampleFreq + (lowFreq - piece.baseSampleFreq)*x;
			}
		}

		float phase = 0.0f;
		_generateWave(piece.lyric.data(), nullptr, uSumLen, freqMap, noteBuf, 0, phase, true);

		delete[] freqMap;

		// Envolope
		for (unsigned pos = 0; pos < uSumLen; pos++)
		{
			float x2 = (float)pos / sumLen;
			float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);
			noteBuf->m_data[pos] *= amplitude;
		}

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
				float v = noteBuf->m_data[pos];
				v *= 10.0f;
				if (v > maxV) v = maxV;
				if (v < -maxV) v = -maxV;
				noteBuf->m_data[pos] = v;
			}
		}
	}

	virtual void GenerateWave_SingConsecutive(SingingPieceInternalList pieceList, VoiceBuffer* noteBuf)
	{
		unsigned *lens = new unsigned[pieceList.size()];
		float sumAllLen=0.0f;
		unsigned uSumAllLen;

		float firstNoteHead = this->getFirstNoteHeadSamples(pieceList[0]->lyric.data());
		
		for (unsigned j = 0; j < pieceList.size(); j++)
		{
			SingingPieceInternal& piece = *pieceList[j];

			float sumLen = 0.0f;
			for (size_t i = 0; i < piece.notes.size(); i++)
				sumLen += piece.notes[i].fNumOfSamples;

			if (j == 0)	sumLen += firstNoteHead;

			float oldSumAllLen = sumAllLen;
			sumAllLen += sumLen;
			
			lens[j] = (unsigned)ceilf(sumAllLen) - (unsigned)ceilf(oldSumAllLen);
		}
		uSumAllLen = (unsigned)ceilf(sumAllLen);

		noteBuf->m_sampleNum = uSumAllLen;
		noteBuf->m_alignPos = (unsigned)firstNoteHead;
		noteBuf->Allocate();

		unsigned noteBufPos = 0;
		float phase = 0.0f;

		for (unsigned j = 0; j < pieceList.size(); j++)
		{
			SingingPieceInternal& piece = *pieceList[j];
			
			unsigned uSumLen = lens[j];
			float *freqMap = new float[uSumLen];

			unsigned pos = 0;
			float targetPos = j == 0 ? firstNoteHead : 0.0f;
			float sampleFreq;
			for (size_t i = 0; i < piece.notes.size(); i++)
			{
				sampleFreq = piece.notes[i].sampleFreq;
				targetPos += piece.notes[i].fNumOfSamples;

				for (; (float)pos < targetPos && pos<uSumLen; pos++)
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
			if (m_transition > 0.0f && m_transition < 1.0f)
			{
				targetPos = j == 0 ? firstNoteHead : 0.0f;
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

				if (j < pieceList.size() - 1)
				{
					size_t i = piece.notes.size() - 1;
					float sampleFreq0 = piece.notes[i].sampleFreq;
					float sampleFreq1 = pieceList[j + 1]->notes[0].sampleFreq;

					float transStart = (float)uSumLen - m_transition*piece.notes[i].fNumOfSamples;

					for (unsigned pos = (unsigned)ceilf(transStart); pos<uSumLen; pos++)
					{
						float k = (cosf(((float)pos - (float)uSumLen) / ((float)uSumLen - transStart)   * (float)PI) + 1.0f)*0.5f;
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

			const char* lyric_next = nullptr;
			if (j < pieceList.size() - 1)
			{
				lyric_next = pieceList[j + 1]->lyric.data();
			}

			_generateWave(piece.lyric.data(), lyric_next, uSumLen, freqMap, noteBuf, noteBufPos, phase, j==0);

			delete[] freqMap;

			noteBufPos += uSumLen;
			
		}

		// Envolope
		for (unsigned pos = 0; pos < uSumAllLen; pos++)
		{
			float x2 = (float)(uSumAllLen-1 - pos) / (float)lens[pieceList.size()-1];
			float amplitude = 1.0f - expf(-x2*10.0f);
			noteBuf->m_data[pos] *= amplitude;
		}
		delete[] lens;
	}

	virtual void GenerateWave_RapConsecutive(RapPieceInternalList pieceList, VoiceBuffer* noteBuf)
	{
		unsigned *lens = new unsigned[pieceList.size()];
		float sumAllLen = 0.0f;
		unsigned uSumAllLen;

		float firstNoteHead = this->getFirstNoteHeadSamples(pieceList[0]->lyric.data());

		for (unsigned j = 0; j < pieceList.size(); j++)
		{
			RapPieceInternal& piece = *pieceList[j];
			float sumLen = piece.fNumOfSamples;

			if (j == 0)	sumLen += firstNoteHead;

			float oldSumAllLen = sumAllLen;
			sumAllLen += sumLen;
			lens[j] = (unsigned)ceilf(sumAllLen) - (unsigned)ceilf(oldSumAllLen);
		}
		uSumAllLen = (unsigned)ceilf(sumAllLen);

		noteBuf->m_sampleNum = uSumAllLen;
		noteBuf->m_alignPos = (unsigned)firstNoteHead;
		noteBuf->Allocate();

		unsigned noteBufPos = 0;
		float phase = 0.0f;

		for (unsigned j = 0; j < pieceList.size(); j++)
		{
			RapPieceInternal& piece = *pieceList[j];

			unsigned uSumLen = lens[j];
			float *freqMap = new float[uSumLen];

			if (piece.tone <= 1)
			{
				for (unsigned i = 0; i < uSumLen; i++)
				{
					freqMap[i] = piece.baseSampleFreq;
				}
			}
			else if (piece.tone == 2)
			{
				float lowFreq = piece.baseSampleFreq*0.7f;
				for (unsigned i = 0; i < uSumLen; i++)
				{
					float x = (float)i / (float)(uSumLen - 1);
					freqMap[i] = lowFreq + (piece.baseSampleFreq - lowFreq)*x;
				}
			}
			else if (piece.tone == 3)
			{
				float highFreq = piece.baseSampleFreq*0.75f;
				float lowFreq = piece.baseSampleFreq*0.5f;
				for (unsigned i = 0; i < uSumLen; i++)
				{
					float x = (float)i / (float)(uSumLen - 1);
					freqMap[i] = lowFreq + (highFreq - lowFreq)*x;
				}
			}
			else if (piece.tone == 4)
			{
				float lowFreq = piece.baseSampleFreq*0.5f;
				for (unsigned i = 0; i < uSumLen; i++)
				{
					float x = (float)i / (float)(uSumLen - 1);
					freqMap[i] = piece.baseSampleFreq + (lowFreq - piece.baseSampleFreq)*x;
				}
			}

			/// Transition
			if (m_transition > 0.0f && m_transition < 1.0f)
			{
				if (j < pieceList.size() - 1)
				{
					RapPieceInternal& piece_next = *pieceList[j+1];

					float sampleFreq_next;
					if (piece_next.tone <= 2)
					{
						sampleFreq_next = piece_next.baseSampleFreq;
					}
					else if (piece_next.tone == 3)
					{
						sampleFreq_next = piece_next.baseSampleFreq*0.75f;
					}
					else if (piece_next.tone == 4)
					{
						sampleFreq_next = piece_next.baseSampleFreq*0.5f;
					}

					float transStart = (float)uSumLen - m_transition*(float)uSumLen;

					for (unsigned pos = (unsigned)ceilf(transStart); pos <uSumLen; pos++)
					{
						float k = (cosf(((float)pos - (float)uSumLen) / ((float)uSumLen - transStart)   * (float)PI) + 1.0f)*0.5f;
						freqMap[pos] = (1.0f - k)* freqMap[pos] + k*sampleFreq_next;
					}
				}
			}

			const char* lyric_next = nullptr;
			if (j < pieceList.size() - 1)
			{
				lyric_next = pieceList[j + 1]->lyric.data();
			}

			_generateWave(piece.lyric.data(), lyric_next, uSumLen, freqMap, noteBuf, noteBufPos, phase, j==0);

			delete[] freqMap;

			noteBufPos += lens[j];
		}

		// Envolope
		for (unsigned pos = 0; pos < uSumAllLen; pos++)
		{
			float x2 = (float)(uSumAllLen - 1 - pos) / (float)lens[pieceList.size() - 1];
			float amplitude = 1.0f - expf(-x2*10.0f);
			noteBuf->m_data[pos] *= amplitude;
		}

		delete[] lens;

		/// Distortion 
		if (m_rap_distortion > 1.0f)
		{
			float maxV = 0.0f;
			for (unsigned pos = 0; pos < uSumAllLen; pos++)
			{
				float v = noteBuf->m_data[pos];
				if (fabsf(v) > maxV) maxV = v;
			}

			for (unsigned pos = 0; pos < uSumAllLen; pos++)
			{
				float v = noteBuf->m_data[pos];
				v *= 10.0f;
				if (v > maxV) v = maxV;
				if (v < -maxV) v = -maxV;
				noteBuf->m_data[pos] = v;
			}
		}

	}

private:
	float getFirstNoteHeadSamples(const char* lyric)
	{
		if (m_OtoMap->find(lyric) == m_OtoMap->end()) lyric = m_defaultLyric.data();
		VoiceLocation loc;
		FrqData frq;
		Buffer source;
		float srcbegin, srcend;

		{
			loc = (*m_OtoMap)[lyric];

			char frq_path[2048];
			memcpy(frq_path, loc.filename.data(), loc.filename.length() - 4);
			memcpy(frq_path + loc.filename.length() - 4, "_wav.frq", strlen("_wav.frq") + 1);

			frq.ReadFromFile(frq_path);

			if (!ReadWavLocToBuffer(loc, source, srcbegin, srcend)) return 0.0f;
		}

		float overlap_pos = loc.overlap* (float)source.m_sampleRate*0.001f;
		float preutter_pos = loc.preutterance * (float)source.m_sampleRate*0.001f;
		if (preutter_pos < overlap_pos) preutter_pos = overlap_pos;

		return preutter_pos - overlap_pos;

	}

	void _generateWave(const char* lyric, const char* lyric_next, unsigned uSumLen, float* freqMap, VoiceBuffer* noteBuf, unsigned noteBufPos, float& phase, bool firstNote)
	{
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
			float sampleFreq;
			sampleFreq = freqMap[pos];

			float speed = sampleFreq / minSampleFreq;
			pos_tmpBuf += speed;
			stretchingMap[pos] = pos_tmpBuf;
		}

		bool hasNextSample = lyric_next != nullptr;

		if (m_OtoMap->find(lyric) == m_OtoMap->end()) lyric = m_defaultLyric.data();

		// Current sample
		VoiceLocation loc;
		FrqData frq;
		Buffer source;
		float srcbegin, srcend;

		{
			loc = (*m_OtoMap)[lyric];

			char frq_path[2048];
			memcpy(frq_path, loc.filename.data(), loc.filename.length() - 4);
			memcpy(frq_path + loc.filename.length() - 4, "_wav.frq", strlen("_wav.frq") + 1);

			frq.ReadFromFile(frq_path);

			if (!ReadWavLocToBuffer(loc, source, srcbegin, srcend)) return;
		}


		//Next sample
		VoiceLocation loc_next;
		FrqData frq_next;
		Buffer source_next;
		float nextbegin, nextend;

		if (hasNextSample)
		{
			if (m_OtoMap->find(lyric_next) == m_OtoMap->end()) lyric_next = m_defaultLyric.data();
			loc_next = (*m_OtoMap)[lyric_next];

			char frq_path_next[2048];
			memcpy(frq_path_next, loc_next.filename.data(), loc_next.filename.length() - 4);
			memcpy(frq_path_next + loc_next.filename.length() - 4, "_wav.frq", strlen("_wav.frq") + 1);

			frq_next.ReadFromFile(frq_path_next);
		
			if (!ReadWavLocToBuffer(loc_next, source_next, nextbegin, nextend)) return;

		}

		float total_len = srcend - srcbegin;
		float overlap_pos = loc.overlap* (float)source.m_sampleRate*0.001f;
		float preutter_pos = loc.preutterance * (float)source.m_sampleRate*0.001f;
		if (preutter_pos < overlap_pos) preutter_pos = overlap_pos;		

		float note_head = preutter_pos - overlap_pos;
		float sumLenWithoutHead = firstNote ? (float)uSumLen - note_head : (float)uSumLen;

		float note_len = total_len - preutter_pos;
		float fixed_end = loc.consonant* (float)source.m_sampleRate*0.001f;
		float fixed_len = fixed_end - preutter_pos;
		float vowel_len = note_len - fixed_len;

		float overlap_pos_next;
		float preutter_pos_next;
		if (hasNextSample)
		{
			overlap_pos_next = loc_next.overlap* (float)source.m_sampleRate*0.001f;
			preutter_pos_next = loc_next.preutterance * (float)source.m_sampleRate*0.001f;
			if (preutter_pos_next < overlap_pos_next) preutter_pos_next = overlap_pos_next;

			fixed_len += preutter_pos_next - overlap_pos_next;
			note_len = vowel_len + fixed_len;
		}


		float k = 1.0f;
		if (sumLenWithoutHead > note_len)
		{
			float k2 = vowel_len / (sumLenWithoutHead - fixed_len);
			if (k2 < k) k = k2;
		}
		float vowel_Weight = 1.0f / (k* fixed_len + vowel_len);
		float fixed_Weight = k* vowel_Weight;
		float headerWeight;

		if (firstNote)
		{
			vowel_Weight *= sumLenWithoutHead / (float)uSumLen;
			fixed_Weight *= sumLenWithoutHead / (float)uSumLen;

			headerWeight = 1.0f / (float)uSumLen;
		}

		class SymmetricWindowWithPosition : public SymmetricWindow
		{
		public:
			float m_pos;
		};

		std::vector<SymmetricWindowWithPosition> windows;
		float fPeriodCount = 0.0f;
		float logicalPos = firstNote ? (-overlap_pos*headerWeight): ( - preutter_pos* fixed_Weight);		

		for (unsigned srcPos = 0; srcPos < source.m_data.size(); srcPos++)
		{
			float srcSampleFreq;
			float srcFreqPos = (srcbegin + (float)srcPos) / (float)frq.m_window_interval;
			unsigned uSrcFreqPos = (unsigned)srcFreqPos;
			float fracSrcFreqPos = srcFreqPos - (float)uSrcFreqPos;

			float freq1 = (float)frq[uSrcFreqPos].freq;
			if (freq1 <= 55.0f) freq1 = (float)frq.m_key_freq;

			float freq2 = (float)frq[uSrcFreqPos + 1].freq;
			if (freq2 <= 55.0f) freq2 = (float)frq.m_key_freq;

			float sampleFreq1 = freq1 / (float)source.m_sampleRate;
			float sampleFreq2 = freq2 / (float)source.m_sampleRate;

			srcSampleFreq = sampleFreq1*(1.0f - fracSrcFreqPos) + sampleFreq2*fracSrcFreqPos;

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

			if (firstNote && (float)srcPos < preutter_pos)
			{
				logicalPos += headerWeight;
			}
			else if ((float)srcPos < fixed_end)
			{
				logicalPos += fixed_Weight;
			}
			else
			{
				logicalPos += vowel_Weight;
			}
		}

		std::vector<SymmetricWindowWithPosition> windows_next;

		if (hasNextSample)
		{
			float fPeriodCount = 0.0f;
			float logicalPos = 1.0f - preutter_pos_next*fixed_Weight;

			for (unsigned srcPos = 0; (float)srcPos < preutter_pos_next; srcPos++)
			{
				float srcSampleFreq;
				float srcFreqPos = (nextbegin + (float)srcPos) / (float)frq_next.m_window_interval;
				unsigned uSrcFreqPos = (unsigned)srcFreqPos;
				float fracSrcFreqPos = srcFreqPos - (float)uSrcFreqPos;

				float freq1 = (float)frq_next[uSrcFreqPos].freq;
				if (freq1 <= 55.0f) freq1 = (float)frq_next.m_key_freq;

				float freq2 = (float)frq_next[uSrcFreqPos + 1].freq;
				if (freq2 <= 55.0f) freq2 = (float)frq_next.m_key_freq;

				float sampleFreq1 = freq1 / (float)source_next.m_sampleRate;
				float sampleFreq2 = freq2 / (float)source_next.m_sampleRate;

				srcSampleFreq = sampleFreq1*(1.0f - fracSrcFreqPos) + sampleFreq2*fracSrcFreqPos;

				unsigned winId = (unsigned)fPeriodCount;
				if (winId >= windows_next.size())
				{
					float srcHalfWinWidth = 1.0f / srcSampleFreq;
					Window srcWin;
					srcWin.CreateFromBuffer(source_next, (float)srcPos, srcHalfWinWidth);

					SymmetricWindowWithPosition symWin;
					symWin.CreateFromAsymmetricWindow(srcWin);
					symWin.m_pos = logicalPos;

					windows_next.push_back(symWin);
				}
				fPeriodCount += srcSampleFreq;
				logicalPos += fixed_Weight;
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
		unsigned winId0_next = 0;
		unsigned pos_final = 0;

		while (phase > -1.0f) phase -= 1.0f;

		float fTmpWinCenter;
		float transitionEnd = 1.0f - (preutter_pos_next - overlap_pos_next)*fixed_Weight;
		float transitionStart = transitionEnd* (1.0f - m_transition);

		for (fTmpWinCenter = phase*tempHalfWinLen; fTmpWinCenter - tempHalfWinLen <= tempLen; fTmpWinCenter += tempHalfWinLen)
		{
			while (fTmpWinCenter > stretchingMap[pos_final] && pos_final<uSumLen - 1) pos_final++;
			float fWinPos = (float)pos_final / float(uSumLen);

			bool in_transition = hasNextSample && m_transition > 0.0f && m_transition < 1.0f && fWinPos >= transitionStart;

			unsigned winId1 = winId0 + 1;
			while (winId1 < windows.size() && windows[winId1].m_pos < fWinPos)
			{
				winId0++;
				winId1 = winId0 + 1;
			}
			if (winId1 == windows.size()) winId1 = winId0;

			unsigned winId1_next = winId0_next + 1;

			if (in_transition)
			{
				while (winId1_next < windows_next.size() && windows_next[winId1_next].m_pos < fWinPos)
				{
					winId0_next++;
					winId1_next = winId0_next + 1;
				}
				if (winId1_next == windows_next.size()) winId1_next = winId0_next;
			}

			SymmetricWindowWithPosition& win0 = windows[winId0];
			SymmetricWindowWithPosition& win1 = windows[winId1];

			float k;
			if (fWinPos >= win1.m_pos) k = 1.0f;
			else if (fWinPos <= win0.m_pos) k = 0.0f;
			else
			{
				k = (fWinPos - win0.m_pos) / (win1.m_pos - win0.m_pos);
			}
			
			float destSampleFreq;
			destSampleFreq = freqMap[pos_final];
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

			SymmetricWindow *win_final_dest = destWin;
			SymmetricWindow l_win_transit;

			if (in_transition)
			{
				SymmetricWindowWithPosition& win0_next = windows_next[winId0_next];
				SymmetricWindowWithPosition& win1_next = windows_next[winId1_next];

				float k;
				if (fWinPos >= win1_next.m_pos) k = 1.0f;
				else if (fWinPos <= win0_next.m_pos) k = 0.0f;
				else
				{
					k = (fWinPos - win0_next.m_pos) / (win1_next.m_pos - win0_next.m_pos);
				}

				SymmetricWindow shiftedWin0_next;
				SymmetricWindow shiftedWin1_next;
				
				SymmetricWindow l_win_next;
				SymmetricWindow* destWin_next = &l_win_next;

				shiftedWin0_next.Repitch_FormantPreserved(win0_next, destHalfWinLen);

				if (winId0_next == winId1_next)
				{
					destWin_next = &shiftedWin0_next;
				}
				else
				{
					shiftedWin1_next.Repitch_FormantPreserved(win1_next, destHalfWinLen);
					l_win_next.m_halfWidth = destHalfWinLen;
					unsigned u_halfWidth = (unsigned)ceilf(destHalfWinLen);
					l_win_next.m_data.resize(u_halfWidth);

					for (unsigned i = 0; i < destHalfWinLen; i++)
						l_win_next.m_data[i] = (1.0f - k) *shiftedWin0_next.m_data[i] + k* shiftedWin1_next.m_data[i];

				}
				
				float x = (fWinPos - transitionEnd) / (transitionEnd*m_transition);
				if (x > 0.0f) x = 0.0f;
				float k2 = 0.5f*(cosf(x*(float)PI) + 1.0f);

				win_final_dest = &l_win_transit;
				l_win_transit.m_halfWidth = destHalfWinLen;
				unsigned u_halfWidth = (unsigned)ceilf(destHalfWinLen);
				l_win_transit.m_data.resize(u_halfWidth);

				for (unsigned i = 0; i < destHalfWinLen; i++)
				{
					l_win_transit.m_data[i] = (1.0f - k2) * destWin->m_data[i] + k2* destWin_next->m_data[i];
				}

			}

			SymmetricWindow l_win2;
			SymmetricWindow *winToMerge = &l_win2;

			if (destHalfWinLen == tempHalfWinLen)
			{
				winToMerge = win_final_dest;
			}
			else
			{
				l_win2.Scale(*win_final_dest, tempHalfWinLen);
			}

			winToMerge->MergeToBuffer(tempBuf, fTmpWinCenter);
		}

		phase = (fTmpWinCenter - tempLen) / tempHalfWinLen;

		// post processing

		float multFac = m_noteVolume;
		for (unsigned pos = 0; pos < uSumLen; pos++, noteBufPos++)
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
				sum += tempBuf.GetSample(ipos);
			}
			float value = sum / (float)(ipos2 - ipos1 + 1);
			noteBuf->m_data[noteBufPos] = value*multFac;
		}

		delete[] stretchingMap;
	}


	OtoMap* m_OtoMap;

	float m_transition;
	float m_rap_distortion;
};



class UtauDraftInitializer : public SingerInitializer
{
public:
	void SetName(const char *name)
	{
		m_name = name;
		char charFileName[1024];
		sprintf(charFileName,"UTAUVoice/%s/character.txt", name);
		FILE *fp = fopen(charFileName, "r");
		if (fp)
		{
			char line[1024];
			while (!feof(fp) && fgets(line, 1024, fp))
			{
				m_charecter_txt += std::string("\t# ") + line;
			}
			m_charecter_txt += "\n";
			fclose(fp);
		}
	}
	std::string GetDirName()
	{
		return m_name;
	}
	std::string GetComment()
	{
		std::string comment = std::string("\t# A singer based on UtauDraft engine and UTAU voice bank in the directory ") + m_name + "\n";
		comment += m_charecter_txt;
		return comment;
	}

	void BuildOtoMap(const char* path)
	{
		char otoIniPath[2048];
		sprintf(otoIniPath, "%s/oto.ini", path);

		FILE *fp = fopen(otoIniPath, "r");
		if (fp)
		{
			fclose(fp);
			m_OtoMap.AddOtoINIPath(path);
		}


#ifdef _WIN32
		WIN32_FIND_DATAA ffd;
		HANDLE hFind = INVALID_HANDLE_VALUE;

		char searchStr[2048];
		sprintf(searchStr, "%s\\*", path);

		hFind = FindFirstFileA(searchStr, &ffd);
		if (INVALID_HANDLE_VALUE == hFind) return;

		do
		{
			if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY && strcmp(ffd.cFileName, ".") != 0 && strcmp(ffd.cFileName, "..") != 0)
			{
				char subPath[2048];
				sprintf(subPath, "%s\\%s", path, ffd.cFileName);
				BuildOtoMap(subPath);
			}

		} while (FindNextFile(hFind, &ffd) != 0);

#else
		DIR *dir;
		struct dirent *entry;

		if (dir = opendir(path))
		{
			while ((entry = readdir(dir)) != NULL)
			{
				if (entry->d_type == DT_DIR)
				{
					if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
					{
						char subPath[2048];
						sprintf(subPath, "%s/%s", path, entry->d_name);
						BuildOtoMap(subPath);
					}
				}
			}
		}
#endif
	}

	virtual Singer_deferred Init()
	{
		if (m_OtoMap.size() == 0)
		{
			char rootPath[1024];
			sprintf(rootPath, "UTAUVoice/%s", m_name.data());
			BuildOtoMap(rootPath);

			m_charset = "utf-8";
			char charsetFn[1024];
			sprintf(charsetFn, "%s/charset", rootPath);

			FILE* fp_charset = fopen(charsetFn, "r");
			if (fp_charset)
			{
				char charsetName[100];
				fscanf(fp_charset, "%s", charsetName);
				m_charset = charsetName;
				fclose(fp_charset);
			}
		}
		Singer_deferred singer = Singer_deferred::Instance<UtauDraft>();
		singer.DownCast<UtauDraft>()->SetOtoMap(&m_OtoMap);
		singer.DownCast<UtauDraft>()->SetCharset(m_charset.data());
		return singer;
	}

private:
	OtoMap m_OtoMap;
	std::string m_charset;
	std::string m_name;
	std::string m_charecter_txt;
};


PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft)
{
	static std::vector<UtauDraftInitializer> s_initializers;


#ifdef _WIN32
	WIN32_FIND_DATAA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	hFind = FindFirstFileA("UTAUVoice\\*", &ffd);
	if (INVALID_HANDLE_VALUE == hFind) return;

	do
	{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY && strcmp(ffd.cFileName, ".") != 0 && strcmp(ffd.cFileName, "..") != 0)
		{
			UtauDraftInitializer initializer;
			initializer.SetName(ffd.cFileName);
			s_initializers.push_back(initializer);
		}

	} while (FindNextFile(hFind, &ffd) != 0);

#else
	DIR *dir;
	struct dirent *entry;

	if (dir = opendir("UTAUVoice"))
	{
		while ((entry = readdir(dir)) != NULL)
		{
			if (entry->d_type == DT_DIR)
			{
				if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
				{
					UtauDraftInitializer initializer;
					initializer.SetName(entry->d_name);
					s_initializers.push_back(initializer);
				}
			}
		}
	}
#endif

	for (unsigned i = 0; i < s_initializers.size(); i++)
		pyScoreDraft->RegisterSingerClass((s_initializers[i].GetDirName()+"_UTAU").data(), &s_initializers[i], s_initializers[i].GetComment().data());
}
