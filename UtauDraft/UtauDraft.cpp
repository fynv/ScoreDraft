#include <Python.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <dlfcn.h>
#endif

#include "TrackBuffer.h"

#include <string.h>
#include <cmath>
#include <ReadWav.h>
#include <float.h>
#include <memory.h>

#include "PrefixMap.h"
#include "UtauDraft.h"
#include "SentenceGenerator_PSOLA.h"
#include "SentenceGenerator_HNM.h"

#ifdef HAVE_CUDA
#include "SentenceGenerator_CUDA.h"
#endif

bool UtauSourceFetcher::ReadWavLocToBuffer(VoiceLocation loc, Buffer& buf, float& begin, float& end)
{
	Buffer whole;
	float maxV;
	if (!ReadWavToBuffer(loc.filename.data(), whole, maxV)) return false;

	float acc = 0.0f;
	float count = 0.0f;
	for (unsigned i = 0; i < whole.m_data.size(); i++)
	{
		acc += whole.m_data[i] * whole.m_data[i];
		if (whole.m_data[i] != 0.0f)
		{
			count += 1.0f;
		}
	}
	acc = sqrtf(count / acc)*0.3f;

	begin = loc.offset*(float)whole.m_sampleRate*0.001f;
	if (loc.cutoff > 0.0f)
		end = (float)whole.m_data.size() - loc.cutoff*(float)whole.m_sampleRate*0.001f;
	else
		end = begin - loc.cutoff*(float)whole.m_sampleRate*0.001f;

	unsigned uBegin = (unsigned)floorf(begin);
	unsigned uEnd = (unsigned)floorf(end);

	buf.m_sampleRate = whole.m_sampleRate;
	buf.m_data.resize(uEnd - uBegin);

	for (unsigned i = uBegin; i < uEnd; i++)
		buf.m_data[i - uBegin] = (i<whole.m_data.size())? (whole.m_data[i] * acc) : 0.0f;

	return true;
}

bool UtauSourceFetcher::FetchSourceInfo(const char* lyric, SourceInfo& srcInfo, bool czmode, const char* lyric_next) const
{
	if (m_OtoMap->find(lyric) == m_OtoMap->end())
	{
		printf("missied lyic: %s\n", lyric);
		lyric = m_defaultLyric.data();
		if (m_OtoMap->find(lyric) == m_OtoMap->end())
			return false;
	}

	VoiceLocation& loc = srcInfo.loc;
	FrqData& frq = srcInfo.frq;
	Buffer& source = srcInfo.source;
	float& srcbegin = srcInfo.srcbegin;
	float& srcend = srcInfo.srcend;

	{
		loc = (*m_OtoMap)[lyric];
		if (czmode)
		{
			float next_overlap = 80.0;
			if (lyric_next)
			{
				if (m_OtoMap->find(lyric_next) == m_OtoMap->end())
				{
					printf("missied lyic: %s\n", lyric_next);
					lyric_next = m_defaultLyric.data();
					if (m_OtoMap->find(lyric_next) == m_OtoMap->end())
						return false;
				}

				VoiceLocation loc_next = (*m_OtoMap)[lyric_next];
				next_overlap = loc_next.overlap;

				if (next_overlap <= 0.0f) next_overlap = 1.0f;
			}

			loc.consonant = loc.preutterance;
			loc.cutoff = -(loc.preutterance + next_overlap);
		}

		char frq_path[2048];
		memcpy(frq_path, loc.filename.data(), loc.filename.length() - 4);
		memcpy(frq_path + loc.filename.length() - 4, "_wav.frq", strlen("_wav.frq") + 1);

		if (!frq.ReadFromFile(frq_path))
		{
			printf("%s not found.\n", frq_path);
			return false;
		}

		if (!ReadWavLocToBuffer(loc, source, srcbegin, srcend))
		{
			printf("%s not found.\n", loc.filename.data());
			return false;
		}
	}

	return true;
}

void SourceDerivedInfo::DeriveInfo(bool firstNote, bool hasNext, unsigned uSumLen, const SourceInfo& curSrc, const SourceInfo& nextSrc)
{
	float total_len = curSrc.srcend - curSrc.srcbegin;
	overlap_pos = curSrc.loc.overlap* (float)curSrc.source.m_sampleRate*0.001f;
	preutter_pos = curSrc.loc.preutterance * (float)curSrc.source.m_sampleRate*0.001f;
	if (preutter_pos < overlap_pos) preutter_pos = overlap_pos;

	float note_head = preutter_pos - overlap_pos;
	float sumLenWithoutHead = firstNote ? (float)uSumLen - note_head : (float)uSumLen;

	float note_len = total_len - preutter_pos;
	fixed_end = curSrc.loc.consonant* (float)curSrc.source.m_sampleRate*0.001f;
	float fixed_len = fixed_end - preutter_pos;
	float vowel_len = note_len - fixed_len;

	if (hasNext)
	{
		overlap_pos_next = nextSrc.loc.overlap* (float)nextSrc.source.m_sampleRate*0.001f;
		preutter_pos_next = nextSrc.loc.preutterance * (float)nextSrc.source.m_sampleRate*0.001f;
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
	vowel_Weight = 1.0f / (k* fixed_len + vowel_len);
	fixed_Weight = k* vowel_Weight;

	if (firstNote)
	{
		vowel_Weight *= sumLenWithoutHead / (float)uSumLen;
		fixed_Weight *= sumLenWithoutHead / (float)uSumLen;

		headerWeight = 1.0f / (float)uSumLen;
	}

};

UtauDraft::UtauDraft(bool useCUDA)
{
	m_use_CUDA = useCUDA;

	m_transition = 0.1f;
	m_gender = 0.0f;
	m_CZMode = false;
	m_LyricConverter = nullptr;

	m_use_prefix_map = true;
	m_PrefixMap = nullptr;

}

UtauDraft::~UtauDraft()
{
	if (m_LyricConverter) Py_DECREF(m_LyricConverter);
}

void UtauDraft::SetOtoMap(OtoMap* otoMap)
{
	m_OtoMap = otoMap;
	m_defaultLyric = m_OtoMap->begin()->first;
}

void UtauDraft::SetPrefixMap(PrefixMap* prefixMap)
{
	m_PrefixMap = prefixMap;
}

void UtauDraft::SetCharset(const char* charset)
{
	m_lyric_charset = charset;
}

void UtauDraft::SetLyricConverter(PyObject* lyricConverter)
{
	if (m_LyricConverter != nullptr) Py_DECREF(m_LyricConverter);
	m_LyricConverter = lyricConverter;
	if (m_LyricConverter != nullptr) Py_INCREF(m_LyricConverter);
}

bool UtauDraft::Tune(const char* cmd)
{
	if (!Singer::Tune(cmd))
	{
		char command[1024];
		sscanf(cmd, "%s", command);

		if (strcmp(command, "transition") == 0)
		{
			float value;
			if (sscanf(cmd + strlen("transition") + 1, "%f", &value))
				m_transition = value;
		}
		else if (strcmp(command, "prefix_map") == 0)
		{
			char value[100];
			if (sscanf(cmd + strlen("prefix_map") + 1, "%s", value))
			{
				if (strcmp(value, "on") == 0)
				{
					m_use_prefix_map = true;
				}
				if (strcmp(value, "off") == 0)
				{
					m_use_prefix_map = false;
				}
			}
		}
		else if (strcmp(command, "gender") == 0)
		{
			float value;
			if (sscanf(cmd + strlen("gender") + 1, "%f", &value))
				m_gender = value;
		}
		else if (strcmp(command, "constvc") == 0) // left for compatibility to old immature design
		{
			m_CZMode = true;
		}
		else if (strcmp(command, "CZMode") == 0)
		{
			m_CZMode = true;
		}
	}
	return false;
}

SentenceGenerator* UtauDraft::createSentenceGenerator()
{
	SentenceGenerator* sg;

#ifdef HAVE_CUDA
	if (m_use_CUDA)
	{
		sg = new SentenceGenerator_CUDA;
	}
	else
#endif
	{
		sg = new SentenceGenerator_HNM;
	}
	sg->_gender = m_gender;
	sg->_transition = m_transition;
	sg->_CZMode = m_CZMode;
	return sg;
}

void UtauDraft::GenerateWave(SyllableInternal syllable, NoteBuffer* noteBuf)
{
	SyllableInternal_Deferred dSyllable;
	*dSyllable = syllable;
	SyllableInternalList syllableList;
	syllableList.push_back(dSyllable);
	GenerateWave_SingConsecutive(syllableList, noteBuf);
}

void UtauDraft::_floatBufSmooth(float* buf, unsigned size)
{
	static unsigned halfWinSize = 1024;
	static unsigned winSize = halfWinSize * 2;
	float *buf2 = new float[size];
	memset(buf2, 0, sizeof(float)*size);

	for (unsigned i = 0; i < size + halfWinSize; i += halfWinSize)
	{
		float sum = 0.0f;
		for (int j = -(int)halfWinSize; j < (int)halfWinSize; j++)
		{
			float v;
			int bufPos = (int)i + j;
			if (bufPos < 0) v = buf[0];
			else if (bufPos >= (int)size) v = buf[size - 1];
			else v = buf[bufPos];
				
			float x = (float)j / (float)halfWinSize*(float)PI;
			float w = (cosf(x) + 1.0f)*0.5f;

			sum += v*w;
		}
		float ave = sum / (float)halfWinSize;
		for (int j = -(int)halfWinSize; j < (int)halfWinSize; j++)
		{
			int bufPos = (int)i + j;
			if (bufPos < 0 || bufPos >= (int)size) continue;

			float x = (float)j / (float)halfWinSize*(float)PI;
			float w = (cosf(x) + 1.0f)*0.5f;

			buf2[bufPos]+= w*ave;
		}
	}		
		
	memcpy(buf, buf2, sizeof(float)*size);
	delete[] buf2;
}

void UtauDraft::GenerateWave_SingConsecutive(SyllableInternalList syllableList, NoteBuffer* noteBuf)
{
	LyricPieceList lyricList;
	for (unsigned i = 0; i < (unsigned)syllableList.size(); i++)
	{
		LyricPiece_Deferred piece;
		piece->lyric = syllableList[i]->lyric;
		piece->fNumOfSamples=syllableList[i]->GetTotalDuration();
		piece->isVowel = true;
		piece->syllableId = i;
		lyricList.push_back(piece);
	}

	if (m_LyricConverter != nullptr)
	{
		lyricList = _convertLyric(lyricList);
	}
	unsigned numPieces = (unsigned)lyricList.size();

	if (m_PrefixMap != nullptr && m_use_prefix_map)
	{
		std::vector<std::string> prefixes;
		prefixes.resize(syllableList.size());

		for (unsigned i = 0; i < syllableList.size(); i++)
		{
			SyllableInternal& syllable = *syllableList[i];
			float aveFreq = 0.0f;
			float sumLen = 0.0f;
			for (unsigned j = 0; j < syllable.ctrlPnts.size(); j++)
			{
				ControlPointInternal& ctrlPnt = syllable.ctrlPnts[j];
				if (ctrlPnt.fNumOfSamples <= 0.0f) continue;
				float freq1 = ctrlPnt.sampleFreq;
				float freq2 = j < syllable.ctrlPnts.size() - 1 ? syllable.ctrlPnts[j + 1].sampleFreq : freq1;

				aveFreq += (freq1 + freq2)*ctrlPnt.fNumOfSamples;
				sumLen += ctrlPnt.fNumOfSamples;				
			}
			aveFreq *= noteBuf->m_sampleRate/sumLen*0.5f;
			prefixes[i] = m_PrefixMap->GetPrefixFromFreq(aveFreq);
		}

		for (unsigned j = 0; j < numPieces; j++)
		{
			LyricPiece & piece = *lyricList[j];
			piece.lyric += prefixes[piece.syllableId];
		}
	}
	
	std::vector<unsigned> lens;
	lens.resize(numPieces);
	float sumAllLen = 0.0f;

	float firstNoteHead = this->getFirstNoteHeadSamples(lyricList[0]->lyric.data());

	for (unsigned j = 0; j < numPieces; j++)
	{
		LyricPiece & piece = *lyricList[j];
		float sumLen = piece.fNumOfSamples;

		if (j == 0)	sumLen += firstNoteHead;

		float oldSumAllLen = sumAllLen;
		sumAllLen += sumLen;

		lens[j] = (unsigned)ceilf(sumAllLen) - (unsigned)ceilf(oldSumAllLen);
	}
	unsigned uSumAllLen = (unsigned)ceilf(sumAllLen);

	noteBuf->m_sampleNum = uSumAllLen;
	noteBuf->m_alignPos = (unsigned)firstNoteHead;
	noteBuf->Allocate();

	float *freqAllMap = new float[uSumAllLen];	
	float targetPos = firstNoteHead;

	float headFreq = syllableList[0]->ctrlPnts[0].sampleFreq;
	unsigned uPos = 0;
	for (; (float)uPos < targetPos; uPos++)
	{
		freqAllMap[uPos] = headFreq;
	}

	for (unsigned i = 0; i < (unsigned)syllableList.size(); i++)
	{
		SyllableInternal& syllable = *syllableList[i];
		for (unsigned j = 0; j < syllable.ctrlPnts.size(); j++)
		{
			ControlPointInternal& ctrlPnt = syllable.ctrlPnts[j];
			if (ctrlPnt.fNumOfSamples <= 0.0f) continue;
			float startPos = targetPos;
			targetPos += ctrlPnt.fNumOfSamples;
			float freq1 = ctrlPnt.sampleFreq;
			float freq2 = j < syllable.ctrlPnts.size() - 1 ? syllable.ctrlPnts[j + 1].sampleFreq : freq1;
			for (; (float)uPos < targetPos && uPos<uSumAllLen; uPos++)
			{
				float k = ((float)uPos - startPos) / (targetPos - startPos);
				freqAllMap[uPos] = freq1*(1.0f-k)+freq2*k;
			}
		}
	}

	SyllableInternal& lastSyllable = **(syllableList.end()-1);
	ControlPointInternal& lastCtrlPnt = *(lastSyllable.ctrlPnts.end()-1);
	float tailFreq = lastCtrlPnt.sampleFreq;
	for (; uPos < uSumAllLen; uPos++)
	{
		freqAllMap[uPos] = tailFreq;
	}

	_floatBufSmooth(freqAllMap, uSumAllLen);

	std::vector<std::string> lyrics;
	std::vector<unsigned> isVowel;
	std::vector<float> weights;

	lyrics.resize(numPieces);
	isVowel.resize(numPieces);
	weights.resize(numPieces);

	for (unsigned j = 0; j < numPieces; j++)
	{
		LyricPiece& piece = *lyricList[j];
		lyrics[j] = piece.lyric;
		isVowel[j] = piece.isVowel ? 1 : 0;
		weights[j] = piece.weight;
	}

	UtauSourceFetcher srcFetcher;
	srcFetcher.m_OtoMap = m_OtoMap;
	srcFetcher.m_defaultLyric = m_defaultLyric;

	SentenceGenerator* sg = createSentenceGenerator();
	sg->GenerateSentence(srcFetcher, numPieces, lyrics.data(), isVowel.data(), weights.data(), lens.data(), freqAllMap, noteBuf);
	releasSentenceGenerator(sg);

	delete[] freqAllMap;

	// Envolope
	for (unsigned pos = 0; pos < uSumAllLen; pos++)
	{
		float x2 = (float)(uSumAllLen - 1 - pos) / (float)lens[numPieces - 1];
		float amplitude = 1.0f - expf(-x2*10.0f);
		noteBuf->m_data[pos] *= amplitude;
	}

}

UtauDraft::LyricPieceList UtauDraft::_convertLyric(UtauDraft::LyricPieceList syllableList)
{
	PyObject* lyricList = PyList_New(0);
	for (unsigned i = 0; i < (unsigned)syllableList.size(); i++)
	{
		PyObject *byteCode = PyBytes_FromString(syllableList[i]->lyric.data());
		PyList_Append(lyricList, PyUnicode_FromEncodedObject(byteCode, m_lyric_charset.data(), 0));
	}
	PyObject* args = PyTuple_Pack(1, lyricList);
	PyObject* rets = PyObject_CallObject(m_LyricConverter, args);

	LyricPieceList list_converted;
	for (unsigned i = 0; i < (unsigned)syllableList.size(); i++)
	{
		PyObject* tuple = PyList_GetItem(rets, i);
		unsigned count = (unsigned)PyTuple_Size(tuple);

		std::vector<std::string> lyrics;
		std::vector<float> weights;
		std::vector<bool> isVowels;

		float sum_weight = 0.0f;
		for (unsigned j = 0; j < count; j += 3)
		{
			PyObject *byteCode = PyUnicode_AsEncodedString(PyTuple_GetItem(tuple, j), m_lyric_charset.data(), 0);
			std::string lyric = PyBytes_AS_STRING(byteCode);
			lyrics.push_back(lyric);

			float weight = 1.0f;
			if (j + 1 < count)
			{
				weight = (float)PyFloat_AsDouble(PyTuple_GetItem(tuple, j + 1));
			}
			weights.push_back(weight);

			sum_weight += weight;

			bool isVowel = true;
			if (j + 2 < count)
			{
				isVowel = PyObject_IsTrue(PyTuple_GetItem(tuple, j + 2)) != 0;
			}
			isVowels.push_back(isVowel);
		}

		LyricPiece_Deferred syllable = syllableList[i];

		float totalNumSamples = syllable->fNumOfSamples;

		for (unsigned j = 0; j < lyrics.size(); j++)
		{
			LyricPiece_Deferred newPiece;
			newPiece->lyric = lyrics[j];
			newPiece->weight = weights[j] / sum_weight;
			newPiece->fNumOfSamples = totalNumSamples*newPiece->weight;
			newPiece->isVowel = isVowels[j];
			newPiece->syllableId = i;

			list_converted.push_back(newPiece);
		}
	}
	return list_converted;
}

float UtauDraft::getFirstNoteHeadSamples(const char* lyric)
{
	UtauSourceFetcher srcFetcher;
	srcFetcher.m_OtoMap = m_OtoMap;
	srcFetcher.m_defaultLyric = m_defaultLyric;

	SourceInfo srcInfo;
	srcFetcher.FetchSourceInfo(lyric, srcInfo);

	float overlap_pos = srcInfo.loc.overlap* (float)srcInfo.source.m_sampleRate*0.001f;
	float preutter_pos = srcInfo.loc.preutterance * (float)srcInfo.source.m_sampleRate*0.001f;
	if (preutter_pos < overlap_pos) preutter_pos = overlap_pos;

	return preutter_pos - overlap_pos;

}


#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

static bool s_have_cuda = false;

static bool HaveCUDA()
{
#if HAVE_CUDA
	int count;
	cudaGetDeviceCount(&count);
	if (count > 0 && cudaGetLastError() == 0)
	{
		cudaFree(nullptr);
		if (cudaGetLastError() == 0) s_have_cuda = true;
	}
#endif
	return s_have_cuda;
}

class UtauDraftInitializer
{
public:
	void SetPath(const char* path)
	{
		m_path = path;
		char charFileName[1024];
		sprintf(charFileName, "%s/character.txt", m_path.data());
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
		sprintf(searchStr, "%s/*", path);

		hFind = FindFirstFileA(searchStr, &ffd);
		if (INVALID_HANDLE_VALUE == hFind) return;

		do
		{
			if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY && strcmp(ffd.cFileName, ".") != 0 && strcmp(ffd.cFileName, "..") != 0)
			{
				char subPath[2048];
				sprintf(subPath, "%s/%s", path, ffd.cFileName);
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

	UtauDraft* Init(bool useCuda)
	{
		if (useCuda && !HaveCUDA())	useCuda = false;
		if (m_OtoMap.size() == 0)
		{
			BuildOtoMap(m_path.data());

			char prefixMapFn[1024];
			sprintf(prefixMapFn, "%s/prefix.map", m_path.data());
			m_PrefixMap.LoadFromFile(prefixMapFn);

#ifdef _WIN32
			m_charset = "shiftjis";
#else
			m_charset = "utf-8";
#endif
			char charsetFn[1024];
			sprintf(charsetFn, "%s/charset", m_path.data());

			FILE* fp_charset = fopen(charsetFn, "r");
			if (fp_charset)
			{
				char charsetName[100];
				fscanf(fp_charset, "%s", charsetName);
				m_charset = charsetName;
				fclose(fp_charset);
			}
		}
		UtauDraft* singer = new UtauDraft(useCuda);
		singer->SetOtoMap(&m_OtoMap);
		singer->SetCharset(m_charset.data());
		if (m_PrefixMap.size() > 0)
			singer->SetPrefixMap(&m_PrefixMap);
		return singer;
	}

private:
	OtoMap m_OtoMap;
	PrefixMap m_PrefixMap;
	std::string m_charset;

	std::string m_path;

	std::string m_charecter_txt;
};

#include <map>
std::map<std::string, UtauDraftInitializer> s_initializers;

UtauDraftInitializer* GetInitializer(std::string path)
{
	if (s_initializers.find(path) == s_initializers.end())
	{
		UtauDraftInitializer initializer;
		initializer.SetPath(path.data());
		s_initializers[path] = initializer;
	}
	return &s_initializers[path];
}

static PyObject* InitializeUtauDraft(PyObject *self, PyObject *args)
{
	std::string path = _PyUnicode_AsString(PyTuple_GetItem(args,0));
	bool useCuda = PyObject_IsTrue(PyTuple_GetItem(args, 1))!=0;
	UtauDraftInitializer* initializer = GetInitializer(path);
	UtauDraft* singer = initializer->Init(useCuda);
	return PyLong_FromVoidPtr(singer);
}


static PyObject* DestroyUtauDraft(PyObject *self, PyObject *args)
{
	UtauDraft* singer = (UtauDraft*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	delete singer;
	return PyLong_FromLong(0);
}

static PyObject* UtauDraftSetLyricConverter(PyObject *self, PyObject *args)
{
	UtauDraft* singer = (UtauDraft*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	PyObject* LyricConverter = PyTuple_GetItem(args, 1);
	singer->SetLyricConverter(LyricConverter);
	return PyLong_FromUnsignedLong(0);
}

static PyObject* UtauDraftSetUsePrefixMap(PyObject *self, PyObject *args)
{
	UtauDraft* singer = (UtauDraft*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	bool use_prefix_map = PyObject_IsTrue(PyTuple_GetItem(args, 1))!=0;
	singer->SetUsePrefixMap(use_prefix_map);
	return PyLong_FromUnsignedLong(0);
}

static PyObject* UtauDraftSetCZMode(PyObject *self, PyObject *args)
{
	UtauDraft* singer = (UtauDraft*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	bool czmode = PyObject_IsTrue(PyTuple_GetItem(args, 1)) != 0;
	singer->SetCZMode(czmode);
	return PyLong_FromUnsignedLong(0);
}


static PyMethodDef s_Methods[] = {
	{
		"InitializeUtauDraft",
		InitializeUtauDraft,
		METH_VARARGS,
		""
	},
	{
		"DestroyUtauDraft",
		DestroyUtauDraft,
		METH_VARARGS,
		""
	},
	{
		"UtauDraftSetLyricConverter",
		UtauDraftSetLyricConverter,
		METH_VARARGS,
		""
	},
	{
		"UtauDraftSetUsePrefixMap",
		UtauDraftSetUsePrefixMap,
		METH_VARARGS,
		""
	},
	{
		"UtauDraftSetCZMode",
		UtauDraftSetCZMode,
		METH_VARARGS,
		""
	},
	{ NULL, NULL, 0, NULL }
};


static struct PyModuleDef cModPyDem =
{
	PyModuleDef_HEAD_INIT,
	"UtauDraft_module", /* name of module */
	"",          /* module documentation, may be NULL */
	-1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
	s_Methods
};

PyMODINIT_FUNC PyInit_PyUtauDraft(void) {
	return PyModule_Create(&cModPyDem);
}
