#ifndef _UtauDraft_h
#define _UtauDraft_h

#include <Singer.h>

struct _object;
typedef struct _object PyObject;
class PrefixMap;

#include "fft.h"
#include "VoiceUtil.h"
using namespace VoiceUtil;
#include "FrqData.h"
#include "OtoMap.h"

struct SourceInfo
{
	VoiceLocation loc;
	FrqData frq;
	Buffer source;
	float srcbegin;
	float srcend;
};

class UtauSourceFetcher
{
public:
	OtoMap* m_OtoMap;
	std::string m_defaultLyric;

	bool FetchSourceInfo(const char* lyric, SourceInfo& srcInfo, bool czmode=false, const char* lyric_next=nullptr) const;
	static bool ReadWavLocToBuffer(VoiceLocation loc, Buffer& buf, float& begin, float& end);

};

struct SourceDerivedInfo
{
	float overlap_pos;
	float preutter_pos;
	float fixed_end;
	float overlap_pos_next;
	float preutter_pos_next;
	float vowel_Weight;
	float fixed_Weight;
	float headerWeight;

	void DeriveInfo(bool firstNote, bool hasNext, unsigned uSumLen, const SourceInfo& curSrc, const SourceInfo& nextSrc);
};

class SentenceGenerator
{
public:
	SentenceGenerator(){}
	virtual ~SentenceGenerator(){}
	float _transition;
	float _gender;
	bool _CZMode;

	virtual void GenerateSentence(const UtauSourceFetcher& srcFetcher, unsigned numPieces, const std::string* lyrics, const unsigned* isVowel,  const float* weights, const unsigned* lengths, const float *freqAllMap, NoteBuffer* noteBuf) = 0;

};

class UtauDraft : public Singer
{
public:
	UtauDraft(bool useCUDA=false);
	virtual ~UtauDraft();

	void SetOtoMap(OtoMap* otoMap);
	void SetPrefixMap(PrefixMap* prefixMap);
	void SetCharset(const char* charset);
	void SetLyricConverter(PyObject* lyricConverter);

	virtual bool Tune(const char* cmd);

	void SetUsePrefixMap(bool use) { m_use_prefix_map = use; }
	void SetCZMode(bool czmode) { m_CZMode = czmode; }

	virtual void GenerateWave(SyllableInternal syllable, NoteBuffer* noteBuf);
	virtual void GenerateWave_SingConsecutive(SyllableInternalList syllableList, NoteBuffer* noteBuf);

private:
	static void _floatBufSmooth(float* buf, unsigned size);

	struct LyricPiece
	{
		std::string lyric;
		float fNumOfSamples;
		float weight;
		bool isVowel;
		unsigned syllableId;
	};

	typedef Deferred<LyricPiece> LyricPiece_Deferred;
	typedef std::vector<LyricPiece_Deferred> LyricPieceList;

	LyricPieceList _convertLyric(LyricPieceList syllableList);

	float getFirstNoteHeadSamples(const char* lyric);

	SentenceGenerator* createSentenceGenerator();
	void releasSentenceGenerator(SentenceGenerator* sg) { delete sg; }

	OtoMap* m_OtoMap;

	float m_transition;
	float m_gender;
	bool m_CZMode;

	PyObject* m_LyricConverter;

	bool m_use_prefix_map;
	PrefixMap* m_PrefixMap;

	bool m_use_CUDA;

};

#endif
