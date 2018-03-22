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

class UtauDraft : public Singer
{
public:
	UtauDraft(bool useCUDA=false);
	~UtauDraft();

	enum Method
	{
		PSOLA,
		HNM,
		CUDA_HNM
	};

	void SetOtoMap(OtoMap* otoMap);
	void SetPrefixMap(PrefixMap* prefixMap);
	void SetCharset(const char* charset);
	void SetLyricConverter(PyObject* lyricConverter);

	virtual bool Tune(const char* cmd);

	virtual void GenerateWave(SingingPieceInternal piece, NoteBuffer* noteBuf);
	virtual void GenerateWave_Rap(RapPieceInternal piece, NoteBuffer* noteBuf);
	virtual void GenerateWave_SingConsecutive(SingingPieceInternalList pieceList, NoteBuffer* noteBuf);
	virtual void GenerateWave_RapConsecutive(RapPieceInternalList pieceList, NoteBuffer* noteBuf);


private:
	struct GenWaveStruct
	{
		bool _isVowel;
		float _transition;
		float _gender;

		unsigned uSumLen;
		float* freqMap;
		float* _phase;	

		Buffer dstBuf;

		bool firstNote;
		bool hasNextNote;		

		FrqData frq;
		Buffer source;
		float srcbegin;

		FrqData frq_next;
		Buffer source_next;
		float nextbegin;

		float overlap_pos;
		float preutter_pos;
		float fixed_end;

		float overlap_pos_next;
		float preutter_pos_next;

		float vowel_Weight;
		float fixed_Weight;
		float headerWeight;

		void _generateWave_PSOLA();
		void _generateWave_HNM();
		void _generateWave_CUDA_HNM();

	};

	static bool ReadWavLocToBuffer(VoiceLocation loc, Buffer& buf, float& begin, float& end);
	static void _floatBufSmooth(float* buf, unsigned size);
	SingingPieceInternalList _convertLyric_singing(SingingPieceInternalList pieceList);
	RapPieceInternalList _convertLyric_rap(const RapPieceInternalList& inputList);
	float getFirstNoteHeadSamples(const char* lyric);
	void _generateWave(const char* lyric, const char* lyric_next, unsigned uSumLen, float* freqMap, NoteBuffer* noteBuf, unsigned noteBufPos, float& phase, bool firstNote, bool isVowel);

	OtoMap* m_OtoMap;

	float m_transition;
	float m_rap_distortion;
	float m_gender;

	PyObject* m_LyricConverter;

	bool m_use_prefix_map;
	PrefixMap* m_PrefixMap;

	Method m_method;
};

#endif
