#ifndef _scoredraft_Singer_h
#define _scoredraft_Singer_h

#include <vector>
#include <string>
#include <Deferred.h>

class NoteBuffer;
class TrackBuffer;
class Syllable;
class SyllableSequence;

struct ControlPointInternal
{
	float fNumOfSamples;
	float sampleFreq; // periods/sample or 1/samplesPerPeriod
};

struct SyllableInternal
{
	std::string lyric;
	std::vector<ControlPointInternal> ctrlPnts;

	float GetTotalDuration()
	{
		float totalDuration = 0.0f;
		for (size_t i = 0; i < ctrlPnts.size(); i++)
			totalDuration += ctrlPnts[i].fNumOfSamples;

		return totalDuration;
	}
};

typedef Deferred<SyllableInternal> SyllableInternal_Deferred;
typedef std::vector<SyllableInternal_Deferred> SyllableInternalList;

typedef std::vector<std::pair<int, float>> TempoMap;

class Singer
{
public:
	Singer();
	virtual ~Singer();

	void SingSyllable(TrackBuffer& buffer, const Syllable& syllable, unsigned tempo = 80, float RefFreq = 261.626f);
	void SingSyllable(TrackBuffer& buffer, const Syllable& syllable, const TempoMap& tempoMap, int tempoMapOffset=0, float RefFreq = 261.626f);
	void SingConsecutiveSyllables(TrackBuffer& buffer, const SyllableSequence& syllables, unsigned tempo = 80, float RefFreq = 261.626f);
	void SingConsecutiveSyllables(TrackBuffer& buffer, const SyllableSequence& syllables, const TempoMap& tempoMap, int tempoMapOffset = 0, float RefFreq = 261.626f);

	std::string GetLyricCharset()
	{
		return m_lyric_charset;
	}

	virtual bool Tune(const char* cmd);

	void SetDefaultLyric(const char* defaultLyric) { m_defaultLyric = defaultLyric; }
	void SetNoteVolume(float volume) { m_noteVolume = volume; }
	void SetNotePan(float pan) { m_notePan = pan; }

protected:
	void Silence(unsigned numOfSamples, NoteBuffer* noteBuf);
	virtual void GenerateWave(SyllableInternal syllable, NoteBuffer* noteBuf);
	virtual void GenerateWave_SingConsecutive(SyllableInternalList syllableList, NoteBuffer* noteBuf);

	float m_noteVolume;
	float m_notePan;

	std::string m_defaultLyric;
	std::string m_lyric_charset;

};

#endif
