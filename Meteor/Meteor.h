#ifndef _Meteor_h
#define _Meteor_h

#include <Python.h>
#include <vector>
#include <string>
#include <Syllable.h>

struct VisNote
{
	unsigned instrumentId;
	int pitch;
	float start;
	float end;
};

struct VisBeat
{
	unsigned percId;
	float start;
	float end;
};

typedef std::vector<std::pair<int, float>> TempoMap;

struct VisSinging
{
	unsigned singerId;
	std::string lyric;
	std::vector<float> pitch;
	float start;
	float end;

	float CreateFromSyllable(unsigned singerId, float pos, float sampleRate, float pitchShift, unsigned tempo, const Syllable& syllable, TempoMap *tempoMap, int tempoMapOffset);
};

#include "SubListLookUp.h"

class Visualizer
{
public:
	Visualizer() { m_needUpdateSublists = false; }
	void ProcessNoteSeq(unsigned instrumentId, float startPosition, float sampleRate, PyObject *seq_py, unsigned tempo, float RefFreq, TempoMap *tempoMap);
	void ProcessBeatSeq(unsigned percIdList[], float startPosition, float sampleRate, PyObject *seq_py, unsigned tempo, TempoMap *tempoMap);
	void ProcessSingingSeq(unsigned singerId, float startPosition, float sampleRate, PyObject *seq_py, unsigned tempo, float RefFreq, TempoMap *tempoMap);
	void Play(unsigned bufferId);

	const std::vector<VisNote>& GetNotes() const { return m_notes;  }
	const std::vector<VisBeat>& GetBeats() const { return m_beats;  }
	const std::vector<VisSinging>& GetSingings() const { return m_singings; }
	const SubLists& GetNotesSublists() const { return m_notes_sublists;  }
	const SubLists& GetBeatsSublists() const { return m_beats_sublists; }
	const SubLists& GetSingingSublists() const { return m_singing_sublists; }

	void SaveToFile(const char* filename);
	void LoadFromFile(const char* filename);

private:
	std::vector<VisNote> m_notes;
	std::vector<VisBeat> m_beats;
	std::vector<VisSinging> m_singings;
	
	void _updateSublists();

	SubLists m_notes_sublists;
	SubLists m_beats_sublists;
	SubLists m_singing_sublists;

	bool m_needUpdateSublists;

};

#endif
