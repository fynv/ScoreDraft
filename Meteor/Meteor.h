#ifndef _Meteor_h
#define _Meteor_h

#include <Python.h>
#include <vector>

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

class Visualizer
{
public:
	void ProcessNoteSeq(unsigned instrumentId, float startPosition, PyObject *seq_py, unsigned tempo, float RefFreq);	
	void ProcessBeatSeq(unsigned percIdList[], float startPosition, PyObject *seq_py, unsigned tempo);
	void Play(unsigned bufferId) const;

	const std::vector<VisNote>& GetNotes() const { return m_notes;  }
	const std::vector<VisBeat>& GetBeats() const { return m_beats;  }

private:
	std::vector<VisNote> m_notes;
	std::vector<VisBeat> m_beats;

};

#endif
