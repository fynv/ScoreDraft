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

class Visualizer
{
public:
	void ProcessNoteSeq(unsigned instrumentId, float startPosition, PyObject *seq_py, unsigned tempo, float RefFreq);	
	void Play(unsigned bufferId) const;

	const std::vector<VisNote>& GetNotes() const { return m_notes;  }

private:
	std::vector<VisNote> m_notes;

};

#endif
