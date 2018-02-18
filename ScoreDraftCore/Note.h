#ifndef _scoredraft_Note_h
#define _scoredraft_Note_h

#include <vector>

class Note
{
public:
	float m_freq_rel; // frequency factor relative to RefFreq
	int m_duration; // 1 beat = 48
	Note()
	{
		m_freq_rel = -1.0f; // silence
		m_duration = 48; // 1 beat
	}
	Note(float freq, int duration) : m_freq_rel(freq), m_duration(duration) {}
	~Note()	{}
};

class NoteSequence : public std::vector<Note>
{
public:
	friend inline NoteSequence operator + (const NoteSequence& A, const NoteSequence& B);
	inline NoteSequence& operator += (const NoteSequence& B)
	{
		insert(this->end(), B.begin(), B.end());
		return *this;
	}
};


inline NoteSequence operator + (const NoteSequence& A, const NoteSequence& B)
{
	NoteSequence res=A;
	res.insert(res.end(), B.begin(), B.end());
	return res;
}


#endif 