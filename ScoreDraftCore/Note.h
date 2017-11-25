#ifndef _scoredraft_Note_h
#define _scoredraft_Note_h

#include <vector>

class Note
{
public:
	float m_freq_rel; // frequency factor relative to RefFreq
	int m_duration; // 1 beat = 48
	Note();
	~Note();
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