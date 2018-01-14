#ifndef _scoredraft_SingingPiece_h
#define _scoredraft_SingingPiece_h

#include "Note.h"
#include <string>
#include <vector>

class SingingPiece
{
public:
	std::string m_lyric;
	NoteSequence m_notes;

	SingingPiece(){}
	~SingingPiece(){}
};


class SingingSequence : public std::vector<SingingPiece>
{
public:
	friend inline SingingSequence operator + (const SingingSequence& A, const SingingSequence& B);
	inline SingingSequence& operator += (const SingingSequence& B)
	{
		insert(this->end(), B.begin(), B.end());
		return *this;
	}
};


inline SingingSequence operator + (const SingingSequence& A, const SingingSequence& B)
{
	SingingSequence res = A;
	res.insert(res.end(), B.begin(), B.end());
	return res;
}


#endif
