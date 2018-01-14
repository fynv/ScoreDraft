#ifndef _scoredraft_RapPiece_h
#define _scoredraft_RapPiece_h

#include <string>
#include <vector>

class RapPiece
{
public:
	std::string m_lyric;
	int m_tone;
	int m_duration; // 1 beat = 48

	RapPiece()
	{
		m_tone = 1;
		m_duration = 48;
	}
	~RapPiece(){}
};


class RapSequence : public std::vector<RapPiece>
{
public:
	friend inline RapSequence operator + (const RapSequence& A, const RapSequence& B);
	inline RapSequence& operator += (const RapSequence& B)
	{
		insert(this->end(), B.begin(), B.end());
		return *this;
	}
};


inline RapSequence operator + (const RapSequence& A, const RapSequence& B)
{
	RapSequence res = A;
	res.insert(res.end(), B.begin(), B.end());
	return res;
}


#endif
