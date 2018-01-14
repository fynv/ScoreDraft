#ifndef _scoredraft_Beat_h
#define _scoredraft_Beat_h

#include <vector>

class Beat
{
public:
	int m_PercId;
	int m_duration; // 1 beat = 48
	Beat()
	{
		m_PercId = -1; // silence
		m_duration = 48; // 1 beat
	}
	~Beat(){}
};


class BeatSequence : public std::vector<Beat>
{
public:
	friend inline BeatSequence operator + (const BeatSequence& A, const BeatSequence& B);
	inline BeatSequence& operator += (const BeatSequence& B)
	{
		insert(this->end(), B.begin(), B.end());
		return *this;
	}
};


inline BeatSequence operator + (const BeatSequence& A, const BeatSequence& B)
{
	BeatSequence res = A;
	res.insert(res.end(), B.begin(), B.end());
	return res;
}


#endif 