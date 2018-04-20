#ifndef _scoredraft_Syllable_h
#define _scoredraft_Syllable_h

#include <vector>
#include <string>

struct ControlPoint
{
	float m_freq_rel; // frequency factor relative to RefFreq
	int m_duration; // 1 beat = 48
};

class Syllable
{
public:
	std::string m_lyric;
	std::vector<ControlPoint> m_ctrlPnts;
};

class SyllableSequence : public std::vector<Syllable>
{
public:
	friend inline SyllableSequence operator + (const SyllableSequence& A, const SyllableSequence& B);
	inline SyllableSequence& operator += (const SyllableSequence& B)
	{
		insert(this->end(), B.begin(), B.end());
		return *this;
	}
};

inline SyllableSequence operator + (const SyllableSequence& A, const SyllableSequence& B)
{
	SyllableSequence res = A;
	res.insert(res.end(), B.begin(), B.end());
	return res;
}


#endif
