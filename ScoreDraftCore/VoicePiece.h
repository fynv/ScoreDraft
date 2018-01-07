#ifndef _scoredraft_VoicePiece_h
#define _scoredraft_VoicePiece_h

#include "Note.h"
#include <string>

class VoicePiece
{
public:
	std::string m_lyric;
	NoteSequence m_notes;

	VoicePiece();
	~VoicePiece();
};


class VoiceSequence : public std::vector<VoicePiece>
{
public:
	friend inline VoiceSequence operator + (const VoiceSequence& A, const VoiceSequence& B);
	inline VoiceSequence& operator += (const VoiceSequence& B)
	{
		insert(this->end(), B.begin(), B.end());
		return *this;
	}
};


inline VoiceSequence operator + (const VoiceSequence& A, const VoiceSequence& B)
{
	VoiceSequence res = A;
	res.insert(res.end(), B.begin(), B.end());
	return res;
}


#endif
