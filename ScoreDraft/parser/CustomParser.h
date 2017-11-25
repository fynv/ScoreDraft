#ifndef _scoredraft_CustomParser_h
#define _scoredraft_CustomParser_h

#include <string>
#include <vector>
using std::string;

class Note;
class NoteSequence;

class CustomParser
{
public:
	struct NoteDef
	{
		string name;
		float freq_oct_rel;
	};

	void Customize(int count, const char* names[], float freqs[]);
	void ModifyFreq(const char* name, float freq);

	/// Parsing
	bool ParseNote(const char* strNote, Note& note);
	bool ParseSeq(const char* strSeq, NoteSequence& seq, char* errMsg = 0);

private:
	std::vector<NoteDef> m_NoteTable;
};

#endif
