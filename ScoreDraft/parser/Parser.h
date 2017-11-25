#ifndef _scoredraft_Parser_h
#define _scoredraft_Parser_h

#include "parser/Document.h"

class Note;
class NoteSeqTable;

class Parser
{
public:
	Parser();
	virtual ~Parser();

	/// Result
	Document* m_doc;

	/// Parsing
	bool ParseNote(const char* strNote, Note& note);
	bool ParseSeq(const char* strSeq, NoteSequence& seq, char* errMsg = 0);

	bool ParseSet(const char* strSet, char* errMsg = 0);
	bool ParseTrack(const char* strTrack, char* errMsg = 0);

	bool ParseLine(const char* strLine, char* errMsg = 0);

private:
	NoteSeqTable* m_nst;

};


#endif
