#ifndef _scoredraft_NoteSeqTable_h
#define _scoredraft_NoteSeqTable_h

#include <string>
#include <vector>
#include "Deferred.h"

using std::string;

class NoteSequence;
typedef Deferred<NoteSequence> NoteSequence_deferred;
struct NamedNoteSequence
{
	string m_name;
	NoteSequence_deferred m_seq;
};

using std::string;
typedef std::vector<NamedNoteSequence> NamedNoteSequenceList;

class NoteSeqTable
{
public:
	NoteSeqTable();
	~NoteSeqTable();
	bool AddNoteSequence(string name, const NoteSequence_deferred& seq);
	const NoteSequence* FindNoteSequence(string name);
private:
	NamedNoteSequenceList m_list;
};


#endif 