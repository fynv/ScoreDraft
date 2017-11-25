#include "parser/NoteSeqTable.h"
#include "Note.h"
#include <memory>

NoteSeqTable::NoteSeqTable()
{
}

NoteSeqTable::~NoteSeqTable()
{
}

bool NoteSeqTable::AddNoteSequence(string name, const NoteSequence_deferred& seq)
{
	if (FindNoteSequence(name)) return false;
	NamedNoteSequence nseq;
	nseq.m_name=name;
	nseq.m_seq = seq;

	m_list.push_back(nseq);

	return true;
}

const NoteSequence* NoteSeqTable::FindNoteSequence(string name)
{
	unsigned size=m_list.size();
	unsigned i;
	for (i=0;i<size;i++)
	{
		NamedNoteSequence nseq=m_list[i];
		if (nseq.m_name==name) return nseq.m_seq;		
	}
	return 0;
}

