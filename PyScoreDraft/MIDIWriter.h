#ifndef _MIDIWriter_h
#define _MIDIWriter_h

#include <vector>
#include "Deferred.h"

class NoteSequence;
typedef Deferred<NoteSequence> NoteSequence_deferred;
typedef std::vector<NoteSequence_deferred> SequenceList;

void WriteToMidi(const SequenceList& seqList, unsigned tempo, float refFreq, const char* fileName);

#endif
