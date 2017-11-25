#ifndef _scoredraft_NaivePiano_h
#define _scoredraft_NaivePiano_h

#include "Instrument.h"

class NaivePiano : public Instrument
{
public:
	NaivePiano();
	~NaivePiano();

protected:
	virtual void GenerateNoteWave(unsigned numOfSamples, float sampleFreq, NoteBuffer* noteBuf);

};


#endif
