#ifndef _scoredraft_PureSin_h
#define _scoredraft_PureSin_h

#include "Instrument.h"

class PureSin : public Instrument
{
public:
	PureSin();
	~PureSin();

protected:
	virtual void GenerateNoteWave(unsigned numOfSamples, float sampleFreq, NoteBuffer* noteBuf);

};


#endif
