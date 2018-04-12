#ifndef _scoredraft_PureSin_h
#define _scoredraft_PureSin_h

#include "Instrument.h"

class PureSin : public Instrument
{
public:
	PureSin();
	virtual ~PureSin();

protected:
	virtual void GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf);

};


#endif
