#ifndef _scoredraft_BottleBlow_h
#define _scoredraft_BottleBlow_h

#include "Instrument.h"

class BottleBlow : public Instrument
{
public:
	BottleBlow();
	~BottleBlow();

protected:
	virtual void GenerateNoteWave(unsigned numOfSamples, float sampleFreq, NoteBuffer* noteBuf);

};


#endif
