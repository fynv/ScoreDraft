#ifndef _scoredraft_Sawtooth_h
#define _scoredraft_Sawtooth_h

#include "Instrument.h"

class Sawtooth : public Instrument
{
public:
	Sawtooth();
	virtual ~Sawtooth();

protected:
	virtual void GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf);

};


#endif
