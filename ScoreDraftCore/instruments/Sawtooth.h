#ifndef _scoredraft_Sawtooth_h
#define _scoredraft_Sawtooth_h

#include "Instrument.h"

class Sawtooth : public Instrument
{
public:
	Sawtooth();
	~Sawtooth();

protected:
	virtual void GenerateNoteWave(unsigned numOfSamples, float sampleFreq, NoteBuffer* noteBuf);

};


#endif
