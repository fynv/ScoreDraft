#ifndef _scoredraft_Square_h
#define _scoredraft_Square_h

#include "Instrument.h"

class Square : public Instrument
{
public:
	Square();
	~Square();

protected:
	virtual void GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf);

};


#endif
