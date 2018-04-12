#ifndef _scoredraft_Triangle_h
#define _scoredraft_Triangle_h

#include "Instrument.h"

class Triangle : public Instrument
{
public:
	Triangle();
	virtual ~Triangle();

protected:
	virtual void GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf);

};


#endif
