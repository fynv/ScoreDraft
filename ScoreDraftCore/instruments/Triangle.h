#ifndef _scoredraft_Triangle_h
#define _scoredraft_Triangle_h

#include "Instrument.h"

class Triangle : public Instrument
{
public:
	Triangle();
	~Triangle();

protected:
	virtual void GenerateNoteWave(unsigned numOfSamples, float sampleFreq, NoteBuffer* noteBuf);

};


#endif
