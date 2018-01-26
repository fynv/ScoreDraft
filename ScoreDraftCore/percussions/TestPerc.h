#ifndef _scoredraft_TestPerc_h
#define _scoredraft_TestPerc_h

#include "Percussion.h"

class TestPerc : public Percussion
{
public:
	TestPerc();
	~TestPerc();

protected:
	virtual void GenerateBeatWave(float fNumOfSamples, BeatBuffer* beatBuf);

};

#endif

