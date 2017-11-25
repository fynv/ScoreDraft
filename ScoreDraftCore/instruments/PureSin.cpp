#include "instruments/PureSin.h"
#include "Note.h"
#include "TrackBuffer.h"

#include <cmath>

#define PI 3.14159265359f

PureSin::PureSin()
{
}

PureSin::~PureSin()
{
}

void PureSin::GenerateNoteWave(unsigned numOfSamples, float sampleFreq, NoteBuffer* noteBuf)
{
	noteBuf->m_sampleNum=numOfSamples;
	noteBuf->Allocate();

	unsigned j;

	for (j=0;j<noteBuf->m_sampleNum;j++)
	{
		float amplitude=sinf(PI*(float)j/noteBuf->m_sampleNum);
		float wave=cos(2*PI*sampleFreq*j);
		noteBuf->m_data[j]=amplitude*wave;
	}
}

