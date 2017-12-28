#include "instruments/Triangle.h"
#include "Note.h"
#include "TrackBuffer.h"

#include <cmath>

#define PI 3.14159265359f

Triangle::Triangle()
{
}

Triangle::~Triangle()
{
}

void Triangle::GenerateNoteWave(unsigned numOfSamples, float sampleFreq, NoteBuffer* noteBuf)
{
	noteBuf->m_sampleNum = numOfSamples;
	noteBuf->Allocate();

	unsigned j;

	for (j = 0; j<noteBuf->m_sampleNum; j++)
	{
		float amplitude = 1.0f-2.0f*fabsf((float)j / (float)(noteBuf->m_sampleNum - 1)-0.5f);

		float x = sampleFreq*j;
		x = x - floor(x);
		
		float wave = x > 0.5f ? (x-0.75f)*4.0f: (0.25f-x)*4.0f;
		noteBuf->m_data[j] = wave*amplitude* m_noteVolume;
	}
}

