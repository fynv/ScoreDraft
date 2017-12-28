#include "instruments/Square.h"
#include "Note.h"
#include "TrackBuffer.h"

#include <cmath>

#define PI 3.14159265359f

Square::Square()
{
}

Square::~Square()
{
}

void Square::GenerateNoteWave(unsigned numOfSamples, float sampleFreq, NoteBuffer* noteBuf)
{
	noteBuf->m_sampleNum = numOfSamples;
	noteBuf->Allocate();

	unsigned j;

	for (j = 0; j<noteBuf->m_sampleNum; j++)
	{
		float x = sampleFreq*j;
		x = x - floor(x);
		float wave = x > 0.5f ? -1.0f : 1.0f;
		noteBuf->m_data[j] = wave* m_noteVolume;
	}
}

