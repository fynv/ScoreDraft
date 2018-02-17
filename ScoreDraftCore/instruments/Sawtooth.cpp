#include "instruments/Sawtooth.h"
#include "Note.h"
#include "TrackBuffer.h"

#include <cmath>

#define PI 3.14159265359f

Sawtooth::Sawtooth()
{
}

Sawtooth::~Sawtooth()
{
}

void Sawtooth::GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf)
{
	noteBuf->m_sampleNum = (unsigned)ceilf(fNumOfSamples);
	noteBuf->Allocate();

	unsigned j;

	for (j = 0; j<noteBuf->m_sampleNum; j++)
	{
		float amplitude = 1.0f - ((float)j / (float)(noteBuf->m_sampleNum - 1));
		float phase = sampleFreq*j;
		float wave = 1.0f-2.0f*(phase - floor(phase));
		noteBuf->m_data[j] = amplitude*wave;
	}
}

