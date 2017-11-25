#include "instruments/NaivePiano.h"
#include "Note.h"
#include "TrackBuffer.h"

#include <cmath>

#define PI 3.14159265359f

NaivePiano::NaivePiano()
{
}

NaivePiano::~NaivePiano()
{
}

void NaivePiano::GenerateNoteWave(unsigned numOfSamples, float sampleFreq, NoteBuffer* noteBuf)
{
	noteBuf->m_sampleNum = numOfSamples;
	noteBuf->Allocate();

	unsigned j;

	for (j = 0; j<noteBuf->m_sampleNum; j++)
	{
		float x = sampleFreq*j;
		x = x - floor(x);

		float x2 = ((float)j / (float)(noteBuf->m_sampleNum - 1));

		float amplitude = 1.0f - powf(x2-0.5f, 3.0f)*8.0f;
		float wave = (1.0f + 0.5f*cos(2 * PI*x * 5))*sin(PI*x)* powf(1.0f - 2.0f * x,3.0f);

		noteBuf->m_data[j] = amplitude*wave;
	}
}

