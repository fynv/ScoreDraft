#include "Instrument.h"
#include "Note.h"
#include "TrackBuffer.h"
#include <memory.h>
#include <cmath>
#include <vector>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

NoteBuffer::NoteBuffer()
{
	m_sampleNum=0;
	m_data=0;
}

NoteBuffer::~NoteBuffer()
{
	delete[] m_data;
}

void NoteBuffer::Allocate()
{
	delete[] m_data;
	m_data=new float[m_sampleNum];
}

class NoteTableItem
{
public:
	Note m_note;
	NoteBuffer m_noteBuffer;
};

class NoteTable : public std::vector<NoteTableItem*> {};

#include <cmath>
#include <time.h>
Instrument::Instrument()
{
	m_accelerate=true;
	//m_accelerate=false;
	m_NoteTable=new NoteTable;

	srand((unsigned)time(NULL));
}

Instrument::~Instrument()
{
	unsigned i;
	for (i=0;i<m_NoteTable->size();i++)
	{
		delete m_NoteTable->at(i);
	}
	delete m_NoteTable;
}

void Instrument::Silence(unsigned numOfSamples, NoteBuffer* noteBuf)
{
	noteBuf->m_sampleNum=numOfSamples;
	noteBuf->Allocate();
	memset(noteBuf->m_data,0,sizeof(float)*numOfSamples);
}

void Instrument::GenerateNoteWave(unsigned numOfSamples, float sampleFreq, NoteBuffer* noteBuf)
{
	Silence(numOfSamples,noteBuf);
}

inline float rand01()
{
	float f = (float)rand() / (float)RAND_MAX;
	if (f < 0.0000001f) f = 0.0000001f;
	if (f > 0.9999999f) f = 0.9999999f;
	return f;
}

void Instrument::PlayNote(TrackBuffer& buffer, const Note& aNote, unsigned tempo, float RefFreq)
{
	NoteBuffer l_noteBuf;
	NoteBuffer *noteBuf=&l_noteBuf;

	float fduration=fabsf((float)(aNote.m_duration*60))/(float)(tempo*48);
	float fNumOfSamples = buffer.Rate()*fduration;
	unsigned numOfSamples = (unsigned)(fNumOfSamples)+ ((fNumOfSamples - floorf(fNumOfSamples) > rand01())?1:0);

	bool bufferFilled=false;
	if (aNote.m_freq_rel<0.0f)
	{
		if (aNote.m_duration>0) 
		{
			Silence(numOfSamples, noteBuf);
			bufferFilled=true;
		}
		else if (aNote.m_duration<0)
		{
			buffer.SeekSample(-min((long)numOfSamples,buffer.Tell()),SEEK_CUR);
			return;
		}
		else return;
	}

	if (!bufferFilled)
	{
		if (m_accelerate)
		{
			unsigned i;
			for (i=0;i<m_NoteTable->size();i++)
			{
				Note& tabNote=m_NoteTable->at(i)->m_note;
				if (tabNote.m_duration == aNote.m_duration && fabsf(tabNote.m_freq_rel - aNote.m_freq_rel)<0.01f)
				{
					noteBuf=&(m_NoteTable->at(i)->m_noteBuffer);
					bufferFilled=true;
					break;
				}
			}
			if (i==m_NoteTable->size())
			{
				NoteTableItem* nti=new NoteTableItem;
				nti->m_note = aNote;
				noteBuf = &nti->m_noteBuffer;
				m_NoteTable->push_back(nti);
			}
		}

		if (!bufferFilled)
		{
			float freq = RefFreq*aNote.m_freq_rel;
			float sampleFreq=freq/(float)buffer.Rate();				
			GenerateNoteWave(numOfSamples, sampleFreq, noteBuf);
		}
	}
	
	buffer.WriteBlend(noteBuf->m_sampleNum,noteBuf->m_data);

	if (numOfSamples < noteBuf->m_sampleNum)
		buffer.SeekSample(numOfSamples-noteBuf->m_sampleNum,SEEK_CUR);
	else if (numOfSamples > noteBuf->m_sampleNum)
	{
		Silence(numOfSamples - noteBuf->m_sampleNum, &l_noteBuf);
		buffer.WriteBlend(l_noteBuf.m_sampleNum, l_noteBuf.m_data);
	}
		
}

void Instrument::PlayNotes(TrackBuffer& buffer, const NoteSequence& seq, unsigned tempo, float RefFreq)
{
	unsigned i;
	int prog=0;
	for (i=0;i<seq.size();i++)
	{
		int newprog = (i + 1) * 10 / seq.size();
		if (newprog>prog)
		{
			printf("-");
			prog=newprog;
		}
			
		PlayNote(buffer,seq[i],tempo,RefFreq);
	}
	printf("\n");
}