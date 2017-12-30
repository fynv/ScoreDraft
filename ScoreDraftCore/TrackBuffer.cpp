#include "TrackBuffer.h"
#include <memory.h>
#include <cmath>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

static const unsigned s_localBufferSize=65536;

TrackBuffer::TrackBuffer(unsigned rate) : m_rate(rate)
{
	m_fp=tmpfile();

	m_localBuffer=new float[s_localBufferSize];
	m_localBufferPos=(unsigned)(-1);

	m_volume=1.0f;
	m_cursor = 0.0f;
	m_length = 0;
}

TrackBuffer::~TrackBuffer()
{
	delete m_localBuffer;
	fclose(m_fp);
}

unsigned TrackBuffer::Rate() const {return m_rate;}

float TrackBuffer::Volume() const
{
	return m_volume;
}

void TrackBuffer::SetVolume(float vol)
{
	m_volume=vol;
}

void TrackBuffer::_seekToCursor()
{
	size_t upos = (size_t)(m_cursor);
	if (upos <= m_length)
	{
		fseek(m_fp, (long)(upos*sizeof(float)), SEEK_SET);
	}
	else
	{
		fseek(m_fp, 0, SEEK_END);
		float *tmp = new float[upos - m_length];
		memset(tmp, 0, (upos - m_length)*sizeof(float));
		fwrite(tmp, sizeof(float), (upos - m_length), m_fp);
		delete[] tmp;
		m_length = upos;
	}
}

float TrackBuffer::GetCursor()
{
	return m_cursor;
}

void TrackBuffer::SetCursor(float fpos)
{
	m_cursor = fpos;
	if (m_cursor < 0.0f) m_cursor = 0.0f;
	_seekToCursor();
}

void TrackBuffer::MoveCursor(float delta)
{
	SetCursor(m_cursor + delta);
}

void TrackBuffer::WriteSamples(unsigned count, const float* samples, float cursorDelta)
{
	fwrite(samples,sizeof(float),count,m_fp);

	size_t upos = (size_t)(m_cursor)+count;
	if (upos > m_length) m_length = upos;

	MoveCursor(cursorDelta);

	m_localBufferPos = -1;
}

void TrackBuffer::WriteBlend(unsigned count, const float* samples, float cursorDelta)
{
	size_t upos = (size_t)(m_cursor); 
	if (upos == m_length)
	{
		WriteSamples(count, samples, cursorDelta);
		return;
	}

	float *tmpSamples=new float[count];
	memcpy(tmpSamples,samples,sizeof(float)*count);

	unsigned sec = min(count, (unsigned)(m_length - upos));
	float* secbuf = new float[sec];
	fread(secbuf, sizeof(float), sec, m_fp);

	for (unsigned i = 0; i < sec; i++)
		tmpSamples[i] += secbuf[i];

	delete[] secbuf;
	_seekToCursor();

	WriteSamples(count, tmpSamples, cursorDelta);

	delete[] tmpSamples;
}

bool TrackBuffer::CombineTracks(TrackBuffer& sumbuffer, unsigned num, TrackBuffer_deferred* tracks)
{
	float *targetBuffer=new float[s_localBufferSize];
	unsigned *lengths=new unsigned[num];

	// scan
	unsigned i;
	unsigned rate=sumbuffer.Rate();
	float maxCursor = 0.0f;

	for (i=0;i<num;i++)
	{
		if (tracks[i]->Rate() != rate)
		{
			delete[] targetBuffer;
			delete[] lengths;
			return false;
		}
		lengths[i] = tracks[i]->NumberOfSamples();

		float cursor = tracks[i]->GetCursor();
		if (cursor > maxCursor) maxCursor = cursor;
	}

	maxCursor += sumbuffer.GetCursor();

	bool finish=false;
	unsigned sourcePos = 0;
	while (!finish)
	{
		finish=true;
		memset(targetBuffer,0,sizeof(float)*s_localBufferSize);
		unsigned maxCount=0;
		for (i=0;i<num;i++)
		{
			if (lengths[i]>0)
			{
				unsigned count=min(s_localBufferSize,lengths[i]);
				maxCount=max(count,maxCount);
				unsigned j;
				for (j=0;j<count;j++)
				{
					targetBuffer[j] += tracks[i]->Sample(j + sourcePos)* tracks[i]->Volume();
				}
				lengths[i]-=count;
				if (lengths[i]>0) finish=false;
			}
		}	
		sumbuffer.WriteBlend(maxCount, targetBuffer, (float)maxCount);
		sourcePos += maxCount;
	}
	sumbuffer.SetCursor(maxCursor);

	delete[] lengths;
	delete[] targetBuffer;
	
	return true;
}


unsigned TrackBuffer::NumberOfSamples()
{
	return (unsigned)m_length;
}

float TrackBuffer::Sample(unsigned index)
{
	if (m_localBufferPos==(unsigned)(-1) || m_localBufferPos>index || m_localBufferPos+s_localBufferSize<=index)
	{
		m_localBufferPos=(index/s_localBufferSize)*s_localBufferSize;
		unsigned num=NumberOfSamples();

		fseek(m_fp, m_localBufferPos*sizeof(float), SEEK_SET);
		fread(m_localBuffer,sizeof(float),min(s_localBufferSize,num-m_localBufferPos),m_fp);

		_seekToCursor();
	}

	return m_localBuffer[index-m_localBufferPos];
	
}

float TrackBuffer::MaxValue()
{
	unsigned i;
	unsigned num=NumberOfSamples();

	float maxValue=Sample(0);
	for (i=1;i<num;i++)
		maxValue=max(maxValue, fabsf(Sample(i)));
	
	return maxValue;
}