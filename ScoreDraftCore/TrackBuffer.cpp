#include "TrackBuffer.h"
#include <memory.h>
#include <cmath>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

static const unsigned s_localBufferSize = 65536;
unsigned TrackBuffer::GetLocalBufferSize()
{
	return s_localBufferSize;
}

TrackBuffer::TrackBuffer(unsigned rate) : m_rate(rate)
{
	m_fp=tmpfile();

	m_localBuffer=new float[s_localBufferSize];
	m_localBufferPos=(unsigned)(-1);

	m_volume=1.0f;
	m_cursor = 0.0f;
	m_length = 0;
	m_alignPos = (unsigned)(-1);
}

TrackBuffer::~TrackBuffer()
{
	delete m_localBuffer;
	fclose(m_fp);
}


void TrackBuffer::_seek(unsigned upos)
{
	if (upos <= m_length)
	{
		fseek(m_fp, (long)(upos*sizeof(float)), SEEK_SET);
	}
	else
	{
		fseek(m_fp, 0, SEEK_END);
		float *tmp = new float[upos - m_length];
		memset(tmp, 0, (upos - m_length)*sizeof(float));
		fwrite(tmp, sizeof(float), upos - m_length, m_fp);
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
}

void TrackBuffer::MoveCursor(float delta)
{
	SetCursor(m_cursor + delta);
}

void TrackBuffer::_writeSamples(unsigned count, const float* samples, unsigned alignPos)
{
	unsigned upos = (unsigned)(m_cursor)+m_alignPos - alignPos;
	_seek(upos);
	fwrite(samples,sizeof(float),count,m_fp);
	m_length = max(m_length, upos + count);
	m_localBufferPos = -1;
}

void TrackBuffer::WriteBlend(unsigned count, const float* samples, float cursorDelta, unsigned note_alignPos)
{
	if (m_alignPos == (unsigned)(-1))
	{
		m_alignPos = note_alignPos;
	}
	if ((unsigned)(m_cursor)+m_alignPos < note_alignPos)
	{
		unsigned truncate = note_alignPos - (unsigned)(m_cursor)+m_alignPos;
		count -= truncate;
		samples += truncate;
		note_alignPos -= truncate;
	}
	unsigned upos = (unsigned)(m_cursor)+m_alignPos - note_alignPos;
	if (upos >= m_length)
	{
		_writeSamples(count, samples, note_alignPos);
	}
	else
	{

		float *tmpSamples = new float[count];
		memcpy(tmpSamples, samples, sizeof(float)*count);

		unsigned sec = min(count, m_length - upos);
		float* secbuf = new float[sec];
		_seek(upos);
		fread(secbuf, sizeof(float), sec, m_fp);

		for (unsigned i = 0; i < sec; i++)
			tmpSamples[i] += secbuf[i];

		delete[] secbuf;

		_writeSamples(count, tmpSamples, note_alignPos);

		delete[] tmpSamples;
	}

	MoveCursor(cursorDelta);
}

bool TrackBuffer::CombineTracks(TrackBuffer& sumbuffer, unsigned num, TrackBuffer_deferred* tracks)
{
	float *targetBuffer=new float[s_localBufferSize];
	unsigned *lengths=new unsigned[num];
	int* sourcePos = new int[num];
	float* trackVolumes = new float[num];

	// scan
	unsigned i;
	unsigned rate=sumbuffer.Rate();
	float maxCursor = 0.0f;
	unsigned maxAlign = 0;

	for (i=0;i<num;i++)
	{
		if (tracks[i]->Rate() != rate)
		{
			delete[] trackVolumes;
			delete[] sourcePos;
			delete[] targetBuffer;
			delete[] lengths;
			return false;
		}
		lengths[i] = tracks[i]->NumberOfSamples();

		float cursor = tracks[i]->GetCursor();
		if (cursor > maxCursor) maxCursor = cursor;

		unsigned align = tracks[i]->AlignPos();
		if (align > maxAlign) maxAlign = align;

		sourcePos[i] = (int)(align);
		trackVolumes[i] = tracks[i]->AbsoluteVolume();
	}

	for (i = 0; i < num; i++)
	{
		sourcePos[i] -= (int)maxAlign;
	}

	maxCursor += sumbuffer.GetCursor();

	bool finish=false;
	
	while (!finish)
	{
		finish=true;
		memset(targetBuffer,0,sizeof(float)*s_localBufferSize);
		unsigned maxCount=0;
		for (i=0;i<num;i++)
		{
			if (lengths[i] - sourcePos[i]>0)
			{
				unsigned count = min(s_localBufferSize, lengths[i] - sourcePos[i]);
				maxCount=max(count,maxCount);
				unsigned j;
				for (j=0;j<count;j++)
				{
					if ((int)j + sourcePos[i]>0)
						targetBuffer[j] += tracks[i]->Sample((unsigned)((int)j + sourcePos[i]))* trackVolumes[i];
				}
				sourcePos[i] += count;
				if (lengths[i] - sourcePos[i]>0) finish = false;
			}
		}	
		sumbuffer.WriteBlend(maxCount, targetBuffer, (float)(maxCount - maxAlign), maxAlign);
		maxAlign = 0;
	}
	sumbuffer.SetCursor(maxCursor);

	delete[] trackVolumes;
	delete[] sourcePos;
	delete[] lengths;
	delete[] targetBuffer;
	
	return true;
}


float TrackBuffer::Sample(unsigned index)
{
	if (index >= m_length) return 0.0f;
	if (m_localBufferPos==(unsigned)(-1) || m_localBufferPos>index || m_localBufferPos+s_localBufferSize<=index)
	{
		m_localBufferPos=(index/s_localBufferSize)*s_localBufferSize;
		unsigned num=NumberOfSamples();

		fseek(m_fp, m_localBufferPos*sizeof(float), SEEK_SET);
		fread(m_localBuffer,sizeof(float),min(s_localBufferSize,num-m_localBufferPos),m_fp);
	}

	return m_localBuffer[index-m_localBufferPos];
	
}

void TrackBuffer::GetSamples(unsigned startIndex, unsigned length, float* buffer)
{
	while (length > 0)
	{
		if (startIndex >= m_length) break;
		if (m_localBufferPos == (unsigned)(-1) || m_localBufferPos > startIndex || m_localBufferPos + s_localBufferSize <= startIndex)
		{
			m_localBufferPos = (startIndex / s_localBufferSize)*s_localBufferSize;
			unsigned num = NumberOfSamples();

			fseek(m_fp, m_localBufferPos*sizeof(float), SEEK_SET);
			fread(m_localBuffer, sizeof(float), min(s_localBufferSize, num - m_localBufferPos), m_fp);
		}

		unsigned readLength = min(length, m_localBufferPos + s_localBufferSize - startIndex);
		memcpy(buffer, m_localBuffer + (startIndex - m_localBufferPos), sizeof(float)* readLength);
		startIndex += readLength;
		length -= readLength;
		buffer += readLength;
	}
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