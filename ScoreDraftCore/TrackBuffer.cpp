#include "TrackBuffer.h"
#include <memory.h>
#include <cmath>
#include <cassert>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


NoteBuffer::NoteBuffer()
{
	m_sampleRate = 44100.0f;
	m_channelNum = 1;
	m_sampleNum = 0;
	m_data = nullptr;

	m_cursorDelta = 0.0f;
	m_alignPos = 0;	
	m_volume = 1.0f;
	m_pan = 0.0f;
}

NoteBuffer::~NoteBuffer()
{
	delete[] m_data;
}

void NoteBuffer::Allocate()
{
	delete[] m_data;
	m_data = new float[m_sampleNum*m_channelNum];
}

TrackBuffer_deferred::TrackBuffer_deferred(){}
TrackBuffer_deferred::TrackBuffer_deferred(const TrackBuffer_deferred & in) : Deferred<TrackBuffer>(in){}
TrackBuffer_deferred::TrackBuffer_deferred(unsigned rate, unsigned chn): Deferred<TrackBuffer>(new TrackBuffer(rate, chn)){}

static const unsigned s_localBufferSize = 65536;
unsigned TrackBuffer::GetLocalBufferSize()
{
	return s_localBufferSize;
}

TrackBuffer::TrackBuffer(unsigned rate, unsigned chn) : m_rate(rate)
{
	if (chn < 1)
	{
		chn = 1;
	}
	else if (chn>2)
	{
		chn = 2;
	}
	m_chn = chn;

	m_fp = tmpfile();

	m_localBuffer = new float[s_localBufferSize*m_chn];
	m_localBufferPos = (unsigned)(-1);

	m_volume = 1.0f;
	m_pan = 0.0f;
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
		fseek(m_fp, (long)(sizeof(float)*upos*m_chn), SEEK_SET);
	}
	else
	{
		fseek(m_fp, 0, SEEK_END);
		float *tmp = new float[(upos - m_length)*m_chn];
		memset(tmp, 0, (upos - m_length)*m_chn*sizeof(float));
		fwrite(tmp, sizeof(float), (upos - m_length)*m_chn, m_fp);
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
	if (m_alignPos == (unsigned)(-1)) m_alignPos = 0;
	m_cursor = fpos;
	if (m_cursor < 0.0f) m_cursor = 0.0f;
}


void TrackBuffer::MoveCursor(float delta)
{
	SetCursor(m_cursor + delta);
}

void TrackBuffer::SeekToCursor()
{
	unsigned upos = (unsigned)(m_cursor);
	_seek(upos);
}


void TrackBuffer::_writeSamples(unsigned count, const float* samples, unsigned alignPos)
{
	unsigned upos = (unsigned)(m_cursor)+m_alignPos - alignPos;
	_seek(upos);
	fwrite(samples, sizeof(float), count*m_chn, m_fp);
	m_length = max(m_length, upos + count);
	m_localBufferPos = -1;
}


void TrackBuffer::WriteBlend(const NoteBuffer& noteBuf)
{
	assert(noteBuf.m_sampleRate == m_rate);
	unsigned count = noteBuf.m_sampleNum;
	unsigned src_chn = noteBuf.m_channelNum;

	float* samples = noteBuf.m_data;
	unsigned note_alignPos = noteBuf.m_alignPos;
	float cursorDelta = noteBuf.m_cursorDelta;
	float volume = noteBuf.m_volume;

	if (m_alignPos == (unsigned)(-1))
	{
		m_alignPos = note_alignPos;
	}
	if ((unsigned)(m_cursor)+m_alignPos < note_alignPos)
	{
		unsigned truncate = note_alignPos - (unsigned)(m_cursor)+m_alignPos;
		count -= truncate;
		samples += truncate*src_chn;
		note_alignPos -= truncate;
	}
	unsigned upos = (unsigned)(m_cursor)+m_alignPos - note_alignPos;

	float *tmpSamples = new float[count*m_chn];
	for (unsigned i = 0; i < count; i++)
	{
		float sample_l;
		float sample_r;

		if (src_chn == 1)
		{
			sample_l = sample_r = samples[i];
		}
		else if (src_chn == 2)
		{
			sample_l = samples[i * 2];
			sample_r = samples[i * 2 + 1];
		}

		if (m_chn == 1)
		{
			tmpSamples[i] = (sample_l + sample_r)*0.5f * volume;
		}
		else if (m_chn == 2)
		{
			CalcPan(noteBuf.m_pan, sample_l, sample_r);
			tmpSamples[i * 2] = sample_l* volume;
			tmpSamples[i * 2 + 1] = sample_r* volume;
		}

	}

	if (upos < m_length)
	{
		unsigned sec = min(count, m_length - upos);
		float* secbuf = new float[sec * m_chn];
		_seek(upos);
		fread(secbuf, sizeof(float), sec*m_chn, m_fp);

		for (unsigned i = 0; i < sec*m_chn; i++)
			tmpSamples[i] += secbuf[i];

		delete[] secbuf;
	}

	_writeSamples(count, tmpSamples, note_alignPos);

	delete[] tmpSamples;

	MoveCursor(cursorDelta);
}


bool TrackBuffer::CombineTracks(unsigned num, TrackBuffer_deferred* tracks)
{
	NoteBuffer targetBuffer;
	targetBuffer.m_sampleNum = s_localBufferSize;
	targetBuffer.m_channelNum = m_chn;
	targetBuffer.Allocate();

	unsigned *lengths = new unsigned[num];
	int* sourcePos = new int[num];
	float* trackVolumes = new float[num];
	float* trackPans = new float[num];

	// scan
	unsigned i;
	float maxCursor = 0.0f;
	unsigned maxAlign = 0;

	for (i = 0; i<num; i++)
	{
		if (tracks[i]->Rate() != m_rate)
		{
			delete[] trackPans;
			delete[] trackVolumes;
			delete[] sourcePos;
			delete[] lengths;
			return false;
		}
		lengths[i] = tracks[i]->NumberOfSamples();

		float cursor = tracks[i]->GetCursor();
		if (cursor > maxCursor) maxCursor = cursor;

		unsigned align = tracks[i]->AlignPos();
		if (align != (unsigned)(-1) && align > maxAlign) maxAlign = align;

		sourcePos[i] = (int)(align);
		trackVolumes[i] = tracks[i]->AbsoluteVolume();
		trackPans[i] = tracks[i]->Pan();
	}

	for (i = 0; i < num; i++)
	{
		sourcePos[i] -= (int)maxAlign;
	}

	maxCursor += m_cursor;

	bool finish = false;

	while (!finish)
	{
		finish = true;
		memset(targetBuffer.m_data, 0, sizeof(float)*s_localBufferSize*m_chn);
		unsigned maxCount = 0;

		for (i = 0; i<num; i++)
		{
			if ((int)lengths[i] > sourcePos[i])
			{
				int count = min(s_localBufferSize, (int)lengths[i] - sourcePos[i]);
				maxCount = (unsigned)max(count, (int)maxCount);
				int j;
				for (j = 0; j<count; j++)
				{
					if ((int)j + sourcePos[i]>0)
					{
						float samples[2];
						tracks[i]->Sample((unsigned)((int)j + sourcePos[i]), samples);
						float sample_l;
						float sample_r;
						if (tracks[i]->m_chn == 1)
						{
							sample_l = sample_r = samples[0];
						}
						else if (tracks[i]->m_chn == 2)
						{
							sample_l = samples[0];
							sample_r = samples[1];
						}

						if (m_chn == 1)
						{
							targetBuffer.m_data[j] += (sample_l + sample_r)*0.5f* trackVolumes[i];
						}
						else if (m_chn == 2)
						{
							CalcPan(trackPans[i], sample_l, sample_r);
							targetBuffer.m_data[j * 2] += sample_l* trackVolumes[i];
							targetBuffer.m_data[j * 2 + 1] += sample_r* trackVolumes[i];
						}
					}
				}
				sourcePos[i] += count;
				if ((int)lengths[i] > sourcePos[i]) finish = false;
			}
		}
		targetBuffer.m_sampleNum = maxCount;
		targetBuffer.m_cursorDelta = (float)(maxCount - maxAlign);
		targetBuffer.m_alignPos = maxAlign;
		WriteBlend(targetBuffer);
		maxAlign = 0;
	}
	SetCursor(maxCursor);

	delete[] trackPans;
	delete[] trackVolumes;
	delete[] sourcePos;
	delete[] lengths;

	return true;
}


void TrackBuffer::Sample(unsigned index, float* sample)
{
	if (index >= m_length)
	{
		for (unsigned c = 0; c<m_chn; c++)
			sample[c] = 0.0f;
	}
	if (m_localBufferPos == (unsigned)(-1) || m_localBufferPos>index || m_localBufferPos + s_localBufferSize <= index)
	{
		m_localBufferPos = (index / s_localBufferSize)*s_localBufferSize;
		memset(m_localBuffer, 0, sizeof(float)*s_localBufferSize*m_chn);

		if (m_localBufferPos < m_length)
		{
			fseek(m_fp, m_localBufferPos*sizeof(float)*m_chn, SEEK_SET);
			fread(m_localBuffer, sizeof(float), min(s_localBufferSize, m_length - m_localBufferPos)*m_chn, m_fp);
		}
	}

	unsigned readPos = index - m_localBufferPos;
	for (unsigned c = 0; c < m_chn; c++)
		sample[c] = m_localBuffer[readPos * m_chn + c];
}

void TrackBuffer::GetSamples(unsigned startIndex, unsigned length, float* buffer)
{
	while (length > 0)
	{
		if (startIndex >= m_length) break;
		if (m_localBufferPos == (unsigned)(-1) || m_localBufferPos > startIndex || m_localBufferPos + s_localBufferSize <= startIndex)
		{
			m_localBufferPos = (startIndex / s_localBufferSize)*s_localBufferSize;
			memset(m_localBuffer, 0, sizeof(float)*s_localBufferSize*m_chn);

			if (m_localBufferPos < m_length)
			{
				fseek(m_fp, m_localBufferPos*sizeof(float)*m_chn, SEEK_SET);
				fread(m_localBuffer, sizeof(float), min(s_localBufferSize, m_length - m_localBufferPos)*m_chn, m_fp);
			}
		}

		unsigned readLength = min(length, m_localBufferPos + s_localBufferSize - startIndex);
		memcpy(buffer, m_localBuffer + (startIndex - m_localBufferPos)*m_chn, sizeof(float)* readLength*m_chn);
		startIndex += readLength;
		length -= readLength;
		buffer += readLength;
	}
}

float TrackBuffer::MaxValue()
{
	unsigned i;
	float buf[2];
	Sample(0, buf);

	float maxValue;
	if (m_chn == 1)
		maxValue = fabsf(buf[0]);
	else if (m_chn == 2)
		maxValue = max(fabsf(buf[0]), fabsf(buf[1]));

	for (i = 1; i < m_length; i++)
	{
		Sample(i, buf);
		float _max;
		if (m_chn == 1)
			_max = fabsf(buf[0]);
		else if (m_chn == 2)
			_max = max(fabsf(buf[0]), fabsf(buf[1]));

		maxValue = max(maxValue, _max);
	}

	return maxValue;
}
