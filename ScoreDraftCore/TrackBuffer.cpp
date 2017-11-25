#include "TrackBuffer.h"
#include <memory.h>


#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

static const unsigned localBufferSize=65536;

TrackBuffer::TrackBuffer(unsigned rate) : m_rate(rate)
{
	m_fp=tmpfile();

	m_localBuffer=new float[localBufferSize];
	m_curPos=(unsigned)(-1);

	m_volume=1.0f;
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

void TrackBuffer::SeekSample(long offset, int origin)
{
	fseek(m_fp,offset*sizeof(float),origin);
}

long TrackBuffer::Tell()
{
	return ftell(m_fp)/sizeof(float);
}

void TrackBuffer::WriteSamples(unsigned count, const float* samples)
{
	fwrite(samples,sizeof(float),count,m_fp);
}

	
void TrackBuffer::ReadSamples(unsigned count, float* samples)
{
	fread(samples,sizeof(float),count,m_fp);
}

void TrackBuffer::WriteBlend(unsigned count, const float* samples)
{
	long cur=Tell();
	SeekSample(0,SEEK_END);
	long end=Tell();
	if (cur==end) 
	{
		WriteSamples(count,samples);
		return;
	}

	float *tmpSamples=new float[count];
	memcpy(tmpSamples,samples,sizeof(float)*count);

	SeekSample(cur,SEEK_SET);
	unsigned sec=min(count,(unsigned)(end-cur));
	float* secbuf=new float[sec];
	ReadSamples(sec,secbuf);
	unsigned i;
	for (i=0;i<sec;i++)
		tmpSamples[i]+=secbuf[i];
	delete[] secbuf;
	SeekSample(cur,SEEK_SET);
	WriteSamples(count,tmpSamples);

	delete[] tmpSamples;
}

bool TrackBuffer::CombineTracks(TrackBuffer& sumbuffer, unsigned num, TrackBuffer_deferred* tracks)
{
	float *sourceBuffer=new float[localBufferSize];
	float *targetBuffer=new float[localBufferSize];

	unsigned *lengths=new unsigned[num];

	// scan
	unsigned i;
	unsigned rate=sumbuffer.Rate();

	for (i=0;i<num;i++)
	{
		if (tracks[i]->Rate()!=rate) return false;
		tracks[i]->SeekSample(0,SEEK_END);
		lengths[i]=tracks[i]->Tell();
		tracks[i]->SeekSample(0,SEEK_SET);
	}

	bool finish=false;
	while (!finish)
	{
		finish=true;
		memset(targetBuffer,0,sizeof(float)*localBufferSize);
		unsigned maxCount=0;
		for (i=0;i<num;i++)
		{
			if (lengths[i]>0)
			{
				unsigned count=min(localBufferSize,lengths[i]);
				maxCount=max(count,maxCount);
				tracks[i]->ReadSamples(count,sourceBuffer);
				unsigned j;
				for (j=0;j<count;j++)
				{
					targetBuffer[j]+=sourceBuffer[j]*tracks[i]->Volume();
				}
				lengths[i]-=count;
				if (lengths[i]>0) finish=false;
			}
		}	
		sumbuffer.WriteSamples(maxCount,targetBuffer);
	}

	delete[] lengths;
	delete[] targetBuffer;
	delete[] sourceBuffer;
	
	return true;
}



unsigned TrackBuffer::NumberOfSamples()
{
	SeekSample(0,SEEK_END);
	return Tell();
}

float TrackBuffer::Sample(int index)
{
	if (m_curPos==(unsigned)(-1) || (int)m_curPos>index || (int)m_curPos+(int)localBufferSize<=index)
	{
		m_curPos=(index/localBufferSize)*localBufferSize;
		unsigned num=NumberOfSamples();

		SeekSample(m_curPos,SEEK_SET);
		fread(m_localBuffer,sizeof(float),min(localBufferSize,num-m_curPos),m_fp);
	}

	return m_localBuffer[index-m_curPos];
	
}

float TrackBuffer::MaxValue()
{
	unsigned i;
	unsigned num=NumberOfSamples();

	float maxValue=Sample(0);
	for (i=1;i<num;i++)
		maxValue=max(maxValue,Sample(i));
	
	return maxValue;
}