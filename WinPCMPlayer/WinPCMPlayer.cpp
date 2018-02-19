#ifdef _WIN32
#include <windows.h>
#pragma comment(lib, "winmm")

#include "WinPCMPlayer.h"
#include "TrackBuffer.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


class Buffer
{
public:
	unsigned m_size;
	unsigned m_chn;
	short* m_data;

	Buffer()
	{
		m_size=0;
		m_chn = 1;
		m_data=0;
	}
	~Buffer()
	{
		delete[] m_data;
	}
	void Allocate()
	{
		delete[] m_data;
		m_data = new short[m_size*m_chn];
	}
};

#include <queue>

class BufferQueue
{
public:
	BufferQueue()
	{
		m_curPos=0;
		m_totalBufferLenth = 0;
	}
	~BufferQueue()
	{
		while (!m_queue.empty())
		{
			Buffer* buf=m_queue.front();
			m_queue.pop();
			delete buf;
		}
	}

	void AddBuffer(TrackBuffer &track)
	{
		float volume = track.Volume();
		float pan = track.Pan();
		Buffer* newBuffer=new Buffer;
		newBuffer->m_size=track.NumberOfSamples();
		newBuffer->m_chn = track.NumberOfChannels();
		newBuffer->Allocate();

		unsigned i;
		for (i = 0; i < newBuffer->m_size; i++)
		{
			float v[2];
			track.Sample(i, v);
			if (newBuffer->m_chn==1)
				newBuffer->m_data[i] = (short)(max(min(v[0]*volume, 1.0f), -1.0f)*32767.0f);
			else if (newBuffer->m_chn == 2)
			{
				CalcPan(pan, v[0], v[1]);
				newBuffer->m_data[i * 2] = (short)(max(min(v[0] * volume, 1.0f), -1.0f)*32767.0f);
				newBuffer->m_data[i * 2+1] = (short)(max(min(v[1] * volume, 1.0f), -1.0f)*32767.0f);
			}

		}

		m_queue.push(newBuffer);

		m_totalBufferLenth += newBuffer->m_size;
	}

	void GetSample(short sample[2])
	{
		while (!m_queue.empty())
		{
			Buffer* buf=m_queue.front();
			if (m_curPos<buf->m_size) 
			{
				if (buf->m_chn == 1)
				{
					sample[0] = sample[1] = buf->m_data[m_curPos];
				}
				else if (buf->m_chn == 2)
				{
					sample[0] = buf->m_data[m_curPos*2];
					sample[1] = buf->m_data[m_curPos * 2+1];
				}
				m_curPos++;
				return;
			}
			m_curPos=0;
			m_queue.pop();
			m_totalBufferLenth -= buf->m_size;
			delete buf;
		}

		sample[0] = sample[1] = 0;
	}

	unsigned GetRemainingSamples()
	{
		return m_totalBufferLenth - m_curPos;
	}


private:
	std::queue<Buffer*> m_queue;
	unsigned m_curPos;
	unsigned m_totalBufferLenth;

};

static void CALLBACK SoundOutCallBack(HWAVEOUT hwo, UINT uMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR dwParam2)
{
	if (uMsg==WOM_DONE)
	{
		BufferQueue *bq=(BufferQueue *)dwInstance;
		WAVEHDR* pwhr=(WAVEHDR*)dwParam1;

		unsigned size=pwhr->dwBufferLength/(sizeof(short)*2);
		short* pdata=(short*)pwhr->lpData;

		unsigned i;
		for (i = 0; i < size; i++)
		{
			bq->GetSample(pdata+i*2);
		}

		waveOutWrite(hwo, pwhr, sizeof(WAVEHDR) );
	}
}



WinPCMPlayer::WinPCMPlayer(unsigned bufferSize):m_bufferSize(bufferSize)
{
	m_Buffer=new short[bufferSize*2*2];
	m_Buffer1=m_Buffer;
	m_Buffer2=m_Buffer+bufferSize*2;
	memset(m_Buffer,0,sizeof(short)*bufferSize*2*2);

	m_BufferQueue=new BufferQueue;

	m_initialized=false;

	m_Rate = 44100;

}

WinPCMPlayer::~WinPCMPlayer()
{
	delete m_BufferQueue;
	delete[] m_Buffer;
}

void WinPCMPlayer::PlayTrack(TrackBuffer &track)
{
	if (!m_initialized)
	{
		m_Rate = track.Rate();
		waveOutGetNumDevs(); 

		waveOutGetDevCaps (0, &m_WaveOutDevCaps, sizeof(WAVEOUTCAPS));

		m_WaveFormat.wFormatTag = WAVE_FORMAT_PCM;
		m_WaveFormat.nChannels = 2;
		m_WaveFormat.nSamplesPerSec = track.Rate();
		m_WaveFormat.nAvgBytesPerSec = track.Rate() * sizeof(short)*2;
		m_WaveFormat.nBlockAlign = sizeof(short)*2;
		m_WaveFormat.wBitsPerSample = 16;
		m_WaveFormat.cbSize = 0;

		waveOutOpen(&m_WaveOut, 0, &m_WaveFormat, (DWORD_PTR)(SoundOutCallBack), (DWORD_PTR)m_BufferQueue, CALLBACK_FUNCTION);

		m_WaveHeader1.lpData = (char *)m_Buffer1;
		m_WaveHeader1.dwBufferLength = m_bufferSize*sizeof(short)*2;
		m_WaveHeader1.dwFlags = 0;

		waveOutPrepareHeader( m_WaveOut, &m_WaveHeader1, sizeof(WAVEHDR) ); 
		waveOutWrite(m_WaveOut, &m_WaveHeader1, sizeof(WAVEHDR) );

		m_WaveHeader2.lpData = (char *)m_Buffer2;
		m_WaveHeader2.dwBufferLength = m_bufferSize*sizeof(short) * 2;
		m_WaveHeader2.dwFlags = 0;

		waveOutPrepareHeader( m_WaveOut, &m_WaveHeader2, sizeof(WAVEHDR) ); 
		waveOutWrite(m_WaveOut, &m_WaveHeader2, sizeof(WAVEHDR) );

		m_initialized=true;
	}
	m_BufferQueue->AddBuffer(track);

}

float WinPCMPlayer::GetRemainingTime()
{
	return (float)m_BufferQueue->GetRemainingSamples() / (float)m_Rate;
}

#endif // WIN32