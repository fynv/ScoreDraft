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
	short* m_data;

	Buffer()
	{
		m_size=0;
		m_data=0;
	}
	~Buffer()
	{
		delete[] m_data;
	}
	void Allocate()
	{
		delete[] m_data;
		m_data=new short[m_size];
	}
};

#include <queue>

class BufferQueue
{
public:
	BufferQueue()
	{
		m_curPos=0;
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

	void AddBuffer(TrackBuffer &track,float volume)
	{
		Buffer* newBuffer=new Buffer;
		newBuffer->m_size=track.NumberOfSamples();
		newBuffer->Allocate();

		unsigned i;
		for (i=0;i<newBuffer->m_size;i++)
			newBuffer->m_data[i]=(short)(max(min(track.Sample(i)*volume,1.0f),-1.0f)*32767.0f);

		m_queue.push(newBuffer);
	}

	short GetSample()
	{
		while (!m_queue.empty())
		{
			Buffer* buf=m_queue.front();
			if (m_curPos<buf->m_size) 
			{
				short value=buf->m_data[m_curPos];
				m_curPos++;
				return value;
			}
			m_curPos=0;
			m_queue.pop();
			delete buf;
		}
		return 0;
	}


private:
	std::queue<Buffer*> m_queue;
	unsigned m_curPos;

};

static void CALLBACK SoundOutCallBack(HWAVEOUT hwo, UINT uMsg, DWORD dwInstance, DWORD dwParam1, DWORD dwParam2)
{
	if (uMsg==WOM_DONE)
	{
		BufferQueue *bq=(BufferQueue *)dwInstance;
		WAVEHDR* pwhr=(WAVEHDR*)dwParam1;

		unsigned size=pwhr->dwBufferLength/sizeof(short);
		short* pdata=(short*)pwhr->lpData;

		unsigned i;
		for (i=0;i<size;i++)
			pdata[i]=bq->GetSample();

		waveOutWrite(hwo, pwhr, sizeof(WAVEHDR) );
	}
}



WinPCMPlayer::WinPCMPlayer(unsigned bufferSize):m_bufferSize(bufferSize)
{
	m_Buffer=new short[bufferSize*2];
	m_Buffer1=m_Buffer;
	m_Buffer2=m_Buffer+bufferSize;
	memset(m_Buffer,0,sizeof(short)*bufferSize*2);

	m_waveHeaderMem = new WAVEHDR[2];
	m_WaveHeader1 = m_waveHeaderMem;
	m_WaveHeader2 = m_waveHeaderMem + 1;

	m_BufferQueue=new BufferQueue;

	m_initialized=false;

}

WinPCMPlayer::~WinPCMPlayer()
{
	delete m_BufferQueue;
	delete[] m_waveHeaderMem;
	delete[] m_Buffer;
}

void WinPCMPlayer::PlayTrack(TrackBuffer &track)
{
	if (!m_initialized)
	{
		m_Volume=track.Volume();

		waveOutGetNumDevs(); 

		waveOutGetDevCaps (0, &m_WaveOutDevCaps, sizeof(WAVEOUTCAPS));

		m_WaveFormat.wFormatTag = WAVE_FORMAT_PCM;
		m_WaveFormat.nChannels = 1;
		m_WaveFormat.nSamplesPerSec = track.Rate();
		m_WaveFormat.nAvgBytesPerSec = track.Rate() * 2;
		m_WaveFormat.nBlockAlign = 2;
		m_WaveFormat.wBitsPerSample = 16;
		m_WaveFormat.cbSize = 0;

		waveOutOpen(&m_WaveOut, 0, &m_WaveFormat, (DWORD_PTR)(SoundOutCallBack), (DWORD_PTR)m_BufferQueue, CALLBACK_FUNCTION);

		m_WaveHeader1->lpData = (char *)m_Buffer1;
		m_WaveHeader1->dwBufferLength = m_bufferSize*sizeof(short);
		m_WaveHeader1->dwFlags = 0;

		waveOutPrepareHeader( m_WaveOut, m_WaveHeader1, sizeof(WAVEHDR) ); 
		waveOutWrite(m_WaveOut, m_WaveHeader1, sizeof(WAVEHDR) );

		m_WaveHeader2->lpData = (char *)m_Buffer2;
		m_WaveHeader2->dwBufferLength = m_bufferSize*sizeof(short);
		m_WaveHeader2->dwFlags = 0;

		waveOutPrepareHeader( m_WaveOut, m_WaveHeader2, sizeof(WAVEHDR) ); 
		waveOutWrite(m_WaveOut, m_WaveHeader2, sizeof(WAVEHDR) );

		m_initialized=true;
	}
	m_BufferQueue->AddBuffer(track,m_Volume);

}
#endif // WIN32