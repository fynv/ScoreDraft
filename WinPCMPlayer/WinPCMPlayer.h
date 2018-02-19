#ifdef _WIN32
#ifndef _WinPCMPlayer_h
#define _WinPCMPlayer_h

#include <windows.h>
#include <mmsystem.h>

class TrackBuffer;
class BufferQueue;
class WinPCMPlayer
{
public:
	WinPCMPlayer(unsigned bufferSize=20000);
	virtual ~WinPCMPlayer();

	void PlayTrack(TrackBuffer &track);

	float GetRemainingTime();

private:
	unsigned m_bufferSize;
	short  *m_Buffer;
	short  *m_Buffer1,*m_Buffer2;	

	BufferQueue* m_BufferQueue;

	bool m_initialized;

	WAVEOUTCAPS		m_WaveOutDevCaps;
    HWAVEOUT		m_WaveOut;
	WAVEHDR			m_WaveHeader1, m_WaveHeader2;
    WAVEFORMATEX	m_WaveFormat;

	unsigned m_Rate;
	
};


#endif

#endif // WIN32