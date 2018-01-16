#include <queue>
#include <stdio.h>
#include "QtPCMPlayer.h"


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
		m_size = 0;
		m_data = 0;
	}
	~Buffer()
	{
		delete[] m_data;
	}
	void Allocate()
	{
		delete[] m_data;
		m_data = new short[m_size];
	}
};


class BufferQueue
{
public:
	BufferQueue()
	{
		m_curPos = 0;
	}
	~BufferQueue()
	{
		while (!m_queue.empty())
		{
			Buffer* buf = m_queue.front();
			m_queue.pop();
			delete buf;
		}
	}

	void AddBuffer(const char* filename)
	{
		FILE *fp = fopen(filename, "rb");
		fseek(fp, 0, SEEK_END);
		long size = ftell(fp);
		fseek(fp, 0, SEEK_SET);
		size /= 2;
		Buffer* newBuffer = new Buffer;
		newBuffer->m_size = size;
		newBuffer->Allocate();

		fread(newBuffer->m_data, sizeof(short), size, fp);
		fclose(fp);

		m_queue.push(newBuffer);
	}

	short GetSample()
	{
		while (!m_queue.empty())
		{
			Buffer* buf = m_queue.front();
			if (m_curPos<buf->m_size)
			{
				short value = buf->m_data[m_curPos];
				m_curPos++;
				return value;
			}
			m_curPos = 0;
			m_queue.pop();
			delete buf;
		}
		return 0;
	}


private:
	std::queue<Buffer*> m_queue;
	unsigned m_curPos;

};


BufferFeeder::BufferFeeder(BufferQueue* queue, QObject *parent) :QIODevice(parent), m_BufferQueue(queue)
{
	open(QIODevice::ReadOnly);
}

BufferFeeder::~BufferFeeder()
{
	close();
}

qint64 BufferFeeder::readData(char *data, qint64 len)
{
	short* sdata = (short*)data;
	qint64 count = len / sizeof(short);
	qint64 i;
	for (i = 0; i<count; i++)
		sdata[i] = m_BufferQueue->GetSample();
	return count*sizeof(short);
}

qint64 BufferFeeder::writeData(const char *data, qint64 len)
{
	Q_UNUSED(data);
	Q_UNUSED(len);

	return 0;
}


qint64 BufferFeeder::bytesAvailable() const
{
	return 20000 * sizeof(short) + QIODevice::bytesAvailable();
}


QtPCMPlayer::QtPCMPlayer()
{
	m_ui.setupUi(this);

	m_BufferQueue = new BufferQueue;
	m_Feeder = new BufferFeeder(m_BufferQueue, this);
	m_initialized = false;
	m_audioOutput = nullptr;

	_playFile("test");
}

QtPCMPlayer::~QtPCMPlayer()
{
	delete m_audioOutput;
	delete m_Feeder;
	delete m_BufferQueue;
}


void QtPCMPlayer::_playFile(const char* filename)
{
	if (!m_initialized)
	{
		m_format.setSampleRate(44100);
		m_format.setChannelCount(1);
		m_format.setSampleSize(16);
		m_format.setCodec("audio/pcm");
		m_format.setByteOrder(QAudioFormat::LittleEndian);
		m_format.setSampleType(QAudioFormat::SignedInt);

		m_audioOutput = new QAudioOutput(m_format, this);
		m_audioOutput->start(m_Feeder);

		m_initialized = true;
	}
	m_BufferQueue->AddBuffer(filename);

}


