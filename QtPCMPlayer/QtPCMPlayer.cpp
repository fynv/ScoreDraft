#include <queue>
#include <stdio.h>
#include <string.h>
#include <QtNetwork/QLocalSocket>
#include <QtNetwork/QLocalServer>
#include <QFile>

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
		m_totalBufferLenth = 0;
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

		QFile file(filename);
		file.remove();

		m_queue.push(newBuffer);
		m_totalBufferLenth += size;
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
			m_totalBufferLenth -= buf->m_size;
			delete buf;
		}
		return 0;
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
	emit newbufferReady(sdata, (unsigned)count);
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


QtPCMPlayer::QtPCMPlayer(QLocalServer* server) : m_server(server)
{
	m_ui.setupUi(this);

	m_BufferQueue = new BufferQueue;
	m_Feeder = new BufferFeeder(m_BufferQueue, this);
	m_initialized = false;
	m_audioOutput = nullptr;

	connect(server, SIGNAL(newConnection()), this, SLOT(newConnection()));
	connect(m_Feeder, SIGNAL(newbufferReady(short*, unsigned)), this, SLOT(newbufferReady(short*, unsigned)));

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


void SendString(QLocalSocket& socket, const char* str)
{
	QByteArray content = str;

	QByteArray block;
	QDataStream out(&block, QIODevice::WriteOnly);
	out.setVersion(QDataStream::Qt_4_0);
	out << (quint32)0;
	out << content;
	out.device()->seek(0);
	out << (quint32)(block.size() - sizeof(quint32));

	socket.write(block);
	socket.flush();

	socket.waitForBytesWritten(-1);
}


bool GetString(QLocalSocket& socket, QByteArray& str)
{
	socket.waitForReadyRead(-1);

	QDataStream in(&socket);
	in.setVersion(QDataStream::Qt_4_0);

	quint32 blockSize;

	if (socket.bytesAvailable() < (int)sizeof(quint32))	return false;
	in >> blockSize;

	if (socket.bytesAvailable() < blockSize || in.atEnd()) return false;
	in >> str;

	return true;
}

void QtPCMPlayer::newConnection()
{
	QLocalSocket *clientConnection = m_server->nextPendingConnection();
	connect(clientConnection, SIGNAL(disconnected()), clientConnection, SLOT(deleteLater()));

	QByteArray str;
	GetString(*clientConnection, str);

	const char* line = str.data();
	char cmd[100];

	sscanf(line, "%s", &cmd);
	if (strcmp(cmd, "NewBuffer") == 0)
	{
		char fn[100];
		sscanf(line + strlen("NewBuffer") + 1, "%s", fn);
		_playFile(fn);
	}
	else if (strcmp(cmd, "GetRemainingSec") == 0)
	{
		double secs = (double)m_BufferQueue->GetRemainingSamples() / 44100.0;		
		sprintf(cmd, "%f", secs);
		SendString(*clientConnection, cmd);
	}
	
}

void QtPCMPlayer::newbufferReady(short* data, unsigned count)
{
	if (count > 2)
	{
		m_ui.view->m_data.resize(count);
		memcpy(m_ui.view->m_data.data(), data, sizeof(short)*count);
		m_ui.view->update();
	}
}
