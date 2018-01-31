#include <queue>
#include <stdio.h>
#include <string.h>
#include <QtNetwork/QLocalSocket>
#include <QtNetwork/QLocalServer>
#include <QFile>

#include "QtPCMPlayer.h"
#include "BufferQueue.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


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
	QTime time;
	time.start();
	emit feedingPos(time, m_BufferQueue->GetCursor());
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


QtPCMPlayer::QtPCMPlayer(QLocalServer* server) : m_server(server)
{
	m_ui.setupUi(this);

	m_BufferQueue = new BufferQueue;
	m_Feeder = new BufferFeeder(m_BufferQueue, this);
	m_initialized = false;
	m_audioOutput = nullptr;

	connect(server, SIGNAL(newConnection()), this, SLOT(newConnection()));
	connect(m_ui.btnPlayPause, SIGNAL(toggled(bool)), this, SLOT(btnPlayPauseToggled(bool)));
	connect(m_Feeder, SIGNAL(feedingPos(QTime, unsigned)), this, SLOT(feedingPos(QTime, unsigned)));

	m_ui.comboMode->addItem("Spectrum");
	m_ui.comboMode->addItem("Wave Form");
	m_ui.comboMode->setCurrentIndex(0);

	connect(m_ui.comboMode, SIGNAL(currentIndexChanged(int)), m_ui.view, SLOT(SetMode(int)));

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

		connect(m_audioOutput, SIGNAL(stateChanged(QAudio::State)), this, SLOT(playbackStateChanged(QAudio::State)));

		m_initialized = true;
	}

	unsigned AlignPos;
	unsigned size;

	FILE *fp = fopen(filename, "rb");
	fread(&AlignPos, sizeof(unsigned), 1, fp);
	fread(&size, sizeof(unsigned), 1, fp);
	
	AudioBuffer_Deferred newBuffer;
	newBuffer->resize((size_t)size);
	newBuffer->m_AlignPos = AlignPos;

	fread(newBuffer->data(), sizeof(short), size, fp);
	fclose(fp);

	QFile file(filename);
	file.remove();

	m_BufferQueue->AddBuffer(newBuffer);
	m_ui.view->AddBuffer(newBuffer);
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

	sscanf(line, "%s", cmd);
	if (strcmp(cmd, "NewBuffer") == 0)
	{
		const char* pFn= line + strlen("NewBuffer") + 1;
		_playFile(pFn);
	}
	else if (strcmp(cmd, "GetRemainingSec") == 0)
	{
		double secs = (double)m_BufferQueue->GetRemainingSamples() / 44100.0;		
		sprintf(cmd, "%f", secs);
		SendString(*clientConnection, cmd);
	}
	
}


void QtPCMPlayer::playbackStateChanged(QAudio::State state)
{
	if (state != QAudio::State::ActiveState)
	{
		m_ui.view->Freeze();
	}
}

void QtPCMPlayer::btnPlayPauseToggled(bool checked)
{
	if (m_audioOutput == nullptr) return;
	if (checked)
	{
		m_audioOutput->resume();		
	}
	else
	{
		m_audioOutput->suspend();
	}

}

void QtPCMPlayer::feedingPos(QTime time, unsigned pos)
{
	m_ui.view->SetRefPos(time, pos);
}
