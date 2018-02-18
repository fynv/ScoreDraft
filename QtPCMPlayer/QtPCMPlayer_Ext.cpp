#include <Python.h>
#include "PyScoreDraft.h"
#include <QApplication>
#include <QtNetwork/QLocalSocket>
#include <QProcess>
#include <QDataStream>
#include <QThread>


#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

static PyScoreDraft* s_pPyScoreDraft;
static QString s_root;

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


PyObject * QPlayTrackBuffer(PyObject *args)
{
	static unsigned count = 0;

	unsigned BufferId = (unsigned)PyLong_AsUnsignedLong(args);
	TrackBuffer_deferred buffer = s_pPyScoreDraft->GetTrackBuffer(BufferId);
	buffer->SeekToCursor();

	unsigned AlignPos = buffer->AlignPos();

	unsigned size = buffer->NumberOfSamples();
	unsigned chn = buffer->NumberOfChannels();
	short* data = new short[size*chn];

	float volume = buffer->AbsoluteVolume();

	unsigned i;
	for (i = 0; i < size; i++)
	{
		float sample[2];
		buffer->Sample(i, sample);
		if (chn == 1)
		{
			data[i] = (short)(max(min(sample[0]*volume, 1.0f), -1.0f)*32767.0f);
		}
		else if (chn == 2)
		{
			data[2 * i] = (short)(max(min(sample[0] * volume, 1.0f), -1.0f)*32767.0f);
			data[2 * i + 1] = (short)(max(min(sample[1] * volume, 1.0f), -1.0f)*32767.0f);
		}
	}

	char fn[100];
	sprintf(fn, "%s/_tmp%d", s_root.toLocal8Bit().data(), count);
	count++;

	FILE *fp = fopen(fn, "wb");
	fwrite(&AlignPos, sizeof(unsigned), 1, fp);
	fwrite(&size, sizeof(unsigned), 1, fp);
	fwrite(&chn, sizeof(unsigned), 1, fp);
	fwrite(data, sizeof(short), size*chn, fp);
	fclose(fp);

	delete[] data;

	int argc = 0;
	QApplication app(argc, nullptr);
	
	QLocalSocket socket;
	socket.connectToServer("QtPCMPlayer");

	if (!socket.waitForConnected(500))
	{
		FILE *fp = fopen("rootpath", "w");
		fprintf(fp, "\"%s\"\n", s_root.toLocal8Bit().data());
		fclose(fp);

		QString playerPath;
#ifdef _WIN32
		playerPath = s_root+"/QtPCMPlayer.exe";
#else
		playerPath = s_root+"/QtPCMPlayer";
#endif
		if (!QProcess::startDetached(playerPath))
			return PyLong_FromUnsignedLong(0);

		while (!socket.waitForConnected(500))
		{
			QThread::msleep(500);
			socket.connectToServer("QtPCMPlayer");
		}
	}


	char cmd[200];
	sprintf(cmd, "NewBuffer %s", fn);

	SendString(socket, cmd);
	
	socket.disconnectFromServer();	

	return PyLong_FromUnsignedLong(0);
}


PyObject * QPlayGetRemainingTime(PyObject *args)
{
	int argc = 0;
	QApplication app(argc, nullptr);

	QLocalSocket socket;
	socket.connectToServer("QtPCMPlayer");

	if (!socket.waitForConnected(500))
		return PyFloat_FromDouble(-1.0);

	SendString(socket, "GetRemainingSec");

	QByteArray str;
	GetString(socket, str);

	return PyFloat_FromDouble(str.toDouble());
	
	socket.disconnectFromServer();

	return PyLong_FromUnsignedLong(0);
}



PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft, const char* root)
{
	s_pPyScoreDraft = pyScoreDraft;
	s_root = root;

	pyScoreDraft->RegisterInterfaceExtension("QPlayTrackBuffer", QPlayTrackBuffer, "buf", "buf.id",
		"\t'''\n"
		"\tUsing Qt Multimedia API to playback a track-buffer.\n"
		"\tbuf -- an instance of TrackBuffer.\n"
		"\t'''\n");

	pyScoreDraft->RegisterInterfaceExtension("QPlayGetRemainingTime", QPlayGetRemainingTime, "", "",
		"\t'''\n"
		"\tMonitoring how much time in seconds is remaining in current play-back.\n"
		"\t'''\n");

	char QtPluginPath[1024];
	sprintf(QtPluginPath, "%s/QtPlugins", root);

	QApplication::addLibraryPath(QtPluginPath);
}

