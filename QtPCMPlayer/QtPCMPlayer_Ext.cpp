#include <Python.h>
#include <TrackBuffer.h>
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

static PyObject* QPlaySetRoot(PyObject *self, PyObject *args)
{
	const char* root = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
	s_root = root;
	char QtPluginPath[1024];
	sprintf(QtPluginPath, "%s/QtPlugins", root);
	QApplication::addLibraryPath(QtPluginPath);
	return PyLong_FromUnsignedLong(0);
}

static PyObject* QPlayTrackBuffer(PyObject *self, PyObject *args)
{
	static unsigned count = 0;

	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	buffer->SeekToCursor();

	unsigned AlignPos = buffer->AlignPos();

	unsigned size = buffer->NumberOfSamples();
	unsigned chn = buffer->NumberOfChannels();
	short* data = new short[size*chn];

	float volume = buffer->AbsoluteVolume();
	float pan = buffer->Pan();

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
			CalcPan(pan, sample[0], sample[1]);
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


static PyObject* QPlayGetRemainingTime(PyObject *self, PyObject *args)
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


static PyMethodDef s_Methods[] = {
	{
		"QPlaySetRoot",
		QPlaySetRoot,
		METH_VARARGS,
		""
	},
	{
		"QPlayTrackBuffer",
		QPlayTrackBuffer,
		METH_VARARGS,
		""
	},
	{
		"QPlayGetRemainingTime",
		QPlayGetRemainingTime,
		METH_VARARGS,
		""
	},
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef cModPyDem =
{
	PyModuleDef_HEAD_INIT,
	"QtPCMPlayer_module", /* name of module */
	"",          /* module documentation, may be NULL */
	-1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
	s_Methods
};

PyMODINIT_FUNC PyInit_PyQtPCMPlayerExt(void) {
	return PyModule_Create(&cModPyDem);
}
