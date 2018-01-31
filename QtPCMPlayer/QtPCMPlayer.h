#ifndef _QtPCMPlayer_h
#define _QtPCMPlayer_h

#include <qwidget.h>
#include "ui_QtPCMPlayer.h"

#include <QIODevice>
#include <QtMultimedia/QAudioOutput>
#include <QTime>

#include <Deferred.h>
#include <vector>

class AudioBuffer;
typedef Deferred<AudioBuffer> AudioBuffer_Deferred;

class BufferQueue;
class BufferFeeder : public QIODevice
{
	Q_OBJECT
public:
	BufferFeeder(BufferQueue* queue, QObject *parent);
	~BufferFeeder();

	virtual qint64 readData(char *data, qint64 maxlen);
	virtual qint64 writeData(const char *data, qint64 len);
	virtual qint64 bytesAvailable() const;

signals:
	void feedingPos(QTime time, unsigned pos);

private:
	BufferQueue* m_BufferQueue;
};

class QLocalServer;

class QtPCMPlayer : public QWidget
{
	Q_OBJECT
public:
	QtPCMPlayer(QLocalServer* server);
	virtual ~QtPCMPlayer();

private:
	Ui_QtPCMPlayer m_ui;

	QLocalServer* m_server;

	BufferQueue* m_BufferQueue;
	BufferFeeder* m_Feeder;

	QAudioFormat m_format;
	QAudioOutput* m_audioOutput;

	bool m_initialized;

	void _playFile(const char* filename);

private slots:
	void newConnection();

	void playbackStateChanged(QAudio::State state);
	void btnPlayPauseToggled(bool checked);
	void feedingPos(QTime time, unsigned pos);

};

#endif

