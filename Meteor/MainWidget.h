#ifndef _MainWidget_h
#define _MainWidget_h

#include "Meteor.h"
#include <TrackBuffer.h>
#include <vector>

#include <QWidget>
#include <QIODevice>
#include <QtMultimedia/QAudioOutput>
#include <QTime>

#include "ui_MainWidget.h"

class BufferFeeder : public QIODevice
{
	Q_OBJECT
public:
	BufferFeeder(unsigned chn, std::vector<short>& pcm, QObject *parent);
	~BufferFeeder();

	virtual qint64 readData(char *data, qint64 maxlen);
	virtual qint64 writeData(const char *data, qint64 len);
	virtual qint64 bytesAvailable() const;

signals:
	void feedingPos(QTime time, unsigned pos);

private:
	unsigned m_chn;
	unsigned m_feedPos;
	std::vector<short>& m_pcm;
};

class MainWidget : public QWidget
{
	Q_OBJECT
public:
	MainWidget(const Visualizer* visualizer, TrackBuffer* trackBuf);
	~MainWidget();

private:
	void _startPlay();

	Ui_MainWidget m_ui;

	TrackBuffer* m_TrackBuf;
	std::vector<short> m_pcm;

	BufferFeeder* m_Feeder;
	QAudioFormat m_format;
	QAudioOutput* m_audioOutput;

	float m_samples_per_sec;

private slots:
	void feedingPos(QTime time, unsigned pos);

};





#endif

