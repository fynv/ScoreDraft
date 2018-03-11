#include "Meteor.h"
#include "MainWidget.h"
#include "ViewWidget.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


BufferFeeder::BufferFeeder(unsigned chn, std::vector<short>& pcm, QObject *parent) :
	QIODevice(parent),
	m_chn(chn),
	m_pcm(pcm),
	m_feedPos(0)
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
	emit feedingPos(time, m_feedPos);

	short* sdata = (short*)data;
	qint64 count = len / sizeof(short) / 2;
	memset(sdata, 0, count*sizeof(short));

	if (m_feedPos == (unsigned)(-1))
		return count*sizeof(short) * 2;

	qint64 i;
	for (i = 0; i < count; i++, m_feedPos++)
	{
		if (m_feedPos*m_chn == (unsigned)m_pcm.size())
		{
			m_feedPos = (unsigned)(-1);
			break;
		}
		short *psdata = sdata + i * 2;
		if (m_chn == 1)
		{
			psdata[0] = psdata[1] = m_pcm[m_feedPos];
		}
		else if (m_chn == 2)
		{
			psdata[0] = m_pcm[m_feedPos * 2];
			psdata[1] = m_pcm[m_feedPos * 2 + 1];
		}
	}
	return count*sizeof(short) * 2;
}

qint64 BufferFeeder::writeData(const char *data, qint64 len)
{
	Q_UNUSED(data);
	Q_UNUSED(len);

	return 0;
}


qint64 BufferFeeder::bytesAvailable() const
{
	return 40000 * sizeof(short) + QIODevice::bytesAvailable();
}


MainWidget::MainWidget(const Visualizer* visualizer, TrackBuffer_deferred trackBuf)
	: QWidget(nullptr),
	m_TrackBuf(trackBuf),
	m_samples_per_sec(44100.0f)
{
	m_ui.setupUi(this);
	m_ui.display->SetData(visualizer);

	unsigned size = trackBuf->NumberOfSamples();
	unsigned chn = trackBuf->NumberOfChannels();
	m_pcm.resize(size*chn);
	float volume = trackBuf->AbsoluteVolume();
	float pan = trackBuf->Pan();

	unsigned i;
	for (i = 0; i < size; i++)
	{
		float sample[2];
		trackBuf->Sample(i, sample);
		if (chn == 1)
		{
			m_pcm[i] = (short)(max(min(sample[0] * volume, 1.0f), -1.0f)*32767.0f);
		}
		else if (chn == 2)
		{
			CalcPan(pan, sample[0], sample[1]);
			m_pcm[2 * i] = (short)(max(min(sample[0] * volume, 1.0f), -1.0f)*32767.0f);
			m_pcm[2 * i + 1] = (short)(max(min(sample[1] * volume, 1.0f), -1.0f)*32767.0f);
		}
	}

	m_Feeder = new BufferFeeder(chn, m_pcm, this);
	connect(m_Feeder, SIGNAL(feedingPos(QTime, unsigned)), this, SLOT(feedingPos(QTime, unsigned)));

	_startPlay();
}


MainWidget::~MainWidget()
{
	m_audioOutput->stop();
}

void MainWidget::_startPlay()
{
	m_format.setSampleRate(44100);
	m_format.setChannelCount(2);
	m_format.setSampleSize(16);
	m_format.setCodec("audio/pcm");
	m_format.setByteOrder(QAudioFormat::LittleEndian);
	m_format.setSampleType(QAudioFormat::SignedInt);

	m_audioOutput = new QAudioOutput(m_format, this);
	m_audioOutput->start(m_Feeder);
}


void MainWidget::feedingPos(QTime time, unsigned pos)
{
	if (pos == (unsigned)(-1))
	{
		this->close();
	}
	else
	{
		float bufferTime = ((float)pos - (float)m_TrackBuf->AlignPos()) / m_samples_per_sec;
		m_ui.display->SetRefTime(bufferTime, time);
	}

}




