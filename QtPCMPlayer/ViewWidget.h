#ifndef _ViewWidget_h
#define _ViewWidget_h

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QTime>
#include "BufferQueue.h"

class ViewWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT
public:
	ViewWidget(QWidget* parent);
	~ViewWidget();

	enum Mode
	{
		Spectrum,
		WaveForm
	};

public slots:
	void AddBuffer(AudioBuffer_Deferred buf)
	{
		m_BufferQueue.AddBuffer(buf);
	}
	void SetRefPos(QTime time, unsigned refPos)
	{
		m_timer = time;
		m_refPos = refPos;
	}
	void Freeze();

	void SetMode(int index);
	
protected:
	void initializeGL() override;
	void resizeGL(int w, int h) override;
	void paintGL() override;

	BufferQueue m_BufferQueue;

	unsigned m_refPos;
	QTime m_timer;
	unsigned m_renderedPos;

	float m_samples_per_ms;

	Mode m_Mode;

};

#endif
