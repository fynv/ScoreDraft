#include "ViewWidget.h"
#include <QPainter>
#include <fft.h>

ViewWidget::ViewWidget(QWidget* parent) : QOpenGLWidget(parent)
{
	m_refPos = 0;
	m_samples_per_ms = 44.1f;

	m_Mode = Spectrum;
}

ViewWidget::~ViewWidget()
{

}

void ViewWidget::initializeGL()
{
	initializeOpenGLFunctions();
}


void ViewWidget::resizeGL(int w, int h)
{
	glViewport(0, 0, w, h);
}

void ViewWidget::paintGL()
{
	QPainter painter;
	painter.begin(this);
	painter.beginNativePainting();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glDisable(GL_DEPTH_TEST);

	m_renderedPos = m_refPos + (unsigned)((float)m_timer.elapsed()*m_samples_per_ms);
	m_BufferQueue.SetCursor(m_renderedPos);

	switch (m_Mode)
	{
	case Spectrum:
		{
			// spectrum visualization
			DComp* fftData = new DComp[2048];
			for (unsigned i = 0; i < 2048; i++)
			{
				float x = (float)((int)i - 1024) / 1024.0f;
				float win = 0.5f*(cosf(x*(float)PI) + 1.0f);
				fftData[i].Re = win*(float)m_BufferQueue.GetSample() / 32768.0f;
				fftData[i].Im = 0.0f;
			}

			fft(fftData, 11);

			float* barv = new float[100];

			for (unsigned i = 0; i < 100; i++)
			{
				float fstart = powf(2.0f, (float)i*0.1f);
				float fstop = powf(2.0f, (float)(i + 1)*0.1f);

				unsigned ustart = (unsigned)ceilf(fstart);
				unsigned ustop = (unsigned)ceilf(fstop);

				float ave = 0.0f;
				if (ustart == ustop)
					ave = DCAbs(&fftData[ustart]);
				else
				{
					for (unsigned j = ustart; j < ustop; j++)
					{
						ave += DCAbs(&fftData[j]);
					}
					ave /= (float)(ustop - ustart);
				}

				barv[i] = logf(ave*10.0f) / 10.0f;

			}
			delete[] fftData;

			glBegin(GL_QUADS);

			for (unsigned i = 0; i < 100.0f; i++)
			{
				float center = ((float)i + 0.5f) / 100.0f;
				float halfWidth = 0.4f / 100.0f;
				float left = center - halfWidth;
				float right = center + halfWidth;

				left = left*2.0f - 1.0f;
				right = right*2.0f - 1.0f;

				float bottom = -1.0f;

				float v = barv[i];
				if (v > 1.0f) v = 1.0f;
				float top = v * 2.0f - 1.0f;

				glColor3f(0.5f, 1.0f, 0.0f);
				glVertex2f(right, bottom);
				glVertex2f(left, bottom);

				glColor3f(0.5f + 0.5f*v, 1.0f - v, 0.0f);
				glVertex2f(left, top);
				glVertex2f(right, top);
			}
			glEnd();

			delete[] barv;

		}
		break;
	case WaveForm:
		{
			// waveform visualization
			unsigned winSize = 1024;

			glColor3f(1.0f, 1.0f, 0.0f);
			glBegin(GL_LINE_STRIP);

			for (unsigned i = 0; i < winSize; i++)
			{
				float x = (float)i / (float)(winSize - 1)*2.0f - 1.0f;
				float y = (float)m_BufferQueue.GetSample() / 40000.0f;

				glVertex2f(x, y);
			}

			glEnd();
		}
		break;

	}

	painter.endNativePainting();
	painter.end();

	update();
}

void ViewWidget::Freeze()
{
	m_refPos = m_renderedPos;
	m_timer = QTime();
	
}

void ViewWidget::SetMode(int index)
{
	m_Mode = (Mode)index;
}