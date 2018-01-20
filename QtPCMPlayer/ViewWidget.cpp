#include "ViewWidget.h"
#include <QPainter>

ViewWidget::ViewWidget(QWidget* parent) : QOpenGLWidget(parent)
{
	m_refPos = 0;
	m_samples_per_ms = 44.1f;
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

	unsigned winSize = 1024;

	glColor3f(1.0f, 1.0f, 0.0f);
	glBegin(GL_LINE_STRIP);

	for (unsigned i = 0; i < winSize; i++)
	{
		float x = (float)i / (float)(winSize - 1)*2.0f - 1.0f;
		float y = (float)m_BufferQueue.GetSample()/ 40000.0f;

		glVertex2f(x, y);		
	}

	glEnd();	

	painter.endNativePainting();
	painter.end();

	update();
}

void ViewWidget::Freeze()
{
	m_refPos = m_renderedPos;
	m_timer = QTime();
	
}