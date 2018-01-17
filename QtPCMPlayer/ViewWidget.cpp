#include "ViewWidget.h"
#include <QPainter>

ViewWidget::ViewWidget(QWidget* parent) : QOpenGLWidget(parent)
{
	m_data.push_back(0);
	m_data.push_back(0);
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

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glDisable(GL_DEPTH_TEST);

	if (m_data.size() >= 2)
	{
		glColor3f(1.0f, 1.0f, 0.0f);
		glBegin(GL_LINE_STRIP);

		unsigned size = (unsigned)m_data.size();
		if (size > 1024) size = 1024;

		for (unsigned i = 0; i < size; i++)
		{
			float x = (float)i / (float)(size - 1)*2.0f - 1.0f;
			float y = (float)m_data[i] / 40000.0f;

			glVertex2f(x, y);		
		}

		glEnd();
	}

	painter.endNativePainting();
	painter.end();
}


