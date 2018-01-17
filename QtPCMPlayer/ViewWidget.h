#ifndef _ViewWidget_h
#define _ViewWidget_h

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <vector>

class ViewWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT
public:
	ViewWidget(QWidget* parent);
	~ViewWidget();

	std::vector<short> m_data;

protected:
	void initializeGL() override;
	void resizeGL(int w, int h) override;
	void paintGL() override;

};

#endif
