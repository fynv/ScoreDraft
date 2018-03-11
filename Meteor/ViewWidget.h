#ifndef _ViewWidget_h
#define _ViewWidget_h

#include "Meteor.h"

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QTime>

#include "SubListLookUp.h"
#include <map>

typedef std::map<unsigned, unsigned char*> ColorMap;

struct Pos2D
{
	float x;
	float y;
};

class ViewWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT
public:
	ViewWidget(QWidget* parent);
	~ViewWidget();

	void SetData(const Visualizer* data);
	void SetRefTime(float refTime, const QTime& timer);

protected:
	void initializeGL() override;
	void resizeGL(int w, int h) override;
	void paintGL() override;

private:
	void _buildColorMap();
	void _draw_key(float left, float right, float bottom, float top, float lineWidth, bool black = false);
	void _draw_flash(float centerx, float centery, float radius, unsigned char color[3], float alpha);

	static unsigned char s_ColorBank[15][3];

	int m_w, m_h;
	const Visualizer* m_data;

	SubLists<VisNote> m_notes_sublists;
	ColorMap m_InstColorMap;

	SubLists<VisBeat> m_beats_sublists;
	ColorMap m_PercColorMap;
	std::vector<Pos2D> m_beats_centers;

	float m_refTime;
	QTime m_timer;

	// UI settings
	float m_whiteKeyWidth;
	float m_blackKeyWidth;
	float m_whiteKeyHeight;
	float m_blackKeyHeight;
	float m_cornerSize;
	float m_whiteKeyPressedDelta;
	float m_blackKeyPressedDelta;
	float m_pressedLineWidth;

	float m_showTime;
	float m_meteorHalfWidth;

	float m_percussion_flash_size_factor;
	float m_percussion_flash_limit;
};


#endif
