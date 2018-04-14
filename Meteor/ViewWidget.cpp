#include "Meteor.h"
#include "ViewWidget.h"
#include <QPainter>
#include <set>
#include <cmath>
#include <float.h>


#define PI 3.1415926535897932384626433832795

inline float rand01()
{
	float f = (float)rand() / (float)RAND_MAX;
	if (f < 0.0000001f) f = 0.0000001f;
	if (f > 0.9999999f) f = 0.9999999f;
	return f;
}


ViewWidget::ViewWidget(QWidget* parent)
	: QOpenGLWidget(parent),
	m_data(nullptr),
	m_refTime(0.0f)
{
	m_whiteKeyWidth = 18.0f;
	m_blackKeyWidth = 14.0f;
	m_whiteKeyHeight = 80.0f;
	m_blackKeyHeight = 50.0f;
	m_cornerSize = 3.0f;
	m_whiteKeyPressedDelta = 3.0f;
	m_blackKeyPressedDelta = 2.0f;
	m_pressedLineWidth = 3.0f;

	m_showTime = 1.0f;
	m_meteorHalfWidth = 5.0f;

	m_percussion_flash_size_factor = 0.15f;
	m_percussion_flash_limit = 0.3f;

	m_singing_half_width = 8.0f;

}

ViewWidget::~ViewWidget()
{

}

void ViewWidget::SetData(const Visualizer* data)
{
	m_data = data;
	if (data != nullptr)
	{
		_buildColorMap();
	}
}


unsigned char ViewWidget::s_ColorBank[15][3] =
{
	{ 0x41, 0x8C, 0xF0 },
	{ 0xFC, 0xB4, 0x41 },
	{ 0xDF, 0x3A, 0x02 },
	{ 0x05, 0x64, 0x92 },
	{ 0xBF, 0xBF, 0xBF },
	{ 0x1A, 0x3B, 0x69 },
	{ 0xFF, 0xE3, 0x82 },
	{ 0x12, 0x9C, 0xDD },
	{ 0xCA, 0x6B, 0x4B },
	{ 0x00, 0x5C, 0xDB },
	{ 0xF3, 0xD2, 0x88 },
	{ 0x50, 0x63, 0x81 },
	{ 0xF1, 0xB9, 0xA8 },
	{ 0xE0, 0x83, 0x0A },
	{ 0x78, 0x93, 0xBE }
};


void ViewWidget::_buildColorMap()
{
	if (m_data == nullptr) return;

	unsigned bankRef = 0;
	const std::vector<VisBeat>&  beats = m_data->GetBeats();
	m_beats_centers.clear();
	for (unsigned i = 0; i < (unsigned)beats.size(); i++)
	{
		unsigned perc = beats[i].percId;
		if (m_PercColorMap.find(perc) == m_PercColorMap.end())
		{
			m_PercColorMap[perc] = s_ColorBank[bankRef];
			bankRef++;
			if (bankRef >= 15) bankRef = 0;
		}

		float x = rand01();
		float y = rand01();

		m_beats_centers.push_back({ x, y });
	}

	const std::vector<VisSinging>&  singings = m_data->GetSingings();
	for (unsigned i = 0; i < (unsigned)singings.size(); i++)
	{
		unsigned singer = singings[i].singerId;
		if (m_SingerColorMap.find(singer) == m_SingerColorMap.end())
		{
			m_SingerColorMap[singer] = s_ColorBank[bankRef];
			bankRef++;
			if (bankRef >= 15) bankRef = 0;
		}
	}

	const std::vector<VisNote>&  notes = m_data->GetNotes();
	for (unsigned i = 0; i < (unsigned)notes.size(); i++)
	{
		unsigned inst = notes[i].instrumentId;
		if (m_InstColorMap.find(inst) == m_InstColorMap.end())
		{
			m_InstColorMap[inst] = s_ColorBank[bankRef];
			bankRef++;
			if (bankRef >= 15) bankRef = 0;
		}
	}
}

void ViewWidget::SetRefTime(float refTime, const QTime& timer)
{
	m_refTime = refTime;
	m_timer = timer;
}

void ViewWidget::initializeGL()
{
	initializeOpenGLFunctions();
}


void ViewWidget::resizeGL(int w, int h)
{
	glViewport(0, 0, w, h);

	m_w = w;
	m_h = h;
}


void ViewWidget::_draw_key(float left, float right, float bottom, float top, float lineWidth, bool black)
{
	if (black)
		glColor3f(0.0f, 0.0f, 0.0f);
	else
		glColor3f(1.0f, 1.0f, 1.0f);
	glBegin(GL_QUADS);
	// top
	glVertex2f(left + m_cornerSize, top);
	glVertex2f(right - m_cornerSize, top);
	glVertex2f(right, top - m_cornerSize);
	glVertex2f(left, top - m_cornerSize);
	// mid
	glVertex2f(left, top - m_cornerSize);
	glVertex2f(right, top - m_cornerSize);
	glVertex2f(right, bottom + m_cornerSize);
	glVertex2f(left, bottom + m_cornerSize);
	// bottom
	glVertex2f(left, bottom + m_cornerSize);
	glVertex2f(right, bottom + m_cornerSize);
	glVertex2f(right - m_cornerSize, bottom);
	glVertex2f(left + m_cornerSize, bottom);
	glEnd();

	// outline
	if (black)
		glColor3f(1.0f, 1.0f, 1.0f);
	else
		glColor3f(0.0f, 0.0f, 0.0f);
	glLineWidth(lineWidth);
	glBegin(GL_LINE_STRIP);
	glVertex2f(left + m_cornerSize, top);
	glVertex2f(right - m_cornerSize, top);
	glVertex2f(right, top - m_cornerSize);
	glVertex2f(right, bottom + m_cornerSize);
	glVertex2f(right - m_cornerSize, bottom);
	glVertex2f(left + m_cornerSize, bottom);
	glVertex2f(left, bottom + m_cornerSize);
	glVertex2f(left, top - m_cornerSize);
	glVertex2f(left + m_cornerSize, top);
	glEnd();

}

void ViewWidget::_draw_flash(float centerx, float centery, float radius, unsigned char color[3], float alpha)
{
	unsigned div = 36;
	unsigned char uAlpha = (unsigned char)(alpha*255.0f);

	glBegin(GL_TRIANGLES);
	for (unsigned i = 0; i < div; i++)
	{
		float theta1 = (float)i / (float)div * 2.0f*(float)PI;
		float theta2 = (float)(i+1) / (float)div * 2.0f*(float)PI;

		float x1 = centerx + cosf(theta1)*radius;
		float y1 = centery + sinf(theta1)*radius;

		float x2 = centerx + cosf(theta2)*radius;
		float y2 = centery + sinf(theta2)*radius;

		glColor4ub(color[0], color[1], color[2], uAlpha);
		glVertex2f(centerx, centery);
		glColor4ub(color[0], color[1], color[2], 0);
		glVertex2f(x1, y1);
		glVertex2f(x2, y2);
	}

	glEnd();
}

void ViewWidget::paintGL()
{
	if (m_data == nullptr) return;

	QPainter painter;
	painter.begin(this);
	painter.beginNativePainting();

	glEnable(GL_MULTISAMPLE);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, (double)m_w, 0.0, (double)m_h, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glDisable(GL_DEPTH_TEST);

	const SubLists& notes_sublists = m_data->GetNotesSublists();
	const SubLists& beats_sublists = m_data->GetBeatsSublists();
	const SubLists& singing_sublists = m_data->GetSingingSublists();

	float note_inTime = m_refTime + (float)m_timer.elapsed()*0.001f;
	unsigned note_intervalId = notes_sublists.GetIntervalId(note_inTime);
	float note_outTime = note_inTime - m_showTime;
	unsigned note_intervalId_min = notes_sublists.GetIntervalId(note_outTime);

	/// draw meteors
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	//notes
	if (notes_sublists.m_subLists.size()>0)
	{
		std::set<const VisNote*> visiableNotes;

		for (unsigned i = note_intervalId_min; i <= note_intervalId; i++)
		{
			const SubList& subList = notes_sublists.m_subLists[i];
			for (unsigned j = 0; j < (unsigned)subList.size(); j++)
			{
				const VisNote& note = m_data->GetNotes()[subList[j]];
				if (note.start<note_inTime && note.end> note_outTime)
					visiableNotes.insert(&note);
			}
		}

		glBegin(GL_QUADS);

		float keyPos[12] = { 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.5f, 4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f };

		std::set<const VisNote*>::iterator iter;
		for (iter = visiableNotes.begin(); iter != visiableNotes.end(); iter++)
		{
			float startY = ((*iter)->start - note_inTime) / -m_showTime* ((float)m_h - m_whiteKeyHeight) + m_whiteKeyHeight;
			float endY = ((*iter)->end - note_inTime) / -m_showTime* ((float)m_h - m_whiteKeyHeight) + m_whiteKeyHeight;

			unsigned instId = (*iter)->instrumentId;
			unsigned char* color = m_InstColorMap[instId];

			int pitch = (*iter)->pitch;
			int octave = 0;
			while (pitch < 0)
			{
				pitch += 12;
				octave--;
			}
			while (pitch >= 12)
			{
				pitch -= 12;
				octave++;
			}

			float x = (float)m_w*0.5f + ((float)octave*7.0f + keyPos[pitch])*m_whiteKeyWidth;

			glColor4ub(color[0], color[1], color[2], 255);
			glVertex2f(x, startY);
			glVertex2f(x + m_meteorHalfWidth, startY - m_meteorHalfWidth);
			glColor4ub(color[0], color[1], color[2], 0);
			glVertex2f(x, endY);
			glColor4ub(color[0], color[1], color[2], 255);
			glVertex2f(x - m_meteorHalfWidth, startY - m_meteorHalfWidth);

		}

		glEnd();
	}

	// beats
	if (beats_sublists.m_subLists.size()>0)
	{
		unsigned beat_intervalId = beats_sublists.GetIntervalId(note_inTime);

		const SubList& subList = beats_sublists.m_subLists[beat_intervalId];
		for (unsigned i = 0; i < (unsigned)subList.size(); i++)
		{
			unsigned beatIndex = subList[i];
			const VisBeat& beat = m_data->GetBeats()[beatIndex];
			
			float start = beat.start;
			float end = beat.end;

			// limting percussion flash time
			if (end - start > m_percussion_flash_limit)
				end = start + m_percussion_flash_limit;

			if (note_inTime >= start && note_inTime <= end)
			{
				float centerx = m_beats_centers[beatIndex].x*m_w;
				float centery = m_beats_centers[beatIndex].y*(m_h - m_whiteKeyHeight) + m_whiteKeyHeight;
				float radius = m_w*m_percussion_flash_size_factor;

				unsigned char* color = m_PercColorMap[beat.percId];
				float alpha = (end - note_inTime) / (end - start);
				_draw_flash(centerx, centery, radius, color, alpha);
			}
		}
	}


	// singing
	if (singing_sublists.m_subLists.size()>0)
	{
		unsigned singing_intervalId = singing_sublists.GetIntervalId(note_inTime);
		unsigned singing_intervalId_min = singing_sublists.GetIntervalId(note_outTime);

		std::set<const VisSinging*> visiableNotes;

		for (unsigned i = singing_intervalId_min; i <= singing_intervalId; i++)
		{
			const SubList& subList = singing_sublists.m_subLists[i];
			for (unsigned j = 0; j < (unsigned)subList.size(); j++)
			{
				const VisSinging& note = m_data->GetSingings()[subList[j]];
				if (note.start<note_inTime && note.end> note_outTime)
					visiableNotes.insert(&note);
			}
		}

		float pixelPerPitch = m_whiteKeyWidth*7.0f / 12.0f;

		glBegin(GL_QUADS);

		std::set<const VisSinging*>::iterator iter;
		for (iter = visiableNotes.begin(); iter != visiableNotes.end(); iter++)
		{
			float startY = ((*iter)->start - note_inTime) / -m_showTime* ((float)m_h - m_whiteKeyHeight) + m_whiteKeyHeight;
			float endY = ((*iter)->end - note_inTime) / -m_showTime* ((float)m_h - m_whiteKeyHeight) + m_whiteKeyHeight;

			unsigned singerId = (*iter)->singerId;
			unsigned char* color = m_SingerColorMap[singerId];

			const float* pitches = &(*iter)->pitch[0];
			unsigned num_pitches = (unsigned) (*iter)->pitch.size();

			for (unsigned i = 0; i < num_pitches - 1; i++)
			{
				float x1 = pitches[i] * pixelPerPitch + (float)m_w*0.5f + m_whiteKeyWidth*0.5f;
				float x2 = pitches[i + 1] * pixelPerPitch + (float)m_w*0.5f + m_whiteKeyWidth*0.5f;
				
				float k1 = (float)i / (float)(num_pitches - 1);
				float y1 = startY*(1.0f - k1) + endY*k1;

				float k2 = (float) (i+1) / (float)(num_pitches - 1);
				float y2 = startY*(1.0f - k2) + endY*k2;

				glColor4ub(color[0], color[1], color[2], (unsigned char)((1.0f - k1)*255.0f));
				glVertex2f(x1 - m_singing_half_width, y1);
				glVertex2f(x1 + m_singing_half_width, y1);
				glVertex2f(x2 + m_singing_half_width, y2);
				glVertex2f(x2 - m_singing_half_width, y2);

			}

		}
		
		glEnd();
	}

	/// draw keyboard
	glDisable(GL_BLEND);

	static int whitePitchs[7] = { 0, 2, 4, 5, 7, 9, 11 };
	static int blackPitchs[5] = { 1, 3, 6, 8, 10 };
	static int blackPos[5] = { 1, 2, 4, 5, 6 };

	float center = (float)m_w *0.5f;
	float octaveWidth = m_whiteKeyWidth*7.0f;

	int minOctave = -(int)ceilf(center / octaveWidth);
	int maxOctave = (int)floorf(center / octaveWidth);
	int numKeys = (maxOctave - minOctave + 1) * 12;
	int indexShift = -minOctave * 12;

	bool* pressed = new bool[numKeys];
	memset(pressed, 0, sizeof(bool)* numKeys);

	// notes
	if (notes_sublists.m_subLists.size()>0)
	{
		const SubList& subList = notes_sublists.m_subLists[note_intervalId];
		for (unsigned i = 0; i < (unsigned)subList.size(); i++)
		{
			const VisNote& note = m_data->GetNotes()[subList[i]];
			float start = note.start;
			float end = note.end;

			// early key-up movement
			end -= (end - start)*0.1f;

			if (note_inTime >= start && note_inTime <= end)
			{
				int index = note.pitch + indexShift;
				if (index >= 0 && index < numKeys)
				{
					pressed[index] = true;
				}
			}
		}

	}


	for (int i = minOctave; center + (float)i*octaveWidth < m_w; i++)
	{
		float octaveLeft = center + (float)i*octaveWidth;
		for (int j = 0; j < 7; j++)
		{
			int index = whitePitchs[j] + i * 12 + indexShift;
			bool keyPressed = pressed[index];

			float left = octaveLeft + (float)j*m_whiteKeyWidth;
			float right = left + m_whiteKeyWidth;
			float bottom = keyPressed ? m_whiteKeyPressedDelta : 0.0f;
			float top = m_whiteKeyHeight;
			_draw_key(left, right, bottom, top, keyPressed ? m_pressedLineWidth : 1.0f);
		}
		for (int j = 0; j < 5; j++)
		{
			int index = blackPitchs[j] + i * 12 + indexShift;
			bool keyPressed = pressed[index];

			float keyCenter = octaveLeft + (float)blackPos[j] * m_whiteKeyWidth;
			float left = keyCenter - m_blackKeyWidth / 2.0f;
			float right = keyCenter + m_blackKeyWidth / 2.0f;

			float bottom = keyPressed ? m_whiteKeyHeight - m_blackKeyHeight + m_blackKeyPressedDelta : m_whiteKeyHeight - m_blackKeyHeight;
			float top = m_whiteKeyHeight;
			_draw_key(left, right, bottom, top, keyPressed ? m_pressedLineWidth : 1.0f, true);
		}
	}

	delete[] pressed;

	painter.endNativePainting();

	// singing
	if (singing_sublists.m_subLists.size()>0)
	{
		unsigned singing_intervalId = singing_sublists.GetIntervalId(note_inTime);
		unsigned singing_intervalId_min = singing_sublists.GetIntervalId(note_outTime);

		std::set<const VisSinging*> visiableNotes;

		for (unsigned i = singing_intervalId_min; i <= singing_intervalId; i++)
		{
			const SubList& subList = singing_sublists.m_subLists[i];
			for (unsigned j = 0; j < (unsigned)subList.size(); j++)
			{
				const VisSinging& note = m_data->GetSingings()[subList[j]];
				if (note.start<note_inTime && note.start> note_outTime)
					visiableNotes.insert(&note);
			}
		}

		float pixelPerPitch = m_whiteKeyWidth*7.0f / 12.0f;

		QFont font = painter.font();
		font.setPointSize(font.pointSize() * 2);
		painter.setFont(font);

		painter.drawText(QPoint(-100,-100), QString("dummy"));

		std::set<const VisSinging*>::iterator iter;
		for (iter = visiableNotes.begin(); iter != visiableNotes.end(); iter++)
		{
			float startY = ((*iter)->start - note_inTime) / -m_showTime* ((float)m_h - m_whiteKeyHeight) + m_whiteKeyHeight;
		
			unsigned singerId = (*iter)->singerId;
			unsigned char* color = m_SingerColorMap[singerId];

			float x = (*iter)->pitch[0] * pixelPerPitch + (float)m_w*0.5f + m_whiteKeyWidth*0.5f + m_singing_half_width;
			std::string lyric = (*iter)->lyric;

			painter.setPen(QColor(color[0], color[1], color[2]));
			painter.drawText(QPoint((int)x, m_h - 1 - (int)startY), QString::fromUtf8(lyric.data(), (int)lyric.length()));

		}
	}

	painter.end();

	update();
}

	