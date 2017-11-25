#include "Note.h"

Note::Note()
{
	m_freq_rel = -1.0f; // silence
	m_duration=48; // 1 beat
}

Note::~Note()
{
}
