#include <stdio.h>
#include <cmath>

#include "Note.h"
#include "TrackBuffer.h"
#include "parser/Document.h"
#include "parser/CustomParser.h"

#include "WinWavWriter.h"

void Composite(const Document& doc, TrackBuffer& buffer);


void FlyMeToTheMoon_eq()
{
	static const char* names[] =
	{
		"do",
		"re",
		"mi",
		"fa",
		"so",
		"la",
		"ti",
		"+so",
		"-ti"
	};

	static float freqs[] =
	{
		1.0f,
		powf(2.0f, 2.0f / 12.0f),
		powf(2.0f, 4.0f / 12.0f),
		powf(2.0f, 5.0f / 12.0f),
		powf(2.0f, 7.0f / 12.0f),
		powf(2.0f, 9.0f / 12.0f),
		powf(2.0f, 11.0f / 12.0f),
		powf(2.0f, 8.0f / 12.0f),
		powf(2.0f, 10.0f / 12.0f),
	};

	Document doc;
	doc.m_RefFreq = 264.0f *1.25f;
	doc.m_tempo = 120;

	CustomParser parser;
	parser.Customize(9, names, freqs);

	Track track1;
	NoteSequence& seq1 = *track1.m_note_seq;

	Track track2;
	NoteSequence& seq2 = *track2.m_note_seq;

	parser.ParseSeq("do6.72 ti5.24 la5.24 so5.72", seq1);
	parser.ParseSeq("la3.192 BK144 mi4.48 so4.48 do5.48", seq2);

	parser.ParseSeq("fa5.96 BL24 so5.24 la5.24 do6.24", seq1);
	parser.ParseSeq("re3.192 BK144 la3.48 do4.48 fa4.48", seq2);

	parser.ParseSeq("ti5.72 la5.24 so5.24 fa5.72", seq1);
	parser.ParseSeq("so3.192 BK144 re4.48 fa4.48 ti4.48", seq2);

	parser.ParseSeq("mi5.144 BL48", seq1);
	parser.ParseSeq("do3.192 BK144 so3.48 do4.48 mi4.48", seq2);

	parser.ParseSeq("la5.72 so5.24 fa5.24 mi5.72", seq1);
	parser.ParseSeq("fa3.192 BK144 do4.48 mi4.48 la4.48", seq2);

	parser.ParseSeq("re5.72 mi5.24 fa5.24 la5.72", seq1);
	parser.ParseSeq("re3.192 BK144 la3.48 do4.48 fa4.48", seq2);

	parser.ParseSeq("+so5.72 fa5.24 mi5.24 re5.72", seq1);
	parser.ParseSeq("mi3.192 BK144 ti3.48 re4.48 +so4.48", seq2);

	parser.ParseSeq("do5.144 BL48", seq1);
	parser.ParseSeq("la3.192 BK144 mi4.48 so4.48 do5.48", seq2);

	parser.ParseSeq("re5.24 la5.72 la5.96", seq1);
	parser.ParseSeq("re3.192 BK144 la3.48 do4.48 fa4.48", seq2);

	parser.ParseSeq("BL96 do6.24 ti5.72", seq1);
	parser.ParseSeq("so3.192 BK144 re4.48 fa4.48 ti4.48", seq2);

	parser.ParseSeq("so5.144 BL48", seq1);
	parser.ParseSeq("mi3.192 BK144 ti3.48 re4.48 so4.48", seq2);

	parser.ParseSeq("la4.24 fa5.72 fa5.96", seq1);
	parser.ParseSeq("re3.192 BK144 la3.48 do4.48 fa4.48", seq2);

	parser.ParseSeq("BL96 la5.72 so5.24", seq1);
	parser.ParseSeq("so3.192 BK144 re4.48 fa4.48 ti4.48", seq2);

	parser.ParseSeq("fa5.24 mi5.120 BL48", seq1);
	parser.ParseSeq("do3.192 BK144 so3.48 do4.48 mi4.48", seq2);

	parser.ParseSeq("do6.72 ti5.24 la5.24 so5.72", seq1);
	parser.ParseSeq("la3.192 BK144 mi4.48 so4.48 do5.48", seq2);

	parser.ParseSeq("fa5.96 BL24 so5.24 la5.24 do6.24", seq1);
	parser.ParseSeq("re3.192 BK144 la3.48 do4.48 fa4.48", seq2);

	parser.ParseSeq("ti5.72 la5.24 so5.24 fa5.72", seq1);
	parser.ParseSeq("so3.192 BK144 re4.48 fa4.48 ti4.48", seq2);

	parser.ParseSeq("mi5.144 BL48", seq1);
	parser.ParseSeq("do3.192 BK144 so3.48 do4.48 mi4.48", seq2);

	parser.ParseSeq("la5.72 so5.24 fa5.24 mi5.72", seq1);
	parser.ParseSeq("fa3.192 BK144 do4.48 mi4.48 la4.48", seq2);

	parser.ParseSeq("re5.72 mi5.24 fa5.24 la5.72", seq1);
	parser.ParseSeq("re3.192 BK144 la3.48 do4.48 fa4.48", seq2);

	parser.ParseSeq("+so5.72 la5.24 ti5.24 ti5.72", seq1);
	parser.ParseSeq("mi3.192 BK144 ti3.48 re4.48 +so4.48", seq2);

	parser.ParseSeq("do6.24 ti5.24 la5.96 BL48", seq1);
	parser.ParseSeq("la3.192 BK144 mi4.48 so4.48 do5.48", seq2);

	parser.ParseSeq("la5.24 so5.72 la5.24 so5.24 fa5.48", seq1);
	parser.ParseSeq("re3.192 BK144 la3.48 do4.48 fa4.48", seq2);

	parser.ParseSeq("BL96 do6.24 ti5.72", seq1);
	parser.ParseSeq("so3.192 BK144 re4.48 fa4.48 ti4.48", seq2);

	parser.ParseSeq("mi6.144 BL48", seq1);
	parser.ParseSeq("fa3.192 BK144 do4.48 mi4.48 la4.48", seq2);

	parser.ParseSeq("mi6.24 do6.72 do6.96", seq1);
	parser.ParseSeq("re3.192 BK144 la3.48 do4.48 fa4.48", seq2);

	parser.ParseSeq("BL96 ti5.24 re6.72", seq1);
	parser.ParseSeq("so3.192 BK144 re4.48 fa4.48 ti4.48", seq2);

	parser.ParseSeq("do6.192", seq1);
	parser.ParseSeq("do3.192 BK180 so3.180 BK168 do4.168 BK156 mi4.156 BK144 so4.144 BK132 do5.132", seq2);

	doc.m_tracks.push_back(track1);
	doc.m_tracks.push_back(track2);

	TrackBuffer tb;
	Composite(doc, tb);

	WriteToWav(tb, "FlyMeToTheMoon_eq.wav");

}
