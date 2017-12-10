#include <stdio.h>

#include "Note.h"
#include "TrackBuffer.h"
#include "parser/Document.h"
#include "parser/CustomParser.h"

#include "WinWavWriter.h"

void Composite(const Document& doc, TrackBuffer& buffer);


void FlyMeToTheMoon_just()
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
		//9.0f / 8.0f,
		10.0f / 9.0f,
		5.0f / 4.0f,
		4.0f / 3.0f,
		3.0f / 2.0f,
		5.0f / 3.0f,
		15.0f / 8.0f,
		25.0f / 16.0f, //#so
		7.0f / 4.0f // bti
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

	parser.ModifyFreq("re", 9.0f / 8.0f);
	parser.ModifyFreq("fa", 21.0f / 16.0f);

	parser.ParseSeq("ti5.72 la5.24 so5.24 fa5.72", seq1);
	parser.ParseSeq("so3.192 BK144 re4.48 fa4.48 ti4.48", seq2);

	parser.ModifyFreq("fa", 4.0f / 3.0f);

	parser.ParseSeq("mi5.144 BL48", seq1);
	parser.ParseSeq("do3.192 BK144 so3.48 do4.48 mi4.48", seq2);

	parser.ModifyFreq("re", 10.0f / 9.0f);

	parser.ParseSeq("la5.72 so5.24 fa5.24 mi5.72", seq1);
	parser.ParseSeq("fa3.192 BK144 do4.48 mi4.48 la4.48", seq2);

	parser.ParseSeq("re5.72 mi5.24 fa5.24 la5.72", seq1);
	parser.ParseSeq("re3.192 BK144 la3.48 do4.48 fa4.48", seq2);

	parser.ModifyFreq("re", 35.0f / 32.0f);
	parser.ParseSeq("+so5.72 fa5.24 mi5.24 re5.72", seq1);
	parser.ParseSeq("mi3.192 BK144 ti3.48 re4.48 +so4.48", seq2);

	parser.ModifyFreq("re", 10.0f / 9.0f);
	parser.ParseSeq("do5.144 BL48", seq1);
	parser.ParseSeq("la3.192 BK144 mi4.48 so4.48 do5.48", seq2);

	parser.ParseSeq("re5.24 la5.72 la5.96", seq1);
	parser.ParseSeq("re3.192 BK144 la3.48 do4.48 fa4.48", seq2);

	parser.ModifyFreq("re", 9.0f / 8.0f);
	parser.ModifyFreq("fa", 21.0f / 16.0f);

	parser.ParseSeq("BL96 do6.24 ti5.72", seq1);
	parser.ParseSeq("so3.192 BK144 re4.48 fa4.48 ti4.48", seq2);

	parser.ModifyFreq("fa", 4.0f / 3.0f);
	parser.ParseSeq("so5.144 BL48", seq1);
	parser.ParseSeq("mi3.192 BK144 ti3.48 re4.48 so4.48", seq2);

	parser.ModifyFreq("re", 10.0f / 9.0f);
	parser.ParseSeq("la4.24 fa5.72 fa5.96", seq1);
	parser.ParseSeq("re3.192 BK144 la3.48 do4.48 fa4.48", seq2);

	parser.ModifyFreq("re", 9.0f / 8.0f);
	parser.ModifyFreq("fa", 21.0f / 16.0f);
	parser.ModifyFreq("la", 27.0f / 16.0f);

	parser.ParseSeq("BL96 la5.72 so5.24", seq1);
	parser.ParseSeq("so3.192 BK144 re4.48 fa4.48 ti4.48", seq2);

	parser.ParseSeq("fa5.24 mi5.120 BL48", seq1);
	parser.ParseSeq("do3.192 BK144 so3.48 do4.48 mi4.48", seq2);

	parser.ModifyFreq("re", 10.0f / 9.0f);
	parser.ModifyFreq("fa", 4.0f / 3.0f);
	parser.ModifyFreq("la", 5.0f / 3.0f);

	parser.ParseSeq("do6.72 ti5.24 la5.24 so5.72", seq1);
	parser.ParseSeq("la3.192 BK144 mi4.48 so4.48 do5.48", seq2);

	parser.ParseSeq("fa5.96 BL24 so5.24 la5.24 do6.24", seq1);
	parser.ParseSeq("re3.192 BK144 la3.48 do4.48 fa4.48", seq2);

	parser.ModifyFreq("re", 9.0f / 8.0f);
	parser.ModifyFreq("fa", 21.0f / 16.0f);

	parser.ParseSeq("ti5.72 la5.24 so5.24 fa5.72", seq1);
	parser.ParseSeq("so3.192 BK144 re4.48 fa4.48 ti4.48", seq2);

	parser.ModifyFreq("fa", 4.0f / 3.0f);

	parser.ParseSeq("mi5.144 BL48", seq1);
	parser.ParseSeq("do3.192 BK144 so3.48 do4.48 mi4.48", seq2);

	parser.ModifyFreq("re", 10.0f / 9.0f);

	parser.ParseSeq("la5.72 so5.24 fa5.24 mi5.72", seq1);
	parser.ParseSeq("fa3.192 BK144 do4.48 mi4.48 la4.48", seq2);

	parser.ParseSeq("re5.72 mi5.24 fa5.24 la5.72", seq1);
	parser.ParseSeq("re3.192 BK144 la3.48 do4.48 fa4.48", seq2);

	parser.ModifyFreq("re", 35.0f / 32.0f);
	parser.ParseSeq("+so5.72 la5.24 ti5.24 ti5.72", seq1);
	parser.ParseSeq("mi3.192 BK144 ti3.48 re4.48 +so4.48", seq2);

	parser.ModifyFreq("re", 10.0f / 9.0f);
	parser.ParseSeq("do6.24 ti5.24 la5.96 BL48", seq1);
	parser.ParseSeq("la3.192 BK144 mi4.48 so4.48 do5.48", seq2);

	parser.ParseSeq("la5.24 so5.72 la5.24 so5.24 fa5.48", seq1);
	parser.ParseSeq("re3.192 BK144 la3.48 do4.48 fa4.48", seq2);

	parser.ModifyFreq("re", 9.0f / 8.0f);
	parser.ModifyFreq("fa", 21.0f / 16.0f);
	parser.ParseSeq("BL96 do6.24 ti5.72", seq1);
	parser.ParseSeq("so3.192 BK144 re4.48 fa4.48 ti4.48", seq2);

	parser.ModifyFreq("re", 10.0f / 9.0f);
	parser.ModifyFreq("fa", 4.0f / 3.0f);
	parser.ParseSeq("mi6.144 BL48", seq1);
	parser.ParseSeq("fa3.192 BK144 do4.48 mi4.48 la4.48", seq2);

	parser.ParseSeq("mi6.24 do6.72 do6.96", seq1);
	parser.ParseSeq("re3.192 BK144 la3.48 do4.48 fa4.48", seq2);

	parser.ModifyFreq("re", 9.0f / 8.0f);
	parser.ModifyFreq("fa", 21.0f / 16.0f);
	parser.ParseSeq("BL96 ti5.24 re6.72", seq1);
	parser.ParseSeq("so3.192 BK144 re4.48 fa4.48 ti4.48", seq2);

	parser.ModifyFreq("fa", 11.0f / 8.0f);
	parser.ParseSeq("do6.384", seq1);
	parser.ParseSeq("do3.384 BK360 so3.360 BK336 do4.336 BK312 mi4.312 BK288 so4.288 BK264 -ti4.264 BK240 do5.240 BK216 re5.216 BK192 mi5.192 BK168 fa5.168 BK144 so5.144", seq2);


	doc.m_tracks.push_back(track1);
	doc.m_tracks.push_back(track2);

	TrackBuffer tb;
	Composite(doc, tb);

	WriteToWav(tb, "FlyMeToTheMoon_just.wav");

}
