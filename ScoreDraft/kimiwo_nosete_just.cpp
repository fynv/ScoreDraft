#include <stdio.h>

#include "Note.h"
#include "TrackBuffer.h"
#include "parser/Document.h"
#include "parser/CustomParser.h"

#include "WinWavWriter.h"

void Composite(const Document& doc, TrackBuffer& buffer);

void kimiwo_nosete_just()
{
	static const char* names[] =
	{
		"do",
		"re",
		"+re",
		"mi",
		"fa",
		"+fa",
		"so",
		"+so",
		"la",
		"ti"
	};

	static float freqs[] =
	{
		1.0f,
		10.0f / 9.0f,
		75.0f / 64.0f,
		5.0f / 4.0f,
		4.0f / 3.0f,
		45.0f / 32.0f,
		3.0f / 2.0f,
		25.0f / 16.0f,
		5.0f / 3.0f,
		15.0f / 8.0f
	};

	Document doc;
	doc.m_RefFreq = 2475.0f / 8.0; // bE
	doc.m_tempo = 100;

	CustomParser parser;
	parser.Customize(10, names, freqs);

	NoteSequence Am;
	parser.ParseSeq("la3.192 BK168 mi4.24 la4.24 do5.24 la4.24 do5.24 la4.24 mi4.24", Am);

	NoteSequence Em;
	parser.ParseSeq("mi3.192 BK168 ti3.24 mi4.24 so4.24 mi4.24 so4.24 mi4.24 ti3.24", Em);

	NoteSequence Dm;
	parser.ParseSeq("re3.192 BK168 la3.24 re4.24 fa4.24 re4.24 fa4.24 re4.24 la3.24", Dm);

	parser.ModifyFreq("re", 9.0f / 8.0f);

	NoteSequence C;
	parser.ParseSeq("do3.192 BK168 so3.24 do4.24 mi4.24 do4.24 mi4.24 do4.24 so3.24", C);

	NoteSequence G;
	parser.ParseSeq("so3.192 BK168 re4.24 so4.24 ti4.24 so4.24 ti4.24 so4.24 re4.24", G);

	NoteSequence F;
	parser.ParseSeq("fa3.192 BK168 do4.24 fa4.24 la4.24 fa4.24 la4.24 fa4.24 do4.24", F);

	NoteSequence E;
	parser.ParseSeq("mi3.192 BK168 ti3.24 mi4.24 +so4.24 mi4.24 +so4.24 mi4.24 ti3.24", E);

	NoteSequence B;
	parser.ParseSeq("ti3.192 BK168 +fa4.24 ti4.24 +re5.24 ti4.24 +re5.24 ti4.24 +fa4.24", B);

	Track track1;
	NoteSequence& seq1 = *track1.m_note_seq;

	NoteSequence main;

	parser.ModifyFreq("re", 10.0f / 9.0f);
	parser.ParseSeq("la5.24 ti5.24 do6.72 ti5.24 do6.48 mi6.48 ti5.96 BL48", main);
	parser.ParseSeq("mi5.48        la5.72 so5.24 la5.48 do6.48 so5.96 BL48", main);
	parser.ParseSeq("mi5.48        fa5.72 mi5.24 fa5.48 do6.48 mi5.96 BL48", main);
	parser.ParseSeq("do6.48        ti5.72 +fa5.24 +fa5.48 ti5.48 ti5.96 BL48", main);
	parser.ParseSeq("la5.24 ti5.24 do6.72 ti5.24 do6.48 mi6.48 ti5.96 BL48", main);
	parser.ParseSeq("so5.48        la5.72 so5.24 la5.48 do6.48 so5.96 BL48", main);
	parser.ParseSeq("mi5.48        fa5.72 do6.24 ti5.48 do6.48 re6.48 mi6.24 do6.72", main);
	parser.ParseSeq("BL48          do6.24 ti5.24 la5.48 ti5.48 +so5.48 la5.144", main);

	seq1 = main;

	parser.ModifyFreq("re", 9.0f / 8.0f);
	parser.ParseSeq("do6.24 re6.24 mi6.72 re6.24 mi6.48 so6.48 re6.96  BL48", seq1);
	parser.ParseSeq("so5.48        do6.72 ti5.24 do6.48 mi6.48 mi6.144 BL48", seq1);
	parser.ParseSeq("la5.24 ti5.24 do6.48 ti5.24 do6.24 re6.24 re6.24  do6.72 so5.24 so5.96", seq1);
	parser.ModifyFreq("re", 10.0f / 9.0f);
	parser.ParseSeq("fa6.48 mi6.48 re6.48 do6.48                       mi6.144", seq1);

	parser.ParseSeq("mi6.48        la6.96 so6.96   mi6.24 re6.24 do6.96", seq1);
	parser.ModifyFreq("re", 9.0f / 8.0f);
	parser.ParseSeq("BL48          re6.72 do6.24   re6.48 so6.48 mi6.144", seq1);
	parser.ParseSeq("mi6.48        la6.96 so6.96   mi6.24 re6.24 do6.96", seq1);
	parser.ParseSeq("BL48          re6.72 do6.24   re6.48 ti5.48 la5.144", seq1);

	seq1 += main;

	Track track2;
	track2.m_vol = 0.6f;
	NoteSequence& seq2 = *track2.m_note_seq;
	parser.ParseSeq("BL48", seq2);

	NoteSequence accompany_main;

	accompany_main += Am + Em;
	accompany_main += F + C;
	accompany_main += Dm + C;
	accompany_main += B + E;
	accompany_main += Am + G;
	accompany_main += F + C;
	accompany_main += Dm + Am;
	accompany_main += E + Am;

	seq2 += accompany_main;

	seq2 += C + G;
	seq2 += Am + Em;
	seq2 += F + C;
	seq2 += Dm + E;

	seq2 += Am + F;
	seq2 += G + C;
	seq2 += Am + F;
	seq2 += G + Am;

	seq2 += accompany_main;

	doc.m_tracks.push_back(track1);
	doc.m_tracks.push_back(track2);

	TrackBuffer tb;
	Composite(doc, tb);

	/*WinPCMPlayer player;
	player.PlayTrack(tb);

	system("pause");*/

	WriteToWav(tb, "kimiwo nosete2_just.wav");
}