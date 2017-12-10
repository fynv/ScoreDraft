#include <stdio.h>

#include "Note.h"
#include "TrackBuffer.h"
#include "parser/Document.h"
#include "parser/CustomParser.h"

#include "WinWavWriter.h"

void Composite(const Document& doc, TrackBuffer& buffer);

void AirBird_just()
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
		"+so"
	};

	static float freqsC[] =
	{
		1.0f,
		9.0f / 8.0f,
		5.0f / 4.0f,
		4.0f / 3.0f,
		3.0f / 2.0f,
		5.0f / 3.0f,
		15.0f / 8.0f
	};

	static float fD = freqsC[1];

	static float freqsD[] =
	{
		fD* freqsC[0],
		fD* freqsC[1],
		fD* freqsC[2],
		fD* freqsC[3],
		fD* freqsC[4],
		fD* freqsC[5],
		fD* freqsC[6],
	};

	static float fB = freqsD[5] * 0.5f;

	static float freqsB[] =
	{
		fB* freqsC[0],
		fB* freqsC[1],
		fB* freqsC[2],
		fB* freqsC[3],
		fB* freqsC[4],
		fB* freqsC[5],
		fB* freqsC[6],
	};

	static float fCS = freqsB[6] * 0.6f;

	static float freqsCS[] =
	{
		fCS,
		fCS* 10.0f / 9.0f,
		fCS* 5.0f / 4.0f,
		fCS* 4.0f / 3.0f,
		fCS* 40.0f / 27.0f,
		freqsB[6],
		fCS* 50.0f / 27.0f,
	};

	static float fE = freqsCS[4] * 0.8f;

	static float freqsE[] =
	{
		fE* freqsC[0],
		fE* freqsC[1],
		fE* freqsC[2],
		fE* freqsC[3],
		fE* freqsC[4],
		fE* freqsC[5],
		fE* freqsC[6],
		fE* freqsC[2] * 1.25f
	};

	Document doc;
	doc.m_RefFreq = 264.0f;
	doc.m_tempo = 100;

	Track track1;
	NoteSequence& seq1 = *track1.m_note_seq;

	Track track2;
	NoteSequence& seq2 = *track2.m_note_seq;

	CustomParser parser;
	// #2
	parser.Customize(7, names, freqsD);
	parser.ParseSeq("la5.48 ti5.24 do6.24 so6.24 fa5.192 BK192 la5.192 BK192 do6.192 BK192 mi6.48 mi6.24 re6.12 mi6.156", seq1);
	parser.ParseSeq("BL120 fa3.192 BK168 do4.96 BK72 fa4.96 BK72 la4.96 BK48 do4.48 BK24 fa4.48 BK24 la4.24", seq2);

	// #4
	parser.ParseSeq("re6.24 mi6.24 so6.24 do6.24 ti5.24 do6.24", seq1);
	parser.ParseSeq("so3.192 BK168 re4.96 BK72 so4.96 BK72 ti4.96 BK48 re4.48 BK24 so4.48", seq2);

	parser.ParseSeq("do5.48 BK48 mi5.48 BK48 ti5.48 ti5.24 la5.24 mi5.168", seq1);
	parser.ParseSeq("la3.192 BK168 do4.96 BK72 mi4.96 BK72 la4.96 BK48 do4.48 BK24 mi4.48 BK24 la4.24", seq2);
	parser.ParseSeq("la3.192 BK168 mi4.96 BK72 la4.24 do5.24 ti4.24 la4.48 BK24 mi4.48", seq2);

	// #7
	parser.ParseSeq("la5.48 ti5.24 do6.24 so6.24 fa5.192 BK192 la5.192 BK192 mi6.48 mi6.24 re6.12 mi6.156", seq1);
	parser.ParseSeq("fa3.192 BK168 do4.96 BK72 fa4.96 BK72 la4.96 BK48 do4.48 BK24 fa4.48 BK24 la4.24", seq2);

	parser.ParseSeq("re6.24 mi6.24 so6.24 mi6.24 so6.24 do7.24", seq1);
	parser.ParseSeq("so3.192 BK168 re4.96 BK72 so4.96 BK72 ti4.96 BK48 re4.48 BK24 so4.48", seq2);

	parser.ParseSeq("ti6.32 la6.32 mi6.104 BK168 mi6.32 mi6.32 do6.104 re6.24", seq1);
	parser.ParseSeq("la3.192 BK168 mi4.72 BK48 la4.72 BK48 do5.48 mi4.72 BK48 la4.48 BK24 do5.24", seq2);

	parser.ParseSeq("mi6.48 so6.48 la6.48 ti6.48 BK192 do6.48 do6.48 mi6.48 so6.48 BK48 mi6.48", seq1);
	parser.ParseSeq("la3.144 BK120 mi4.144 BK120 la4.120 BK96 ti4.24 do5.24 mi5.24 la5.24 mi3.24", seq2);

	parser.ParseSeq("fa5.192 BK192 la5.192 BK192 do6.192 BK192 mi6.48 mi6.24 re6.12 mi6.156", seq1);
	parser.ParseSeq("fa3.192 BK168 do4.96 BK72 fa4.96 BK72 la4.96 BK48 do4.48 BK24 fa4.48 BK24 la4.24", seq2);

	// #12
	parser.ParseSeq("re6.24 mi6.24 so6.24 do6.24 ti5.24 do6.24", seq1);
	parser.ParseSeq("so3.192 BK168 re4.96 BK72 so4.96 BK72 ti4.96 BK48 re4.48 BK24 so4.48", seq2);

	parser.ParseSeq("do5.240 BK240 mi5.72 BK72 ti5.48 ti5.24 la5.24 mi5.144 BL24", seq1);
	parser.ParseSeq("la3.192 BK168 do4.72 BK48 mi4.72 BK48 la4.72 BK24 do4.72 BK48 mi4.48 BK24 la4.24", seq2);
	parser.ParseSeq("la3.192 BK168 mi4.96 BK72 la4.24 do5.24 ti4.24 la4.48 BK24 mi4.48", seq2);

	// #15
	parser.ParseSeq("la5.48 ti5.24 do6.24 so6.24 fa5.192 BK192 la5.192 BK192 mi6.48 mi6.24 re6.12 mi6.156", seq1);
	parser.ParseSeq("fa3.192 BK168 do4.72 BK48 fa4.72 BK48 la4.72 BK24 do4.48 BK24 fa4.48 BK24 la4.24", seq2);

	parser.ParseSeq("re6.24 mi6.24 so6.24 mi6.24 so6.24 do7.24", seq1);
	parser.ParseSeq("so3.192 BK168 re4.72 BK48 so4.72 BK48 ti4.72 BK24 re4.48 BK24 so4.48", seq2);

	parser.ParseSeq("mi6.192 BK192 so6.192 BK192 ti6.192", seq1);
	parser.ParseSeq("la3.192 BK168 mi4.72 BK48 la4.72 BK48 do5.48 mi4.72 BK48 la4.48 BK24 do5.24", seq2);

	parser.ParseSeq("do6.144 BK144 mi6.144 BK144 la6.144 do6.48 BK48 mi6.48 BK48 la6.48", seq1);
	parser.ParseSeq("la3.192 BK168 mi4.168 BK144 la4.144 BK120 ti4.24 do5.24 mi5.24 la5.24 ti5.24", seq2);

	parser.ParseSeq("re6.168 BK168 so6.168 BK168 ti6.168 la6.24", seq1);
	parser.ParseSeq("la3.192 BK168 mi4.72 BK48 so4.72 BK48 re5.48 mi4.72 BK48 so4.48 BK24 re5.24", seq2);

	parser.Customize(7, names, freqsB);

	parser.ParseSeq("mi6.168 BK168 so6.168 BK168 do6.168 BL24", seq1);
	parser.ParseSeq("do3.96 BK96 do4.96 BK72 so4.72 BK48 re5.48 BK24 fa5.24 do4.96 BK96 do5.96 BK96 mi5.96 BK96 so5.96", seq2);

	// #21
	parser.ParseSeq("BL72 la6.24 la6.32 la6.32 la6.32", seq1);
	parser.ParseSeq("fa4.192 BK168 do5.72 BK48 fa5.72 BK48 la5.72 BK24 do5.72 BK48 fa5.48 BK24 la5.24", seq2);

	parser.ParseSeq("la6.48 so6.24 so6.48 la6.48 ti6.24", seq1);
	parser.ParseSeq("so4.192 BK168 re5.72 BK48 so5.72 BK48 ti5.72 BK24 re5.48 BK24 so5.48", seq2);

	parser.ParseSeq("do6.120 BK120 mi6.96 BK96 so6.72 BK72 do7.48 ti6.24 la6.48 mi6.24 mi5.144 BK144 la5.144 BK144 mi6.144", seq1);
	parser.ParseSeq("la4.192 BK168 mi5.72 BK48 la5.72 BK48 do.72 BK24 mi5.72 BK48 la5.48 BK24 do6.24", seq2);

	parser.ParseSeq("ti5.48 BK48 re6.48 do6.96", seq1);
	parser.ParseSeq("la3.72 BK48 la4.48 BK24 mi5.48 BK24 la5.24 so3.72 BK48 so4.48 BK24 re5.48 BK24 so5.24", seq2);

	// #25
	parser.ParseSeq("BL24 la6.24 la6.32 la6.32 la6.32", seq1);
	parser.ParseSeq("fa4.192 BK168 do5.72 BK48 fa5.72 BK48 la5.72 BK24 do5.72 BK48 fa5.48 BK24 la5.24", seq2);

	parser.ParseSeq("la6.48 so6.24 so6.48 mi6.48 so6.24", seq1);
	parser.ParseSeq("so4.192 BK168 re5.72 BK48 so5.72 BK48 ti5.72 BK24 re5.48 BK24 so5.48", seq2);

	parser.ParseSeq("do6.48 BK48 mi6.48 BK48 la6.48 ti6.24 mi6.312 BK312 la6.312 BK312 do7.312", seq1);
	parser.ParseSeq("la4.192 BK168 mi5.72 BK48 la5.72 BK48 do.72 BK24 mi5.72 BK48 la5.48 BK24 do6.24", seq2);
	parser.ParseSeq("la3.144 BK120 mi4.168 BK144 la4.96 BK72 mi5.120 BK96 la5.96 BK72 do5.72 BK48 la4.48 BK24 so3.24", seq2);

	// #29
	parser.ParseSeq("BL72 la6.24 la6.32 la6.32 la6.32", seq1);
	parser.ParseSeq("fa3.144 BK120 do5.72 BK48 fa5.72 BK48 la5.72 BK24 do5.72 BK48 fa5.48 BK24 la5.24", seq2);

	parser.ParseSeq("la6.48 so6.24 so6.48 la6.48 ti6.24", seq1);
	parser.ParseSeq("so3.144 BK120 re5.72 BK48 so5.72 BK48 ti5.72 BK24 re5.48 BK24 so5.48 BK24 ti5.24", seq2);

	parser.ParseSeq("do6.120 BK120 mi6.96 BK96 so6.72 BK72 do7.48 ti6.24 la6.48 mi6.24 mi5.144 BK144 la5.144 BK144 mi6.144", seq1);
	parser.ParseSeq("la3.144 BK120 mi5.72 BK48 la5.72 BK48 do.72 BK24 mi5.72 BK48 la5.48 BK24 do6.24", seq2);

	parser.ParseSeq("ti5.48 BK48 re6.48 do6.96", seq1);
	parser.ParseSeq("so3.96 BK72 la4.48 BK24 mi5.48 BK24 la5.24 fa3.72 BK48 so4.48 BK24 re5.48 BK24 so5.24", seq2);

	// #33
	parser.ParseSeq("BL24 mi6.24 mi6.32 mi6.32 mi6.32", seq1);
	parser.ParseSeq("fa3.144 BK120 do5.72 BK48 fa5.72 BK48 la5.72 BK24 do5.48 BK24 fa5.48 BK24 la5.24", seq2);

	parser.ParseSeq("re6.48 do6.24 so6.48 mi6.48 so6.24", seq1);
	parser.ParseSeq("so3.144 BK120 re5.72 BK48 so5.72 BK48 ti5.72 BK24 re5.48 BK24 so5.24 ti5.24", seq2);

	parser.ParseSeq("do6.96 BK96 mi6.96 BK96 la6.96 mi6.96 BK96 la6.48 BK48 do7.48 ti6.24 la6.24", seq1);
	parser.ParseSeq("la3.192 BK168 mi4.168 BK144 la4.144 BK120 mi5.72 BK48 la5.48 BK24 do5.24 mi5.48", seq2);

	parser.Customize(7, names, freqsCS);
	parser.ParseSeq("ti5.144 BK144 re6.144 BK144 la6.24 so6.120 fa6.48", seq1);
	parser.ParseSeq("so3.192 BK192 so4.192 BK168 ti4.168 BK144 re5.144 BK120 so5.24 re6.24 so5.24 ti5.24 fa5.24", seq2);

	// #37
	parser.ParseSeq("mi6.24 mi6.24 mi6.24 re6.12 mi6.144 BL12", seq1);
	parser.ParseSeq("fa3.192 BK168 fa4.168 BK144 la4.48 BK24 do5.24 mi5.24 la4.24 do5.24 la4.24", seq2);

	parser.ParseSeq("mi6.48 fa6.48 so6.48", seq1);
	parser.ParseSeq("fa3.192 BK168 fa4.168 BK144 la4.48 BK24 do5.24 mi5.24 la4.24 do5.24 la4.24", seq2);

	parser.ModifyFreq("mi", fCS*(100.0f / 81.0f));

	parser.ParseSeq("re6.72 ti5.24 ti5.144", seq1);
	parser.ParseSeq("mi3.192 BK168 mi4.168 BK144 so4.48 BK24 ti4.24 re5.24 so4.24 ti4.24 so4.24", seq2);

	parser.ParseSeq("BL24 ti5.24 do6.24 re6.72", seq1);
	parser.ParseSeq("mi3.192 BK168 mi4.168 BK144 so4.48 BK24 ti4.24 re5.24 so4.24 ti4.24 so4.24", seq2);

	// #41
	parser.ModifyFreq("mi", fCS*(5.0f / 4.0f));

	parser.ParseSeq("mi6.24 mi6.24 mi6.24 la5.12 mi6.240 BL12", seq1);
	parser.ParseSeq("re3.192 BK168 re4.168 BK144 la4.48 BK24 do5.24 mi5.24 la4.24 do5.24 la4.24", seq2);

	parser.ParseSeq("re6.24 do6.24", seq1);
	parser.ParseSeq("re3.192 BK168 re4.168 BK144 la4.48 BK24 do5.24 mi5.24 la4.24 do5.24 la4.24", seq2);

	parser.ModifyFreq("mi", fCS*(100.0f / 81.0f));

	parser.ParseSeq("re6.48 re6.24 do6.24 re6.24 so6.48 so6.168", seq1);
	parser.ParseSeq("mi3.192 BK168 mi4.168 BK144 so4.48 BK24 ti4.24 re5.24 so4.24 ti4.24 so4.24", seq2);

	parser.ModifyFreq("mi", fCS*(5.0f / 4.0f));

	parser.ParseSeq("BK96 so4.12 la4.12 do5.12 re5.12 do5.12 re5.12 so5.12 la5.12 so5.12 la5.12 do6.12 re6.12", seq1);
	parser.ParseSeq("la3.12 BK12 mi4.12 BL36 so3.12 BK12 do4.12 BL36 fa3.12	BK12 re4.12 BL36 mi3.48 BK48 mi4.48", seq2);

	// #45
	parser.ParseSeq("fa5.96 BK96 la5.96 BK96 mi6.24 mi6.24 mi6.24 re6.12 mi6.144 BL12", seq1);
	parser.ParseSeq("fa3.192 BK168 fa4.168 BK144 la4.48 BK24 do5.24 mi5.24 la4.24 do5.24 la4.24", seq2);

	parser.ParseSeq("mi6.48 BK24 la5.24 fa6.48 BK24 la5.24 so6.48 BK24 la5.24", seq1);
	parser.ParseSeq("fa3.192 BK168 fa4.168 BK144 la4.48 BK24 do5.24 mi5.24 la4.24 do5.24 la4.24", seq2);

	parser.ModifyFreq("mi", fCS*(100.0f / 81.0f));

	parser.ParseSeq("mi5.72 BK72 so5.72 BK72 re6.72 so5.24 BK24 ti5.24 mi5.96 BK96 so5.96 BK96 ti5.96 BK24 so6.24", seq1);
	parser.ParseSeq("mi3.192 BK168 mi4.168 BK144 so4.48 BK24 ti4.24 re5.24 so4.24 ti4.24 so4.24", seq2);

	parser.ParseSeq("mi6.24 BK24 so6.24 BK24 re7.24 BL12 ti5.12 BK12 mi6.12 BK12 so6.12 BK12 ti6.12 BL24 ti5.72 BK72 mi6.72 BK72 so6.72 BK72 ti6.72 BK12 so5.24 la5.12 do6.12 re6.12", seq1);
	parser.ParseSeq("mi3.192 BK168 mi4.168 BK144 so4.48 BK24 ti4.24 re5.24 so4.24 ti4.24 so4.24", seq2);

	// #49
	parser.Customize(8, names, freqsE);
	parser.ParseSeq("la5.24 BK24 mi6.24 la5.24 BK24 mi6.24 la5.24 BK24 mi6.24 re6.12 la5.108 BK108 mi6.108", seq1);
	parser.ParseSeq("fa3.192 BK168 fa4.168 BK144 la4.48 BK24 do5.24 fa5.24 la4.24 do5.24 la4.24", seq2);

	parser.ParseSeq("mi6.24 BK24 mi7.24 mi6.24 BK24 mi7.24 mi6.24 BK24 mi7.24 re6.12 BK12 re7.12 mi6.60 BK60 mi7.60 re6.24 do6.24", seq1);
	parser.ParseSeq("fa3.192 BK168 fa4.168 BK144 la4.48 BK24 do5.24 fa5.24 la4.24 do5.24 la4.24", seq2);

	parser.ParseSeq("ti5.48 BK48 re6.48 ti5.96 BK96 re6.24 do6.24 re6.24 mi6.36 BL12 +so5.96 BK96 ti5.96 BK96 mi6.96", seq1);
	parser.ParseSeq("so3.192 BK168 so4.168 BK144 ti4.48 BK24 re5.24 so5.48 BK24 ti4.24 re5.24 ti.24", seq2);

	parser.ParseSeq("+so3.192 BK168 +so4.168 BK144 ti5.144 BK120 mi6.96 BL24", seq2);
	parser.Customize(7, names, freqsD);
	parser.ParseSeq("la5.48 ti5.24 do5.24 BK24 do6.24 so5.24 BK24 so6.24", seq1);

	// #53
	parser.ParseSeq("la5.240 BK240 do6.240 BK240 mi6.48 mi6.24 re6.12 mi6.156", seq1);
	parser.ParseSeq("fa3.192 BK168 do4.96 BK72 fa4.96 BK72 la4.96 BK48 do4.48 BK24 fa4.48 BK24 la4.24", seq2);

	parser.ParseSeq("re6.24 mi6.24 so6.24 do6.24 ti5.24 do6.24", seq1);
	parser.ParseSeq("so3.192 BK168 re4.96 BK72 so4.96 BK72 ti4.96 BK48 re4.48 BK24 so4.48", seq2);

	parser.ParseSeq("do5.48 BK48 mi5.48 BK48 ti5.48 ti5.24 la5.24 mi5.168", seq1);
	parser.ParseSeq("la3.192 BK168 do4.96 BK72 mi4.96 BK72 la4.96 BK48 do4.48 BK24 mi4.48 BK24 la4.24", seq2);

	parser.ParseSeq("la4.48 BK48 la5.48 ti4.24 BK24 ti5.24 do5.24 BK24 do6.24 so5.24 BK24 so6.24", seq1);
	parser.ParseSeq("la3.192 BK168 mi4.96 BK72 la4.24 do5.24 ti4.24 la4.48 BK24 mi4.48", seq2);

	// #57
	parser.ParseSeq("fa5.192 BK192 la5.192 BK192 mi6.48 mi6.24 re6.12 mi6.156", seq1);
	parser.ParseSeq("fa3.192 BK168 do4.96 BK72 fa4.96 BK72 la4.96 BK48 do4.48 BK24 fa4.48 BK24 la4.24", seq2);

	parser.ParseSeq("so5.24 BK24 ti5.24 BK24 re6.24 so5.24 BK24 ti5.24 BK24 mi6.24 so5.24 BK24 so6.24 so5.24 BK24 mi6.24 so5.24 BK24 mi6.24 BK24 so6.24 do6.24 BK24 so6.24 BK24 do7.24", seq1);
	parser.ParseSeq("so3.192 BK168 re4.96 BK72 so4.96 BK72 ti4.96 BK48 re4.48 BK24 so4.48", seq2);

	parser.ParseSeq("do6.32 BK32 mi6.32 BK32 ti6.32 do6.32 BK32 mi6.32 BK32 la6.32 la5.104 BK104 do6.104 BK104 mi6.104 re6.24", seq1);
	parser.ParseSeq("la3.192 BK168 mi4.72 BK48 la4.72 BK48 do5.48 mi4.72 BK48 la4.48 BK24 do5.24", seq2);

	parser.ParseSeq("la5.48 BK48 do6.48 BK48 mi6.48 la5.48 BK48 do6.48 BK48 so6.48 la5.48 BK48 do6.48 BK48 mi6.48 BK48 la6.48 la5.48 BK48 do6.48 BK48 so6.48 BK48 ti6.48", seq1);
	parser.ParseSeq("la3.196 BK168 mi4.168 BK144 la4.144 BK120 ti4.24 do5.24 mi5.24 la5.24 mi3.24", seq2);

	// #61
	parser.ParseSeq("fa5.48 BK48 la5.48 BK48 mi6.48 fa5.24 BK24 la5.24 BK24 mi6.24 re6.12 fa5.156 BK156 la5.156 BK156 mi6.156", seq1);
	parser.ParseSeq("fa3.192 BK168 do4.96 BK72 fa4.96 BK72 la4.96 BK48 do4.48 BK24 fa4.48 BK24 la4.24", seq2);

	parser.ParseSeq("so5.24 BK24 re6.24 so5.24 BK24 mi6.24 so5.24 BK24 ti5.24 BK24 so6.24 re5.24 BK24 so5.24 BK24 do6.24 re5.24 BK24 so5.24 BK24 ti5.24 re5.24 BK24 so5.24 BK24 do6.24", seq1);
	parser.ParseSeq("so3.192 BK168 re4.96 BK72 so4.96 BK72 ti4.96 BK48 re4.48 BK24 so4.48", seq2);

	parser.ParseSeq("do5.48 BK48 mi5.48 BK48 ti5.48 do5.24 BK24 mi5.24 BK24 ti5.24 la5.24 do5.144 BK144 mi5.144 BL24", seq1);
	parser.ParseSeq("la3.192 BK168 do4.72 BK48 mi4.72 BK48 la4.72 BK24 do4.72 BK48 mi4.48 BK24 la4.24", seq2);

	parser.ParseSeq("la4.48 BK48 la5.48 ti4.24 BK24 ti5.24 do5.24 BK24 do6.24 so5.24 BK24 so6.24", seq1);
	parser.ParseSeq("la3.192 BK168 mi4.96 BK72 la4.24 do5.24 ti4.24 la4.48 BK24 mi4.48", seq2);

	// #65
	parser.ParseSeq("fa5.48 BK48 la5.48 BK48 mi6.48 fa5.24 BK24 la5.24 BK24 mi6.24 re6.12 fa5.156 BK156 la5.156 BK156 mi6.156", seq1);
	parser.ParseSeq("fa3.192 BK168 do4.96 BK72 fa4.96 BK72 la4.96 BK48 do4.48 BK24 fa4.48 BK24 la4.24", seq2);

	parser.ParseSeq("re5.24 BK24 re6.24 mi5.24 BK24 mi6.24 so5.24 BK24 so6.24 mi5.24 BK24 mi6.24 so5.24 BK24 so6.24 do6.24 BK24 do7.24", seq1);
	parser.ParseSeq("so3.192 BK168 re4.72 BK48 so4.72 BK48 ti4.72 BK24 re4.48 BK24 so4.48", seq2);

	parser.ParseSeq("ti5.168 BK168 re6.168 BK168 so6.168 BK168 ti6.168 BL24", seq1);
	parser.ParseSeq("la3.192 BK168 mi4.72 BK48 la4.72 BK48 do5.48 mi4.72 BK48 la4.48 BK24 do5.24", seq2);

	parser.ParseSeq("la5.108 BK108 do6.108 BK108 mi6.108 BK108 la6.108 mi5.12 la5.12 do6.12 mi6.16 la6.16 do7.16", seq1);
	parser.ParseSeq("la3.192 BK168 mi4.168 BK144 la4.144 BK120 ti4.24 do5.24 mi5.24 la5.24 ti5.24", seq2);

	parser.ParseSeq("ti5.168 BK168 mi6.168 BK168 so6.168 BK168 ti6.168 la6.24", seq1);
	parser.ParseSeq("la3.192 BK168 mi4.72 BK48 la4.72 BK48 re5.48 mi4.72 BK48 la4.48 BK24 re5.24", seq2);

	parser.Customize(7, names, freqsB);

	parser.ParseSeq("do6.120 BK120 mi6.24 BK24 so6.24 BK24 do7.48 BK24 so6.24 mi6.72 BK48 do7.48", seq1);
	parser.ParseSeq("do3.96 BK96 do4.96 BK72 so4.72 BK48 re5.48 BK24 fa5.24 do4.96 BK96 do5.96 BK96 mi5.96 BK96 so5.96", seq2);

	parser.Customize(7, names, freqsD);
	parser.ParseSeq("ti6.24 do7.24 so7.12 BL12 ", seq1);

	// #71
	parser.ParseSeq("fa6.192 BK192 la6.192 BK192 mi7.48 mi7.24 re7.12 mi7.156", seq1);
	parser.ParseSeq("fa5.120 BK96 do6.96 BK72 fa6.72 la5.72 BK48 la6.48", seq2);

	parser.ParseSeq("re7.24 mi7.24 so7.24 do7.24 ti6.24 do7.24", seq1);
	parser.ParseSeq("so5.120 BK96 re5.96 BK72 so6.72 ti5.24 re6.48", seq2);

	parser.ParseSeq("ti6.48 ti6.24 la6.24 mi6.168", seq1);
	parser.ParseSeq("la5.24 mi6.96 mi5.24 la5.24 do6.24", seq2);

	parser.ParseSeq("la6.48 ti6.24 do7.24 so7.12 BL12", seq1);
	parser.ParseSeq("mi6.24 la5.24 la6.24 BL48 mi6.24 la5.24 mi5.24", seq2);

	// #75
	parser.ParseSeq("fa6.240 BK240 la6.240 BK240 mi7.48 mi7.24 re7.12 mi7.156", seq1);
	parser.ParseSeq("fa5.120 BK96 do6.96 BK72 fa6.72 la5.72 BK48 la6.48", seq2);

	parser.ParseSeq("re7.24 mi7.24 so7.24 mi7.24 so7.24 do8.24", seq1);
	parser.ParseSeq("so5.96 BK72 re6.72 BK48 so6.48 BL24 so6.24 do7.24 mi7.24", seq2);

	parser.ParseSeq("ti7.12 la7.12 mi7.12 do7.12 ti6.12 la6.12 mi6.12 do6.12 ti5.12 la5.12 mi5.12 do5.12 ti4.12 la4.12 mi4.12 do4.12", seq1);
	parser.ParseSeq("mi6.24 BK24 do7.24 BL24 mi5.24 BK24 do6.24 BL24 mi4.24 BK24 do5.24 BL24 mi3.24 BK24 do4.24 BL24", seq2);

	parser.ParseSeq("la3.192 BK192 do4.192 BK192 mi4.192 BK192 la4.192", seq1);
	parser.ParseSeq("la2.192", seq2);

	doc.m_tracks.push_back(track1);
	doc.m_tracks.push_back(track2);

	TrackBuffer tb;
	Composite(doc, tb);

	WriteToWav(tb, "AirBird_just.wav");


}

