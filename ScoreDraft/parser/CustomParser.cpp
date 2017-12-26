#include "parser/CustomParser.h"
#include "Note.h"

void CustomParser::Customize(int count, const char* names[], float freqs[])
{
	m_NoteTable.resize(count);
	for (int i = 0; i < count; i++)
	{
		NoteDef def;
		def.name = names[i];
		def.freq_oct_rel = freqs[i];
		m_NoteTable[i] = def;
	}
}

void CustomParser::ModifyFreq(const char* name, float freq)
{
	for (int i = 0; i < (int) m_NoteTable.size(); i++)
	{
		if (m_NoteTable[i].name == name)
		{
			m_NoteTable[i].freq_oct_rel = freq;
			break;
		}
	}
}

bool CustomParser::ParseNote(const char* strNote, Note& note)
{
	string stringNote = strNote;

	int noteCount = (int)m_NoteTable.size();

	int i;
	for (i = 0; i<noteCount; i++)
	{
		string namei = m_NoteTable[i].name;
		if (stringNote.substr(0, namei.length()) == namei) break;
	}
	if (i >= noteCount)
	{
		if (stringNote.substr(0, 2) == "BL")
		{
			const char* pDur = strNote + 2;
			note.m_freq_rel = -1.0f;
			note.m_duration = atoi(pDur);
			return true;
		}
		else if (stringNote.substr(0, 2) == "BK")
		{
			const char* pDur = strNote + 2;
			note.m_freq_rel = -1.0f;
			note.m_duration = -atoi(pDur);
			return true;
		}
		return false;
	}

	float freq_oct_rel = m_NoteTable[i].freq_oct_rel;

	const char* pOct = strNote + m_NoteTable[i].name.length();

	const char* pEdOct = strchr(pOct, '.');
	if (!pEdOct) return false;

	char strOct[10];

	for (i = 0; pOct != pEdOct; i++, pOct++)
		strOct[i] = *pOct;
	strOct[i] = 0;

	unsigned Oct = atoi(strOct);
	note.m_freq_rel = freq_oct_rel*pow(m_OctBase, ((float)Oct - 5.0f));

	const char* pDur = pEdOct + 1;
	note.m_duration = atoi(pDur);

	return true;
}

bool CustomParser::ParseSeq(const char* strSeq, NoteSequence& seq, char* errMsg)
{
	std::vector<string> subStrings;

	char subStr[100];

	while (sscanf(strSeq, "%s", subStr) != EOF)
	{
		string subString = subStr;
		subStrings.push_back(subString);
		strSeq = strstr(strSeq, subStr) + strlen(subStr);
	}

	unsigned NumOfNotes = (unsigned)subStrings.size();
	size_t originalSize = seq.size();
	seq.resize(NumOfNotes + originalSize);

	unsigned j = 0;

	for (unsigned i = 0; i<subStrings.size(); i++)
	{
		bool result = ParseNote(subStrings[i].data(), seq[j + originalSize]);
		if (!result)
		{
			if (errMsg) sprintf(errMsg, "Note parse failure: %s", subStrings[i].data());
			seq.clear();
			return false;
		}
		j++;
	}
	return true;
}
