#include "DrawText.h"

#include <ft2build.h>
#include FT_FREETYPE_H

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "NotoSansSC-Bold.hpp"

inline bool exists_test(const char* name)
{
	if (FILE *file = fopen(name, "r"))
	{
		fclose(file);
		return true;
	}
	else
	{
		return false;
	}
}


class LibHolder
{
public:
	static LibHolder& singlton()
	{
		thread_local static LibHolder* lh = nullptr;
		if (lh == nullptr)
		{
			lh = new LibHolder;
		}
		return *lh;
	}

	FT_Library get_library() const
	{
		return m_library;
	}

	void add_face(const std::string& name, const std::string& path_ttf)
	{
		FT_New_Face(m_library, path_ttf.c_str(), 0, &m_faces[name]);
	}

	void set_current_face(const std::string& name)
	{
		m_name_current_face = name;
	}

	FT_Face get_face(const std::string& name = "") const
	{
		std::string fnt_name = name;
		if (fnt_name == "") fnt_name = m_name_current_face;
		auto iter = m_faces.find(fnt_name);
		if (iter == m_faces.end())
		{
			return m_faces.begin()->second;
		}
		else
		{
			return iter->second;
		}
	}

private:
	FT_Library  m_library;
	std::unordered_map<std::string, FT_Face>  m_faces;
	std::string m_name_current_face;

	LibHolder()
	{
		FT_Init_FreeType(&m_library);
		FT_New_Memory_Face(m_library, (const FT_Byte*)noto_sans_bold, noto_sans_bold_size, 0, &m_faces["noto_sans_bold"]);
	}

	~LibHolder()
	{
		for (auto iter = m_faces.begin(); iter != m_faces.end(); iter++)
		{
			FT_Done_Face(iter->second);
		}
		FT_Done_FreeType(m_library);
	}
};

#define FT_LIB LibHolder::singlton().get_library()
#define FT_FACE LibHolder::singlton().get_face()

void FTAddFont(const std::string& name, const std::string& path_ttf)
{
	LibHolder::singlton().add_face(name, path_ttf);
}

void FTSetCurrentFont(const std::string& name)
{
	LibHolder::singlton().set_current_face(name);
}

std::wstring UTF8_to_wchar(const char * in)
{
	std::wstring out;
	unsigned int codepoint;
	while (*in != 0)
	{
		unsigned char ch = static_cast<unsigned char>(*in);
		if (ch <= 0x7f)
			codepoint = ch;
		else if (ch <= 0xbf)
			codepoint = (codepoint << 6) | (ch & 0x3f);
		else if (ch <= 0xdf)
			codepoint = ch & 0x1f;
		else if (ch <= 0xef)
			codepoint = ch & 0x0f;
		else
			codepoint = ch & 0x07;
		++in;
		if (((*in & 0xc0) != 0x80) && (codepoint <= 0x10ffff))
		{
			if (sizeof(wchar_t) > 2)
				out.append(1, static_cast<wchar_t>(codepoint));
			else if (codepoint > 0xffff)
			{
				out.append(1, static_cast<wchar_t>(0xd800 + (codepoint >> 10)));
				out.append(1, static_cast<wchar_t>(0xdc00 + (codepoint & 0x03ff)));
			}
			else if (codepoint < 0xd800 || codepoint >= 0xe000)
				out.append(1, static_cast<wchar_t>(codepoint));
		}
	}
	return out;
}


struct Character
{
	int width = 0;
	int height = 0;
	int offset_x = 0;
	int offset_y = 0;
	std::vector<unsigned char> data;
};


Text::Text(const char* text, int size)
{
	std::wstring wtext = UTF8_to_wchar(text);
	size_t total_chars = wtext.size();

	FT_Set_Char_Size(FT_FACE, 0, size * 64, 72, 72);
	FT_GlyphSlot slot = FT_FACE->glyph;
	std::vector<Character> characters(total_chars);

	int min_x = 0x7FFFFFFF;
	int max_x = 0x80000000;
	int min_y = 0x7FFFFFFF;
	int max_y = 0x80000000;

	int off_x = 0;
	int off_y = size;

	for (size_t j = 0; j < total_chars; j++)
	{
		wchar_t c = wtext[j];
		FT_Load_Char(FT_FACE, c, FT_LOAD_DEFAULT);
		FT_Render_Glyph(slot, FT_RENDER_MODE_NORMAL);
		Character& character = characters[j];
		character.offset_x = off_x + slot->bitmap_left;
		character.offset_y = off_y - slot->bitmap_top;
		character.width = (int)slot->bitmap.width;
		character.height = (int)slot->bitmap.rows;
		character.data.resize(character.width*character.height);
		if (character.width > 0 && character.height > 0)
			memcpy(character.data.data(), slot->bitmap.buffer, character.width*character.height);
		if (character.offset_x < min_x) min_x = character.offset_x;
		if (character.offset_y < min_y) min_y = character.offset_y;
		if (character.offset_x + character.width > max_x) max_x = character.offset_x + character.width;
		if (character.offset_y + character.height > max_y) max_y = character.offset_y + character.height;
		off_x += (slot->advance.x >> 6);
		off_y += slot->advance.y >> 6;
	}

	m_width = max_x - min_x;
	m_height = max_y - min_y;
	m_offset_x = min_x;
	m_offset_y = min_y;
	m_data.resize(m_width*m_height, 0);

	for (size_t i = 0; i < total_chars; i++)
	{
		const Character& character = characters[i];
		int off_x = character.offset_x - m_offset_x;
		int off_y = character.offset_y - m_offset_y;

		for (int y = 0; y < character.height; y++)
		{
			for (int x = 0; x < character.width; x++)
			{
				int pos_x = x + off_x;
				int pos_y = y + off_y;
				m_data[pos_x + pos_y * m_width] = character.data[x + y * character.width];
			}
		}
	}	
}

Text::~Text()
{

}

