#pragma once

#include <vector>

class Text
{
public:
	Text(const char* text, int size = 32);
	~Text();

	int m_width = 0;
	int m_height = 0;
	int m_offset_x = 0;
	int m_offset_y = 0;
	std::vector<unsigned char> m_data;
};
