#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>

inline void base64_encode(const std::vector<uint8_t>& input, std::string& output)
{
	static const char kEncodeLookup[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	static const char kPadCharacter = '=';

	output = "";
	output.reserve(((input.size() / 3) + (input.size() % 3 > 0)) * 4);

	std::uint32_t temp{};
	auto it = input.begin();

	for(std::size_t i = 0; i < input.size() / 3; ++i)
	{
		temp  = (*it++) << 16;
		temp += (*it++) << 8;
		temp += (*it++);
		output.append(1, kEncodeLookup[(temp & 0x00FC0000) >> 18]);
		output.append(1, kEncodeLookup[(temp & 0x0003F000) >> 12]);
		output.append(1, kEncodeLookup[(temp & 0x00000FC0) >> 6 ]);
		output.append(1, kEncodeLookup[(temp & 0x0000003F)      ]);
	}

	switch(input.size() % 3)
	{
	case 1:
		temp = (*it++) << 16;
		output.append(1, kEncodeLookup[(temp & 0x00FC0000) >> 18]);
		output.append(1, kEncodeLookup[(temp & 0x0003F000) >> 12]);
		output.append(2, kPadCharacter);
		break;
	case 2:
		temp  = (*it++) << 16;
		temp += (*it++) << 8;
		output.append(1, kEncodeLookup[(temp & 0x00FC0000) >> 18]);
		output.append(1, kEncodeLookup[(temp & 0x0003F000) >> 12]);
		output.append(1, kEncodeLookup[(temp & 0x00000FC0) >> 6 ]);
		output.append(1, kPadCharacter);
		break;
	}
}

