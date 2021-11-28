#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

inline void blob_write(std::vector<uint8_t>& blob, const void* data, size_t size)
{
	blob.resize(blob.size() + size);
	memcpy(blob.data() + blob.size() - size, data, size);
}

