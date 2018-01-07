cmake_minimum_required (VERSION 3.0)

set(SOURCES
ReadWav.cpp
WriteWav.cpp
)

set(HEADERS 
ReadWav.h
WriteWav.h
)

set (INCLUDE_DIR
.
)

if (WIN32) 
set (DEFINES  ${DEFINES}
-D"_CRT_SECURE_NO_DEPRECATE"  
-D"_SCL_SECURE_NO_DEPRECATE" 
)
else()
add_definitions(-std=c++0x)
add_compile_options(-fPIC)
endif()

include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})
add_library (WavUtil ${SOURCES} ${HEADERS})





