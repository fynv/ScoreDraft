cmake_minimum_required (VERSION 3.0)

project(ScoreDraftCore)

set(SOURCES
api.cpp
TrackBuffer.cpp
ReadWav.cpp
WriteWav.cpp
)

set(HEADERS 
api.h
utils.h
WavBuffer.h
TrackBuffer.h
ReadWav.h
WriteWav.h
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

add_definitions(${DEFINES})
add_library (ScoreDraftCore SHARED ${SOURCES} ${HEADERS})

if (WIN32) 
target_compile_definitions(ScoreDraftCore PRIVATE SCOREDRAFTCORE_DLL_EXPORT)
endif()

if (WIN32) 
install(TARGETS ScoreDraftCore RUNTIME DESTINATION ScoreDraft)
else()
install(TARGETS ScoreDraftCore DESTINATION ScoreDraft)
endif()

