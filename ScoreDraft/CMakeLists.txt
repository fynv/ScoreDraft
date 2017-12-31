cmake_minimum_required (VERSION 3.0)

set(SOURCES
parser/CustomParser.cpp
parser/NoteSeqTable.cpp
parser/Parser.cpp
WinWavWriter.cpp
MIDIWriter.cpp
WinPCMPlayer.cpp
main.cpp
AirBird_eq.cpp
AirBird_just.cpp
FlyMeToTheMoon_eq.cpp
FlyMeToTheMoon_just.cpp
kimiwo_nosete_just.cpp
)

set(HEADERS 
parser/CustomParser.h
parser/Document.h
parser/NoteSeqTable.h
parser/Parser.h
WinWavWriter.h
MIDIWriter.h
WinPCMPlayer.h

)

set (INCLUDE_DIR
.
../ScoreDraftCore
)

set (LINK_LIBS ScoreDraftCore)

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
add_executable (ScoreDraft ${SOURCES} ${HEADERS})
target_link_libraries(ScoreDraft ${LINK_LIBS})





