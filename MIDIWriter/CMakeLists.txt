cmake_minimum_required (VERSION 3.0)

project(MIDIWriter)

if (WIN32) 
set (DEFINES  ${DEFINES}
-D"_CRT_SECURE_NO_DEPRECATE"  
-D"_SCL_SECURE_NO_DEPRECATE" 
)
else()
add_definitions(-std=c++14)
add_compile_options(-fPIC)
endif()

include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})
add_library (MIDIWriter SHARED MIDIWriter.cpp)

if (WIN32) 
install(TARGETS MIDIWriter RUNTIME DESTINATION ScoreDraft)
else()
install(TARGETS MIDIWriter DESTINATION ScoreDraft)
endif()

