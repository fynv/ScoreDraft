cmake_minimum_required (VERSION 3.0)

find_package(PythonLibs 3 REQUIRED)

set(SOURCES
SF2.cpp
Presets.cpp
Synth.cpp
SF2Synth.cpp
SF2Instrument.cpp
)

set(HEADERS 
SF2.h
Presets.h
Synth.h
SF2Synth.h
)

set (INCLUDE_DIR
${PYTHON_INCLUDE_DIRS}
.
../ScoreDraftCore
../PyScoreDraft
)


set (LINK_LIBS 
${PYTHON_LIBRARIES}
ScoreDraftCore)


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
add_library (SF2Instrument SHARED ${SOURCES} ${HEADERS})
if (WIN32) 
set_target_properties(SF2Instrument PROPERTIES SUFFIX ".dll")
else()
set_target_properties(SF2Instrument PROPERTIES SUFFIX ".so")
endif()
target_link_libraries(SF2Instrument ${LINK_LIBS})

install(TARGETS SF2Instrument DESTINATION ./ScoreDraft/Extensions)
