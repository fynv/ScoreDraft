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
add_library (PySF2Instrument SHARED ${SOURCES} ${HEADERS})
if (WIN32) 
set_target_properties(PySF2Instrument PROPERTIES SUFFIX ".pyd")
else()
set_target_properties(PySF2Instrument PROPERTIES SUFFIX ".so")
set_target_properties(PySF2Instrument PROPERTIES PREFIX "")
endif()
target_link_libraries(PySF2Instrument ${LINK_LIBS})

install(TARGETS PySF2Instrument DESTINATION ./ScoreDraft)
