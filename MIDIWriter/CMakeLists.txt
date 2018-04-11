cmake_minimum_required (VERSION 3.0)

find_package(PythonLibs 3 REQUIRED)

set(SOURCES
MIDIWriter.cpp
)

set (INCLUDE_DIR
${PYTHON_INCLUDE_DIRS}
.
../ScoreDraftCore
../PyScoreDraft
)

set (LINK_LIBS 
${PYTHON_LIBRARIES}
ScoreDraftCore
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
add_library (MIDIWriter SHARED ${SOURCES})
if (WIN32) 
set_target_properties(MIDIWriter PROPERTIES SUFFIX ".dll")
else()
set_target_properties(MIDIWriter PROPERTIES SUFFIX ".so")
endif()
target_link_libraries(MIDIWriter ${LINK_LIBS})

install(TARGETS MIDIWriter DESTINATION Extensions)



