cmake_minimum_required (VERSION 3.0)

find_package(PythonLibs 3 REQUIRED)

set(SOURCES
WinPCMPlayer.cpp
Initializer.cpp
)


set(HEADERS 
WinPCMPlayer.h
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

set (DEFINES  ${DEFINES}
-D"_CRT_SECURE_NO_DEPRECATE"  
-D"_SCL_SECURE_NO_DEPRECATE" 
)

include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})
add_library (WinPCMPlayer SHARED ${SOURCES} ${HEADERS})
target_link_libraries(WinPCMPlayer ${LINK_LIBS})

install(TARGETS WinPCMPlayer DESTINATION Extensions)



