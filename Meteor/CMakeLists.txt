cmake_minimum_required (VERSION 3.0)

project(Meteor)

set(SOURCES
api.cpp
Meteor.cpp
MeteorPlayer.cpp
DrawText.cpp
)

set(HEADERS 
SubListLookUp.h
Meteor.h
MeteorPlayer.h
DrawText.h
)


set (INCLUDE_DIR
../ScoreDraftCore
../thirdparty/portaudio/include
../thirdparty/glfw/include
../thirdparty/fonts
)


if (WIN32) 
set (DEFINES  ${DEFINES}
-D"_CRT_SECURE_NO_DEPRECATE"  
-D"_SCL_SECURE_NO_DEPRECATE" 
)

set (INCLUDE_DIR
${INCLUDE_DIR}
../thirdparty/freetype/include/freetype2
)

set (LIB_DIR
../thirdparty/freetype/lib/
)
else()
add_definitions(-std=c++0x)
add_compile_options(-fPIC)

set (INCLUDE_DIR
${INCLUDE_DIR}
/usr/include/freetype2
)
endif()

include_directories(${INCLUDE_DIR})
link_directories(${LIB_DIR})
add_definitions(${DEFINES})
add_library (Meteor SHARED  ${SOURCES} ${HEADERS})
target_link_libraries(Meteor ScoreDraftCore portaudio_static glfw freetype)

if (WIN32) 
target_link_libraries(Meteor opengl32)
else()
target_link_libraries(Meteor GL)
endif()

if (WIN32) 
target_compile_definitions(Meteor PUBLIC SCOREDRAFTCORE_DLL_IMPORT)
endif()

if (WIN32) 
install(TARGETS Meteor RUNTIME DESTINATION ScoreDraft)
else()
install(TARGETS Meteor DESTINATION ScoreDraft)
endif()

