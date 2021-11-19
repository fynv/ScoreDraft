cmake_minimum_required (VERSION 3.0)

project(PCMPlayer)

set(SOURCES
../DSPUtil/complex.cpp
../DSPUtil/fft.cpp
api.cpp
PCMPlayer.cpp
)

set(HEADERS 
../DSPUtil/complex.h
../DSPUtil/fft.h
PCMPlayer.h
)


set (INCLUDE_DIR
../ScoreDraftCore
../DSPUtil
../thirdparty/portaudio/include
../thirdparty/glfw/include
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
add_library (PCMPlayer SHARED  ${SOURCES} ${HEADERS})
target_link_libraries(PCMPlayer ScoreDraftCore portaudio_static glfw)

if (WIN32) 
target_link_libraries(PCMPlayer opengl32)
else()
target_link_libraries(PCMPlayer GL)
endif()

if (WIN32) 
target_compile_definitions(PCMPlayer PUBLIC SCOREDRAFTCORE_DLL_IMPORT)
endif()

if (WIN32) 
install(TARGETS PCMPlayer RUNTIME DESTINATION ScoreDraft)
else()
install(TARGETS PCMPlayer DESTINATION ScoreDraft)
endif()

