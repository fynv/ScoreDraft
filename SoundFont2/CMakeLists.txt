cmake_minimum_required (VERSION 3.0)

project(SoundFont2)


set(SOURCES
api.cpp
SF2.cpp
Presets.cpp
Synth.cpp
SF2Synth.cpp
)

set(HEADERS 
SF2.h
Presets.h
Synth.h
SF2Synth.h
)


set (INCLUDE_DIR
../ScoreDraftCore
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
add_library (SoundFont2 SHARED  ${SOURCES} ${HEADERS})
target_link_libraries(SoundFont2 ScoreDraftCore)

if (WIN32) 
target_compile_definitions(SoundFont2 PUBLIC SCOREDRAFTCORE_DLL_IMPORT)
endif()

if (WIN32) 
install(TARGETS SoundFont2 RUNTIME DESTINATION ScoreDraft)
else()
install(TARGETS SoundFont2 DESTINATION ScoreDraft)
endif()

