cmake_minimum_required (VERSION 3.0)

set(SOURCES
PercussionSampler.cpp
)

set (INCLUDE_DIR
.
../ScoreDraftCore
../WavUtil
../PyScoreDraft
)

set (LINK_LIBS 
ScoreDraftCore
WavUtil
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
add_library (PercussionSampler SHARED ${SOURCES})
target_link_libraries(PercussionSampler ${LINK_LIBS})

install(TARGETS PercussionSampler DESTINATION Extensions)


