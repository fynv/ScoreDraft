cmake_minimum_required (VERSION 3.0)

set(SOURCES
complex.cpp
fft.cpp
InstrumentSampler.cpp
)

set(HEADERS 
complex.h
fft.h
)

set (INCLUDE_DIR
.
../ScoreDraftCore
../PyScoreDraft
)

set (LINK_LIBS 
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
add_library (InstrumentSampler SHARED ${SOURCES})
target_link_libraries(InstrumentSampler ${LINK_LIBS})

install(TARGETS InstrumentSampler DESTINATION Extensions)



