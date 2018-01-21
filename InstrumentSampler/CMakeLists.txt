cmake_minimum_required (VERSION 3.0)

set(SOURCES
InstrumentSample.cpp
InstrumentSingleSampler.cpp
InstrumentMultiSampler.cpp
InstrumentSamplerFactory.cpp
FrequencyDetection.cpp
)

set(HEADERS 
InstrumentSample.h
InstrumentSingleSampler.h
InstrumentMultiSampler.h
FrequencyDetection.h
)

set (INCLUDE_DIR
.
../ScoreDraftCore
../WavUtil
../DSPUtil
../PyScoreDraft
)

set (LINK_LIBS 
ScoreDraftCore
WavUtil
DSPUtil)

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
add_library (InstrumentSampler SHARED ${SOURCES} ${HEADERS})
target_link_libraries(InstrumentSampler ${LINK_LIBS})

install(TARGETS InstrumentSampler DESTINATION Extensions)



