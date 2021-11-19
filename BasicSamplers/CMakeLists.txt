cmake_minimum_required (VERSION 3.0)

project(BasicSamplers)


set(SOURCES
../DSPUtil/complex.cpp 
../DSPUtil/fft.cpp
api.cpp
FrequencyDetection.cpp
PercussionSampler.cpp
InstrumentSingleSampler.cpp
InstrumentMultiSampler.cpp
)

set(HEADERS 
../DSPUtil/complex.h
../DSPUtil/fft.h
FrequencyDetection.h
Sample.h
PercussionSampler.h
InstrumentSingleSampler.h
InstrumentMultiSampler.h
)


set (INCLUDE_DIR
../ScoreDraftCore
../DSPUtil
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
add_library (BasicSamplers SHARED  ${SOURCES} ${HEADERS})
target_link_libraries(BasicSamplers ScoreDraftCore)

if (WIN32) 
target_compile_definitions(BasicSamplers PUBLIC SCOREDRAFTCORE_DLL_IMPORT)
endif()

if (WIN32) 
install(TARGETS BasicSamplers RUNTIME DESTINATION ScoreDraft)
else()
install(TARGETS BasicSamplers DESTINATION ScoreDraft)
endif()

