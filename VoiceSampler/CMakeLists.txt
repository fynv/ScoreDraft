cmake_minimum_required (VERSION 3.0)
project(VoiceSampler)

set(USE_CUDA true CACHE BOOL "Voice Sampler Use CUDA")

if (USE_CUDA) 
find_package(CUDA REQUIRED)
endif ()


set(SOURCES
../DSPUtil/complex.cpp 
../DSPUtil/fft.cpp
api.cpp
FrequencyDetection.cpp
SentenceGeneratorGeneral.cpp
SentenceGeneratorCPU.cpp
)

set(HEADERS 
../DSPUtil/complex.h
../DSPUtil/fft.h
FrequencyDetection.h
VoiceUtil.h
SentenceDescriptor.h
SentenceGeneratorGeneral.h
SentenceGeneratorCPU.h
)


set (INCLUDE_DIR
.
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


if (USE_CUDA) 

set(SOURCES ${SOURCES}
HNM.cu
SentenceGeneratorCUDA.cpp
)

set(HEADERS ${HEADERS}
helper_math.h
VoiceUtil.cuh
fft.cuh
DVVector.hpp
SentenceGeneratorCUDA.h
)

set (DEFINES ${DEFINES} -D"HAVE_CUDA")

endif ()


include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})
if (USE_CUDA) 
cuda_add_library (VoiceSampler SHARED ${SOURCES} ${HEADERS})
else()
add_library (VoiceSampler SHARED  ${SOURCES} ${HEADERS})
endif()
target_link_libraries(VoiceSampler ScoreDraftCore)

if (WIN32) 
target_compile_definitions(VoiceSampler PUBLIC SCOREDRAFTCORE_DLL_IMPORT)
endif()

if (WIN32) 
install(TARGETS VoiceSampler RUNTIME DESTINATION ScoreDraft)
else()
install(TARGETS VoiceSampler DESTINATION ScoreDraft)
endif()

