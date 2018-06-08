cmake_minimum_required (VERSION 3.0)

set (USE_CUDA true)

if (USE_CUDA) 
find_package(CUDA REQUIRED)
endif ()

find_package(PythonLibs 3 REQUIRED)

set(SOURCES
UtauDraft.cpp
OtoMap.cpp
FrqData.cpp
PrefixMap.cpp
SentenceGenerator_CPU.cpp
SentenceGenerator_PSOLA.cpp
SentenceGenerator_HNM.cpp
)

set(HEADERS 
UtauDraft.h
OtoMap.h
FrqData.h
PrefixMap.h
SentenceGenerator_CPU.h
SentenceGenerator_PSOLA.h
SentenceGenerator_HNM.h
)

set (INCLUDE_DIR
${PYTHON_INCLUDE_DIRS}
.
../ScoreDraftCore
../WavUtil
../DSPUtil
../PyScoreDraft
)

set (LINK_LIBS 
${PYTHON_LIBRARIES}
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


if (USE_CUDA) 

set(SOURCES ${SOURCES}
SentenceGenerator_CUDA.cpp
HNM.cu
)

set(HEADERS ${HEADERS}
SentenceGenerator_CUDA.h
helper_math.h
VoiceUtil.cuh
fft.cuh
)

set (DEFINES ${DEFINES} -D"HAVE_CUDA")

endif ()

include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})
if (USE_CUDA) 
cuda_add_library (UtauDraft SHARED ${SOURCES} ${HEADERS})
else()
add_library (UtauDraft SHARED ${SOURCES} ${HEADERS})
endif()
if (WIN32) 
set_target_properties(UtauDraft PROPERTIES SUFFIX ".dll")
else()
set_target_properties(UtauDraft PROPERTIES SUFFIX ".so")
endif()
target_link_libraries(UtauDraft ${LINK_LIBS})

install(TARGETS UtauDraft DESTINATION ./ScoreDraft/Extensions)


