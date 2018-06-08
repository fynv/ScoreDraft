cmake_minimum_required (VERSION 3.0)

find_package(PythonLibs 3 REQUIRED)

set(SOURCES
KeLa.cpp
FrequencyDetection.cpp
)

set(HEADERS 
FrequencyDetection.h
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

include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})
add_library (KeLa SHARED ${SOURCES} ${HEADERS})
if (WIN32) 
set_target_properties(KeLa PROPERTIES SUFFIX ".dll")
else()
set_target_properties(KeLa PROPERTIES SUFFIX ".so")
endif()
target_link_libraries(KeLa ${LINK_LIBS})

install(TARGETS KeLa DESTINATION ./ScoreDraft/Extensions)



