cmake_minimum_required (VERSION 3.0)

find_package(PythonLibs 3 REQUIRED)

set(SOURCES
PercussionSampler.cpp
)

set (INCLUDE_DIR
${PYTHON_INCLUDE_DIRS}
.
../ScoreDraftCore
../WavUtil
)

set (LINK_LIBS 
${PYTHON_LIBRARIES}
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
add_library (PyPercussionSampler SHARED ${SOURCES})
if (WIN32) 
set_target_properties(PyPercussionSampler PROPERTIES SUFFIX ".pyd")
else()
set_target_properties(PyPercussionSampler PROPERTIES SUFFIX ".so")
set_target_properties(PyPercussionSampler PROPERTIES PREFIX "")
endif()
target_link_libraries(PyPercussionSampler ${LINK_LIBS})

install(TARGETS PyPercussionSampler DESTINATION ./ScoreDraft)


