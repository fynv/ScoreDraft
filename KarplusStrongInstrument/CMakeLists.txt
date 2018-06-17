cmake_minimum_required (VERSION 3.0)

find_package(PythonLibs 3 REQUIRED)

set(SOURCES
KarplusStrongInstrument.cpp
)

set(HEADERS 
)

set (INCLUDE_DIR
${PYTHON_INCLUDE_DIRS}
.
../ScoreDraftCore
../DSPUtil
../PyScoreDraft
)

set (LINK_LIBS 
${PYTHON_LIBRARIES}
ScoreDraftCore
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
add_library (KarplusStrongInstrument SHARED ${SOURCES} ${HEADERS})
if (WIN32) 
set_target_properties(KarplusStrongInstrument PROPERTIES SUFFIX ".dll")
else()
set_target_properties(KarplusStrongInstrument PROPERTIES SUFFIX ".so")
endif()
target_link_libraries(KarplusStrongInstrument ${LINK_LIBS})

install(TARGETS KarplusStrongInstrument DESTINATION ./ScoreDraft/Extensions)



