cmake_minimum_required (VERSION 3.0)

project(SimpleInstruments)

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
add_library (SimpleInstruments SHARED SimpleInstruments.cpp)
target_link_libraries(SimpleInstruments ScoreDraftCore)

if (WIN32) 
target_compile_definitions(SimpleInstruments PUBLIC SCOREDRAFTCORE_DLL_IMPORT)
endif()

if (WIN32) 
install(TARGETS SimpleInstruments RUNTIME DESTINATION ScoreDraft)
else()
install(TARGETS SimpleInstruments DESTINATION ScoreDraft)
endif()

