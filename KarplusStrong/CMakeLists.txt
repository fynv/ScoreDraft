cmake_minimum_required (VERSION 3.0)

project(KarplusStrong)

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
add_library (KarplusStrong SHARED KarplusStrong.cpp ../DSPUtil/complex.cpp ../DSPUtil/fft.cpp)
target_link_libraries(KarplusStrong ScoreDraftCore)

if (WIN32) 
target_compile_definitions(KarplusStrong PUBLIC SCOREDRAFTCORE_DLL_IMPORT)
endif()

if (WIN32) 
install(TARGETS KarplusStrong RUNTIME DESTINATION ScoreDraft)
else()
install(TARGETS KarplusStrong DESTINATION ScoreDraft)
endif()

