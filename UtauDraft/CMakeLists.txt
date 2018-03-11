cmake_minimum_required (VERSION 3.0)

find_package(PythonLibs 3 REQUIRED)

set(SOURCES
UtauDraft.cpp
UtauDraft_PSOLA.cpp
UtauDraft_HNM.cpp
UtauDraft_HNM2.cpp
OtoMap.cpp
FrqData.cpp
PrefixMap.cpp
)

set(HEADERS 
UtauDraft.h
OtoMap.h
FrqData.h
PrefixMap.h
)

set(PYTHON
CVVCChineseConverter.py
JPVCVConverter.py
TsuroVCVConverter.py
TTEnglishConverter.py
XiaYYConverter.py
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
add_library (UtauDraft SHARED ${SOURCES} ${HEADERS})
target_link_libraries(UtauDraft ${LINK_LIBS})

install(TARGETS UtauDraft DESTINATION Extensions)
install(FILES ${PYTHON} DESTINATION .)

