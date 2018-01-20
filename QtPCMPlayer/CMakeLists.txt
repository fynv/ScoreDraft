cmake_minimum_required (VERSION 3.0)


set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
find_package(Qt5Widgets)
find_package(Qt5Network)
find_package(Qt5Multimedia)
find_package(Qt5OpenGL)
find_package(PythonLibs 3 REQUIRED)
find_package(OpenGL)

set(SOURCES
main.cpp
QtPCMPlayer.cpp
ViewWidget.cpp
BufferQueue.cpp
)


set(HEADERS 
QtPCMPlayer.h
ViewWidget.h
BufferQueue.h
)

set (INCLUDE_DIR
.
../DSPUtil
)

set (LINK_LIBS 
DSPUtil
Qt5::Widgets
Qt5::Network
Qt5::Multimedia
Qt5::OpenGL
${OPENGL_gl_LIBRARY}
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
add_executable (QtPCMPlayer ${SOURCES} ${HEADERS} resources/resources.qrc)
target_link_libraries(QtPCMPlayer ${LINK_LIBS})

install(TARGETS QtPCMPlayer DESTINATION .)


set(SOURCES
QtPCMPlayer_Ext.cpp
)


set(HEADERS 
)

set (INCLUDE_DIR
${PYTHON_INCLUDE_DIRS}
.
../ScoreDraftCore
../PyScoreDraft
)


set (LINK_LIBS 
Qt5::Widgets
Qt5::Network
${PYTHON_LIBRARIES}
ScoreDraftCore
)


include_directories(${INCLUDE_DIR})
add_library (QtPCMPlayerExt SHARED ${SOURCES} ${HEADERS})
target_link_libraries(QtPCMPlayerExt ${LINK_LIBS})

install(TARGETS QtPCMPlayerExt DESTINATION Extensions)