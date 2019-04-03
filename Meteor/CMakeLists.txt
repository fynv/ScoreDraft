cmake_minimum_required (VERSION 3.0)

find_package(PythonLibs 3 REQUIRED)


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
Meteor.cpp
MainWidget.cpp
ViewWidget.cpp
)

set(HEADERS 
Meteor.h
MainWidget.h
SubListLookUp.h
ViewWidget.h
)

set (INCLUDE_DIR
${PYTHON_INCLUDE_DIRS}
.
../ScoreDraftCore
)

set (LINK_LIBS 
${PYTHON_LIBRARIES}
Qt5::Widgets
Qt5::Network
Qt5::Multimedia
Qt5::OpenGL
${OPENGL_gl_LIBRARY}
ScoreDraftCore
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
add_library (PyMeteor SHARED ${SOURCES} ${HEADERS})
if (WIN32) 
set_target_properties(PyMeteor PROPERTIES SUFFIX ".pyd")
else()
set_target_properties(PyMeteor PROPERTIES SUFFIX ".so")
set_target_properties(PyMeteor PROPERTIES PREFIX "")
endif()
target_link_libraries(PyMeteor ${LINK_LIBS})

install(TARGETS PyMeteor DESTINATION ./ScoreDraft)


