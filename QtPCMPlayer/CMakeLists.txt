cmake_minimum_required (VERSION 3.0)


set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
find_package(Qt5Widgets)
find_package(Qt5Network)

set(SOURCES
main.cpp
QtPCMPlayer.cpp
)


set(HEADERS 
QtPCMPlayer.h
)

set (INCLUDE_DIR
.
)

set (LINK_LIBS 
Qt5::Widgets
Qt5::Network
)

set (DEFINES  ${DEFINES}
-D"_CRT_SECURE_NO_DEPRECATE"  
-D"_SCL_SECURE_NO_DEPRECATE" 
)

include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})
add_executable (QtPCMPlayer ${SOURCES} ${HEADERS})
target_link_libraries(QtPCMPlayer ${LINK_LIBS})

install(TARGETS QtPCMPlayer DESTINATION .)



