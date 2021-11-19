cmake_minimum_required (VERSION 3.0)
project(ScoreDraft)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX  ../Test CACHE PATH "Install path" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

add_subdirectory(ScoreDraftCore)
add_subdirectory(SimpleInstruments)
add_subdirectory(KarplusStrong)
add_subdirectory(BasicSamplers)
add_subdirectory(SoundFont2)
add_subdirectory(VoiceSampler)

add_subdirectory(MIDIWriter)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(thirdparty/portaudio)
add_subdirectory(thirdparty/glfw)
# add_subdirectory(thirdparty/glew)

add_subdirectory(PCMPlayer)
add_subdirectory(Meteor)

add_subdirectory(python)
