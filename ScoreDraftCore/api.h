#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#pragma warning( disable: 4251 )
#if defined SCOREDRAFTCORE_DLL_EXPORT
#define SCOREDRAFTCORE_API __declspec(dllexport)
#elif defined SCOREDRAFTCORE_DLL_IMPORT
#define SCOREDRAFTCORE_API __declspec(dllimport)
#endif
#endif

#ifndef SCOREDRAFTCORE_API
#define SCOREDRAFTCORE_API
#endif
