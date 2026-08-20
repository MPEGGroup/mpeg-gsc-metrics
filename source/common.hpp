
/* The copyright in this software is being made available under the BSD
 * Licence, included below.  This software may be subject to other third
 * party and contributor rights, including patent rights, and no such
 * rights are granted under this licence.
 *
 * Copyright (c) 2025, ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * * Neither the name of the ISO/IEC nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define _USE_MATH_DEFINES
#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <istream>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <numeric>
#include <variant>

#ifdef WIN32
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <sys/dir.h>
#else
#include <sys/dir.h>
#endif

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
#include <glm/ext/matrix_relational.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#if defined( USE_GLFW )
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

static void getBuffer( GLuint uiBuffer, std::vector<uint32_t>& vec, size_t size ) {
  GLenum error = glGetError();
  if ( error != GL_NO_ERROR ) {
    printf( "OpenGL error code: 0x%x before getBuffer\n", error );
    fflush( stdout );
  }
  glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, uiBuffer );
  void* rawKey = glMapBufferRange( GL_ELEMENT_ARRAY_BUFFER, 0, sizeof( uint32_t ) * size, GL_MAP_READ_BIT );
  if ( !rawKey ) {
    printf( "Error: glMapBufferRange failed, OpenGL error: 0x%x\n", glGetError() );
    fflush( stdout );
  } else {
    if ( vec.size() < size ) { vec.resize( size ); }
    memcpy( vec.data(), rawKey, sizeof( uint32_t ) * size );
  }
  if ( !glUnmapBuffer( GL_ELEMENT_ARRAY_BUFFER ) ) {
    printf( "Error: glUnmapBuffer failed, OpenGL error: 0x%x\n", glGetError() );
    fflush( stdout );
  }
  error = glGetError();
  if ( error != GL_NO_ERROR ) {
    printf( "OpenGL error code: 0x%x after getBuffer\n", error );
    fflush( stdout );
  }
}

static void setBuffer( GLuint uiBuffer, const std::vector<uint32_t>& vec ) {
  GLenum error;
  glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, uiBuffer );
  error = glGetError();
  if ( error != GL_NO_ERROR ) {
    printf( "OpenGL error: Failed to bind buffer 0x%x\n", error );
    return;
  }
  glBufferSubData( GL_ELEMENT_ARRAY_BUFFER, 0, vec.size() * sizeof( uint32_t ), vec.data() );
  error = glGetError();
  if ( error != GL_NO_ERROR ) {
    printf( "OpenGL error: glBufferSubData failed with code 0x%x\n", error );
    return;
  }
  glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
}
#endif

#ifdef WIN32
#define snprintf _snprintf
#endif
template <typename... Args>
std::string stringFormat( const char* pFormat, Args... args ) {
  size_t      iSize = snprintf( nullptr, 0, pFormat, args... );
  std::string pString;
  pString.reserve( iSize + 1 );
  pString.resize( iSize );
  snprintf( &pString[0], iSize + 1, pFormat, args... );
  return pString;
}

static inline bool exist( const std::string& pString ) {
  struct stat buffer;
  return ( stat( pString.c_str(), &buffer ) == 0 );
}

static bool dirExists( const std::string& pString ) {
  struct stat sb;
  if ( stat( pString.c_str(), &sb ) == 0 ) { return true; }
  return false;
}

#ifndef _WIN32
static char getSeparator() { return '/'; }
#else
static char getSeparator() { return '\\'; }
#endif

static char getSeparator( const std::string& eFilename ) {
  auto pos = ( std::min )( eFilename.find_last_of( '/' ), eFilename.find_last_of( '\\' ) );
  return pos != std::string::npos ? eFilename[pos] : getSeparator();
}

static std::string createFilename( const std::string& pFilename, int32_t iFrameIndex ) {
  std::string sFilename = stringFormat( pFilename.c_str(), iFrameIndex );
#ifdef MSVC
  char pNewName[512];
  GetFullPathName( sFilename.c_str(), 512, pNewName, NULL );
  sFilename = pNewName;
#endif
  return sFilename;
}

static bool createDirectory( const std::string& eFilename ) {
  try {
    std::filesystem::path p( eFilename );
    std::filesystem::path dir = p.parent_path();
    if ( dir.empty() ) { return true; }
    if ( !std::filesystem::exists( dir ) ) { return std::filesystem::create_directories( dir ); }
    return true;
  } catch ( const std::filesystem::filesystem_error& e ) {
    std::cerr << "Error: " << e.what() << std::endl;
    return false;
  }
}

static std::string getExtension( const std::string& eFilename ) {
  auto position = eFilename.find_last_of( '.' );
  if ( position != std::string::npos ) {
    return eFilename.substr( position + 1 );
  } else {
    return std::string( "" );
  }
}

static std::string getRemoveExtension( const std::string& eFilename ) {
  auto position = eFilename.find_last_of( '.' );
  if ( position != std::string::npos ) { return eFilename.substr( 0, position ); }
  return eFilename;
}

static std::string getDirectory( const std::string& string ) {
  auto position = string.find_last_of( getSeparator( string ) );
  if ( position != std::string::npos ) { return string.substr( 0, position ); }
  return string;
}

static std::string getBasename( const std::string& string ) {
  auto position = string.find_last_of( getSeparator( string ) );
  if ( position != std::string::npos ) { return string.substr( position + 1, string.length() ); }
  return string;
}

static std::string getLabel( const std::string& path ) {
  const auto parent = getBasename( getDirectory( path ) );
  return parent.empty() ? getBasename( path ) : parent + getSeparator( path ) + getBasename( path );
}

static std::string formatCode( const std::string& eCode ) {
  std::istringstream shaderStream( eCode );
  std::string        line;
  std::string        output;
  int32_t            lineIndex = 1;
  while ( std::getline( shaderStream, line ) ) output += stringFormat( "%4d: %s \n", lineIndex++, line.c_str() );
  return output;
}

static void logMat4( const std::string& name, const glm::mat4& mat ) {
  printf( "glm::mat4 %-8s( %12.8f, %12.8f, %12.8f, %12.8f, \n", name.c_str(), mat[0][0], mat[0][1], mat[0][2],
          mat[0][3] );
  printf( "                    %12.8f, %12.8f, %12.8f, %12.8f, \n", mat[1][0], mat[1][1], mat[1][2], mat[1][3] );
  printf( "                    %12.8f, %12.8f, %12.8f, %12.8f, \n", mat[2][0], mat[2][1], mat[2][2], mat[2][3] );
  printf( "                    %12.8f, %12.8f, %12.8f, %12.8f );\n", mat[3][0], mat[3][1], mat[3][2], mat[3][3] );
}

static void logVec3( const std::string& name, const glm::vec3& vec ) {
  printf( "glm::vec3 %-8s( %12.8f, %12.8f, %12.8f ); \n", name.c_str(), vec[0], vec[1], vec[2] );
}

static void logVec2( const std::string& name, const glm::vec2& vec ) {
  printf( "glm::vec2 %-8s( %12.8f, %12.8f ); \n", name.c_str(), vec[0], vec[1] );
}

inline float deterministic_exp(float x) {
    if (x < -87.33654f) return 0.0f;
    if (x >  88.72283f) return 3.4028235e38f;
    const float INV_LN2 = 1.4426950408889634f;
    float k_f = std::floor(x * INV_LN2 + 0.5f);
    int32_t k = static_cast<int32_t>(k_f);
    const float LN2_HI = 0.6931457519f;
    const float LN2_LO = 1.4286067653e-6f;
    float r = (x - k_f * LN2_HI) - k_f * LN2_LO;
    float p = 0.04166666f;
    p = p * r + 0.16666667f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    p = std::ldexp(p, k); 
    return std::max(p, 0.0f);
}

static size_t getPeakMemoryUsage() {
#if defined(WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.PeakWorkingSetSize;
    }
    return 0;
#elif defined(__linux__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<size_t>(usage.ru_maxrss) * 1024;
    }
    return 0;
#else
    return 0; 
#endif
}

typedef float ( *CastFunction )( uint8_t* pPointer );

static inline float castUChar( uint8_t* pPointer ) { return (float)( *( (uint8_t*)pPointer ) ); }
static inline float castChar( uint8_t* pPointer ) { return (float)( *( (char*)pPointer ) ); }
static inline float castUShort( uint8_t* pPointer ) { return (float)( *( (uint16_t*)pPointer ) ); }
static inline float castShort( uint8_t* pPointer ) { return (float)( *( (short*)pPointer ) ); }
static inline float castUInt( uint8_t* pPointer ) { return (float)( *( (uint32_t*)pPointer ) ); }
static inline float castInt( uint8_t* pPointer ) { return (float)( *( (int32_t*)pPointer ) ); }
static inline float castFloat( uint8_t* pPointer ) { return (float)( *( (float*)pPointer ) ); }
static inline float castDouble( uint8_t* pPointer ) { return (float)( *( (double*)pPointer ) ); }
static inline float castZero( uint8_t* ) { return 0; }

static void getSize( const std::string& pFormat, int32_t& iSize, CastFunction& eCastFunction ) {
  // clang-format off
  if      ( pFormat == "float"   ) { iSize = 4; eCastFunction = castFloat;   } 
  else if ( pFormat == "uchar"   ) { iSize = 1; eCastFunction = castUChar;   } 
  else if ( pFormat == "uint8"   ) { iSize = 1; eCastFunction = castUChar;   }   
  else if ( pFormat == "char"    ) { iSize = 1; eCastFunction = castChar;    } 
  else if ( pFormat == "int8"    ) { iSize = 1; eCastFunction = castChar;    } 
  else if ( pFormat == "ushort"  ) { iSize = 2; eCastFunction = castUShort;  } 
  else if ( pFormat == "uint16"  ) { iSize = 2; eCastFunction = castUShort;  } 
  else if ( pFormat == "short"   ) { iSize = 2; eCastFunction = castShort;   } 
  else if ( pFormat == "int16"   ) { iSize = 2; eCastFunction = castShort;   } 
  else if ( pFormat == "uint"    ) { iSize = 4; eCastFunction = castUInt;    }
  else if ( pFormat == "uint32"  ) { iSize = 4; eCastFunction = castUInt;    } 
  else if ( pFormat == "int"     ) { iSize = 4; eCastFunction = castInt;     } 
  else if ( pFormat == "int32"   ) { iSize = 4; eCastFunction = castInt;     }
  else if ( pFormat == "float32" ) { iSize = 4; eCastFunction = castFloat;   }
  else if ( pFormat == "float64" ) { iSize = 8; eCastFunction = castDouble;  }
  else if ( pFormat == "double"  ) { iSize = 8; eCastFunction = castDouble;  } 
  else {    iSize         = 0;    eCastFunction = castZero;  }
  // clang-format on
}

#define SHADER_VERSION_MAJOR 4
#define SHADER_VERSION_MINOR 6

#define SHADER_VERSION ( ( SHADER_VERSION_MAJOR * 100 ) + ( SHADER_VERSION_MINOR * 10 ) )

#define SHADER( shader ) "#version " + std::to_string( SHADER_VERSION ) + "\n" #shader
#define SHADERR( shader ) R"(#version )" + std::to_string( SHADER_VERSION ) + shader