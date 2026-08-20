cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

get_filename_component(MYNAME ${CMAKE_CURRENT_LIST_DIR} NAME)
string(REPLACE " " "_" MYNAME ${MYNAME})
project(${MYNAME} C CXX)

# Building options
set(GSCM_VERSION_MAJOR            4 )
set(GSCM_VERSION_MINOR            0 )
set(CMAKE_CXX_STANDARD            23) 
set(CMAKE_CXX_STANDARD_REQUIRED   OFF)
set(CMAKE_CXX_EXTENSIONS          OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_COMPILER_WORKS      1)
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE )
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type: Debug, Release, RelWithDebInfo, MinSizeRel" FORCE)
endif()

option( USE_CPU_ONLY "CPU only, disbale GPU rendering and display mode"   OFF ) 
option( USE_OPENMP   "Use OpenMP library if available"                    ON )

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if (MSVC)
  set(CMAKE_C_FLAGS_RELEASE             "${CMAKE_C_FLAGS_RELEASE}             /MD")
  set(CMAKE_C_FLAGS_DEBUG               "${CMAKE_C_FLAGS_DEBUG}               /MDd")
  set(CMAKE_CXX_FLAGS_RELEASE           "${CMAKE_CXX_FLAGS_RELEASE}           /MD /DNDEBUG /DNOMINMAX")
  set(CMAKE_CXX_FLAGS_DEBUG             "${CMAKE_CXX_FLAGS_DEBUG}             /MDd")
  set(CMAKE_EXE_LINKER_FLAGS            "${CMAKE_EXE_LINKER_FLAGS}            /NODEFAULTLIB:LIBCMT")
  set(CMAKE_CXX_FLAGS_RELEASE           "${CMAKE_CXX_FLAGS_RELEASE}           /arch:AVX2 /O2 /Ob2 /Oi /Ot /Gy /fp:strict /openmp") # 
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE    "${CMAKE_EXE_LINKER_FLAGS_RELEASE}    /OPT:REF /OPT:ICF /INCREMENTAL:NO")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /OPT:REF /OPT:ICF /INCREMENTAL:NO")
    add_compile_options(/utf-8)
elseif(APPLE)
  add_definitions(-DOS_MACOS)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Xpreprocessor -fno-fast-math -ffp-contract=off ")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fno-tree-vectorize -fno-fast-math -ffp-contract=off -mavx2 -mno-fma")
  # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fno-fast-math -ffp-contract=off -mavx2 -mno-fma")    
  # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -fno-tree-vectorize -fno-fast-math -ffp-contract=off -mavx2 -mno-fma")
  # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fno-tree-vectorize -fno-fast-math -ffp-contract=off -mavx2 -mno-fma")
endif()
add_definitions(-DGLM_FORCE_ALIGNED_GENTYPES)

# Open Multi-Processing (OpenMP)
message("USE_CPU_ONLY = ${USE_CPU_ONLY}")
message("USE_OPENMP   = ${USE_OPENMP}")
if( USE_OPENMP )
  find_package(OpenMP REQUIRED)
  if( OPENMP_FOUND )
    set( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}"   )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
    message("OpenMP library found add : ${OpenMP_C_FLAGS} and ${OpenMP_CXX_FLAGS} to cmake flags")
  else()
    message( FATAL_ERROR "Package OpenMP required, but not found!, please install apt install libomp-dev libomp5 or brew install libomp-dev libomp5" )  
  endif()
endif()

# Cmake dependencies
include(${CMAKE_SOURCE_DIR}/dependencies/cmake/CPM.cmake)
include(${CMAKE_SOURCE_DIR}/dependencies/cmake/clang-format.cmake)
include(${CMAKE_SOURCE_DIR}/dependencies/cmake/cppcheck.cmake)
include(${CMAKE_SOURCE_DIR}/dependencies/cmake/glm.cmake)
if ( NOT USE_CPU_ONLY )
  include(${CMAKE_SOURCE_DIR}/dependencies/cmake/glfw.cmake)
endif()
include(${CMAKE_SOURCE_DIR}/dependencies/cmake/fmtlib.cmake)
add_subdirectory(${CMAKE_SOURCE_DIR}/dependencies/pmbb-base libpmbb-base)
add_subdirectory(${CMAKE_SOURCE_DIR}/dependencies/pmbb-core libpmbb-core)
add_subdirectory(${CMAKE_SOURCE_DIR}/dependencies/pmbb-ivqm libpmbb-ivqm)

# Include 
include_directories(${PROJECT_SOURCE_DIR}/external/stb)
if( APPLE )
  if( USE_OPENMP )
    link_libraries(${OpenMP_CXX_LIBRARIES})
  endif()
endif()

# Determine the software version
set(VERSION_FILE ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)
add_custom_target( genversion
                   COMMAND ${CMAKE_COMMAND} -D OUTPUT=${CMAKE_CURRENT_BINARY_DIR}
                                            -D VERSION_NUMBER=v${GSCM_VERSION_MAJOR}.${GSCM_VERSION_MINOR}
                                            -P ${PROJECT_SOURCE_DIR}/dependencies/cmake/genversion.cmake
                   BYPRODUCTS ${VERSION_FILE}
                   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} )

# Main applications 

link_directories( ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} ${GLFW_LIBRARY_DIR} )

file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/source/*.cpp 
                    ${PROJECT_SOURCE_DIR}/source/util/*.cpp 
                    ${PROJECT_SOURCE_DIR}/dependencies/program-options-lite/*.cpp 
                    ${PROJECT_SOURCE_DIR}/dependencies/glad/src/glad.c                    
                    ${PROJECT_SOURCE_DIR}/dependencies/stb_image/stb_image.cpp)

find_package(Threads REQUIRED)
SET( LIBS Threads::Threads)
if ( NOT USE_CPU_ONLY ) 
  if (WIN32)
    list(APPEND LIBS ${GLFW_LIBRARIES}) 
  else()
    list(APPEND LIBS glfw) 
  endif()
else()
  message("CPU only mode, disabling GLFW linking")
  if (WIN32)
    list (APPEND LIBS gdi32 user32)
  else()
    list (APPEND LIBS dl)
  endif()
endif()
list(APPEND LIBS ${LIB_PMBB_IVQM_NAME})
list(APPEND LIBS ${LIB_PMBB_CORE_NAME})
list(APPEND LIBS ${LIB_PMBB_BASE_NAME})
list(APPEND LIBS fmt::fmt)
if(USE_OPENMP)
  list(APPEND LIBS OpenMP::OpenMP_CXX)
  if(APPLE)
    list(APPEND LIBS omp)
  endif()
endif()

include_directories(${PROJECT_SOURCE_DIR}/source
                    ${PROJECT_SOURCE_DIR}/source/util
                    ${PROJECT_SOURCE_DIR}/dependencies
                    ${PROJECT_SOURCE_DIR}/dependencies/glm
                    ${PROJECT_SOURCE_DIR}/dependencies/glad/include
                    ${PROJECT_SOURCE_DIR}/dependencies/stb_image
                    ${PROJECT_SOURCE_DIR}/dependencies/sse2neon
                    ${GLFW_INCLUDE_DIR} 
                    ${PROJECT_SOURCE_DIR}/dependencies/fmtlib/include
                    ${${LIB_PMBB_BASE_NAME}_SOURCE_DIR}/src
                    ${${LIB_PMBB_CORE_NAME}_SOURCE_DIR}/src
                    ${${LIB_PMBB_IVQM_NAME}_SOURCE_DIR}/src
                    ${CMAKE_CURRENT_BINARY_DIR} )
add_executable( mpeg-gsc-metrics ${CPP_FILES} ${VERSION_FILE})
add_dependencies(mpeg-gsc-metrics genversion)    
if ( NOT USE_CPU_ONLY )
  target_compile_definitions( mpeg-gsc-metrics PRIVATE USE_GLFW)
endif()
if ( USE_OPENMP )
  target_compile_definitions( mpeg-gsc-metrics PUBLIC USE_OPENMP)
  if (APPLE)
    target_compile_options(mpeg-gsc-metrics PRIVATE -Xpreprocessor -fopenmp)
    target_include_directories(mpeg-gsc-metrics PRIVATE /opt/homebrew/opt/libomp/include)
    target_link_directories(mpeg-gsc-metrics PRIVATE /opt/homebrew/opt/libomp/lib)
  elseif(MSVC)
    target_compile_options(mpeg-gsc-metrics PRIVATE /openmp)
  endif()
endif()
target_link_libraries( mpeg-gsc-metrics ${LIBS} )
install( TARGETS mpeg-gsc-metrics DESTINATION bin ) 


