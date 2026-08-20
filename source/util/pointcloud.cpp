
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
#include "pointcloud.hpp"
#if defined( USE_OPENMP )
#include <omp.h>
#endif
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC target("fma")
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

  bool Pointcloud::read( const std::string& path, int32_t frameIndex, bool verbose ) {
    auto filename = createFilename( path, frameIndex );
    if ( !exist( filename ) ) {
      printf( "Error: The input file does not exist: \"%s\"\n", filename.c_str() );
      return false;
    }
    std::string   line, s1, s2;
    std::ifstream inputPly;
    if ( verbose ) {
      printf( "Pointcloud reader: Reading %s \n", filename.c_str() );
      fflush( stdout );
    }
    inputPly.open( filename.c_str(), std::ios::in | std::ios::binary );
    if ( !inputPly.is_open() ) {
      printf( "Pointcloud reader: Couldn't open %s \n", filename.c_str() );
      return false;
    }
    bool bFormatError = false, bAscii = false, bLittleEndian = false, bInVertexElements = false,
         bInHeaderElements = false;
    int iNumPoints = 0, iSize = 0, iOrder = 0, iIndex = 0, iNumFaces = 0, iNumHeader = 0, iSizeHeader = 0;
    const std::vector<std::string> pProperty = {
        "x",         "y",         "z",         "nx",        "ny",        "nz",        "f_dc_0",    "f_dc_1",
        "f_dc_2",    "f_rest_0",  "f_rest_1",  "f_rest_2",  "f_rest_3",  "f_rest_4",  "f_rest_5",  "f_rest_6",
        "f_rest_7",  "f_rest_8",  "f_rest_9",  "f_rest_10", "f_rest_11", "f_rest_12", "f_rest_13", "f_rest_14",
        "f_rest_15", "f_rest_16", "f_rest_17", "f_rest_18", "f_rest_19", "f_rest_20", "f_rest_21", "f_rest_22",
        "f_rest_23", "f_rest_24", "f_rest_25", "f_rest_26", "f_rest_27", "f_rest_28", "f_rest_29", "f_rest_30",
        "f_rest_31", "f_rest_32", "f_rest_33", "f_rest_34", "f_rest_35", "f_rest_36", "f_rest_37", "f_rest_38",
        "f_rest_39", "f_rest_40", "f_rest_41", "f_rest_42", "f_rest_43", "f_rest_44", "opacity",   "scale_0",
        "scale_1",   "scale_2",   "rot_0",     "rot_1",     "rot_2",     "rot_3",
    };
    std::vector<int>          pSize( pProperty.size() + 1, 0 );
    std::vector<int>          pOrder( pProperty.size(), -1 );
    std::vector<int>          pIndex( pProperty.size(), 0 );
    std::vector<CastFunction> pCast( pProperty.size() + 1, castZero );
    getline( inputPly, line );
    while ( line.find( "end_header" ) != 0 ) {
      line = line.erase( line.size() );
      line.erase( std::remove( line.begin(), line.end(), '\n' ), line.end() );
      line.erase( std::remove( line.begin(), line.end(), '\r' ), line.end() );
      if ( static_cast<int>( line.find( "format " ) ) > -1 ) {
        std::string format = line.substr(
            line.find( ' ' ) + 1, line.find_first_of( ' ', line.find_first_of( ' ' ) + 1 ) - line.find( ' ' ) - 1 );
        if ( format == "ascii" ) {
          bAscii = true;
        } else if ( format == "binary_little_endian" ) {
          bLittleEndian = true;
        } else {
          printf(
              "GaussianSpatReader: Format is not ascii or "
              "binary_little_endian: format = \"%s\" \n",
              format.c_str() );
          bFormatError = true;
        }
      } else if ( static_cast<int>( line.find( "element" ) ) == 0 ) {
        if ( static_cast<int>( line.find( "header_data " ) ) > 0 ) {
          line              = line.substr( line.find( "header_data " ) );
          iNumHeader        = atoi( line.substr( line.find( ' ' ) ).c_str() );
          bInHeaderElements = true;
          bInVertexElements = false;
        } else if ( static_cast<int>( line.find( "vertex " ) ) > 0 ) {
          line              = line.substr( line.find( "vertex " ) );
          iNumPoints        = atoi( line.substr( line.find( ' ' ) ).c_str() );
          bInHeaderElements = false;
          bInVertexElements = true;
        } else if ( static_cast<int>( line.find( "face " ) ) > 0 ) {
          line              = line.substr( line.find( "face " ) );
          iNumFaces         = atoi( line.substr( line.find( ' ' ) ).c_str() );
          bInHeaderElements = false;
          bInVertexElements = false;
        } else {
          bInHeaderElements = false;
          bInVertexElements = false;
        }
      } else if ( static_cast<int>( line.find( "comment" ) ) == 0 ) {
        std::string key = "camera_position ";
        if ( static_cast<int>( line.find( key ) ) > 0 ) {
          line = line.substr( line.find( key ) + key.length() );
          std::istringstream ss( line );
          CameraPosition     el;
          if ( ss >> el.index_ >> el.cameraId_ >> el.position_.x >> el.position_.y >> el.position_.z >>
               el.orientation_.w >> el.orientation_.x >> el.orientation_.y >> el.orientation_.z >> el.focal_.x >>
               el.focal_.y >> el.width_ >> el.height_ >> el.center_.x >> el.center_.y >>el.name_ ) {
            cameraPosition_.push_back( el );
          } else {
            std::cerr << "Error: Can't read camera position line.\n";
          }
        }
      } else {
        if ( bInVertexElements && static_cast<int>( line.find( "property " ) ) == 0 ) {
          line           = line.substr( line.find( ' ' ) + 1, line.size() - line.find( ' ' ) - 1 );
          std::string eV = line.substr( 0, line.find_last_of( ' ' ) ), eT = line.substr( line.find_last_of( ' ' ) + 1 );
          CastFunction eCast;
          getSize( eV, iSize, eCast );
          bool find = false;
          for ( size_t i = 0; i < pProperty.size() && !find; i++ ) {
            if ( eT == pProperty[i] ) {
              pSize[i]  = iSize;
              pOrder[i] = iOrder;
              pIndex[i] = iIndex;
              pCast[i]  = eCast;
              find      = true;
            }
          }
          if ( !find ) {
            pSize[pProperty.size()] += iSize;
            pCast[pProperty.size()] = eCast;
            printf( "GaussianSplatReader: Property \"%s\" is not supported ( %s ) \n", eT.c_str(), eV.c_str() );
            fflush( stdout );
          }
          iIndex += iSize;
          iOrder++;
        } else if ( bInHeaderElements && static_cast<int>( line.find( "property " ) ) == 0 ) {
          line           = line.substr( line.find( ' ' ) + 1, line.size() - line.find( ' ' ) - 1 );
          std::string eV = line.substr( 0, line.find_last_of( ' ' ) ), eT = line.substr( line.find_last_of( ' ' ) + 1 );
          CastFunction eCast;
          getSize( eV, iSize, eCast );
          iSizeHeader += iSize;
        }
      }
      getline( inputPly, line );
    }
    if ( pSize[0] == 0 ) {
      printf( "GaussianSplatReader: X must be define \n" );
      bFormatError = true;
    }
    if ( pSize[1] == 0 ) {
      printf( "GaussianSplatReader: Y must be define \n" );
      bFormatError = true;
    }
    if ( pSize[2] == 0 ) {
      printf( "GaussianSplatReader: Z must be define \n" );
      bFormatError = true;
    }
    if ( iNumFaces > 0 ) {
      printf(
          "GaussianSplatReader: number of face must be equal to 0 (iNumFaces "
          "= %d) \n",
          iNumFaces );
      bFormatError = true;
    }
    if ( bFormatError ) {
      printf( "GaussianSplatReader: format not yet supported \n" );
      fflush( stdout );
      reset();
      return false;
    }
    gaussianData_.resize( iNumPoints );
    if ( bAscii ) {
      printf( "Gaussian splat ascii is not yet supported \n" );
      fflush( stdout );
      exit( -1 );
    } else if ( bLittleEndian ) {
      std::vector<unsigned char> pC;
      pC.resize( iIndex );
      std::array<float, 48> pSh;
      if ( iNumHeader > 0 ) { inputPly.ignore( iNumHeader * iSizeHeader ); }
      for ( size_t i = 0; i < static_cast<size_t>( iNumPoints ); i++ ) {
        auto& dst = gaussianData_[i];
        inputPly.read( reinterpret_cast<char*>( pC.data() ), iIndex * sizeof( unsigned char ) );

        int k       = 0;
        dst.xyz_[0] = pCast[k + 0]( pC.data() + pIndex[k + 0] );  // x
        dst.xyz_[1] = pCast[k + 1]( pC.data() + pIndex[k + 1] );  // y
        dst.xyz_[2] = pCast[k + 2]( pC.data() + pIndex[k + 2] );  // z
        dst.xyz_[3] = 1;
        k += 3;

        // dst.normal_[0] = pCast[k + 0]( pC.data() + pIndex[k + 0] );  // nx
        // dst.normal_[1] = pCast[k + 1]( pC.data() + pIndex[k + 1] );  // ny
        // dst.normal_[2] = pCast[k + 2]( pC.data() + pIndex[k + 2] );  // nz
        k += 3;
        for ( int j = 0; j < 48; j++, k++ ) {
          float sh = pCast[k]( pC.data() + pIndex[k] );  // sh
          pSh[j]   = sh;
        }

        dst.opacity_ = 1.0 / ( 1.0 + deterministic_exp( -pCast[k]( pC.data() + pIndex[k] ) ) );  // o
        k++;

        dst.scale_ = glm::vec4( deterministic_exp( pCast[k + 0]( pC.data() + pIndex[k + 0] ) ),       // scale 0
                                deterministic_exp( pCast[k + 1]( pC.data() + pIndex[k + 1] ) ),       // scale 1
                                deterministic_exp( pCast[k + 2]( pC.data() + pIndex[k + 2] ) ), 0 );  // scale 2
        k += 3;

        float qw    = pCast[k + 0]( pC.data() + pIndex[k + 0] );
        float qx    = pCast[k + 1]( pC.data() + pIndex[k + 1] );
        float qy    = pCast[k + 2]( pC.data() + pIndex[k + 2] );
        float qz    = pCast[k + 3]( pC.data() + pIndex[k + 3] );
        dst.rotate_ = glm::normalize( glm::quat( qw, qx, qy, qz ) );
        k += 4;
        dst.sh_[0][0] = pSh[0];
        dst.sh_[0][1] = pSh[1];
        dst.sh_[0][2] = pSh[2];
        for ( size_t sh_idx = 1; sh_idx < 16; ++sh_idx ) {
          dst.sh_[sh_idx][0] = pSh[3 + sh_idx - 1];
          dst.sh_[sh_idx][1] = pSh[3 + sh_idx + 14];
          dst.sh_[sh_idx][2] = pSh[3 + sh_idx + 29];
        }
      }
      pC.clear();
    }
    inputPly.close();
    return true;
  }

  bool Pointcloud::write( const std::string& path, int32_t frameIndex ) {
    auto          filename = createFilename( path, frameIndex );
    std::ofstream file( filename, std::ios::out | std::ios::binary );
    if ( !file.is_open() ) { throw std::runtime_error( "Error: can't open: " + filename ); }
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << gaussianData_.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float nx\n";
    file << "property float ny\n";
    file << "property float nz\n";
    for ( int i = 0; i < 3; ++i ) file << "property float f_dc_" << i << "\n";
    for ( int i = 0; i < 45; ++i ) file << "property float f_rest_" << i << "\n";
    file << "property float opacity\n";
    file << "property float scale_0\n";
    file << "property float scale_1\n";
    file << "property float scale_2\n";
    file << "property float rot_0\n";
    file << "property float rot_1\n";
    file << "property float rot_2\n";
    file << "property float rot_3\n";
    file << "end_header\n";

    std::array<float, 48> sh;
    std::array<float, 3>  normal;
    for ( const auto& data : gaussianData_ ) {
      sh[0] = data.sh_[0][0];  
      sh[1] = data.sh_[0][1];   
      sh[2] = data.sh_[0][2];   
      for ( size_t sh_idx = 1; sh_idx < 16; ++sh_idx ) {
        sh[3 + sh_idx - 1]  = data.sh_[sh_idx][0];  
        sh[3 + sh_idx + 14] = data.sh_[sh_idx][1];   
        sh[3 + sh_idx + 29] = data.sh_[sh_idx][2];  
      }
      file.write( reinterpret_cast<const char*>( &data.xyz_ ), sizeof( data.xyz_ ) );
      file.write( reinterpret_cast<const char*>( normal.data() ), sizeof( normal ) );
      file.write( reinterpret_cast<const char*>( sh.data() ), sizeof( sh ) );
      file.write( reinterpret_cast<const char*>( &data.opacity_ ), sizeof( data.opacity_ ) );
      file.write( reinterpret_cast<const char*>( &data.scale_ ), sizeof( data.scale_ ) );
      file.write( reinterpret_cast<const char*>( &data.rotate_[3] ), sizeof( data.rotate_[3] ) );
      file.write( reinterpret_cast<const char*>( &data.rotate_[0] ), sizeof( data.rotate_[0] ) );
      file.write( reinterpret_cast<const char*>( &data.rotate_[1] ), sizeof( data.rotate_[1] ) );
      file.write( reinterpret_cast<const char*>( &data.rotate_[2] ), sizeof( data.rotate_[2] ) );
    }
    file.close();
    return true;
  }

void Pointcloud::trace() {
  printf( "trace: num points = %d \n", size() );
  size_t modulo = 10000;
  int    num = 10, idx = 0;
  for ( int32_t i = 0; i < size() && idx < num; i += modulo, idx++ ) {
    auto& gs = gaussianData_[i];
    printf( "%7d: %6.2f %6.2f %6.2f ", i, gs.xyz_[0], gs.xyz_[1], gs.xyz_[2] );
    printf( "o: %6.2f ", gs.opacity_ );
    printf( "s: %6.2f %6.2f %6.2f ", gs.scale_[0], gs.scale_[1], gs.scale_[2] );
    printf( "r: %6.2f %6.2f %6.2f %6.2f ", gs.rotate_[0], gs.rotate_[1], gs.rotate_[2], gs.rotate_[3] );
    printf( "dc: %6.2f %6.2f %6.2f ", gs.sh_[0].x, gs.sh_[0].x, gs.sh_[0].x );
    printf( "rest: %6.2f %6.2f %6.2f ", gs.sh_[1].y, gs.sh_[1].y, gs.sh_[1].y );
    printf( " %6.2f %6.2f %6.2f ", gs.sh_[2].z, gs.sh_[2].z, gs.sh_[2].z );
    printf( "\n" );
  }
  auto [minBound, maxBound] = getBoundingBox();
  printf( "Bounding Box = ( %6.2f %6.2f %6.2f ) ( %6.2f %6.2f %6.2f ) \n", minBound.x, minBound.y, minBound.z,
          maxBound.x, maxBound.y, maxBound.z );
  fflush( stdout );
}

#if 0

std::vector<uint32_t> Pointcloud::sort( const glm::mat4& modelViewProj, const glm::vec2& nearFar, bool inverse ) const {
  int32_t               numPoints = size();
  int32_t               count     = 0;
  std::vector<float>    depth( numPoints );
  std::vector<uint32_t> index( numPoints );
  const uint32_t        maxValue = std::numeric_limits<uint32_t>::max();
  for ( int32_t i = 0; i < numPoints; ++i ) {
    glm::vec4 p = modelViewProj * gaussianData_[i].xyz_;
    if ( p.z > nearFar.x && p.z < nearFar.y ) {
      depth[count] = p.z;
      index[count] = i;
      count++;
    }
  }
  std::vector<uint32_t> indices( count );
  std::iota( indices.begin(), indices.end(), 0 );
  if ( inverse )
    std::stable_sort( indices.begin(), indices.end(), [&]( uint32_t a, uint32_t b ) { return depth[a] < depth[b]; } );
  else
    std::stable_sort( indices.begin(), indices.end(), [&]( uint32_t a, uint32_t b ) { return depth[a] > depth[b]; } );
  std::vector<uint32_t> sorted( count );
  std::transform( indices.begin(), indices.end(), sorted.begin(), [&]( uint32_t idx ) { return index[idx]; } );
  return sorted;
}

#else 

void radix_sort_64_and_unpack(uint64_t* data, uint64_t* temp, uint32_t* sorted_out, uint32_t count) {
  uint32_t counts[8][256] = {0};
  for (uint32_t i = 0; i < count; ++i) {
    uint64_t val = data[i];
    counts[0][(val >>  0) & 0xFF]++;
    counts[1][(val >>  8) & 0xFF]++;
    counts[2][(val >> 16) & 0xFF]++;
    counts[3][(val >> 24) & 0xFF]++;
    counts[4][(val >> 32) & 0xFF]++;
    counts[5][(val >> 40) & 0xFF]++;
    counts[6][(val >> 48) & 0xFF]++;
    counts[7][(val >> 56) & 0xFF]++;
  }
  uint32_t offsets[8][256] = {0};
  for (int b = 0; b < 8; ++b) {
    offsets[b][0] = 0;
    for (int i = 1; i < 256; ++i) {
      offsets[b][i] = offsets[b][i - 1] + counts[b][i - 1];
    }
  }
  uint64_t* source = data;
  uint64_t* dest = temp;
  for (int b = 0; b < 8; ++b) {
    int shift = b * 8;
    if (b == 7) {
      for (uint32_t i = 0; i < count; ++i) {
        uint8_t val = (source[i] >> shift) & 0xFF;
        sorted_out[offsets[b][val]++] = static_cast<uint32_t>(source[i] & 0xFFFFFFFFULL);
      }
    } else {
      for (uint32_t i = 0; i < count; ++i) {
        uint8_t val = (source[i] >> shift) & 0xFF;
        dest[offsets[b][val]++] = source[i];
      }
      std::swap(source, dest);
    }
  }
}

std::vector<uint32_t> Pointcloud::sort( const glm::mat4& modelViewProj, const glm::vec2& nearFar, bool inverse ) const {
  int32_t numPoints = static_cast<int32_t>(size());
  int num_threads = 1;
#if defined( USE_OPENMP )
  #pragma omp parallel
  {
    #pragma omp single
    num_threads = omp_get_num_threads();
  }
#endif
  std::vector<std::vector<uint64_t>> local_depth_index(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    local_depth_index[t].reserve(numPoints / num_threads + 256);
  }
#if defined( USE_OPENMP )
  #pragma omp parallel for schedule(dynamic, 1024)
#endif
  for (int32_t i = 0; i < numPoints; ++i) {
    int tid = 0;
#if defined( USE_OPENMP )
    tid = omp_get_thread_num();
#endif
    glm::vec4 p = modelViewProj * gaussianData_[i].xyz_;    
    if ( p.z > nearFar.x && p.z < nearFar.y ) {
      uint32_t depth_u32;
      std::memcpy(&depth_u32, &p.z, sizeof(float));
      if (!inverse) {
        depth_u32 = 0xFFFFFFFFU - depth_u32;
      }
      uint64_t packed_val = (static_cast<uint64_t>(depth_u32) << 32) | static_cast<uint32_t>(i);
      local_depth_index[tid].push_back(packed_val);
    }
  }
  uint32_t total_count = 0;
  for (int t = 0; t < num_threads; ++t) {
    total_count += local_depth_index[t].size();
  }
  if (total_count == 0) return std::vector<uint32_t>();
  std::vector<uint64_t> depth_index(total_count);
  std::vector<uint64_t> depth_index_tmp(total_count);
  std::vector<uint32_t> sorted_output(total_count);
  uint32_t current_offset = 0;
  for (int t = 0; t < num_threads; ++t) {
    size_t size = local_depth_index[t].size();
    if (size > 0) {
      std::memcpy(&depth_index[current_offset], local_depth_index[t].data(), size * sizeof(uint64_t));
      current_offset += size;
    }
  }
  radix_sort_64_and_unpack( depth_index.data(), depth_index_tmp.data(), sorted_output.data(), total_count);
  return sorted_output;
}

#endif