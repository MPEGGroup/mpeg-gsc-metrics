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

#include "common.hpp"

struct alignas( 16 ) GaussianData {
  glm::vec4 xyz_;
  glm::vec4 scale_;
  glm::quat rotate_;
  glm::vec4 sh_[16];
  float     opacity_;
};

struct CameraPosition {
  int32_t     index_;
  int32_t     cameraId_;
  glm::vec3   position_;
  glm::quat   orientation_;
  glm::vec2   focal_;
  glm::vec2   center_;
  int32_t     width_;
  int32_t     height_;
  std::string name_;
  void        log() const {
    printf(
        "index = %8d id = %8d xyz= %16.10f %16.10f %16.10f quat = %16.10f %16.10f %16.10f %16.10f focal = %16.10f "
        "%16.10f width = %8d height = %8d center = %16.10f %16.10f%s \n",
        index_, cameraId_, position_.x, position_.y, position_.z, orientation_.x, orientation_.y, orientation_.z,
        orientation_.w, focal_.x, focal_.y, width_, height_, center_.x, center_.y, name_.c_str() );
  }

  void updateView( glm::vec3& eEye, glm::vec3& eCenter, glm::vec3& eUp ) const {
    glm::mat3 rotation = glm::transpose( glm::mat3_cast( orientation_ ) );
    glm::vec3 forward  = rotation * glm::vec3( 0, 0, 1 );
    glm::vec3 right    = rotation * glm::vec3( 1, 0, 0 );
    glm::vec3 up       = rotation * glm::vec3( 0, -1, 0 );
    eEye               = position_;
    eCenter            = position_ + forward;
    eUp                = up;
  }

  glm::vec3 pos() const {
    glm::vec3 eEye, eCenter, eUp;
    updateView( eEye, eCenter, eUp );
    return eEye;
  }
  glm::vec3 view() const {
    glm::vec3 eEye, eCenter, eUp;
    updateView( eEye, eCenter, eUp );
    return eCenter;
  }
  glm::vec3 up() const {
    glm::vec3 eEye, eCenter, eUp;
    updateView( eEye, eCenter, eUp );
    return eUp;
  }
  
  bool operator==( const CameraPosition& other ) const {
    return index_ == other.index_ && cameraId_ == other.cameraId_ && position_ == other.position_ &&
           orientation_ == other.orientation_ && focal_ == other.focal_ && 
           width_ == other.width_ && height_ == other.height_ &&   
           center_ == other.center_ && name_ == other.name_;
  }
  
  bool operator!=( const CameraPosition& other ) const { return !( *this == other ); }
  
  CameraPosition& operator=( const CameraPosition& other ) {
    if ( this != &other ) {
      index_       = other.index_;
      cameraId_    = other.cameraId_;
      position_    = other.position_;
      orientation_ = other.orientation_;
      focal_       = other.focal_;
      center_      = other.center_;
      width_       = other.width_;   
      height_      = other.height_;
      name_        = other.name_;
    }
    return *this;
  }
};

static void traceSH( const GaussianData& gs ) {
  for ( size_t j = 0; j < 16; j++ )
    printf( "SH[%zu] = %12.6f %12.6f %12.6f %12.6f ", j, gs.sh_[j][0], gs.sh_[j][1], gs.sh_[j][2], gs.sh_[j][3] );
  printf( "\n" );
}

class Pointcloud {
 public:
  Pointcloud() {}
  Pointcloud( const std::string& path, int32_t frameIndex ) { read( path, frameIndex ); }
  ~Pointcloud() { reset(); }

  auto                       begin() { return gaussianData_.begin(); }
  auto                       end() { return gaussianData_.end(); }
  auto                       begin() const { return gaussianData_.cbegin(); }
  auto                       end() const { return gaussianData_.cend(); }
  inline GaussianData&       operator[]( int32_t index ) { return gaussianData_[index]; }
  inline const GaussianData& operator[]( int32_t index ) const { return gaussianData_[index]; }
  inline int32_t             size() const { return (int32_t)gaussianData_.size(); }
  inline void*               data() const { return (void*)gaussianData_.data(); }
  inline void                reset() { gaussianData_.clear(); }
  inline const auto&         getCameraPosition() const { return cameraPosition_; }
  inline auto&               getCameraPosition() { return cameraPosition_; }

  std::pair<glm::vec3, glm::vec3> getBoundingBox() {
    glm::vec4 minBound = gaussianData_[0].xyz_;
    glm::vec4 maxBound = gaussianData_[0].xyz_;
    for ( const auto& data : gaussianData_ ) {
      minBound = glm::min( minBound, data.xyz_ );
      maxBound = glm::max( maxBound, data.xyz_ );
    }
    return { minBound.xyz(), maxBound.xyz() };
  }

  bool read( const std::string& path, int32_t frameIndex, bool verbose = false );
  bool write( const std::string& path, int32_t frameIndex );
  void trace() ;
  std::vector<uint32_t> sort( const glm::mat4& modelViewProj, const glm::vec2& nearFar, bool inverse = false ) const;

 private:
  std::vector<GaussianData>   gaussianData_;
  std::vector<CameraPosition> cameraPosition_;
};