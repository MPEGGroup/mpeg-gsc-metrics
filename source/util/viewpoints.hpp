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
#include "camera.hpp"
#include "pointcloud.hpp"

class GLFWwindow;

class Viewpoint {
 public:
  void set( glm::vec3 pos, glm::vec3 view, glm::vec3 up, glm::vec2 focal, int32_t originalWidth, int32_t originalHeight, glm::vec2 center = glm::vec2(0.f) ) {
    pos_ = pos;
    view_ = view;
    up_ = up;
    focal_ = focal;
    originalWidth_ = originalWidth;
    originalHeight_ = originalHeight;
    center_ = center;
  }

  auto pos() const { return pos_; }
  auto view() const { return view_; }
  auto up() const { return up_; }
  auto focal() const { return focal_; }
  auto originalWidth() const { return originalWidth_; }
  auto originalHeight() const { return originalHeight_; }
  auto center() const { return center_; }

  glm::vec2 getAdjustedFocal(int32_t currentHeight) const {
    if (originalHeight_ > 0) {
      float scale = static_cast<float>(currentHeight) / static_cast<float>(originalHeight_);
      return focal_ * scale;
    }
    return focal_;
  }

  void trace() {
    printf( " - pos = %12.8f %12.8f %12.8f ", pos_.x, pos_.y, pos_.z );
    printf( "view = %12.8f %12.8f %12.8f ", view_.x, view_.y, view_.z );
    printf( "up = %12.8f %12.8f %12.8f ", up_.x, up_.y, up_.z );
    printf( "focal = %12.8f %12.8f ", focal_.x, focal_.y );
    printf( "center = %12.8f %12.8f ", center_.x, center_.y );
    printf( "original = %dx%d \n", originalWidth_, originalHeight_ );
  }

 private:
  glm::vec3 pos_;
  glm::vec3 view_;
  glm::vec3 up_;
  glm::vec2 focal_;
  glm::vec2 center_;
  int32_t originalWidth_ = 0;
  int32_t originalHeight_ = 0;
};

class Viewpoints {
 public:
  Viewpoints() { instance_ = this; }
  ~Viewpoints() = default;

  void create( int numPoints, glm::vec3 center, float radius = 5.f, float focal = 2000.f, int32_t originalWidth = 0, int32_t originalHeight = 0 ) {
    viewpoints_.resize( numPoints );
    if ( numPoints == 1 ) {
      viewpoints_[0].set( center + glm::vec3( radius, 0.0f, 0.0f ), center, glm::vec3( 0.0f, 1.0f, 0.0f ),
                          glm::vec2( focal, focal ), originalWidth, originalHeight );
      return;
    }

    const float goldenAngle = glm::pi<float>() * ( 3.0f - glm::sqrt( 5.0f ) );
    for ( int i = 0; i < numPoints; ++i ) {
      float     z   = 1.0f - ( 2.0f * i ) / ( numPoints - 1 );
      float     r   = glm::sqrt( 1.0f - z * z );
      float     phi = i * goldenAngle;
      glm::vec3 pos = center + radius * glm::vec3( r * glm::cos( phi ), r * glm::sin( phi ), z );
      printf( "phi = %f => pos = %f %f %f \n", phi, pos.x, pos.y, pos.z );
      glm::vec3 view    = center;
      glm::vec3 worldUp = glm::vec3( 0.0f, 1.0f, 0.0f );
      glm::vec3 right   = glm::normalize( glm::cross( worldUp, view ) );
      glm::vec3 up      = glm::cross( view, right );
      viewpoints_[i].set( pos, view, up, glm::vec2( focal, focal ), originalWidth, originalHeight );
    }
  }

  void             resize( size_t size ) { viewpoints_.resize( size ); }
  auto             size() const { return viewpoints_.size(); }
  auto             begin() { return viewpoints_.begin(); }
  auto             end() { return viewpoints_.end(); }
  auto             begin() const { return viewpoints_.cbegin(); }
  auto             end() const { return viewpoints_.cend(); }
  Viewpoint&       operator[]( int32_t index ) { return viewpoints_[index]; }
  const Viewpoint& operator[]( int32_t index ) const { return viewpoints_[index]; }

  void write( const std::string& filename ) const {
    std::ofstream file( filename );
    if ( !file.is_open() ) {
      std::cerr << "Error : can't open: " << filename << "\n";
      return;
    }
    file << viewpoints_.size() << "\n";
    for ( const auto& el : viewpoints_ ) {
      file << el.pos().x << " " << el.pos().y << " " << el.pos().z << " ";
      file << el.view().x << " " << el.view().y << " " << el.view().z << " ";
      file << el.up().x << " " << el.up().y << " " << el.up().z << "\n";
      file << el.focal().x << " " << el.focal().y << "\n";
      file << el.center().x << " " << el.center().y << "\n";
      file << el.originalWidth() << " " << el.originalHeight() << "\n";
    }
    file.close();
  }

  void read( const std::string& filename ) {
    std::ifstream file( filename );
    if ( !file.is_open() ) {
      std::cerr << "Error : can't open: " << filename << "\n";
      return;
    }
    int numPoints = 0;
    file >> numPoints;
    viewpoints_.resize( numPoints );
    for ( auto& el : viewpoints_ ) {
      glm::vec3 pos, view, up;
      glm::vec2 focal;
      glm::vec2 center;
      int32_t originalWidth, originalHeight;
      file >> pos.x >> pos.y >> pos.z;
      file >> view.x >> view.y >> view.z;
      file >> up.x >> up.y >> up.z;
      file >> focal.x >> focal.y;
      file >> center.x >> center.y;
      file >> originalWidth >> originalHeight;
      el.set( pos, view, up, focal, originalWidth, originalHeight, center );
    }
    file.close();
  }

  void trace() {
    printf( "Viewpoints[ %4zu ]: \n", size() );
    for ( auto& el : viewpoints_ ) { el.trace(); }
  }

  void        display( const Pointcloud& pc, int width, int height );
  static void mouseCallback( GLFWwindow* window, double xpos, double ypos );
  static void scrollCallback( GLFWwindow* window, double xoffset, double yoffset );
  static void keyCallback( GLFWwindow* window, int key, int scancode, int action, int mods );

 private:
  std::vector<Viewpoint>      viewpoints_;
  std::vector<CameraPosition> cameraPosition_;
  Camera                      camera_;
  static Viewpoints*          instance_;
  float                       lastX_;
  float                       lastY_;
  bool                        firstMouse_     = true;
  int32_t                     viewpointIndex_ = 0;
  int32_t                     cameraIndex_    = 0;
  glm::vec2                   focal_          = glm::vec2( 2000.f, 2000.f );
  glm::vec2                   center_         = glm::vec2( 0.f );
  int32_t                     originalWidth_  = 0;
  int32_t                     originalHeight_ = 0;
};