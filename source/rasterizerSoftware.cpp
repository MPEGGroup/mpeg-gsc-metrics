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
#include "rasterizerSoftware.hpp"
#include "pointcloud.hpp"
#include "image.hpp"
#include "viewpoints.hpp"
#if defined( USE_OPENMP )
#include <omp.h>
#endif

RasterizerSoftware::RasterizerSoftware() {}

RasterizerSoftware::~RasterizerSoftware() {}

#define BLOCK_SIZE 16

template <typename T>
T clamp( T v, T a, T b ) {
  return ( ( v < a ) ? a : ( ( v > b ) ? b : v ) );
}

glm::vec4 computeRadiance( const glm::vec4 sh[16], const glm::vec3 direction, float opacity ) {
  const float C[14] = { 0.28209479177387814f, 0.4886025119029199f,  1.0925484305920792f, -1.0925484305920792f,
                        0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f, -0.5900435899266435f,
                        2.890611442640554f,   -0.4570457994644658f, 0.3731763325901154f, -0.4570457994644658f,
                        1.445305721320277f,   -0.5900435899266435f };
  glm::vec3   v     = glm::normalize( direction );
  float       xx = v.x * v.x, yy = v.y * v.y, zz = v.z * v.z, xy = v.x * v.y, yz = v.y * v.z, xz = v.x * v.z;
  glm::vec3   color = C[0] * sh[0].xyz();
  color += C[1] * ( -v.y * sh[1].xyz() + v.z * sh[2].xyz() - v.x * sh[3].xyz() );  //
  color += C[2] * xy * sh[4].xyz()                                                 //
           + C[3] * yz * sh[5].xyz()                                               //
           + C[4] * ( 2.0f * zz - xx - yy ) * sh[6].xyz()                          //
           + C[5] * xz * sh[7].xyz()                                               //
           + C[6] * ( xx - yy ) * sh[8].xyz();                                     //
  color += C[7] * v.y * ( 3.0f * xx - yy ) * sh[9].xyz()                           //
           + C[8] * xy * v.z * sh[10].xyz()                                        //
           + C[9] * v.y * ( 4.0f * zz - xx - yy ) * sh[11].xyz()                   //
           + C[10] * v.z * ( 2.0f * zz - 3.0f * xx - 3.0f * yy ) * sh[12].xyz()    //
           + C[11] * v.x * ( 4.0f * zz - xx - yy ) * sh[13].xyz()                  //
           + C[12] * v.z * ( xx - yy ) * sh[14].xyz()                              //
           + C[13] * v.x * ( xx - 3.0f * yy ) * sh[15].xyz();                      //
  color += 0.5f;
  color = glm::max( color, 0.0f );
  return glm::vec4( color, opacity );
}

glm::mat3 quatToRotation( glm::quat q ) {
  float xx = q.x * q.x, xy = q.x * q.y, xz = q.x * q.z;
  float yy = q.y * q.y, yz = q.y * q.z, zz = q.z * q.z;
  float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
  return glm::transpose( glm::mat3( 1. - 2. * ( yy + zz ), 2. * ( xy - wz ), 2. * ( xz + wy ), 2. * ( xy + wz ),
                                    1. - 2. * ( xx + zz ), 2. * ( yz - wx ), 2. * ( xz - wy ), 2. * ( yz + wx ),
                                    1. - 2. * ( xx + yy ) ) );
}

glm::mat3 computeCovariance3D( glm::quat quat, glm::vec4 scale ) {
  glm::mat3 R = quatToRotation( quat );
  glm::mat3 M = glm::mat3( R[0] * scale.x, R[1] * scale.y, R[2] * scale.z );
  return M * glm::transpose( M );
}

glm::mat2 computeCovariance2D( glm::vec3 mean, glm::mat3 covar, glm::mat3 viewMat, glm::vec2 focal, glm::vec2 window ) {
  glm::vec2   center = window * 0.5f;
  glm::vec2   limit  = 1.3f * center / focal;
  float       rz     = 1.0f / mean.z;
  float       rz2    = rz * rz;
  glm::vec2   t      = mean.z * clamp( mean.xy() * rz, -limit, limit );
  glm::mat3x2 J      = glm::mat3x2( -focal.x * rz, 0.0, 0.0, -focal.y * rz, -focal.x * t.x * rz2, focal.y * t.y * rz2 );
  glm::mat3   cov    = viewMat * covar * glm::transpose( viewMat );
  cov = glm::mat3( cov[0][0], -cov[0][1], -cov[0][2], -cov[1][0], cov[1][1], cov[1][2], -cov[2][0], cov[2][1],
                   cov[2][2] );
  return J * cov * transpose( J );
}

glm::vec3 computeConic( glm::mat2 covar2d, float& radius ) {
  covar2d[0][0] += 0.3;
  covar2d[1][1] += 0.3;
  float det = std::max( covar2d[0][0] * covar2d[1][1] - covar2d[0][1] * covar2d[1][0], 1e-10f );
  float b   = 0.5f * ( covar2d[0][0] + covar2d[1][1] );
  float s   = sqrt( std::max( b * b - det, 0.1f ) );
  radius    = ceil( 3.0 * sqrt( std::max( b - s, b + s ) ) );
  return glm::vec3( covar2d[1][1] / det, -( covar2d[0][1] + covar2d[1][0] ) / ( 2.0 * det ), covar2d[0][0] / det );
}

template <typename T, int N>
void RasterizerSoftware::render( const Pointcloud&  pc,
                                 const Viewpoint&   viewpoint,
                                 Image<T, N>&       image,
                                 Image<uint8_t, 1>& ocm ) {
  int     width     = image.width();
  int     height    = image.height();
  int32_t numPoints = pc.size();
  int32_t count     = 0;
  auto    focal     = viewpoint.focal();
  auto    center    = viewpoint.center();  
  if ( viewpoint.originalHeight() > 0 ) {
    float scale = static_cast<float>(height) / static_cast<float>(viewpoint.originalHeight());
    focal = focal * scale;
    center = center * scale;
  }
  
  Camera  camera;
  camera.initialize( image.width(), image.height() );
  camera.setLookAt( viewpoint.pos(), viewpoint.view(), viewpoint.up() );

  float     zNear = 0.01f, zFar = 10000000000.0f;
  glm::mat4 projMat;
  if ( center.x > 0.f && center.y > 0.f && focal.x > 0.f && focal.y > 0.f ) {
    float left   = -zNear * center.x / focal.x;
    float right  =  zNear * ( (float)image.width() - center.x ) / focal.x;
    float bottom = -zNear * ( (float)image.height() - center.y ) / focal.y;
    float top    =  zNear * center.y / focal.y;
    projMat = glm::frustum( left, right, bottom, top, zNear, zFar );
  } else {
    float fov = 2 * atan( 1.0f / ( 2 * focal[1] / image.height() ) );
    camera.create( 10.f, fov );
    projMat = camera.projMat();
  }

  glm::mat4 viewMat = camera.viewMat();
  auto      pos     = camera.getPosition();
  auto      nearFar = camera.nearFar();

  // Project sorted splat
  const glm::vec2     scale = glm::vec2( 2.0 / width, 2.0 / height );
  Image<glm::vec4, 1> buffer( width, height, glm::vec4( 0, 0, 0, 1 ) );
  Image<uint8_t, 1>   done( width, height, 0 );
  ocm.fill( 0 );

  const glm::vec2 window( width, height ), halfWindow( width * 0.5f, height * 0.5f );
  const glm::vec2 offsets[4] = { glm::vec2( 1, 1 ), glm::vec2( -1, 1 ), glm::vec2( 1, -1 ), glm::vec2( -1, -1 ) };

  // Sort
  auto modelViewProj = projMat * viewMat;
  auto sorted        = pc.sort( modelViewProj, nearFar, true );

  for ( const auto& i : sorted ) {
    // Vertex shader
    float       radius;
    const auto& gs      = pc[i];
    glm::vec4   t       = viewMat * gs.xyz_;
    glm::vec4   p       = projMat * t;
    glm::vec3   ndc     = glm::vec3( p.x / p.w, p.y / p.w, p.z / p.w );
    glm::mat3   covar3d = computeCovariance3D( gs.rotate_, gs.scale_ );
    glm::mat2   covar2d = computeCovariance2D( t.xyz(), covar3d, glm::mat3( viewMat ), focal, window );
    glm::vec3   conic   = computeConic( covar2d, radius );
    glm::vec4   color   = computeRadiance( gs.sh_, gs.xyz_.xyz() - pos, gs.opacity_ );
    glm::vec2   screen  = halfWindow * ndc.xy() + halfWindow;

    // Frament shader
    glm::vec3 P[4];
    float     z = ndc.z * 0.5f + 0.5f;
    for ( size_t j = 0; j < 4; j++ ) P[j] = glm::vec3( screen + offsets[j].xy() * radius, z );
    int32_t minU = std::max( (int)( std::floor( std::min( { P[0].x, P[1].x, P[2].x, P[3].x } ) ) ), 0 );
    int32_t minV = std::max( (int)( std::floor( std::min( { P[0].y, P[1].y, P[2].y, P[3].y } ) ) ), 0 );
    int32_t maxU = std::min( (int)( std::ceil( std::max( { P[0].x, P[1].x, P[2].x, P[3].x } ) ) ), width - 1 );
    int32_t maxV = std::min( (int)( std::ceil( std::max( { P[0].y, P[1].y, P[2].y, P[3].y } ) ) ), height - 1 );

    // Rendering
    for ( int32_t v = minV; v <= maxV; ++v ) {
      for ( int32_t u = minU; u <= maxU; ++u ) {
        if ( done[0].get( u, v ) == 0 ) {
          glm::vec2 d     = screen - glm::vec2( u + 0.5, v + 0.5 );
          float     sigma = 0.5 * ( conic.x * d.x * d.x + conic.z * d.y * d.y ) - conic.y * d.x * d.y;
          if ( sigma < 0.0f ) continue;
          float alpha = std::min( static_cast<float>( color.w * std::exp( -sigma ) ), 0.999f );
          if ( alpha >= ( 1.0f / 255.0f ) ) {
            if ( alpha >= ( 10.0f / 255.0f ) ) ocm[0].get( u, v ) = 255;
            auto& value  = buffer[0].get( u, v );
            float t      = value.w;
            float test_t = t * ( 1.0 - alpha );
            if ( test_t <= 1e-4f ) {
              done[0].get( u, v ) = 1;
            } else {
              value = glm::vec4( value.xyz() + color.xyz() * alpha * t, test_t );
            }
          }
        }
      }
    }
  }
  image.from( buffer );
}

template void RasterizerSoftware::render<uint8_t>( Pointcloud const&,
                                                   const Viewpoint&,
                                                   Image<uint8_t>&,
                                                   Image<uint8_t, 1>& );
template void RasterizerSoftware::render<uint16_t>( Pointcloud const&,
                                                    const Viewpoint&,
                                                    Image<uint16_t>&,
                                                    Image<uint8_t, 1>& );
template void RasterizerSoftware::render<float>( Pointcloud const&,
                                                 const Viewpoint&,
                                                 Image<float>&,
                                                 Image<uint8_t, 1>& );