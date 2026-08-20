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
#if defined(__GNUC__) || defined(__clang__)
  #pragma GCC target("fma")
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#elif defined(__APPLE__) && defined(__arm64__)
inline __m128 _mm_fmadd_ps(__m128 a, __m128 b, __m128 c) {
    return _mm_add_ps(_mm_mul_ps(a, b), c);
}
#endif
#include <smmintrin.h> 

inline float deterministic_sqrt(float x) {
  __m128 v = _mm_set_ss(x);
  return _mm_cvtss_f32(_mm_sqrt_ss(v));
}

inline float deterministic_ceil(float x) {
  __m128 v = _mm_set_ss(x);
  return _mm_cvtss_f32(_mm_round_ss(v, v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
}

inline float deterministic_floor(float x) {
  __m128 v = _mm_set_ss(x);
  return _mm_cvtss_f32(_mm_round_ss(v, v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
}

inline float deterministic_log(float x) {
  if (x <= 0.0f) return -87.3365f;  
  uint32_t ix;
  std::memcpy(&ix, &x, sizeof(float));
  int iexp = static_cast<int>((ix >> 23) & 0xFF) - 126;    
  uint32_t mx = (ix & 0x007FFFFF) | 0x3F000000;
  float luxury_u;
  std::memcpy(&luxury_u, &mx, sizeof(float));  
  float s = (luxury_u - 1.0f) / (luxury_u + 1.0f);
  float s2 = s * s;
  float r = s * (2.0000000f + s2 * (0.6666667f + s2 * (0.4000000f + s2 * 0.2857143f)));  
  return r + static_cast<float>(iexp) * 0.69314718f;
}

inline glm::vec3 deterministic_normalize(const glm::vec3& v) {
  float lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
  if (lenSq > 0.0f) {
    return v * (1.0f / deterministic_sqrt(lenSq));
  }
  return v;
}

RasterizerSoftware::RasterizerSoftware() {}

RasterizerSoftware::~RasterizerSoftware() {}

template <typename T>
T clamp( T v, T a, T b ) {
  return ( ( v < a ) ? a : ( ( v > b ) ? b : v ) );
}

glm::vec4 computeRadiance( const glm::vec4 sh[16], const glm::vec3 direction, float opacity ) {
  const float C[14] = { 0.28209479177387814f, 0.4886025119029199f,  1.0925484305920792f, -1.0925484305920792f,
                        0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f, -0.5900435899266435f,
                        2.890611442640554f,   -0.4570457994644658f, 0.3731763325901154f, -0.4570457994644658f,
                        1.445305721320277f,   -0.5900435899266435f };
  glm::vec3   v     = deterministic_normalize( direction );  
  float       xx   = v.x * v.x, yy = v.y * v.y, zz = v.z * v.z, xy = v.x * v.y, yz = v.y * v.z, xz = v.x * v.z;
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
  return glm::transpose( glm::mat3( 1.f - 2.f * ( yy + zz ), 2.f * ( xy - wz ), 2.f * ( xz + wy ), 2.f * ( xy + wz ),
                                    1.f - 2.f * ( xx + zz ), 2.f * ( yz - wx ), 2.f * ( xz - wy ), 2.f * ( yz + wx ),
                                    1.f - 2.f * ( xx + yy ) ) );
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
  glm::mat3x2 J      = glm::mat3x2( -focal.x * rz, 0.0f, 0.0f, -focal.y * rz, -focal.x * t.x * rz2, focal.y * t.y * rz2 );
  glm::mat3   cov    = viewMat * covar * glm::transpose( viewMat );
  cov = glm::mat3( cov[0][0], -cov[0][1], -cov[0][2], -cov[1][0], cov[1][1], cov[1][2], -cov[2][0], cov[2][1],
                   cov[2][2] );
  return J * cov * transpose( J );
}

glm::vec3 computeConic( glm::mat2 covar2d, float& radius ) {
  covar2d[0][0] += 0.3f;
  covar2d[1][1] += 0.3f;
  float det = std::max( covar2d[0][0] * covar2d[1][1] - covar2d[0][1] * covar2d[1][0], 1e-10f );
  float b   = 0.5f * ( covar2d[0][0] + covar2d[1][1] );
  float s   = deterministic_sqrt( std::max( b * b - det, 0.1f ) );
  radius    = deterministic_ceil( 3.0f * deterministic_sqrt( std::max( b - s, b + s ) ) );  
  return glm::vec3( covar2d[1][1] / det, -( covar2d[0][1] + covar2d[1][0] ) / ( 2.0f * det ), covar2d[0][0] / det );
}

inline __m128 deterministic_exp_sse(__m128 x) {
  __m128  ln2_inv = _mm_set1_ps(1.4426950408f);
  __m128  ln2_hi  = _mm_set1_ps(0.6931457519f);
  __m128  ln2_lo  = _mm_set1_ps(1.4286068203e-06f);
  __m128i n       = _mm_cvtps_epi32(_mm_mul_ps(x, ln2_inv));
  __m128  fn      = _mm_cvtepi32_ps(n);
  __m128  f       = _mm_fnmadd_ps(fn, ln2_hi, x);  
  f               = _mm_fnmadd_ps(fn, ln2_lo, f);    
  __m128 y        = _mm_set1_ps(0.008333333f);
  y               = _mm_fmadd_ps(y, f, _mm_set1_ps(0.041666667f));
  y               = _mm_fmadd_ps(y, f, _mm_set1_ps(0.166666667f));
  y               = _mm_fmadd_ps(y, f, _mm_set1_ps(0.5f));
  y               = _mm_fmadd_ps(y, f, _mm_set1_ps(1.0f));
  y               = _mm_fmadd_ps(y, f, _mm_set1_ps(1.0f));    
  __m128i i       = _mm_slli_epi32(_mm_add_epi32(n, _mm_set1_epi32(127)), 23);
  return _mm_mul_ps(y, _mm_castsi128_ps(i));
}

template <typename T, int N>
void RasterizerSoftware::render( const Pointcloud&  pc,
                                 const Viewpoint&   viewpoint,
                                 Image<T, N>&       image,
                                 Image<uint8_t, 1>& ocm ) {
  const int32_t width  = image.width();
  const int32_t height = image.height();
  auto    focal        = viewpoint.focal();  
  auto    center       = viewpoint.center();  
  if ( viewpoint.originalHeight() > 0 ) {
    float scale = static_cast<float>(height) / static_cast<float>(viewpoint.originalHeight());
    focal       = focal * scale;
    center      = center * scale;
  }
  float atan_input         = 1.0f / ( 2.f * focal[1] / static_cast<float>(height) );
  float atan_x             = std::clamp(atan_input, -1.0f, 1.0f); 
  float atan_x2            = atan_x * atan_x;
  float deterministic_atan = atan_x * (0.9998660f + atan_x2 * (-0.3302995f + atan_x2 * (0.1801410f - atan_x2 * 0.0851330f)));
  float fov                = 2.f * deterministic_atan;

  Camera  camera;
  camera.initialize( image.width(), image.height() );
  camera.setLookAt( viewpoint.pos(), viewpoint.view(), viewpoint.up() );
  glm::vec3 deterministic_dir = viewpoint.view() - viewpoint.pos();
  glm::vec3 f = deterministic_normalize(deterministic_dir);
  glm::vec3 s = deterministic_normalize(glm::cross(f, viewpoint.up()));
  glm::vec3 u = glm::cross(s, f);
  
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

  glm::mat4 viewMat(1.0f);
  viewMat[0][0] = s.x;  viewMat[1][0] = s.y;  viewMat[2][0] = s.z;
  viewMat[0][1] = u.x;  viewMat[1][1] = u.y;  viewMat[2][1] = u.z;
  viewMat[0][2] = -f.x; viewMat[1][2] = -f.y; viewMat[2][2] = -f.z;
  viewMat[3][0] = -glm::dot(s, viewpoint.pos());
  viewMat[3][1] = -glm::dot(u, viewpoint.pos());
  viewMat[3][2] =  glm::dot(f, viewpoint.pos());

  auto      pos     = viewpoint.pos();
  auto      nearFar = camera.nearFar();
  float n           = nearFar.x;
  float f_dist      = nearFar.y;
  float tanHalfFovY = static_cast<float>(height) / (2.f * focal[1]);
  float tanHalfFovX = static_cast<float>(width) / (2.f * focal[0]);
  const glm::vec2 window( width, height ), halfWindow( width * 0.5f, height * 0.5f );
  ocm.fill( 0 );

  // Sort
  auto modelViewProj = projMat * viewMat;
  auto sorted        = pc.sort( modelViewProj, nearFar, true );

  // Vertex
  const int32_t TILE_SIZE = 64; 
  int32_t gridWidth  = (width  + TILE_SIZE - 1) / TILE_SIZE;
  int32_t gridHeight = (height + TILE_SIZE - 1) / TILE_SIZE;
  int32_t totalTiles = gridHeight * gridWidth;  
  projected_.resize(sorted.size());
  tileBins_.resize(totalTiles);
  for (auto& bin : tileBins_) {
    bin.clear(); 
  }     
#if defined( USE_OPENMP )
  #pragma omp parallel for schedule(dynamic, 256)
#endif
  for (int32_t idx    = 0; idx < (int32_t)sorted.size(); ++idx) {
    float radius;
    int32_t i         = sorted[idx];
    const auto& gs    = pc[i];
    glm::vec4 t       = viewMat * gs.xyz_;
    glm::vec4 p       = projMat * t;
    glm::vec3 ndc     = glm::vec3( p.x / p.w, p.y / p.w, p.z / p.w );    
    glm::mat3 covar3d = computeCovariance3D( gs.rotate_, gs.scale_ );
    glm::mat2 covar2d = computeCovariance2D( t.xyz(), covar3d, glm::mat3( viewMat ), focal, window );
    glm::vec3 conic   = computeConic( covar2d, radius );
    glm::vec4 color   = computeRadiance( gs.sh_, gs.xyz_.xyz() - pos, gs.opacity_ );
    glm::vec2 screen  = halfWindow * ndc.xy() + halfWindow;
    glm::ivec2 minUV;
    minUV.x = std::max(static_cast<int32_t>(deterministic_floor(screen.x - radius)), 0);
    minUV.y = std::max(static_cast<int32_t>(deterministic_floor(screen.y - radius)), 0);
    glm::ivec2 maxUV;
    maxUV.x = std::min(static_cast<int32_t>(deterministic_ceil(screen.x + radius)), width - 1);
    maxUV.y = std::min(static_cast<int32_t>(deterministic_ceil(screen.y + radius)), height - 1);
    float maxSigma    = deterministic_log(255.0f * color.w) + 0.05f;
    projected_[idx]   = { screen, conic, color, maxSigma, minUV, maxUV };   
  }

  for (int32_t idx = 0; idx < (int32_t)sorted.size(); ++idx) {
    const auto& s = projected_[idx];
    auto minUV = s.minUV_ / TILE_SIZE;
    auto maxUV = s.maxUV_ / TILE_SIZE;
    for (int32_t ty = minUV.y; ty <= maxUV.y; ++ty) {
      for (int32_t tx = minUV.x; tx <= maxUV.x; ++tx) {
        tileBins_[ty * gridWidth + tx].push_back(idx);
      }
    }
  }
  std::vector<int32_t> tileBinsFlat;
  std::vector<int32_t> tileOffsets(totalTiles + 1, 0);
  int32_t offset = 0;
  for (int32_t tilesIdx = 0; tilesIdx < totalTiles; ++tilesIdx) {
    tileOffsets[tilesIdx] = offset;
    offset += tileBins_[tilesIdx].size();
  }
  tileOffsets[totalTiles] = offset;
  tileBinsFlat.resize(offset);
  for (int32_t tilesIdx = 0; tilesIdx < totalTiles; ++tilesIdx) {
    const auto& bin = tileBins_[tilesIdx];
    if (!bin.empty()) {
      std::memcpy(&tileBinsFlat[tileOffsets[tilesIdx]], bin.data(), bin.size() * sizeof(int32_t));
    }
  }

  // Fragment
#if defined( USE_OPENMP )
  #pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int32_t tileIdx = 0; tileIdx < totalTiles; ++tileIdx) {
    int32_t ty           = tileIdx / gridWidth;
    int32_t tx           = tileIdx % gridWidth;
    int32_t startIdx     = tileOffsets[tileIdx];
    int32_t endIdx       = tileOffsets[tileIdx + 1]; 
    if (startIdx == endIdx) continue; 
    int32_t tileMinU     = tx * TILE_SIZE;
    int32_t tileMinV     = ty * TILE_SIZE;
    int32_t tileMaxU     = std::min(tileMinU + TILE_SIZE - 1, width  - 1);
    int32_t tileMaxV     = std::min(tileMinV + TILE_SIZE - 1, height - 1);
    int32_t tileWidth    = tileMaxU - tileMinU + 1;
    int32_t tileHeight   = tileMaxV - tileMinV + 1;
    int32_t activePixels = tileWidth * tileHeight;

    // Local buffer
    alignas(16) float localR[TILE_SIZE][TILE_SIZE] = {};
    alignas(16) float localG[TILE_SIZE][TILE_SIZE] = {};
    alignas(16) float localB[TILE_SIZE][TILE_SIZE] = {};
    alignas(16) float localA[TILE_SIZE][TILE_SIZE];
    std::fill(&localA[0][0], &localA[0][0] + TILE_SIZE * TILE_SIZE, 1.0f);
    alignas(16) uint8_t localDone[TILE_SIZE][TILE_SIZE] = {};
    alignas(16) uint8_t localOcm[TILE_SIZE][TILE_SIZE]  = {};

    // Loop for all splats
    for (int32_t iter = startIdx; iter < endIdx; ++iter) {
      if (activePixels <= 0) break;
      int32_t idx       = tileBinsFlat[iter];
      const auto& s     = projected_[idx];
      int32_t workMinU  = std::max(tileMinU, s.minUV_.x);
      int32_t workMaxU  = std::min(tileMaxU, s.maxUV_.x);
      int32_t workMinV  = std::max(tileMinV, s.minUV_.y);
      int32_t workMaxV  = std::min(tileMaxV, s.maxUV_.y);
      if (workMinU > workMaxU || workMinV > workMaxV) continue;
      int32_t localMinU = workMinU - tileMinU;
      int32_t localMaxU = workMaxU - tileMinU;
      __m128 vConicX    = _mm_set1_ps(s.conic_.x);
      __m128 vConicY    = _mm_set1_ps(s.conic_.y);
      __m128 vConicZ    = _mm_set1_ps(s.conic_.z);
      __m128 vCw        = _mm_set1_ps(s.color_.w);
      __m128 vMaxSig    = _mm_set1_ps(s.maxSigma_);
      __m128 vHalf      = _mm_set1_ps(0.5f);
      __m128 vZero      = _mm_setzero_ps();
      float dy          = s.screen_.y - (workMinV + 0.5f);
      float dxStart     = s.screen_.x - (workMinU + 0.5f);

      // Loop V
      for (int32_t v = workMinV; v <= workMaxV; ++v, dy -= 1.0f) {
        int32_t lv                = v - tileMinV; 
        const float conicZDy2     = s.conic_.z * dy * dy;
        uint8_t* __restrict pDone = &localDone[lv][0];
        uint8_t* __restrict pOcm  = &localOcm[lv][0];
        __m128 vDx                = _mm_setr_ps(dxStart, dxStart - 1.0f, dxStart - 2.0f, dxStart - 3.0f);
        __m128 vStep              = _mm_set1_ps(4.0f);
        __m128 vDy                = _mm_set1_ps(dy);
        __m128 vConicZ_dy2        = _mm_mul_ps(vConicZ, _mm_mul_ps(vDy, vDy));

        // Loop U
        for (int32_t lu = localMinU; lu <= localMaxU; lu += 4) {
          uint32_t doneVal;
          std::memcpy(&doneVal, &pDone[lu], sizeof(uint32_t));
          if (doneVal == 0x01010101) {
            vDx = _mm_sub_ps(vDx, vStep);
            continue;
          }
          // Rasterization
          __m128 dx2            = _mm_mul_ps(vDx, vDx);
          __m128 term1          = _mm_add_ps(_mm_mul_ps(vConicX, dx2), vConicZ_dy2);
          __m128 term2          = _mm_mul_ps(_mm_mul_ps(vConicY, vDx), vDy);
          __m128 vSigma         = _mm_sub_ps(_mm_mul_ps(vHalf, term1), term2);
          __m128 vMask          = _mm_and_ps(_mm_cmpgt_ps(vSigma, _mm_setzero_ps()), 
                                             _mm_cmple_ps(vSigma, vMaxSig));
          __m128 vAlpha         = deterministic_exp_sse(_mm_sub_ps(_mm_setzero_ps(), vSigma));
          vAlpha                = _mm_mul_ps(vAlpha, vCw);
          vAlpha                = _mm_min_ps(vAlpha, _mm_set1_ps(0.999f));
          __m128 vInv255        = _mm_set1_ps(1.0f / 255.0f);
          __m128 vAlphaValid    = _mm_cmpge_ps(vAlpha, vInv255);
          __m128 vCombinedMask  = _mm_and_ps(vMask, vAlphaValid);
          if (_mm_movemask_ps(vCombinedMask) == 0) {
            vDx = _mm_sub_ps(vDx, vStep);
            continue;
          }
          float* __restrict pR  = &localR[lv][0];
          float* __restrict pG  = &localG[lv][0];
          float* __restrict pB  = &localB[lv][0];
          float* __restrict pA  = &localA[lv][0];
          __m128 rowR           = _mm_loadu_ps(&pR[lu]);  
          __m128 rowG           = _mm_loadu_ps(&pG[lu]);  
          __m128 rowB           = _mm_loadu_ps(&pB[lu]); 
          __m128 rowA           = _mm_loadu_ps(&pA[lu]); 
          __m128 vTestT         = _mm_fnmadd_ps(vAlpha, rowA, rowA);
          __m128 vAlphaT        = _mm_mul_ps(vAlpha, rowA);    
          __m128 vNewR          = _mm_add_ps(rowR, _mm_mul_ps(_mm_set1_ps(s.color_.x), vAlphaT));
          __m128 vNewG          = _mm_add_ps(rowG, _mm_mul_ps(_mm_set1_ps(s.color_.y), vAlphaT));
          __m128 vNewB          = _mm_add_ps(rowB, _mm_mul_ps(_mm_set1_ps(s.color_.z), vAlphaT));

          // Store results
          alignas(16) float alphas[4];
          alignas(16) float masks[4];
          alignas(16) float finalR[4];
          alignas(16) float finalG[4];
          alignas(16) float finalB[4];
          alignas(16) float finalA[4];
          _mm_storeu_ps(masks,  vMask);
          _mm_storeu_ps(alphas, vAlpha);
          _mm_storeu_ps(finalR, vNewR);
          _mm_storeu_ps(finalG, vNewG);
          _mm_storeu_ps(finalB, vNewB);
          _mm_storeu_ps(finalA, vTestT); 
          for (int32_t i = 0; i < 4 && (lu + i) <= localMaxU; ++i) {
            int32_t currentLu = lu + i;
            if (masks[i] <= 0.0f || alphas[i] < (1.0f / 255.0f) || pDone[currentLu] == 1) continue;
            if (alphas[i] >= 10.0f / 255.0f) pOcm[currentLu] = 255;
            if (finalA[i] <= 1e-4f) {
              pDone[currentLu] = 1;
              activePixels--;
            } else {
              pR[currentLu] = finalR[i];
              pG[currentLu] = finalG[i];
              pB[currentLu] = finalB[i];
              pA[currentLu] = finalA[i];
            }
          }
          vDx = _mm_sub_ps(vDx, vStep);
        }
      }
    }

    // Copy to image and ocm  
    auto* pO = ocm[0].data();
    auto* pR = image[0].data();
    auto* pG = image[1].data();
    auto* pB = image[2].data();   
    T* pA = nullptr;
    if constexpr ( N == 4 ) {
      pA = image[3].data();
    }
    for (int32_t lv = 0, gv = tileMinV * width; lv < tileHeight; ++lv, gv += width) {
      for (int32_t lu = 0, gu = gv + tileMinU; lu < tileWidth; ++lu, ++gu) {
        pO[gu] = localOcm[lv][lu];
        if constexpr ( std::is_same_v<T, uint8_t> ) {
          const float scale = 255.f;
          pR[gu] = static_cast<T>( std::clamp( std::round( localR[lv][lu] * scale ), 0.f, 255.f ) );
          pG[gu] = static_cast<T>( std::clamp( std::round( localG[lv][lu] * scale ), 0.f, 255.f ) );
          pB[gu] = static_cast<T>( std::clamp( std::round( localB[lv][lu] * scale ), 0.f, 255.f ) );
          if constexpr ( N == 4 ) {
            pA[gu] = static_cast<T>( std::round( std::clamp( localA[lv][lu] * scale + 0.5f, 0.f, 255.f ) ) );
          }
        } else if constexpr ( std::is_same_v<T, uint16_t> ) {
          const float scale = 65535.f;
          pR[gu] = static_cast<T>( std::clamp( std::round( localR[lv][lu] * scale ), 0.f, 65535.f ) );
          pG[gu] = static_cast<T>( std::clamp( std::round( localG[lv][lu] * scale ), 0.f, 65535.f ) );
          pB[gu] = static_cast<T>( std::clamp( std::round( localB[lv][lu] * scale ), 0.f, 65535.f ) );
          if constexpr ( N == 4 ) {
            pA[gu] = static_cast<T>( std::clamp( std::round( localA[lv][lu] * scale ), 0.f, 65535.f ) );
          }
        } else if constexpr ( std::is_same_v<T, float> ) {
          pR[gu] = std::clamp( localR[lv][lu], 0.f, 1.f );
          pG[gu] = std::clamp( localG[lv][lu], 0.f, 1.f );
          pB[gu] = std::clamp( localB[lv][lu], 0.f, 1.f );
          if constexpr ( N == 4 ) { 
            pA[gu] = std::clamp( localA[lv][lu], 0.f, 1.f ); 
          }
        }
      }
    }
  }
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
