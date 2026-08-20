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
#include "metric.hpp"
#include "pointcloud.hpp"
#include "image.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#if defined( USE_OPENMP )
#include <omp.h>
#endif

#include "xSSIM.h"

#include "xColorSpace.h"
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

Metric::Metric( bool computeOm, bool computeSsim, bool computeIvssimMetrics, bool computeQmiv ) :
    computeOmMetrics_( computeOm ),
    computeSsimMetrics_( computeSsim ),
    computeIvssimMetrics_( computeIvssimMetrics ),
    computeQmiv_( computeQmiv ) {}

Metric::~Metric() { results_.clear(); }

inline glm::dvec3 log10( const glm::dvec3& v ) {
  return glm::dvec3( std::log10( v.x ), std::log10( v.y ), std::log10( v.z ) );
};

// Generate Gaussian window
std::vector<std::vector<double>> generateGaussianWindow( int windowSize, double sigma ) {
  std::vector<std::vector<double>> window( windowSize, std::vector<double>( windowSize ) );
  double                           sum      = 0.0;
  int                              halfSize = windowSize / 2;

  for ( int i = -halfSize; i <= halfSize; i++ ) {
    for ( int j = -halfSize; j <= halfSize; j++ ) {
      double value                       = deterministic_exp( -( i * i + j * j ) / ( 2 * sigma * sigma ) );
      window[i + halfSize][j + halfSize] = value;
      sum += value;
    }
  }

  // Normalize
  for ( auto& row : window ) {
    for ( auto& val : row ) { val /= sum; }
  }

  return window;
}

// Preserve original maskless version
template <typename T, int32_t N>
void Metric::compute( const Image<T, N>& imageA, const Image<T, N>& imageB, const int32_t frameIndex ) {
  Image<uint8_t, 1> emptyMask( imageA.width(), imageA.height(), 1 );
  compute( imageA, imageB, emptyMask, frameIndex );
}

// Implementation of compute function with mask
template <typename T, int32_t N>
void Metric::compute( const Image<T, N>&       imageA,
                      const Image<T, N>&       imageB,
                      const Image<uint8_t, 1>& mask,
                      const int32_t            frameIndex ) {
  Results res;
  res.frameIndex_ = frameIndex;
  if ( imageA.size() != imageB.size() || ( computeOmMetrics_ && imageA.size() != mask.size() ) ) {
    printf( "Error: Images or mask are not the same size\n" );
    return;
  } else {
    double peak = 1.0;
    if constexpr ( std::is_integral_v<T> ) {
      peak = static_cast<double>( std::numeric_limits<T>::max() );
    } else if constexpr ( std::is_floating_point_v<T> ) {
      peak = 1.0;
    }
    const double peak2  = peak * peak;
    const auto   size   = imageA.size();
    const int    width  = imageA.width();
    const int    height = imageA.height();

    // Initialize statistical variables
    int32_t    validPixelCount = 0;
    glm::dvec3 mseRgbOm        = { 0.0, 0.0, 0.0 };
    glm::dvec3 mseYuvOm        = { 0.0, 0.0, 0.0 }; 
    double mseRgbX = 0.0, mseRgbY = 0.0, mseRgbZ = 0.0;
    double mseYuvX = 0.0, mseYuvY = 0.0, mseYuvZ = 0.0;
    double mseRgbOmX = 0.0, mseRgbOmY = 0.0, mseRgbOmZ = 0.0;
    double mseYuvOmX = 0.0, mseYuvOmY = 0.0, mseYuvOmZ = 0.0;
    int32_t localValidPixelCount = 0;

    // Original MSE and OM-MSE computation
#if defined( USE_OPENMP )
    #pragma omp parallel for reduction(+:mseRgbX, mseRgbY, mseRgbZ, mseYuvX, mseYuvY, mseYuvZ, \
                                         mseRgbOmX, mseRgbOmY, mseRgbOmZ, mseYuvOmX, mseYuvOmY, mseYuvOmZ, \
                                         localValidPixelCount) schedule(static)
#endif
    for ( int32_t index = 0; index < size; index++ ) {
      const glm::dvec3 rgb = imageA.getD( index ) - imageB.getD( index );
      const glm::dvec3 yuv = computeQmiv_ ? imageA.getYuvQmivD( index ) - imageB.getYuvQmivD( index )
                                          : imageA.getYuvD( index ) - imageB.getYuvD( index );
      mseRgbX += rgb.x * rgb.x;
      mseRgbY += rgb.y * rgb.y;
      mseRgbZ += rgb.z * rgb.z;
      
      mseYuvX += yuv.x * yuv.x;
      mseYuvY += yuv.y * yuv.y;
      mseYuvZ += yuv.z * yuv.z;

      // Use mask to detect valid pixels
      if ( computeOmMetrics_ && mask[0].get( index ) != 0 ) {
        localValidPixelCount++;
        mseRgbOmX += rgb.x * rgb.x;
        mseRgbOmY += rgb.y * rgb.y;
        mseRgbOmZ += rgb.z * rgb.z;
        
        mseYuvOmX += yuv.x * yuv.x;
        mseYuvOmY += yuv.y * yuv.y;
        mseYuvOmZ += yuv.z * yuv.z;
      }
    }
    res.mseRgb_     = { mseRgbX, mseRgbY, mseRgbZ };
    res.mseYuv_     = { mseYuvX, mseYuvY, mseYuvZ };
    validPixelCount = localValidPixelCount;
    mseRgbOm        = { mseRgbOmX, mseRgbOmY, mseRgbOmZ };
    mseYuvOm        = { mseYuvOmX, mseYuvOmY, mseYuvOmZ };

    // Computation PSNR full image
    res.mseRgb_ /= size;
    res.mseYuv_ /= size;
    res.psnrRgb_ = glm::min( glm::dvec3( 999.99 ), 10.0 * log10( ( peak2 ) / res.mseRgb_ ) );
    res.psnrYuv_ = glm::min( glm::dvec3( 999.99 ), 10.0 * log10( ( peak2 ) / res.mseYuv_ ) );

    // Compute RGB combined channel PSNR (QMIV = 1:1:1 MPEG = 1:1:1)
    if ( computeQmiv_ ) {
      res.psnrRgbAvg_ = ( res.psnrRgb_.x + res.psnrRgb_.y + res.psnrRgb_.z ) / 3.0;
    } else {
      double mseRgbAvg = ( res.mseRgb_.x + res.mseRgb_.y + res.mseRgb_.z ) / 3.0;
      res.psnrRgbAvg_  = ( mseRgbAvg > 0 ) ? std::min( 999.99, 10.0 * std::log10( ( peak2 ) / mseRgbAvg ) ) : 999.99;
    }

    // Compute YUV combined channel PSNR (QMIV = 4:1:1 MPEG = 1:1:1)
    if ( computeQmiv_ ) {
      res.psnrYuvAvg_ = ( 4.0 * res.psnrYuv_.x + res.psnrYuv_.y + res.psnrYuv_.z ) / 6.0;
    } else {
      double mseYuvAvg = ( res.mseYuv_.x + res.mseYuv_.y + res.mseYuv_.z ) / 3.0;
      res.psnrYuvAvg_  = ( mseYuvAvg > 0 ) ? std::min( 999.99, 10.0 * std::log10( ( peak2 ) / mseYuvAvg ) ) : 999.99;
    }

    // Compute OM-PSNR for valid pixels
    if ( computeOmMetrics_ && validPixelCount > 0 ) {
      // Compute RGB MSE
      res.mseRgbOm_  = mseRgbOm / static_cast<double>( validPixelCount );
      res.psnrRgbOm_ = glm::min( glm::dvec3( 999.99 ), 10.0 * log10( ( peak2 ) / ( res.mseRgbOm_ ) ) );

      // Compute YUV MSE
      res.mseYuvOm_  = mseYuvOm / static_cast<double>( validPixelCount );
      res.psnrYuvOm_ = glm::min( glm::dvec3( 999.99 ), 10.0 * log10( ( peak2 ) / ( res.mseYuvOm_ ) ) );

      // Compute RGB combined channel OM-PSNR (QMIV = 1:1:1 MPEG = 1:1:1)
      if ( computeQmiv_ ) {
        res.psnrRgbOmAvg_ = ( res.psnrRgbOm_.x + res.psnrRgbOm_.y + res.psnrRgbOm_.z ) / 3.0;
      } else {
        double mseRgbOmAvg = ( res.mseRgbOm_.x + res.mseRgbOm_.y + res.mseRgbOm_.z ) / 3.0;
        res.psnrRgbOmAvg_ =
            ( mseRgbOmAvg > 0 ) ? std::min( 999.99, 10.0 * std::log10( ( peak2 ) / mseRgbOmAvg ) ) : 999.99;
      }

      // Compute YUV combined channel OM-PSNR (QMIV = 4:1:1 MPEG = 1:1:1)
      if ( computeQmiv_ ) {
        res.psnrYuvOmAvg_ = ( 4.0 * res.psnrYuvOm_.x + res.psnrYuvOm_.y + res.psnrYuvOm_.z ) / 6.0;
      } else {
        double mseYuvOmAvg = ( res.mseYuvOm_.x + res.mseYuvOm_.y + res.mseYuvOm_.z ) / 3.0;
        res.psnrYuvOmAvg_ =
            ( mseYuvOmAvg > 0 ) ? std::min( 999.99, 10.0 * std::log10( ( peak2 ) / mseYuvOmAvg ) ) : 999.99;
      }

      // Compute occupancy rate
      res.occupancyRate_ = 100.0 * validPixelCount / size;
    }

    // Compute SSIM and OM-SSIM metrics
    if ( computeSsimMetrics_ ) {
      // Create YUV images
      if ( yuvA_.width() != width || yuvA_.height() != height ) {
        yuvA_.resize( width, height );
        yuvB_.resize( width, height );
      }      
      if (computeQmiv_){
        for ( int32_t index = 0; index < size; index++ ) {
          yuvA_.setF( index, imageA.getYuvQmivF( index ) );
          yuvB_.setF( index, imageB.getYuvQmivF( index ) );
        }
      } else {
        for ( int32_t index = 0; index < size; index++ ) {
          yuvA_.setF( index, imageA.getYuvF( index ) );
          yuvB_.setF( index, imageB.getYuvF( index ) );
        }
      }

      // Compute SSIM for each channel      
      for ( int c = 0; c < 3; c++ ) { 
        res.ssim_[c] = computeSSIM( yuvA_[c], yuvB_[c] ); 
      }
      // Compute SSIM weighted average (QMIV = 4:1:1 MPEG = 1:1:1)
      if ( computeQmiv_ ) {
        res.ssimAvg_ = ( 4.0 * res.ssim_[0] + res.ssim_[1] + res.ssim_[2] ) / 6.0;
      } else {
        res.ssimAvg_ = ( res.ssim_[0] + res.ssim_[1] + res.ssim_[2] ) / 3.0;
      }
      if ( computeOmMetrics_ ) {
        // Compute OM-SSIM for each channel
        for ( int c = 0; c < 3; c++ ) { res.ssimOm_[c] = computeMaskedSSIM( yuvA_[c], yuvB_[c], mask.plane( 0 ) ); }
        // Compute OM-SSIM weighted average (QMIV = 4:1:1 MPEG = 1:1:1)
        if ( computeQmiv_ ) {
          res.ssimOmAvg_ = ( 4.0 * res.ssimOm_[0] + res.ssimOm_[1] + res.ssimOm_[2] ) / 6.0;
        } else {
          res.ssimOmAvg_ = ( res.ssimOm_[0] + res.ssimOm_[1] + res.ssimOm_[2] ) / 3.0;
        }
      }      
    }

    // Compute IVSSIM metrics
    if ( computeIvssimMetrics_ ) { 
      computeIVSSIM( res, imageA, imageB ); 
    }
  }
  results_.push_back( res );
}

template <typename T, int32_t N>
void convertBuffer( PMBB::xPicP& dst, const Image<T, N>& src ) {
  using namespace PMBB;

  const int32 width     = dst.getWidth();
  const int32 height    = dst.getHeight();
  const int32 bitDepth  = dst.getBitDepth();
  const int32 dstStride = dst.getStride();

  for ( int32 c = 0; c < 3; c++ ) {
    const Plane<T>& srcPlane = src.plane( c );
    uint16*         dstPtr   = dst.getAddr( eCmp( c ) );

    for ( int32 y = 0; y < height; y++ ) {
      for ( int32 x = 0; x < width; x++ ) {
        T srcA = srcPlane.get( x, y );
        if constexpr ( std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ) {
          dstPtr[x] = srcA;
        } else if constexpr ( std::is_same_v<T, float> ) {
          dstPtr[x] = round( srcA * xBitDepth2MaxValue( bitDepth ) );
        }
      }
      dstPtr += dstStride;
    }
  }
}

template <typename>
inline constexpr bool always_false_v = false;

template <typename T, int32_t N>
void Metric::computeIVSSIM( Results& result, const Image<T, N>& imageA, const Image<T, N>& imageB ) {
  using namespace PMBB;

  const int32 width  = imageA.width();
  const int32 height = imageA.height();

  int32 bitDepth = NOT_VALID;
  if constexpr ( std::is_same_v<T, uint8_t> ) {
    bitDepth = 8;
  } else if constexpr ( std::is_same_v<T, uint16_t> ) {
    bitDepth = 16;
  } else if constexpr ( std::is_same_v<T, float> ) {
    bitDepth = 16;
  } else {
    static_assert( always_false_v<T>, "Unsupported Image<T, N> type" );
  }

  static_assert( N == 3, "Unsupported number of components" );

  // create PMBB image buffers
  xPicP picA( { width, height }, bitDepth, 4 );
  xPicP picB( { width, height }, bitDepth, 4 );

  // create IVSSIM calculator
  xIVSSIM procIvSsim;
  procIvSsim.create( { width, height }, bitDepth, 4, false );
  procIvSsim.createThrdPoolIntf( nullptr, 0 );  // create empty thread pool interface, don't use parallel processing
  procIvSsim.setStructSimParams( xSSIM::eMode::BlockAveraged, false, 8, 4 );
  procIvSsim.initRowBuffers( height );

  // copy/convert data from Image<T, N> to xPicP
  convertBuffer<T, N>( picA, imageA );
  assert( picA.check( "imageA" ) );
  picA.extend();

  convertBuffer<T, N>( picB, imageB );
  assert( picB.check( "imageB" ) );
  picB.extend();

  // calculate RGB IV-SSIM
  procIvSsim.setCmpWeightsSearch( xIVSSIM::c_EqualCmpWeights );
  procIvSsim.setCmpWeightsAverage( xIVSSIM::c_EqualCmpWeights );
  double rgbIvSsim = procIvSsim.calcPicIVSSIM( &picA, &picB );

  // convert colorspace (in-place)
  xColorSpace::ConvertRGB2YCbCr( picA.getAddr( eCmp::LM ),  //
                                 picA.getAddr( eCmp::CB ),  //
                                 picA.getAddr( eCmp::CR ),  //
                                 picA.getAddr( eCmp::R ),   //
                                 picA.getAddr( eCmp::G ),   //
                                 picA.getAddr( eCmp::B ),   //
                                 picA.getStride(),          //
                                 picA.getStride(),          //
                                 picA.getWidth(),           //
                                 picA.getHeight(),          //
                                 picA.getBitDepth(),        //
                                 eClrSpcLC::BT709 );
  picA.extend();
  xColorSpace::ConvertRGB2YCbCr( picB.getAddr( eCmp::LM ),  //
                                 picB.getAddr( eCmp::CB ),  //
                                 picB.getAddr( eCmp::CR ),  //
                                 picB.getAddr( eCmp::R ),   //
                                 picB.getAddr( eCmp::G ),   //
                                 picB.getAddr( eCmp::B ),   //
                                 picB.getStride(),          //
                                 picB.getStride(),          //
                                 picB.getWidth(),           //
                                 picB.getHeight(),          //
                                 picB.getBitDepth(),        //
                                 eClrSpcLC::BT709 );
  picB.extend();

  // calculate YCbCr IV-SSIM
  procIvSsim.setCmpWeightsSearch( xIVSSIM::c_DefaultCmpWeights );
  procIvSsim.setCmpWeightsAverage( xIVSSIM::c_DefaultCmpWeights );
  double ycbcrIvSsim = procIvSsim.calcPicIVSSIM( &picA, &picB );

  // store results
  result.rgbIvSsim_ = rgbIvSsim;
  result.yuvIvSsim_ = ycbcrIvSsim;

  // cleanup
  procIvSsim.destroy();
}

// Standard SSIM computation function
double Metric::computeSSIM( const Plane<float>& img1, const Plane<float>& img2 ) const {
  const int    width      = img1.width();
  const int    height     = img1.height();
  const int    windowSize = 11;
  const float  k1         = 0.01f;
  const float  k2         = 0.03f;
  const float  l          = 255.0f;
  const float  c1         = ( k1 * l ) * ( k1 * l );
  const float  c2         = ( k2 * l ) * ( k2 * l );
  if ( gaussianWindow_.empty() || windowSize_ != windowSize || gaussianSigma_ != 1.5 ) {
    windowSize_     = windowSize;
    gaussianSigma_  = 1.5;
    gaussianWindow_ = generateGaussianWindow( windowSize, gaussianSigma_ );
  }
  float localWin[11][11];
  for ( int j = 0; j < 11; j++ ) {
    for ( int i = 0; i < 11; i++ ) {
      localWin[j][i] = static_cast<float>(gaussianWindow_[j][i]);
    }
  }
  double    ssimTotal    = 0.0; 
  int       validWindows = 0;
  const int halfWin      = windowSize / 2;
  const int pWidth        = width + 2 * halfWin;
  const int pHeight       = height + 2 * halfWin;
  const size_t neededSize = static_cast<size_t>(pWidth) * pHeight;
  if ( ssimBuf1_.size() != neededSize ) {
    ssimBuf1_.resize( neededSize );
    ssimBuf2_.resize( neededSize );
  }
  for ( int y = 0; y < height; y++ ) {
    for ( int x = 0; x < width; x++ ) {
      ssimBuf1_[( y + halfWin ) * pWidth + ( x + halfWin )] = img1.get( x, y );
      ssimBuf2_[( y + halfWin ) * pWidth + ( x + halfWin )] = img2.get( x, y );
    }
  }  
  const int srcX    = std::min( halfWin, width - 1 );
  const int srcY    = std::min( halfWin, height - 1 );
  const int bottomY = height + halfWin;
  const int rightX  = width + halfWin;
  for ( int y = 0; y < halfWin; y++ ) {
    const int yt = std::min( std::max( y - halfWin, 0 ), srcY );
    const int yb = std::min( std::max( y - halfWin, 0 ), height - 1 );
    for ( int x = 0; x < pWidth; x++ ) {
      const int x0 = std::min( std::max( x - halfWin, 0 ), width - 1 );
      ssimBuf1_[y * pWidth + x              ] = img1.get( x0, yt );
      ssimBuf2_[y * pWidth + x              ] = img2.get( x0, yt );
      ssimBuf1_[( bottomY + y ) * pWidth + x] = img1.get( x0, yb );
      ssimBuf2_[( bottomY + y ) * pWidth + x] = img2.get( x0, yb );
    }
  }
  for ( int x = 0; x < halfWin; x++ ) {
    for ( int y = 0; y < pHeight; y++ ) {
      const int y0 = std::min( std::max( y - halfWin, 0 ), height - 1 );
      ssimBuf1_[y * pWidth + x             ] = img1.get( srcX, y0 );
      ssimBuf2_[y * pWidth + x             ] = img2.get( srcX, y0 );
      ssimBuf1_[y * pWidth + ( rightX + x )] = img1.get( width - 1 - x, y0 );
      ssimBuf2_[y * pWidth + ( rightX + x )] = img2.get( width - 1 - x, y0 );
    }
  }
  const __m256 v_c1 = _mm256_set1_ps(c1);
  const __m256 v_c2 = _mm256_set1_ps(c2);
#if defined( USE_OPENMP )
  #pragma omp parallel for schedule(static) reduction(+:ssimTotal, validWindows)
#endif
  for ( int y = 0; y < height; y++ ) {
    int x = 0;
    for ( ; x <= width - 8; x += 8 ) {
      __m256 v_mu1    = _mm256_setzero_ps();
      __m256 v_mu2    = _mm256_setzero_ps();
      __m256 v_var1   = _mm256_setzero_ps();
      __m256 v_var2   = _mm256_setzero_ps();
      __m256 v_cov12  = _mm256_setzero_ps();
      for ( int j = 0; j < 11; j++ ) {
        int py = y + j;
        const float* __restrict b1_ptr = &ssimBuf1_[py * pWidth + x];
        const float* __restrict b2_ptr = &ssimBuf2_[py * pWidth + x];
        for ( int i = 0; i < 11; i++ ) {
          __m256 v_w    = _mm256_set1_ps(localWin[j][i]);
          __m256 v_val1 = _mm256_loadu_ps(&b1_ptr[i]);
          __m256 v_val2 = _mm256_loadu_ps(&b2_ptr[i]);
          __m256 v_w_val1 = _mm256_mul_ps(v_w, v_val1);
          __m256 v_w_val2 = _mm256_mul_ps(v_w, v_val2);
          v_mu1 = _mm256_add_ps(v_mu1, v_w_val1);
          v_mu2 = _mm256_add_ps(v_mu2, v_w_val2);
          v_var1  = _mm256_fmadd_ps(v_w_val1, v_val1, v_var1);
          v_var2  = _mm256_fmadd_ps(v_w_val2, v_val2, v_var2);
          v_cov12 = _mm256_fmadd_ps(v_w_val1, v_val2, v_cov12);
        }
      }
      __m256 v_mu1Sq   = _mm256_mul_ps(v_mu1, v_mu1);
      __m256 v_mu2Sq   = _mm256_mul_ps(v_mu2, v_mu2);
      __m256 v_mu1_mu2 = _mm256_mul_ps(v_mu1, v_mu2);
      __m256 v_sigma1Sq = _mm256_sub_ps(v_var1, v_mu1Sq);
      __m256 v_sigma2Sq = _mm256_sub_ps(v_var2, v_mu2Sq);
      __m256 v_sigma12  = _mm256_sub_ps(v_cov12, v_mu1_mu2);
      __m256 v_num = _mm256_mul_ps(
          _mm256_fmadd_ps(_mm256_set1_ps(2.0f), v_mu1_mu2, v_c1),
          _mm256_fmadd_ps(_mm256_set1_ps(2.0f), v_sigma12, v_c2)
      );
      __m256 v_den = _mm256_mul_ps(
          _mm256_add_ps(_mm256_add_ps(v_mu1Sq, v_mu2Sq), v_c1),
          _mm256_add_ps(_mm256_add_ps(v_sigma1Sq, v_sigma2Sq), v_c2)
      );
      alignas(32) float res_num[8];
      alignas(32) float res_den[8];
      _mm256_store_ps(res_num, v_num);
      _mm256_store_ps(res_den, v_den);
      for ( int k = 0; k < 8; k++ ) {
        if ( res_den[k] > 0.0f ) {
          ssimTotal += static_cast<double>(res_num[k] / res_den[k]);
          validWindows++;
        }
      }
    }
    for ( ; x < width; x++ ) {
      float mu1 = 0.0f, mu2 = 0.0f;
      float var1 = 0.0f, var2 = 0.0f, cov12 = 0.0f;
      for ( int j = 0; j < 11; j++ ) {
        int py = y + j;
        const float* __restrict b1_ptr = &ssimBuf1_[py * pWidth + x];
        const float* __restrict b2_ptr = &ssimBuf2_[py * pWidth + x];
        for ( int i = 0; i < 11; i++ ) {
          float weight = localWin[j][i];
          float val1   = b1_ptr[i];
          float val2   = b2_ptr[i];
          float w_val1 = weight * val1;
          float w_val2 = weight * val2;
          mu1   += w_val1;
          mu2   += w_val2;
          var1  += w_val1 * val1;
          var2  += w_val2 * val2;
          cov12 += w_val1 * val2;
        }
      }
      float sigma1Sq    = var1 - ( mu1 * mu1 );
      float sigma2Sq    = var2 - ( mu2 * mu2 );
      float sigma12     = cov12 - ( mu1 * mu2 );
      float numerator   = ( 2.0f * mu1 * mu2 + c1 ) * ( 2.0f * sigma12 + c2 );
      float denominator = ( mu1 * mu1 + mu2 * mu2 + c1 ) * ( sigma1Sq + sigma2Sq + c2 );
      if ( denominator > 0.0f ) {
        ssimTotal += static_cast<double>(numerator / denominator);
        validWindows++;
      }
    }
  }

  return ( validWindows > 0 ) ? ssimTotal / validWindows : 1.0;
}

// Masked SSIM computation function (OM-SSIM)
double Metric::computeMaskedSSIM( const Plane<float>&   img1,
                                  const Plane<float>&   img2,
                                  const Plane<uint8_t>& mask ) const {
  const int    width      = img1.width();
  const int    height     = img1.height();
  const int    windowSize = 11;
  const float  k1         = 0.01f;
  const float  k2         = 0.03f;
  const float  l          = 255.0f;
  const float  c1         = ( k1 * l ) * ( k1 * l );
  const float  c2         = ( k2 * l ) * ( k2 * l );
  const float  threshold  = 0.5f;  
  if ( gaussianWindow_.empty() || windowSize_ != windowSize || gaussianSigma_ != 1.5 ) {
    windowSize_     = windowSize;
    gaussianSigma_  = 1.5;
    gaussianWindow_ = generateGaussianWindow( windowSize, gaussianSigma_ );
  }
  float localWin[11][11];
  for ( int j = 0; j < 11; j++ ) {
    for ( int i = 0; i < 11; i++ ) {
      localWin[j][i] = static_cast<float>(gaussianWindow_[i][j]);
    }
  }

  const int halfWin      = windowSize / 2;
  double    ssimTotal    = 0.0;  
  int       validWindows = 0;
  const int pWidth       = width + 2 * halfWin;
  const int pHeight      = height + 2 * halfWin;
  paddedImg1_.resize( pWidth, pHeight, 0.0f );
  paddedImg2_.resize( pWidth, pHeight, 0.0f );
  paddedMask_.resize( pWidth, pHeight, 0.0f );
  for ( int y = 0; y < height; y++ ) {
    for ( int x = 0; x < width; x++ ) {
      paddedImg1_.set( x + halfWin, y + halfWin, img1.get( x, y ) );
      paddedImg2_.set( x + halfWin, y + halfWin, img2.get( x, y ) );
      paddedMask_.set( x + halfWin, y + halfWin, static_cast<float>(mask.get( x, y )) );
    }
  }
  const int srcX    = std::min( halfWin, width - 1 );
  const int srcY    = std::min( halfWin, height - 1 );
  const int bottomY = height + halfWin;
  const int rightX  = width + halfWin;
  for ( int y = 0; y < halfWin; y++ ) {
    const int yt = std::min( std::max( y - halfWin, 0 ), srcY );
    const int yb = std::min( std::max( y - halfWin, 0 ), height - 1 );
    for ( int x = 0; x < pWidth; x++ ) {
      const int x0 = std::min( std::max( x - halfWin, 0 ), width - 1 );
      paddedImg1_.set( x, y, img1.get( x0, yt ) );
      paddedImg2_.set( x, y, img2.get( x0, yt ) );
      paddedMask_.set( x, y, static_cast<float>(mask.get( x0, yt )) );
      paddedImg1_.set( x, bottomY + y, img1.get( x0, yb ) );
      paddedImg2_.set( x, bottomY + y, img2.get( x0, yb ) );
      paddedMask_.set( x, bottomY + y, static_cast<float>(mask.get( x0, yb )) );
    }
  }
  for ( int x = 0; x < halfWin; x++ ) {
    for ( int y = 0; y < pHeight; y++ ) {
      const int y0 = std::min( std::max( y - halfWin, 0 ), height - 1 );
      paddedImg1_.set( x, y, img1.get( srcX, y0 ) );
      paddedImg2_.set( x, y, img2.get( srcX, y0 ) );
      paddedMask_.set( x, y, static_cast<float>(mask.get( srcX, y0 )) );
      paddedImg1_.set( rightX + x, y, img1.get( width - 1 - x, y0 ) );
      paddedImg2_.set( rightX + x, y, img2.get( width - 1 - x, y0 ) );
      paddedMask_.set( rightX + x, y, static_cast<float>(mask.get( width - 1 - x, y0 )) );
    }
  }
  for ( int y = 0; y < height; y++ ) {
    for ( int x = 0; x < width; x++ ) {
      float mu1 = 0.0f, mu2 = 0.0f;
      float sigma1Sq = 0.0f, sigma2Sq = 0.0f, sigma12 = 0.0f;
      float maskSum = 0.0f;
      for ( int j = 0; j < windowSize; j++ ) {
        int py = y + j;
        for ( int i = 0; i < windowSize; i++ ) {
          int   px         = x + i;
          float weight     = localWin[j][i];
          float maskVal    = paddedMask_.get( px, py );
          float val1       = paddedImg1_.get( px, py );
          float float_val2 = paddedImg2_.get( px, py );
          float w_mask     = weight * maskVal;
          mu1     += w_mask * val1;
          mu2     += w_mask * float_val2;
          maskSum += w_mask;
        }
      }
      if ( maskSum < threshold ) { continue; }
      float invMaskSum = 1.0f / maskSum;
      mu1 *= invMaskSum;
      mu2 *= invMaskSum;
      for ( int j = 0; j < windowSize; j++ ) {
        int py = y + j;
        for ( int i = 0; i < windowSize; i++ ) {
          int   px      = x + i;
          float weight  = localWin[j][i];
          float maskVal = paddedMask_.get( px, py );
          float val1    = paddedImg1_.get( px, py );
          float float_val2    = paddedImg2_.get( px, py );
          float diff1  = val1 - mu1;
          float diff2  = float_val2 - mu2;
          float w_mask = weight * maskVal;
          sigma1Sq += w_mask * diff1 * diff1;
          sigma2Sq += w_mask * diff2 * diff2;
          sigma12  += w_mask * diff1 * diff2;
        }
      }
      sigma1Sq *= invMaskSum;
      sigma2Sq *= invMaskSum;
      sigma12  *= invMaskSum;
      float numerator   = ( 2.0f * mu1 * mu2 + c1 ) * ( 2.0f * sigma12 + c2 );
      float denominator = ( mu1 * mu1 + mu2 * mu2 + c1 ) * ( sigma1Sq + sigma2Sq + c2 );
      if ( denominator > 0.0f ) {
        ssimTotal += static_cast<double>(numerator / denominator);
        validWindows++;
      }
    }
  }
  return ( validWindows > 0 ) ? ssimTotal / validWindows : 1.0;
}

Results Metric::getAverage() {
  Results res;
  for ( auto& el : results_ ) res += el;
  res /= results_.size();
  return res;
}

Results Metric::getLastFrame() {
  Results res;
  if ( results_.size() == 0 ) return res;
  int32_t frameIndex = results_.back().frameIndex_;
  int32_t numFrames  = 0;
  for ( auto& el : results_ )
    if ( el.frameIndex_ == frameIndex ) {
      res += el;
      numFrames++;
    }
  res /= numFrames;
  return res;
}

// Explicit instantiations
template void Metric::compute<uint8_t, 3>( const Image<uint8_t, 3>&,
                                           const Image<uint8_t, 3>&,
                                           const Image<uint8_t, 1>&,
                                           const int32_t );
template void Metric::compute<uint16_t, 3>( const Image<uint16_t, 3>&,
                                            const Image<uint16_t, 3>&,
                                            const Image<uint8_t, 1>&,
                                            const int32_t );
template void Metric::compute<float, 3>( const Image<float, 3>&,
                                         const Image<float, 3>&,
                                         const Image<uint8_t, 1>&,
                                         const int32_t );

template void Metric::compute<uint8_t, 3>( const Image<uint8_t, 3>&, const Image<uint8_t, 3>&, const int32_t );
template void Metric::compute<uint16_t, 3>( const Image<uint16_t, 3>&, const Image<uint16_t, 3>&, const int32_t );
template void Metric::compute<float, 3>( const Image<float, 3>&, const Image<float, 3>&, const int32_t );
