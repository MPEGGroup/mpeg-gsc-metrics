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
#include "image.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>

// Generate Gaussian window
std::vector<std::vector<double>> generate_gaussian_window( int window_size, double sigma );

inline std::ostream& operator<<( std::ostream& os, const glm::dvec3& v ) {
  std::ios_base::fmtflags f( os.flags() );
  std::streamsize         p = os.precision();
  os << std::setw( 20 ) << v.x << " " << std::setw( 20 ) << v.y << " " << std::setw( 20 ) << v.z;
  os.flags( f );
  os.precision( p );
  return os;
}
template <typename T, int32_t N>
class Image;

struct Results {
  // Frame
  int32_t frameIndex_ = -1;

  // RGB / YUV MSE and PSNR
  glm::dvec3 mseRgb_  = { 0.0, 0.0, 0.0 };
  glm::dvec3 psnrRgb_ = { 0.0, 0.0, 0.0 };
  glm::dvec3 mseYuv_  = { 0.0, 0.0, 0.0 };
  glm::dvec3 psnrYuv_ = { 0.0, 0.0, 0.0 };

  // RGB / YUV MSE and PSNR (Occupancy Map)
  glm::dvec3 mseRgbOm_  = { 0.0, 0.0, 0.0 };
  glm::dvec3 psnrRgbOm_ = { 0.0, 0.0, 0.0 };
  glm::dvec3 mseYuvOm_  = { 0.0, 0.0, 0.0 };
  glm::dvec3 psnrYuvOm_ = { 0.0, 0.0, 0.0 };

  // Occupancy rate
  double occupancyRate_ = 0.0;

  // SSIM
  glm::dvec3 ssim_   = { 0.0, 0.0, 0.0 };
  glm::dvec3 ssimOm_ = { 0.0, 0.0, 0.0 };

  // IVSSIM
  double rgbIvSsim_ = 0.0;
  double yuvIvSsim_ = 0.0;

  // Weighted Averages:
  double psnrRgbAvg_   = 0.0;  // weight MPEG = 1:1:1 QMIV = 1:1:1
  double psnrYuvAvg_   = 0.0;  // weight MPEG = 1:1:1 QMIV = 4:1:1
  double psnrRgbOmAvg_ = 0.0;  // weight MPEG = 1:1:1 QMIV = 1:1:1
  double psnrYuvOmAvg_ = 0.0;  // weight MPEG = 1:1:1 QMIV = 4:1:1
  double ssimAvg_      = 0.0;  // weight MPEG = 1:1:1 QMIV = 4:1:1
  double ssimOmAvg_    = 0.0;  // weight MPEG = 1:1:1 QMIV = 4:1:1

  Results& operator+=( const Results& other ) {
    // RGB / YUV MSE and PSNR
    mseRgb_ += other.mseRgb_;
    psnrRgb_ += other.psnrRgb_;
    mseYuv_ += other.mseYuv_;
    psnrYuv_ += other.psnrYuv_;
    // RGB / YUV MSE and PSNR (Occupancy Map)
    mseRgbOm_ += other.mseRgbOm_;
    psnrRgbOm_ += other.psnrRgbOm_;
    mseYuvOm_ += other.mseYuvOm_;
    psnrYuvOm_ += other.psnrYuvOm_;
    // Occupancy
    occupancyRate_ += other.occupancyRate_;
    // SSIM
    ssim_ += other.ssim_;
    ssimOm_ += other.ssimOm_;
    // Weighted Averages
    psnrRgbAvg_ += other.psnrRgbAvg_;
    psnrYuvAvg_ += other.psnrYuvAvg_;
    psnrRgbOmAvg_ += other.psnrRgbOmAvg_;
    psnrYuvOmAvg_ += other.psnrYuvOmAvg_;
    ssimAvg_ += other.ssimAvg_;
    ssimOmAvg_ += other.ssimOmAvg_;
    // IV SSIM
    rgbIvSsim_ += other.rgbIvSsim_;
    yuvIvSsim_ += other.yuvIvSsim_;
    return *this;
  }

  Results& operator/=( double divisor ) {
    // RGB / YUV MSE and PSNR
    mseRgb_ /= divisor;
    psnrRgb_ /= divisor;
    mseYuv_ /= divisor;
    psnrYuv_ /= divisor;
    // RGB / YUV MSE and PSNR (Occupancy Map)
    mseRgbOm_ /= divisor;
    psnrRgbOm_ /= divisor;
    mseYuvOm_ /= divisor;
    psnrYuvOm_ /= divisor;
    // Occupancy
    occupancyRate_ /= divisor;
    // SSIM
    ssim_ /= divisor;
    ssimOm_ /= divisor;
    ssimAvg_ /= divisor;
    ssimOmAvg_ /= divisor;
    // IV SSIM
    rgbIvSsim_ /= divisor;
    yuvIvSsim_ /= divisor;
    // Weighted Averages
    psnrRgbAvg_ /= divisor;
    psnrYuvAvg_ /= divisor;
    psnrRgbOmAvg_ /= divisor;
    psnrYuvOmAvg_ /= divisor;
    return *this;
  }

  void log( const std::string& name = "", bool isSmall = false ) const { writeMetrics( std::cout, name, isSmall ); }
  void saveToFile( const std::string& filename, const std::string& name = "" ) const {
    createDirectory( filename );
    std::ofstream file( filename, std::ios::app );
    if ( file.is_open() ) {
      writeMetrics( file, name );
      file.close();
    }
  }

 private:
  void writeMetrics( std::ostream& os, const std::string& name = "", bool isSmall = false ) const {
    os << std::fixed << std::setprecision( 12 );
    if ( isSmall ) {
      os << "Psnr RGB = " << psnrRgb_ << " Psnr YUV = " << psnrYuv_ << "\n";
    } else {
      os << "Metric " << ( name.empty() ? "" : name ) << ":\n"
         << "Mse RGB           = " << mseRgb_ << "\n"
         << "Psnr RGB          = " << psnrRgb_ << "\n"
         << "Psnr RGB (avg)    = " << std::setw( 20 ) << psnrRgbAvg_ << "\n"
         << "Mse YUV           = " << mseYuv_ << "\n"
         << "Psnr YUV          = " << psnrYuv_ << "\n"
         << "Psnr YUV (avg)    = " << std::setw( 20 ) << psnrYuvAvg_ << "\n"
         << "OM-Mse RGB        = " << mseRgbOm_ << "\n"
         << "OM-Psnr RGB       = " << psnrRgbOm_ << "\n"
         << "OM-Psnr RGB (avg) = " << std::setw( 20 ) << psnrRgbOmAvg_ << "\n"
         << "OM-Mse YUV        = " << mseYuvOm_ << "\n"
         << "OM-Psnr YUV       = " << psnrYuvOm_ << "\n"
         << "OM-Psnr YUV (avg) = " << std::setw( 20 ) << psnrYuvOmAvg_ << "\n"
         << "SSIM              = " << ssim_ << "\n"
         << "SSIM (avg)        = " << std::setw( 20 ) << ssimAvg_ << "\n"
         << "OM-SSIM           = " << ssimOm_ << "\n"
         << "OM-SSIM (avg)     = " << std::setw( 20 ) << ssimOmAvg_ << "\n"
         << "IVSSIM RGB        = " << std::setw( 20 ) << rgbIvSsim_ << "\n"
         << "IVSSIM YUV        = " << std::setw( 20 ) << yuvIvSsim_ << "\n";
      os << std::fixed << std::setprecision( 2 );
      os << "Occupancy         = " << std::setw( 20 ) << occupancyRate_ << "%\n";
    }
  }
};

class Metric {
 public:
  Metric( bool computeOm_ = true, bool computeSsim_ = true, bool computeIvssimMetrics = true, bool computeQmiv = true );
  ~Metric();

  void reset() { results_.clear(); }
  void setMetricsOptions( bool computeOm, bool computeSsim ) {
    computeOmMetrics_   = computeOm;
    computeSsimMetrics_ = computeSsim;
  }

  // Compute the image quality metrics with mask
  template <typename T, int32_t N>
  void compute( const Image<T, N>&       imageA,
                const Image<T, N>&       imageB,
                const Image<uint8_t, 1>& mask,
                const int32_t            frameIndex );

  // Compute the image quality metrics without a mask
  template <typename T, int32_t N>
  void compute( const Image<T, N>& imageA, const Image<T, N>& imageB, const int32_t frameIndex );

  Results getAverage();
  Results getLastFrame();

 private:
  inline bool isValidPixel( const glm::dvec3& pixel ) const {
    return ( pixel.r != 0.0 ) || ( pixel.g != 0.0 ) || ( pixel.b != 0.0 );
  }

  // Compute the SSIM metric
  double computeSSIM( const Plane<float>& img1, const Plane<float>& img2 ) const;

  // Compute the SSIM index with a mask (OM-SSIM)
  double computeMaskedSSIM( const Plane<float>& img1, const Plane<float>& img2, const Plane<uint8_t>& mask ) const;

  // Compute the IV-SSIM metric
  template <typename T, int32_t N>
  void computeIVSSIM( Results& Result, const Image<T, N>& imageA, const Image<T, N>& imageB );

  mutable std::vector<std::vector<double>> gaussianWindow_;
  mutable int                              windowSize_           = 11;
  mutable double                           gaussianSigma_        = 1.5;
  bool                                     computeOmMetrics_     = true;
  bool                                     computeSsimMetrics_   = true;
  bool                                     computeIvssimMetrics_ = true;
  bool                                     computeQmiv_          = true;
  std::vector<Results>                     results_;
  mutable Image<float, 3>                  yuvA_;
  mutable Image<float, 3>                  yuvB_;
  mutable std::vector<float>               ssimBuf1_;
  mutable std::vector<float>               ssimBuf2_;
  mutable Plane<float>                     paddedImg1_;
  mutable Plane<float>                     paddedImg2_;
  mutable Plane<float>                     paddedMask_;
};
