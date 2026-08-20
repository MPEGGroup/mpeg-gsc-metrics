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
#include "text.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#elif defined(__APPLE__) && defined(__arm64__)
#include "sse2neon.h" 
#else
#include <immintrin.h>
#endif

#include "stb_image.h"
#include "stb_image_write.h"

namespace detail {
class MD5 {
  uint32_t state[4];
  uint32_t count[2];
  uint8_t  buffer[64];

  static void transform(uint32_t st[4], const uint8_t block[64]) {
    uint32_t a = st[0], b = st[1], c = st[2], d = st[3], x[16];
    for (int i = 0; i < 16; ++i)
      x[i] = block[i*4] | (block[i*4+1] << 8) | (block[i*4+2] << 16) | (block[i*4+3] << 24);

    auto f = [](uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); };
    auto g = [](uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); };
    auto h = [](uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; };
    auto i_func = [](uint32_t x, uint32_t y, uint32_t z) { return y ^ (x | ~z); };

    auto step = [](auto func, uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac) {
      a += func(b, c, d) + x + ac;
      a = b + ((a << s) | (a >> (32 - s)));
    };

    // Round 1
    step(f, a, b, c, d, x[0], 7, 0xd76aa478);   step(f, d, a, b, c, x[1], 12, 0xe8c7b756);
    step(f, c, d, a, b, x[2], 17, 0x242070db);  step(f, b, c, d, a, x[3], 22, 0xc1bdceee);
    step(f, a, b, c, d, x[4], 7, 0xf57c0faf);   step(f, d, a, b, c, x[5], 12, 0x4787c62a);
    step(f, c, d, a, b, x[6], 17, 0xa8304613);  step(f, b, c, d, a, x[7], 22, 0xfd469501);
    step(f, a, b, c, d, x[8], 7, 0x698098d8);   step(f, d, a, b, c, x[9], 12, 0x8b44f7af);
    step(f, c, d, a, b, x[10], 17, 0xffff5bb1); step(f, b, c, d, a, x[11], 22, 0x895cd7be);
    step(f, a, b, c, d, x[12], 7, 0x6b901122);  step(f, d, a, b, c, x[13], 12, 0xfd987193);
    step(f, c, d, a, b, x[14], 17, 0xa679438e); step(f, b, c, d, a, x[15], 22, 0x49b40821);

    // Round 2
    step(g, a, b, c, d, x[1], 5, 0xf61e2562);   step(g, d, a, b, c, x[6], 9, 0xc040b340);
    step(g, c, d, a, b, x[11], 14, 0x265e5a51); step(g, b, c, d, a, x[0], 20, 0xe9b6c7aa);
    step(g, a, b, c, d, x[5], 5, 0xd62f105d);   step(g, d, a, b, c, x[10], 9, 0x02441453);
    step(g, c, d, a, b, x[15], 14, 0xd8a1e681); step(g, b, c, d, a, x[4], 20, 0xe7d3fbc8);
    step(g, a, b, c, d, x[9], 5, 0x21e1cde6);   step(g, d, a, b, c, x[14], 9, 0xc33707d6);
    step(g, c, d, a, b, x[3], 14, 0xf4d50d87);  step(g, b, c, d, a, x[8], 20, 0x455a14ed);
    step(g, a, b, c, d, x[13], 5, 0xa9e3e905);  step(g, d, a, b, c, x[2], 9, 0xfcefa3f8);
    step(g, c, d, a, b, x[7], 14, 0x676f02d9);  step(g, b, c, d, a, x[12], 20, 0x8d2a4c8a);

    // Round 3
    step(h, a, b, c, d, x[5], 4, 0xfffa3942);   step(h, d, a, b, c, x[8], 11, 0x8771f681);
    step(h, c, d, a, b, x[11], 16, 0x6d9d6122); step(h, b, c, d, a, x[14], 23, 0xfde5380c);
    step(h, a, b, c, d, x[1], 4, 0xa4beea44);   step(h, d, a, b, c, x[4], 11, 0x4bdecfa9);
    step(h, c, d, a, b, x[7], 16, 0xf6bb4b60);  step(h, b, c, d, a, x[10], 23, 0xbebfbc70);
    step(h, a, b, c, d, x[13], 4, 0x289b7ec6);  step(h, d, a, b, c, x[0], 11, 0xeaa127fa);
    step(h, c, d, a, b, x[3], 16, 0xd4ef3085);  step(h, b, c, d, a, x[6], 23, 0x04881d05);
    step(h, a, b, c, d, x[9], 4, 0xd9d4d039);   step(h, d, a, b, c, x[12], 11, 0xe6db99e5);
    step(h, c, d, a, b, x[15], 16, 0x1fa27cf8); step(h, b, c, d, a, x[2], 23, 0xc4ac5665);

    // Round 4
    step(i_func, a, b, c, d, x[0], 6, 0xf4292244);   step(i_func, d, a, b, c, x[7], 10, 0x432aff97);
    step(i_func, c, d, a, b, x[14], 15, 0xab9423a7); step(i_func, b, c, d, a, x[5], 21, 0xfc93a039);
    step(i_func, a, b, c, d, x[12], 6, 0x655b59c3);  step(i_func, d, a, b, c, x[3], 10, 0x8f0ccc92);
    step(i_func, c, d, a, b, x[10], 15, 0xffeff47d); step(i_func, b, c, d, a, x[1], 21, 0x85845dd1);
    step(i_func, a, b, c, d, x[8], 6, 0x6fa87e4f);   step(i_func, d, a, b, c, x[15], 10, 0xfe2ce6e0);
    step(i_func, c, d, a, b, x[6], 15, 0xa3014314);  step(i_func, b, c, d, a, x[13], 21, 0x4e0811a1);
    step(i_func, a, b, c, d, x[4], 6, 0xf7537e82);   step(i_func, d, a, b, c, x[11], 10, 0xbd3af235);
    step(i_func, c, d, a, b, x[2], 15, 0x2ad7d2bb);  step(i_func, b, c, d, a, x[9], 21, 0xeb86d391);

    st[0] += a; st[1] += b; st[2] += c; st[3] += d;
  }

public:
  MD5() {
    state[0] = 0x67452301; state[1] = 0xefcdab89; state[2] = 0x98badcfe; state[3] = 0x10325476;
    count[0] = count[1] = 0;
  }

  void update(const uint8_t* input, size_t inputLen) {
    size_t i, index, partLen;
    index = (count[0] >> 3) & 0x3F;
    if ((count[0] += ((uint32_t)inputLen << 3)) < ((uint32_t)inputLen << 3)) count[1]++;
    count[1] += ((uint32_t)inputLen >> 29);
    partLen = 64 - index;
    if (inputLen >= partLen) {
      std::memcpy(&buffer[index], input, partLen);
      transform(state, buffer);
      for (i = partLen; i + 63 < inputLen; i += 64) transform(state, &input[i]);
      index = 0;
    } else i = 0;
    std::memcpy(&buffer[index], &input[i], inputLen - i);
  }

  std::string finalize() {
    uint8_t bits[8];
    for (int i = 0; i < 8; ++i) bits[i] = (uint8_t)((count[i >= 4 ? 1 : 0] >> ((i % 4) * 8)) & 0xFF);
    uint32_t index = (count[0] >> 3) & 0x3f;
    uint32_t padLen = (index < 56) ? (56 - index) : (120 - index);
    static const uint8_t PADDING[64] = { 0x80 };
    update(PADDING, padLen);
    update(bits, 8);
    
    static const char hex[] = "0123456789abcdef";
    std::string res;
    res.reserve(32);
    for (int i = 0; i < 16; ++i) {
      uint8_t byte = (state[i/4] >> ((i%4)*8)) & 0xFF;
      res.push_back(hex[byte >> 4]);
      res.push_back(hex[byte & 0x0F]);
    }
    return res;
  }
};
} 

enum class ImageFormat { PNG, BMP, TGA, JPG };

template <typename T>
struct Plane {
  void zero() { fill( T( 0 ) ); }
  void fill( T v ) { std::fill( buffer_.begin(), buffer_.end(), v ); }
  void resize( int32_t w = 0, int32_t h = 0, T initValue = T( 0 ) ) {
    width_  = w;
    height_ = h;
    buffer_.resize( w * h, initValue );
  }

  template <typename D>
  Plane& operator=( const Plane<D>& src ) {
    resize( src.width(), src.height() );
    for ( size_t i = 0; i < buffer_.size(); i++ ) set( i, (T)src.get( i ) );
    return *this;
  }

  inline int32_t               size() const { return (int32_t)buffer_.size(); }
  inline T                     get( int32_t i ) const { return buffer_[i]; }
  inline T&                    get( int32_t i ) { return buffer_[i]; }
  inline T                     get( int32_t u, int32_t v ) const { return buffer_[v * width_ + u]; }
  inline T&                    get( int32_t u, int32_t v ) { return buffer_[v * width_ + u]; }
  inline T                     operator()( int32_t u, int32_t v ) const { return buffer_[v * width_ + u]; }
  inline T&                    operator()( int32_t u, int32_t v ) { return buffer_[v * width_ + u]; }
  inline T                     operator()( int32_t i ) const { return buffer_[i]; }
  inline T&                    operator()( int32_t i ) { return buffer_[i]; }
  inline T                     operator[]( int32_t i ) const { return buffer_[i]; }
  inline T&                    operator[]( int32_t i ) { return buffer_[i]; }
  inline T*                    data( int32_t u, int32_t v ) { return buffer_.data() + ( v * width_ + u ); }
  inline const T*              data( int32_t u, int32_t v ) const { return buffer_.data() + ( v * width_ + u ); }
  inline void                  set( int32_t index, T e ) { buffer_[index] = e; }
  inline void                  set( int32_t u, int32_t v, T e ) { buffer_[v * width_ + u] = e; }
  inline T*                    data() { return buffer_.data(); }
  inline const T*              data() const { return buffer_.data(); }
  inline std::vector<T>&       buffer() { return buffer_; }
  inline const std::vector<T>& buffer() const { return buffer_; }
  inline int32_t               width() const { return width_; }
  inline int32_t               height() const { return height_; }
  inline void                  clear() { buffer_.clear(); }

  Plane operator*( T scalar ) const {
    Plane result;
    result.resize( width_, height_ );
    for ( int32_t i = 0; i < size(); i++ ) { result.get( i ) = this->get( i ) * scalar; }
    return result;
  }
  Plane& operator*=( T scalar ) {
    for ( auto& value : buffer_ ) { value *= scalar; }
    return *this;
  }
  void log( const std::string& str, int32_t u0 = 0, int32_t v0 = 0, int32_t number = 6 ) const {
    printf( "%s: %dx%d \n", str.c_str(), width_, height_ );
    printf( "%4s: ", "Comp" );
    for ( int32_t u = u0; u < std::min( width_, u0 + number ); u++ )
      if constexpr ( std::is_floating_point<T>::value )
        printf( "%6d ", u );
      else
        printf( "%4d ", u );
    printf( "\n" );
    fflush( stdout );
    for ( int32_t v = v0; v < std::min( height_, v0 + number ); v++ ) {
      printf( "%4d: ", v );
      for ( int32_t u = u0; u < std::min( width_, u0 + number ); u++ ) {
        if constexpr ( std::is_floating_point<T>::value )
          printf( "%6.2f ", static_cast<float>( get( u, v ) ) );
        else
          printf( "%4d ", static_cast<int32_t>( get( u, v ) ) );
      }
      printf( "\n" );
    }
  }

 private:
  int32_t        width_;
  int32_t        height_;
  std::vector<T> buffer_;
};

template <typename T, int N = 3>
class Image {
 public:
  Image( const Image& ) = default;
  Image( int32_t width = 0, int32_t height = 0, T value = 0 ) { resize( width, height, value ); }
  ~Image() = default;

  Image( const Image& imageA, const Image& imageB, bool butterfly ) {
    if ( imageA.height_ != imageB.height_ ) {
      printf( "Heigth are not the same \n" );
      fflush( stdout );
      exit( -1 );
    }
    resize( butterfly ? imageA.width_ : ( imageA.width_ + imageB.width_ ), imageA.height_ );
    for ( int c = 0; c < N; c++ ) {
      for ( int v = 0; v < height_; v++ ) {
        if ( butterfly ) {
          int width2 = imageA.width_ / 2;
          for ( int u = 0; u < width2; u++ ) planes_[c].set( u, v, imageA.planes_[c].get( u, v ) );
          for ( int u = 0; u < width2; u++ )
            planes_[c].set( u + width2, v, imageB.planes_[c].get( width2 - 1 - u, v ) );
        } else {
          for ( int u = 0; u < imageA.width_; u++ ) planes_[c].set( u, v, imageA.planes_[c].get( u, v ) );
          for ( int u = 0; u < imageB.width_; u++ )
            planes_[c].set( u + imageA.width_, v, imageB.planes_[c].get( u, v ) );
        }
      }
    }
  }

  void resize( int32_t width, int32_t height, T value = 0 ) {
    width_  = width;
    height_ = height;
    planes_.resize( N );
    for ( auto& plane : planes_ ) plane.resize( width_, height_, value );
  };

  inline void zero() {
    for ( auto& p : planes_ ) { p.zero(); }
  }
  inline void fill( T v ) {
    for ( auto& p : planes_ ) { p.fill( v ); }
  }
  inline void clear() {
    for ( auto& plane : planes_ ) { plane.clear(); }
  }

  Image& operator=( const Image& ) = default;

  template <typename D, int M = 3>
  Image& operator=( const Image<D, M>& src ) {
    resize( src.width(), src.height() );
    for ( int32_t i = 0; i < src.planeCount(); i++ ) { planes_[i] = src.plane( i ); }
    return *this;
  }

  inline void swap( Image<T, N>& src ) {
    std::swap( width_, src.width_ );
    std::swap( height_, src.height_ );
    std::swap( planes_, src.planes_ );
  }

  inline Plane<T>&       plane( int32_t index ) { return planes_[index]; }
  inline const Plane<T>& plane( int32_t index ) const { return planes_[index]; }
  inline auto            begin() { return planes_.begin(); }
  inline auto            end() { return planes_.end(); }
  inline auto            begin() const { return planes_.cbegin(); }
  inline auto            end() const { return planes_.cend(); }
  inline Plane<T>&       operator[]( int32_t index ) { return planes_[index]; }
  inline const Plane<T>& operator[]( int32_t index ) const { return planes_[index]; }
  inline int32_t         planeCount() const { return int32_t( planes_.size() ); }
  inline int32_t         width() const { return width_; }
  inline int32_t         height() const { return height_; }
  inline int32_t         size() const { return width_ * height_; }
  inline T*              data( int32_t c, int32_t u, int32_t v ) { return planes_[c].data( u, v ); }
  inline const T*        data( int32_t c, int32_t u, int32_t v ) const { return planes_[c].data( u, v ); }
  inline T               get( int32_t c, int32_t u, int32_t v ) const { return planes_[c].get( u, v ); }
  inline void            set( int32_t c, int32_t u, int32_t v, T e ) { planes_[c].set( u, v, e ); }
  inline glm::dvec3 getD( int32_t index ) const {
    return glm::dvec3( planes_[0].get( index ), planes_[1].get( index ), planes_[2].get( index ) );
  }
  inline void setD( int32_t index, const glm::dvec3& e ) {
    for ( int32_t i = 0; i < 3; i++ ) { planes_[i].set( index, e[i] ); }
  }
  inline glm::vec3 getF( int32_t index ) const {
    return glm::vec3( planes_[0].get( index ), planes_[1].get( index ), planes_[2].get( index ) );
  }
  inline void setF( int32_t index, const glm::vec3& e ) {
    for ( int32_t i = 0; i < 3; i++ ) { planes_[i].set( index, e[i] ); }
  }

  inline glm::dvec3 getYuvD( int32_t index ) const {
    glm::dvec3 rgb( static_cast<double>(planes_[0][index]), 
                    static_cast<double>(planes_[1][index]), 
                    static_cast<double>(planes_[2][index]) );                    
    return glm::dvec3(  0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2],
                       -0.1146 * rgb[0] - 0.3854 * rgb[1] + 0.5000 * rgb[2] + 128.0,
                        0.5000 * rgb[0] - 0.4542 * rgb[1] - 0.0458 * rgb[2] + 128.0 );
  }

  inline glm::dvec3 getYuvQmivD( int32_t index ) const {
    int32_t r = planes_[0][index];
    int32_t g = planes_[1][index];
    int32_t b = planes_[2][index];
    int32_t y = std::clamp( ( ( 13933 * r + 46871 * g + 4732 * b + 32768 ) >> 16 ), 0, 255 );
    int32_t u = std::clamp( ( ( -7509 * r - 25259 * g + ( b << 15 ) + 32768 ) >> 16 ) + 128, 0, 255 );
    int32_t v = std::clamp( ( ( ( r << 15 ) - 29763 * g - 3005 * b + 32768 ) >> 16 ) + 128, 0, 255 );
    return glm::dvec3( y, u, v );
  }

  inline glm::vec3 getYuvF( int32_t index ) const { // <- Correction ici : glm::vec3
    glm::vec3 rgb( static_cast<float>(planes_[0][index]), 
                  static_cast<float>(planes_[1][index]), 
                  static_cast<float>(planes_[2][index]) );                    
    
    return glm::vec3(  0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2],
                      -0.1146f * rgb[0] - 0.3854f * rgb[1] + 0.5000f * rgb[2] + 128.0f,
                       0.5000f * rgb[0] - 0.4542f * rgb[1] - 0.0458f * rgb[2] + 128.0f );
  }

  inline glm::vec3 getYuvQmivF( int32_t index ) const {
    int32_t r = static_cast<int32_t>(planes_[0][index]);
    int32_t g = static_cast<int32_t>(planes_[1][index]);
    int32_t b = static_cast<int32_t>(planes_[2][index]);
    
    int32_t y = std::clamp( ( ( 13933 * r + 46871 * g + 4732 * b + 32768 ) >> 16 ), 0, 255 );
    int32_t u = std::clamp( ( ( -7509 * r - 25259 * g + ( b << 15 ) + 32768 ) >> 16 ) + 128, 0, 255 );
    int32_t v = std::clamp( ( ( ( r << 15 ) - 29763 * g - 3005 * b + 32768 ) >> 16 ) + 128, 0, 255 );
    
    return glm::vec3( static_cast<float>(y), static_cast<float>(u), static_cast<float>(v) );
  }

  Image& operator*=( T scalar ) {
    for ( auto& plane : planes_ ) plane *= scalar;
    return *this;
  }
  Image operator*( T scalar ) const {
    Image result;
    result.resize( width_, height_ );
    for ( int32_t i = 0; i < N; i++ ) result[i] = this->planes_[i] * scalar;
    return result;
  }

  bool save( const std::string& prefix ) const {
    auto filename = prefix + "_" + std::to_string( width_ ) + "x" + std::to_string( height_ ) + "_" +
                    std::to_string( sizeof( T ) * 8 ) + "b_p444.rgb";
    std::ofstream file( filename, std::ios::binary );
    if ( !file.is_open() ) {
      printf( "Error: Unable to open file %s for writing.\n", filename.c_str() );
      return false;
    }
#if 0
    for ( const auto& plane : planes_ )
      file.write( reinterpret_cast<const char*>( plane.data() ), plane.size() * sizeof( T ) );
#else
    for ( const auto& plane : planes_ )
      for ( int32_t v = height_ - 1; v >= 0; v-- )
        file.write( reinterpret_cast<const char*>( plane.data( 0, v ) ), width_ * sizeof( T ) );
#endif
    if ( !file ) {
      printf( "Error: Failed to write image data to file: %s.\n", filename.c_str() );
      return false;
    }
    file.close();
    printf( "Save %s \n", filename.c_str() );
    return true;
  }

  void all_equal_zero() const {
    bool zero = true;
    for ( int32_t v = 0; v < height_; v++ )
      for ( int32_t c = 0; c < N; c++ )
        for ( int32_t u = 0; u < width_; u++ )
          if constexpr ( std::is_same_v<T, glm::vec4> ) {
            if ( static_cast<float>( planes_[c].get( u, v )[0] != 0 ) ) zero = false;
            if ( static_cast<float>( planes_[c].get( u, v )[1] != 0 ) ) zero = false;
            if ( static_cast<float>( planes_[c].get( u, v )[2] != 0 ) ) zero = false;
            if ( static_cast<float>( planes_[c].get( u, v )[3] != 0 ) ) zero = false;
          } else {
            if ( planes_[c].get( u, v ) != 0 ) zero = false;
          }
    printf( "all_equal_zero => %s \n", zero ? "TRUE" : "FALSE" );
  }

  void log( const std::string& str, int32_t u0 = 0, int32_t v0 = 0, int32_t number = 6 ) const {
    printf( "%s: %dx%d \n", str.c_str(), width_, height_ );
    for ( int32_t c = 0; c < N; c++ ) {
      printf( "%4s: ", c == 0 ? "R" : c == 1 ? "G" : c == 2 ? "B" : "A" );
      for ( int32_t u = u0; u < std::min( width_, u0 + number ); u++ )
        if constexpr ( std::is_floating_point<T>::value )
          printf( "%6d ", u );
        else
          printf( "%4d ", u );
    }
    printf( "\n" );
    for ( int32_t v = v0; v < std::min( height_, v0 + number ); v++ ) {
      for ( int32_t c = 0; c < N; c++ ) {
        printf( "%4d: ", v );
        for ( int32_t u = u0; u < std::min( width_, u0 + number ); u++ )
          if constexpr ( std::is_same_v<T, glm::vec4> )
            printf( "(%6.2f,%6.2f,%6.2f,%6.2f) ", static_cast<float>( planes_[c].get( u, v )[0] ),
                    static_cast<float>( planes_[c].get( u, v )[1] ), static_cast<float>( planes_[c].get( u, v )[2] ),
                    static_cast<float>( planes_[c].get( u, v )[3] ) );
          else if constexpr ( std::is_floating_point<T>::value )
            printf( "%6.2f ", static_cast<float>( planes_[c].get( u, v ) ) );
          else
            printf( "%4d ", static_cast<int32_t>( planes_[c].get( u, v ) ) );
      }
      printf( "\n" );
    }
    fflush( stdout );
  }

  void draw( char c, int32_t& x, int32_t& y ) {
    const int32_t charIndex = static_cast<int>( c );
    if ( charIndex < 0 || charIndex >= g_textCharacterNumber ) return;
    for ( int j = 0, p = g_textIndex[charIndex]; j < g_textHeight[charIndex]; ++j ) {
      for ( int i = 0; i < g_textWidth[charIndex]; ++i, p++ ) {
        int posX = x + i + g_textLeft[charIndex];
        int posY = height_ - 1 - ( y + 16 + j - g_textTop[charIndex] );
        if ( posX >= 0 && posX < width_ && posY >= 0 && posY < height_ ) {
          planes_[0].set( posX, posY, g_textCharacters[p] );
          planes_[1].set( posX, posY, g_textCharacters[p] );
          planes_[2].set( posX, posY, g_textCharacters[p] );
        }
      }
    }
    x += 11;
  }

  void draw( const std::string& text, int32_t x, int32_t y, bool left ) {
    if ( left ) { x -= text.size() * 11; }
    for ( char c : text ) draw( c, x, y );
  }
  T clamp( T v, T a, T b ) const { return ( ( v < a ) ? a : ( ( v > b ) ? b : v ) ); }

  void from( const Image<glm::vec4, 1>& buffer ) {
    if ( buffer.width() != width_ || buffer.height() != height_ ) {
      fprintf( stderr, "Error: Image dimensions mismatch (%dx%d vs %dx%d)\n", width_, height_, buffer.width(),
               buffer.height() );
      std::fflush( stderr );
      std::exit( -1 );
    }
    const int32_t size = width_ * height_;
    if constexpr ( std::is_same_v<T, uint8_t> ) {
      const float scale = 255.f;
      for ( int i = 0; i < size; ++i ) {
        const glm::vec4& c  = buffer.plane( 0 ).get( i );
        planes_[0].get( i ) = static_cast<T>( std::clamp( std::round( c.x * scale ), 0.f, 255.f ) );
        planes_[1].get( i ) = static_cast<T>( std::clamp( std::round( c.y * scale ), 0.f, 255.f ) );
        planes_[2].get( i ) = static_cast<T>( std::clamp( std::round( c.z * scale ), 0.f, 255.f ) );
        if constexpr ( N == 4 ) {
          planes_[3].get( i ) = static_cast<T>( std::round( std::clamp( c.w * scale + 0.5f, 0.f, 255.f ) ) );
        }
      }
    } else if constexpr ( std::is_same_v<T, uint16_t> ) {
      const float scale = 65535.f;
      for ( int i = 0; i < size; ++i ) {
        const glm::vec4& c  = buffer.plane( 0 ).get( i );
        planes_[0].get( i ) = static_cast<T>( std::clamp( std::round( c.x * scale ), 0.f, 65535.f ) );
        planes_[1].get( i ) = static_cast<T>( std::clamp( std::round( c.y * scale ), 0.f, 65535.f ) );
        planes_[2].get( i ) = static_cast<T>( std::clamp( std::round( c.z * scale ), 0.f, 65535.f ) );
        if constexpr ( N == 4 ) {
          planes_[3].get( i ) = static_cast<T>( std::clamp( std::round( c.w * scale ), 0.f, 65535.f ) );
        }
      }
    } else if constexpr ( std::is_same_v<T, float> ) {
      for ( int i = 0; i < size; ++i ) {
        const glm::vec4& c  = buffer.plane( 0 ).get( i );
        planes_[0].get( i ) = std::clamp( c.x, 0.f, 1.f );
        planes_[1].get( i ) = std::clamp( c.y, 0.f, 1.f );
        planes_[2].get( i ) = std::clamp( c.z, 0.f, 1.f );
        if constexpr ( N == 4 ) { planes_[3].get( i ) = std::clamp( c.w, 0.f, 1.f ); }
      }
    } else {
      fprintf( stderr, "Error: unsupported template type in %s\n", __FILE__ );
      std::fflush( stderr );
      std::exit( -1 );
    }
  }

  bool save( std::ofstream& file, const bool planar = false ) {
    if ( planar ) {
      for ( const auto& plane : planes_ )
        for ( int32_t v = height_ - 1; v >= 0; v-- )
          file.write( reinterpret_cast<const char*>( plane.data( 0, v ) ), width_ * sizeof( T ) );
    } else {
      for ( int32_t v = height_ - 1; v >= 0; v-- ) {
        for ( int32_t u = 0; u < width_; u++ ) {
          if ( N == 3 ) {
            for ( int32_t c = 0; c < N; c++ ) {
              T value = planes_[c].get( u, v );
              file.write( reinterpret_cast<const char*>( &value ), sizeof( T ) );
            }
          } else {
            T value = planes_[0].get( u, v ), zero = 0;
            file.write( reinterpret_cast<const char*>( &value ), sizeof( T ) );
            file.write( reinterpret_cast<const char*>( &zero ), sizeof( T ) );
            file.write( reinterpret_cast<const char*>( &zero ), sizeof( T ) );
          }
        }
      }
    }
    if ( !file ) {
      printf( "Error: Failed to write image data to file. \n" );
      return false;
    }
    return true;
  }

  bool loadPNG(const std::string& filename) {
    if constexpr (!std::is_same_v<T, uint8_t>) {
      printf("Error: loadPNG only supports uint8_t images.\n");
      return false;
    } else {
      int w, h, channels;
      stbi_set_flip_vertically_on_load(true); 
      unsigned char* data = stbi_load(filename.c_str(), &w, &h, &channels, N);
      if (!data) {
        printf("Error: Could not load PNG %s\n", filename.c_str());
        return false;
      }
      if (width_ != w || height_ != h) {
        resize(w, h);
      }
      for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
          for (int c = 0; c < N; ++c) {
            set(c, x, y, data[(y * width_ + x) * N + c]);
          }
        }
      }
      stbi_image_free(data);
      return true;
    }
  }

  bool savePNG(const std::string& filename) const {
    if constexpr (!std::is_same_v<T, uint8_t>) {
      printf("Error: savePNG only supports uint8_t images.\n");
      return false;
    } else {
      std::vector<uint8_t> interleaved(width_ * height_ * N);
      for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
          for (int c = 0; c < N; ++c) {
            interleaved[(y * width_ + x) * N + c] = get(c, x, y);
          }
        }
      }
      stbi_flip_vertically_on_write(true); 
      int stride_in_bytes = width_ * N;
      int success = stbi_write_png(filename.c_str(), width_, height_, N, interleaved.data(), stride_in_bytes);
      
      if (!success) {
        printf("Error: Failed to write PNG data to file: %s.\n", filename.c_str());
        return false;
      }
      return true;
    }
  }
  std::string computeMD5() const {
    detail::MD5 md5;
    for ( const auto& plane : planes_ ) {
      md5.update( reinterpret_cast<const uint8_t*>( plane.data() ), plane.size() * sizeof( T ) );
    }
    return md5.finalize();
  }
 private:
  int32_t               width_;
  int32_t               height_;
  std::vector<Plane<T>> planes_;
};

template <typename T, int N = 3>
class Video {
 public:
  Video( const Video& ) = default;
  Video( int32_t numFrames = 0, int32_t width = 0, int32_t height = 0, T value = 0 ) {
    resize( numFrames, width, height, value );
  }
  ~Video() = default;

  Video( const Video& videoA, const Video& videoB, bool butterfly = false ) {
    if ( videoA.height_ != videoB.height_ ) {
      printf( "Error: Heigth are not the same \n" );
      fflush( stdout );
      exit( -1 );
    }
    if ( videoA.size() != videoB.size() ) {
      printf( "Error: Size are not the same \n" );
      fflush( stdout );
      exit( -1 );
    }
    images_.clear();
    for ( size_t i = 0; i < videoA.size(); i++ ) images_.push_back( Image( videoA[i], videoB[i], butterfly ) );
    width_  = images_.back().width();
    height_ = images_.back().height();
  }

  void resize( int32_t numFrames, int32_t width, int32_t height, T value = 0 ) {
    width_  = width;
    height_ = height;
    images_.resize( numFrames );
    for ( size_t i = 0; i < images_.size(); i++ ) images_[i].resize( width, height, value );
  };

  template <typename D, int M>
  void add( const Image<D, M>& image ) {
    if ( size() > 0 && ( width_ != image.width() || height_ != image.height() ) ) {
      printf( "Error: image size are not the same \n" );
      fflush( stdout );
      exit( 0 );
    }
    if ( image.planeCount() != N || N != M ) {
      printf( "Error: image plane count is not the same \n" );
      fflush( stdout );
      exit( 0 );
    }
    width_  = image.width();
    height_ = image.height();
    if constexpr ( std::is_same_v<T, D> && N == M ) {
      images_.push_back( image );
    } else {
      Image<T, N> convert;
      convert = image;
      images_.push_back( convert );
    }
  }

  inline void zero() {
    for ( auto& el : images_ ) { el.zero(); }
  }
  inline void fill( T v ) {
    for ( auto& el : images_ ) { el.fill( v ); }
  }
  inline void clear() {
    for ( auto& el : images_ ) { el.clear(); }
  }

  void swap( Video<T, N>& src ) {
    std::swap( width_, src.width_ );
    std::swap( height_, src.height_ );
    std::swap( images_, src.images_ );
  }

  Image<T, N>&       plane( int32_t index ) { return images_[index]; }
  const Image<T, N>& plane( int32_t index ) const { return images_[index]; }
  Image<T, N>&       back() { return images_.back(); }
  auto               begin() { return images_.begin(); }
  auto               end() { return images_.end(); }
  auto               begin() const { return images_.cbegin(); }
  auto               end() const { return images_.cend(); }
  Image<T, N>&       operator[]( int32_t index ) { return images_[index]; }
  const Image<T, N>& operator[]( int32_t index ) const { return images_[index]; }
  int32_t            size() const { return int32_t( images_.size() ); }
  int32_t            width() const { return width_; }
  int32_t            height() const { return height_; }

  Video& operator=( const Video& ) = default;

  template <typename D, int M>
  Video& operator=( const Video<D, M>& src ) {
    resize( src.size(), src.width(), src.height() );
    for ( int32_t i = 0; i < src.size(); i++ ) { images_[i] = src[i]; }
    return *this;
  }

  Video& operator+=( Image<T, N>& image ) {
    this->add( image );
    return *this;
  }
  Video& operator*=( T scalar ) {
    for ( auto& image : images_ ) image *= scalar;
    return *this;
  }
  Video operator*( T scalar ) const {
    Video result;
    result.resize( size(), width_, height_ );
    for ( int32_t i = 0; i < N; i++ ) result[i] = this->images_[i] * scalar;
    return result;
  }

  bool save( const std::string& prefix, const bool planar = false ) const {
    auto filename = prefix + "_" + std::to_string( width_ ) + "x" + std::to_string( height_ ) + "_" +
                    std::to_string( sizeof( T ) * 8 ) + "b_" + ( planar ? "p" : "i" ) + "444" + ".rgb";
    createDirectory( filename );
    std::ofstream file( filename, std::ios::binary );
    if ( !file.is_open() ) {
      printf( "Error: Unable to open file %s for writing.\n", filename.c_str() );
      return false;
    }
    bool success = true; 
    for (auto & image : images_)
      success &= image.save( file, planar );
    file.close();
    printf( "Save %s (success = %d) \n", filename.c_str(), success );
    return success;
  }

  void log( const std::string& str, int32_t u0 = 0, int32_t v0 = 0, int32_t number = 6 ) const {
    printf( "%s: %d %dx%d \n", str.c_str(), size(), width_, height_ );
    for ( size_t i = 0; i < size(); i++ ) {
      printf( "Frame %4zu \n", i );
      for ( int32_t c = 0; c < N; c++ ) {
        printf( "%4s: ", c == 0 ? "R" : c == 1 ? "G" : c == 2 ? "B" : "A" );
        for ( int32_t u = u0; u < std::min( width_, u0 + number ); u++ )
          if constexpr ( std::is_floating_point<T>::value )
            printf( "%6d ", u );
          else
            printf( "%4d ", u );
      }
      printf( "\n" );
      for ( int32_t v = v0; v < std::min( height_, v0 + number ); v++ ) {
        for ( int32_t c = 0; c < N; c++ ) {
          printf( "%4d: ", v );
          for ( int32_t u = u0; u < std::min( width_, u0 + number ); u++ )
            if constexpr ( std::is_same_v<T, glm::vec4> )
              printf( "(%6.2f,%6.2f,%6.2f,%6.2f) ", static_cast<float>( images_[i][c].get( u, v )[0] ),
                      static_cast<float>( images_[i][c].get( u, v )[1] ),
                      static_cast<float>( images_[i][c].get( u, v )[2] ),
                      static_cast<float>( images_[i][c].get( u, v )[3] ) );
            else if constexpr ( std::is_floating_point<T>::value )
              printf( "%6.2f ", static_cast<float>( images_[i][c].get( u, v ) ) );
            else
              printf( "%3x ", static_cast<int32_t>( images_[i][c].get( u, v ) ) );
        }
        printf( "\n" );
      }
    }
    fflush( stdout );
  }

  void draw( const std::string& text, int32_t x, int32_t y, bool left = false ) {
    for ( auto& image : images_ ) image.draw( text, x, y, left );
  }
  T clamp( T v, T a, T b ) const { return ( ( v < a ) ? a : ( ( v > b ) ? b : v ) ); }

 private:
  int32_t                  width_;
  int32_t                  height_;
  std::vector<Image<T, N>> images_;
};

template <typename T, int N>
Video<T, N> operator-( const Video<T, N>& a, const Video<T, N>& b ) {
  assert( a.size() == b.size() );
  assert( a.width() == b.width() );
  assert( a.height() == b.height() );
  Video<T, N> result;
  result.resize( a.size(), a.width(), a.height() );
  for ( int32_t i = 0; i < a.size(); ++i )
    for ( int32_t v = 0; v < a.height(); ++v )
      for ( int32_t u = 0; u < a.width(); ++u )
        for ( int32_t c = 0; c < N; ++c ) {
          int diff = int( a[i].get( c, u, v ) ) - int( b[i].get( c, u, v ) );
          result[i].set( c, u, v, static_cast<T>( std::clamp( diff * 10 + 128, 0, 255 ) ) );
        }
  return result;
}