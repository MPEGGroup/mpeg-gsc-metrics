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
#include "rasterizerHardware.hpp"
#include "pointcloud.hpp"
#include "image.hpp"
#include "viewpoints.hpp"
#include "shaders.hpp"

RasterizerHardware::RasterizerHardware() {}

RasterizerHardware::~RasterizerHardware() {
  unload();
  reset();
}

#if defined( USE_GLFW )
static void error_callback( int error, const char* description ) { std::cerr << "Error: " << description << std::endl; }

bool RasterizerHardware::initialize() {
  reset();
  glfwSetErrorCallback( error_callback );

  if ( !glfwInit() ) return false;
  glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, SHADER_VERSION_MAJOR );
  glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, SHADER_VERSION_MINOR );
  glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );
  glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
  glfwWindowHint( GLFW_SRGB_CAPABLE, GLFW_TRUE );
  glfwWindowHint( GLFW_VISIBLE, GLFW_FALSE );
  window_ = glfwCreateWindow( width_, height_, "Window", nullptr, nullptr );
  if ( !window_ ) {
    printf( "Systen can't initilialize glfw windows \n" );
    fflush( stdout );
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent( window_ );
  if ( !gladLoadGLLoader( (GLADloadproc)glfwGetProcAddress ) ) {
    printf( "Systen can't initilialize GLAD \n" );
    fflush( stdout );
    glfwTerminate();
    return false;
  }
  program_.create( "shader", g_splatVertexShader, g_splatFragmentShaderWithOcm );
  glEnable( GL_BLEND );
  glBlendFunc( GL_ONE, GL_ONE_MINUS_SRC_ALPHA );
  glEnable( GL_DEPTH_TEST );
  glDepthMask( GL_TRUE );
  glGenFramebuffers( 1, &fbo_ );
  glBindFramebuffer( GL_FRAMEBUFFER, fbo_ );
  glGenRenderbuffers( 1, &fboTexture_ );
  glBindRenderbuffer( GL_RENDERBUFFER, fboTexture_ );
  glRenderbufferStorage( GL_RENDERBUFFER, GL_RGBA32F, width_, height_ );
  glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, fboTexture_ );
  GLenum status;
  status = glCheckFramebufferStatus( GL_FRAMEBUFFER );
  if ( status != GL_FRAMEBUFFER_COMPLETE ) {
    printf( "Error: FBO not complete \n" );
    fflush( stdout );
    glfwTerminate();
    return false;
  }

  glGenTextures( 1, &ocm_ );
  glBindTexture( GL_TEXTURE_2D, ocm_ );
  glTexStorage2D( GL_TEXTURE_2D, 1, GL_R8, width_, height_ );
  glBindTexture( GL_TEXTURE_2D, 0 );
  glBindImageTexture( 2, ocm_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R8 );
  return true;
}

void RasterizerHardware::reset() {
  if ( window_ ) {
    glDeleteFramebuffers( 1, &fbo_ );
    glDeleteRenderbuffers( 1, &fboTexture_ );
    glfwDestroyWindow( window_ );
    glfwTerminate();
    window_ = nullptr;
  }
}

void RasterizerHardware::load( const Pointcloud& pc ) {
  unload();

  // Create VAO
  glGenVertexArrays( 1, &vao_ );
  glBindVertexArray( vao_ );

  // Create VBO for gaussian data
  glGenBuffers( 1, &vboGaussianData_ );
  glBindBuffer( GL_SHADER_STORAGE_BUFFER, vboGaussianData_ );
  glBufferData( GL_SHADER_STORAGE_BUFFER, sizeof( GaussianData ) * pc.size(), pc.data(), GL_DYNAMIC_DRAW );
  glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, vboGaussianData_ );
  glBindBuffer( GL_SHADER_STORAGE_BUFFER, 0 );

  // Create VBO for index data
  std::vector<uint32_t> index( pc.size() );
  glGenBuffers( 1, &vboIndex_ );
  glBindBuffer( GL_SHADER_STORAGE_BUFFER, vboIndex_ );
  glBufferData( GL_SHADER_STORAGE_BUFFER, sizeof( uint32_t ) * index.size(), index.data(), GL_DYNAMIC_DRAW );
  glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, vboIndex_ );
  glBindBuffer( GL_SHADER_STORAGE_BUFFER, 0 );

  // Disable VAO and VBO
  glBindBuffer( GL_ARRAY_BUFFER, 0 );
  glBindBuffer( GL_SHADER_STORAGE_BUFFER, 0 );
  glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
  glBindBuffer( GL_ATOMIC_COUNTER_BUFFER, 0 );
  glBindVertexArray( 0 );
  loaded_ = true;
}

void RasterizerHardware::unload() {
  if ( loaded_ ) {
    glDeleteBuffers( 1, &vboGaussianData_ );
    glDeleteBuffers( 1, &vboIndex_ );
    glDeleteVertexArrays( 1, &vao_ );
    loaded_ = false;
  }
}

template <typename T, int N>
void RasterizerHardware::render( const Pointcloud&  pc,
                                 const Viewpoint&   viewpoint,
                                 Image<T, N>&       image,
                                 Image<uint8_t, 1>& ocm ) {
  if ( width_ != image.width() || height_ != image.height() ) {
    width_  = image.width();
    height_ = image.height();
    if ( !initialize() ) {
      printf( "Error during the GLFW window creation \n" );
      fflush( stdout );
      exit( -1 );
    }
  }
  load( pc );

  // Camera
  auto      focal = viewpoint.focal();
  auto      center = viewpoint.center();
  if ( viewpoint.originalHeight() > 0 ) {
    float scale = static_cast<float>(image.height()) / static_cast<float>(viewpoint.originalHeight());
    focal = focal * scale;
    center = center * scale;
  }
  Camera camera;
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
  glm::mat4 viewMat       = camera.viewMat();
  auto      pos           = camera.getPosition();
  auto      nearFar       = camera.nearFar();
  auto      modelViewProj = projMat * viewMat;
  int32_t   numPoints     = pc.size();

  // Initialize opengl
  glBindVertexArray( vao_ );
  glBindFramebuffer( GL_FRAMEBUFFER, fbo_ );
  glViewport( 0, 0, width_, height_ );
  glClearColor( 0.f, 0.f, 0.0f, 1.f );
  glClearDepth( 0.0f );
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
  glEnable( GL_DEPTH_TEST );
  glEnable( GL_BLEND );
  glBlendEquation( GL_FUNC_ADD );
  glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
  glEnable( GL_DEPTH_CLAMP );

  // Sort
  auto sorted = pc.sort( modelViewProj, nearFar );
  setBuffer( vboIndex_, sorted );
  int32_t count = (int32_t)sorted.size();

  // Rasterize
  program_.use();
  program_.setUniform( "viewMat", viewMat );
  program_.setUniform( "projMat", projMat );
  program_.setUniform( "eye", pos );
  program_.setUniform( "focal", focal );
  program_.setUniform( "window", glm::vec4( width_, height_, nearFar ) );
  GLubyte zero = 0;
  glClearTexImage( ocm_, 0, GL_RED, GL_UNSIGNED_BYTE, &zero );
  glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, vboGaussianData_ );
  glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, vboIndex_ );
  glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
  glDrawArraysInstanced( GL_TRIANGLE_STRIP, 0, 4, count );
  glBindBuffer( GL_SHADER_STORAGE_BUFFER, 0 );
  glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
  glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );

  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  // Image
  glReadBuffer( GL_COLOR_ATTACHMENT0 );
  Image<glm::vec4, 1> buffer( width_, height_, glm::vec4( 0 ) );
  glReadPixels( 0, 0, width_, height_, GL_RGBA, GL_FLOAT, buffer[0].data() );
  image.from( buffer );

  // Occupancy map
  ocm.fill( 0 );
  glBindTexture( GL_TEXTURE_2D, ocm_ );
  glGetTexImage( GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, ocm[0].data() );

  program_.stop();
  glBindVertexArray( 0 );
  glBindBuffer( GL_ARRAY_BUFFER, 0 );
  glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
  glBindFramebuffer( GL_FRAMEBUFFER, 0 );
}
#else
bool RasterizerHardware::initialize() { return true; }
void RasterizerHardware::reset() {}
void RasterizerHardware::load( const Pointcloud& pc ) {}
void RasterizerHardware::unload() {}
template <typename T, int N>
void RasterizerHardware::render( const Pointcloud&  pc,
                                 const Viewpoint&   viewpoint,
                                 Image<T, N>&       image,
                                 Image<uint8_t, 1>& ocm ) {}
#endif

template void RasterizerHardware::render<uint8_t>( Pointcloud const&,
                                                   const Viewpoint&,
                                                   Image<uint8_t>&,
                                                   Image<uint8_t, 1>& );
template void RasterizerHardware::render<uint16_t>( Pointcloud const&,
                                                    const Viewpoint&,
                                                    Image<uint16_t>&,
                                                    Image<uint8_t, 1>& );
template void RasterizerHardware::render<float>( Pointcloud const&,
                                                 const Viewpoint&,
                                                 Image<float>&,
                                                 Image<uint8_t, 1>& );
