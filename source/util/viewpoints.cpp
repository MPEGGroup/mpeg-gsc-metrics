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
#include "viewpoints.hpp"
#include "program.hpp"
#include "pointcloud.hpp"
#include "shaders.hpp"

class Arrow {
 public:
  Arrow() {}
  ~Arrow() {
    if ( number_ != -1 ) {
      glDeleteBuffers( 1, &vbo_ );
      glDeleteVertexArrays( 1, &vao_ );
      number_ = -1;
    }
  }

  void load( glm::vec3 A, glm::vec3 B, float fA, float fB, glm::vec4 color ) {
    if ( !program_.isLoad() ) {
      static const std::string vertexShader =
          SHADER( layout( location = 0 ) in vec3 vertex_position; uniform mat4 ProjMat; uniform mat4 ModMat;
                  void main() { gl_Position = ProjMat * ModMat * vec4( vertex_position, 1.0 ); } );
      static const std::string fragmentShader =
          SHADER( uniform vec4 color; out vec4 frag_colour; void main() { frag_colour = color; } );
      program_.create( "Color", vertexShader, fragmentShader );
    }
    std::vector<glm::vec3> points;
    points.push_back( A );
    points.push_back( B );
    glm::vec3 X = B - A, Z, Y;
    X *= fA / glm::length( X );
    glm::vec3 V0( fB * ( -X[1] ) / fA, fB * ( X[0] ) / fA, 0 );
    glm::vec3 V1( fB * ( -X[2] ) / fA, 0, fB * ( X[0] ) / fA );
    glm::vec3 V2( 0, fB * ( -X[2] ) / fA, fB * ( X[1] ) / fA );
    float     fNormZ0 = glm::dot( V0, V0 ), fNormZ1 = glm::dot( V1, V1 ), fNormZ2 = glm::dot( V2, V2 );
    if ( fNormZ1 > fNormZ2 ) {
      Z = V1;
      Y = fNormZ0 > fNormZ2 ? V0 : V2;
    } else {
      Z = V2;
      Y = fNormZ0 > fNormZ1 ? V0 : V1;
    }
    points.push_back( B );
    points.push_back( B - X + Y - Z );
    points.push_back( B - X + Y + Z );
    points.push_back( B - X - Y + Z );
    points.push_back( B );
    points.push_back( B - X - Y - Z );
    points.push_back( B - X + Y - Z );
    points.push_back( B - X - Y + Z );
    glGenBuffers( 1, &vbo_ );
    glBindBuffer( GL_ARRAY_BUFFER, vbo_ );
    glBufferData( GL_ARRAY_BUFFER, points.size() * sizeof( glm::vec3 ), points.data(), GL_STATIC_DRAW );
    glGenVertexArrays( 1, &vao_ );
    glBindVertexArray( vao_ );
    glBindBuffer( GL_ARRAY_BUFFER, vbo_ );
    glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, NULL );
    glEnableVertexAttribArray( 0 );
    number_ = (int)points.size();
    color_  = color;
  }

  void draw( glm::mat4& eMatMod, glm::mat4& eMatPro ) {
    program_.use();
    program_.setUniform( "ModMat", eMatMod );
    program_.setUniform( "ProjMat", eMatPro );
    program_.setUniform( "color", color_ );
    glBindVertexArray( vao_ );
    glDrawArrays( GL_TRIANGLE_STRIP, 0, number_ );
    glDrawArrays( GL_LINES, 0, number_ );
    glBindVertexArray( 0 );
    program_.stop();
  }

 private:
  GLuint    vao_ = 0;
  GLuint    vbo_ = 0;
  Program   program_;
  int       number_ = -1;
  glm::vec4 color_;
};

#if defined( USE_GLFW )
void Viewpoints::mouseCallback( GLFWwindow* window, double xpos, double ypos ) {
  if ( instance_ ) {
    int width = 0, height = 0;
    glfwGetWindowSize( window, &width, &height );
    const int r = width < height ? width : height;
    xpos        = ( 2.0 * xpos - width ) / r;
    ypos        = ( height - 2.0 * ypos ) / r;
    if ( glfwGetMouseButton( window, GLFW_MOUSE_BUTTON_LEFT ) == GLFW_PRESS ) {
      if ( instance_->firstMouse_ ) {
        instance_->lastX_      = xpos;
        instance_->lastY_      = ypos;
        instance_->firstMouse_ = false;
      }
      instance_->camera_.rotate( instance_->lastX_, instance_->lastY_, xpos, ypos );
    } else if ( glfwGetMouseButton( window, GLFW_MOUSE_BUTTON_RIGHT ) == GLFW_PRESS ) {
      if ( instance_->firstMouse_ ) {
        instance_->lastX_      = xpos;
        instance_->lastY_      = ypos;
        instance_->firstMouse_ = false;
      }
      instance_->camera_.translate( ( instance_->lastX_ - xpos ) * ( width / 50 ),   //
                                    ( instance_->lastY_ - ypos ) * ( height / 50 ),  //
                                    0.f );
    } else {
      instance_->firstMouse_ = true;
    }
    instance_->lastX_ = xpos;
    instance_->lastY_ = ypos;
  }
}

void Viewpoints::scrollCallback( GLFWwindow* window, double xoffset, double yoffset ) {
  if ( instance_ ) { instance_->camera_.zoom( static_cast<float>( yoffset * 10 ) ); }
}

void Viewpoints::keyCallback( GLFWwindow* window, int key, int scancode, int action, int mods ) {
  if ( instance_ ) {
    if ( action == GLFW_PRESS || action == GLFW_REPEAT ) {
      switch ( key ) {
        case GLFW_KEY_T: {
          if ( instance_->viewpoints_.size() > instance_->viewpointIndex_ ) {
            auto& el = instance_->viewpoints_[instance_->viewpointIndex_];
            instance_->camera_.setLookAt( el.pos(), el.view(), el.up() );
            instance_->focal_          = el.getAdjustedFocal( instance_->originalHeight_ > 0 ? instance_->originalHeight_ : instance_->camera_.getHeight() );
            instance_->originalWidth_  = el.originalWidth();
            instance_->originalHeight_ = el.originalHeight();
            instance_->viewpointIndex_ = ( instance_->viewpointIndex_ + 1 ) % instance_->viewpoints_.size();
            
            if (instance_->originalHeight_ > 0) {
              float scale = static_cast<float>(instance_->camera_.getHeight()) / static_cast<float>(instance_->originalHeight_);
              glm::vec2 adjustedFocal = el.focal() * scale;
              float fov = 2 * atan( 1.0f / ( 2 * adjustedFocal[1] / instance_->camera_.getHeight() ) );
              instance_->camera_.setFov(fov);
            }
          }
        } break;
        case GLFW_KEY_D: {
          std::tuple<glm::vec3, glm::vec3, glm::vec3> value = instance_->camera_.getLookAt();
          printf( "%12.6f %12.6f %12.6f     %12.6f %12.6f %12.6f      %12.6f %12.6f %12.6f \n",  //
                  std::get<0>( value ).x, std::get<0>( value ).y, std::get<0>( value ).z,        //
                  std::get<1>( value ).x, std::get<1>( value ).y, std::get<1>( value ).z,        //
                  std::get<2>( value ).x, std::get<2>( value ).y, std::get<2>( value ).z );
          fflush( stdout );
        } break;
        default: break;
      }
    }
  }
}

glm::vec4 getRandomColor() {
  float r = static_cast<float>( rand() ) / static_cast<float>( RAND_MAX );
  float g = static_cast<float>( rand() ) / static_cast<float>( RAND_MAX );
  float b = static_cast<float>( rand() ) / static_cast<float>( RAND_MAX );
  float a = 1.0f;
  return glm::vec4( r, g, b, a );
}

void Viewpoints::display( const Pointcloud& pc, int width, int height ) {
  // Initialize glfw windows
  if ( !glfwInit() ) {
    std::cerr << "Erreur : impossible d'initialiser GLFW\n";
    return;
  }
  glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, SHADER_VERSION_MAJOR );
  glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, SHADER_VERSION_MINOR );
  glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
#if defined( __APPLE__ )
  glfwWindowHint( GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE );
#endif
  auto window = glfwCreateWindow( width, height, "Points and Viewpoints Display", nullptr, nullptr );
  if ( !window ) {
    std::cerr << "Erreur : impossible de créer la fenêtre GLFW\n";
    glfwTerminate();
    return;
  }
  glfwMakeContextCurrent( window );
  if ( !gladLoadGLLoader( (GLADloadproc)glfwGetProcAddress ) ) {
    std::cerr << "Erreur : impossible d'initialiser GLAD\n";
    return;
  }
  glfwSetCursorPosCallback( window, Viewpoints::mouseCallback );
  glfwSetScrollCallback( window, Viewpoints::scrollCallback );
  glfwSetKeyCallback( window, Viewpoints::keyCallback );

  // Initialize camera
  glm::vec3 eye = glm::vec3( -15, 0, -2 ), center = glm::vec3( 0, 0.5, 0 ), up = glm::vec3( 0, -1, 0 );
  auto      cameraPosition = pc.getCameraPosition();
  
  
  if ( viewpoints_.size() > 0 ) {
    focal_ = viewpoints_[0].focal();
    originalWidth_  = viewpoints_[0].originalWidth();
    originalHeight_ = viewpoints_[0].originalHeight();
    glm::vec2 adjustedFocal = focal_;
    if (originalHeight_ > 0) {
      float scale = static_cast<float>(height) / static_cast<float>(originalHeight_);
      adjustedFocal = focal_ * scale;
    }
    float fov = 2 * atan( 1.0f / ( 2 * adjustedFocal[1] / height ) );
    camera_.create( 10.f, fov );
    camera_.initialize( width, height );
    if (viewpoints_.size() > viewpointIndex_) {
      auto& el = viewpoints_[viewpointIndex_];
      camera_.setLookAt( el.pos(), el.view(), el.up() );
    } else {
      camera_.setLookAt( eye, center, up );
    }
  } else {
    float fov = 2 * atan( 1.0f / ( 2 * focal_[1] / height ) );
    camera_.create( 10.f, fov );
    camera_.initialize( width, height );
    camera_.setLookAt( eye, center, up );
  }

  // Create program
  Program programPoints;
  programPoints.create( "Splat", g_splatVertexShader, g_splatFragmentShader );
  programPoints.use();

  // Create VAO
  GLuint vao, vboGaussianData, vboIndex;
  glGenVertexArrays( 1, &vao );
  glBindVertexArray( vao );
  const uint32_t stride = (uint32_t)sizeof( GaussianData );

  // Create VBO for gaussian data
  glGenBuffers( 1, &vboGaussianData );
  glBindBuffer( GL_SHADER_STORAGE_BUFFER, vboGaussianData );
  glBufferData( GL_SHADER_STORAGE_BUFFER, sizeof( GaussianData ) * pc.size(), pc.data(), GL_DYNAMIC_DRAW );
  glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, vboGaussianData );
  glBindBuffer( GL_SHADER_STORAGE_BUFFER, 0 );

  // Create VBO for index data
  int32_t               numPoints = pc.size();
  const uint32_t        maxValue  = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> depth( numPoints );
  std::vector<uint32_t> index( numPoints );
  glGenBuffers( 1, &vboIndex );
  glBindBuffer( GL_SHADER_STORAGE_BUFFER, vboIndex );
  glBufferData( GL_SHADER_STORAGE_BUFFER, sizeof( uint32_t ) * index.size(), index.data(), GL_DYNAMIC_DRAW );
  glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, vboIndex );
  glBindBuffer( GL_SHADER_STORAGE_BUFFER, 0 );

  // Initialize opengl
  programPoints.stop();
  glClearColor( 0.f, 0.f, 0.f, 1.f );
  glClearDepth( 0.0 );
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
  glEnable( GL_DEPTH_TEST );
  glEnable( GL_BLEND );
  glBlendEquation( GL_FUNC_ADD );
  glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
  glEnable( GL_DEPTH_CLAMP );
  glViewport( 0, 0, width, height );
  glLineWidth( 2 );

  // Create arrows for axes, viewpoints
  glm::vec3          coords[3] = { glm::vec3( 1, 0, 0 ), glm::vec3( 0, 1, 0 ), glm::vec3( 0, 0, 1 ) };
  std::vector<Arrow> axes( 3 );
  for ( size_t i = 0; i < axes.size(); i++ )
    axes[i].load( glm::vec3( 0, 0, 0 ), coords[i], 0.1, 0.05, glm::vec4( coords[i], 1. ) );
  std::vector<Arrow> arrowViewpoints( viewpoints_.size() );
  for ( size_t i = 0; i < viewpoints_.size(); i++ )
    arrowViewpoints[i].load( viewpoints_[i].pos() + glm::normalize( viewpoints_[i].pos() - viewpoints_[i].view() ),
                             viewpoints_[i].pos(), 0.1, 0.05, getRandomColor() );

  glm::vec2 currentFocal = focal_;
  if (originalHeight_ > 0) {
    float scale = static_cast<float>(height) / static_cast<float>(originalHeight_);
    currentFocal = focal_ * scale;
  }

  // Rendering loop
  while ( !glfwWindowShouldClose( window ) ) {
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glClearColor( 0.f, 0.f, 0.f, 1.f );
    auto projMat       = camera_.projMat();
    auto viewMat       = camera_.viewMat();
    auto modelViewProj = projMat * viewMat;
    glBindVertexArray( vao );

    // Sorting
    auto sorted = pc.sort( modelViewProj, camera_.nearFar() );
    setBuffer( vboIndex, sorted );
    int32_t count = ( int32_t )( sorted.size() );

    // Set uniforms
    programPoints.use();
    programPoints.setUniform( "viewMat", viewMat );
    programPoints.setUniform( "projMat", projMat );
    programPoints.setUniform( "eye", camera_.getPosition() );
    programPoints.setUniform( "focal", currentFocal );
    programPoints.setUniform( "window", glm::vec4( (float)width, (float)height, camera_.nearFar() ) );

    // Rendering
    glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, vboGaussianData );
    glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, vboIndex );
    glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
    glDrawArraysInstanced( GL_TRIANGLE_STRIP, 0, 4, count );
    glBindBuffer( GL_SHADER_STORAGE_BUFFER, 0 );
    programPoints.stop();
    glDisable( GL_DEPTH_TEST );
    glDepthMask( GL_FALSE );
    for ( auto& el : axes ) el.draw( viewMat, projMat );
    for ( auto& el : arrowViewpoints ) el.draw( viewMat, projMat );

    glfwSwapBuffers( window );
    glfwPollEvents();
  }
  glDeleteBuffers( 1, &vboGaussianData );
  glDeleteBuffers( 1, &vboIndex );
  glDeleteVertexArrays( 1, &vao );
  glfwDestroyWindow( window );
  glfwTerminate();
}
#else
void Viewpoints::display( const Pointcloud& pc, int width, int height ) {}
#endif

Viewpoints* Viewpoints::instance_ = nullptr;