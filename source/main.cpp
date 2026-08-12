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

#include "common.hpp"
#include "image.hpp"
#include "pointcloud.hpp"
#include "rasterizer.hpp"
#include "viewpoints.hpp"
#include "program-options-lite/program_options_lite.h"
#include "version.hpp"
#include "metric.hpp"
#include "timer.hpp"

#if defined( USE_OPENMP )
#include <omp.h>
#endif

// Parse input command line
struct Parameters {
  bool        verbose_           = false;
  bool        help_              = false;
  bool        useCPU_            = false;
  bool        useFloat_          = false;
  bool        display_           = false;
  bool        saveViewpoints_    = false;
  bool        save_              = false;
  bool        useCameraPosition_ = false;
  bool        compareCpuGpu_     = false;
  std::string sourcePath_        = "";
  std::string decodePath_        = "";
  std::string viewpointPath_     = "";
  int32_t     startFrame_        = 0;
  int32_t     frameCount_        = 1;
  int32_t     numPoints_         = 16;
  int32_t     onlyViewpoint_     = -1;
  int32_t     width_             = 1024;
  int32_t     height_            = 768;
  int32_t     numThreads_        = 0;
  std::string outputPath_        = "";
  bool        computeOm_         = true;
  bool        computeSsim_       = true;
  bool        computeIvssim_     = false;
  bool        computeQmiv_       = false;

  bool parse( int32_t argc, char* argv[] ) {
    namespace po = df::program_options_lite;
    po::Options opts;
    /* clang-format off */
    opts.addOptions()
      (po::Section("Common"))
        ("h,help",            help_,              false,              "This help text")
        ("c,config",          po::parseConfigFile,                    "Configuration file name")
        ("v,verbose",         verbose_,           verbose_,           "Verbose")

      ( po::Section("Input paths") )      
        ( "a,source",         sourcePath_,        sourcePath_,        "Source 3DGS path" )
        ( "b,decode",         decodePath_,        decodePath_,        "Decode 3DGS path" )

      (po::Section("Sequence information"))       
        ("i,startFrame",      startFrame_,        startFrame_,        "First frame number")
        ("f,frameCount",      frameCount_,        frameCount_,        "Number of frames")
        ("width",             width_,             width_,             "Width of the generated images")
        ("height",            height_,            height_,            "Height of the generated images")
      
      (po::Section("Viewpoints"))        
        ("viewpoint",         viewpointPath_,     viewpointPath_,     "Viewpoint path used to load viewpoints")
        ("useCameraPosition", useCameraPosition_, useCameraPosition_, "Use camera position stored in ply file")
        ("n,numPoints",       numPoints_,         numPoints_,         "Create N random viewpoints")
        ("display",           display_,           display_,           "Show 3D model and viewpoints")
        ("saveViewpoints",    saveViewpoints_,    saveViewpoints_,    "Save the generated viewpoints")
        ("onlyViewpoint",     onlyViewpoint_,     onlyViewpoint_,     "Only render the viewpoint")

      (po::Section("Rendering"))  
        ("cpu",               useCPU_,            useCPU_,            "Use CPU rendering")
        ("float",             useFloat_,          useFloat_,          "Use floatting point images")
        ("compareCpuGpu",     compareCpuGpu_,     compareCpuGpu_,     "Compare CPU/GPU rendering")
        ("threads",           numThreads_,        numThreads_,        "Number of threads to use by OpenMP (0=auto)")

      (po::Section("Metrics"))
        ("computeOm",         computeOm_,         computeOm_,         "Compute OM metrics (OM-PSNR and OM-SSIM)")
        ("computeSsim",       computeSsim_,       computeSsim_,       "Compute SSIM metrics")
        ("computeIvssim",     computeIvssim_,     computeIvssim_,     "Compute IVSSIM metrics")
        ("computeQmiv",       computeQmiv_,       computeQmiv_,       "Compute color transform and average as QMIV")
        
      ( po::Section("Output") )      
        ("s,save",            save_,              save_,              "Save generated videos")
        ("o,output",          outputPath_,        outputPath_,        "Output text file for metrics")
      ;

    /* clang-format on */
    po::setDefaults( opts );
    po::ErrorReporter err;
    const std::list<const char*>& argv_unhandled = po::scanArgv( opts, argc, (const char**)argv, err );
    if ( argc == 1 || help_ || sourcePath_.empty() || decodePath_.empty() || argv_unhandled.size() != 0 ||
         err.is_errored ) {
      std::cout << "\nusage: " << argv[0] << " [arguments...] \n";
      po::doHelp( std::cout, opts, 78 );
      std::cout << "\n";
      if ( sourcePath_.empty() || decodePath_.empty() ) printf( "Warning: input files are not defined. \n" );
      for ( const auto* const arg : argv_unhandled ) { err.warn() << "Unhandled argument ignored: " << arg << '\n'; }
      if ( err.is_errored ) err.error() << "Unhandled argument \n";
      return 1;
    }
    if ( useFloat_ && save_ ) {
      printf(
          "Warning: Floating point images are not compatible with the recording option, disabling the creation of "
          "output videos\n" );
      save_ = false;
    }
    if ( ( !useCameraPosition_ ) && viewpointPath_.empty() && numPoints_ == 0 ) {
      printf( "Viewpoints configuration is not correclty defined \n" );
      return 1;
    }
    if ( verbose_ ) {
      dumpCfg( std::cout, opts );
    }
    return 0;
  }
};

std::string getOuputName( const std::string& path ) {
  auto str = getRemoveExtension( path );
  auto pos = str.find( '%' );
  if ( pos == 0 ) return "video";
  if ( pos != std::string::npos )
    if ( str[pos - 1] == '_' ) pos = pos - 1;
  return ( pos == std::string::npos ) ? str : str.substr( 0, pos );
}

int main( int32_t argc, char* argv[] ) {
  printf( "MPEG GSC metric version %s \n", ::gscm::version );
  fflush( stdout );

  Parameters params;
  if ( params.parse( argc, argv ) ) { return 1; }
  auto start = std::chrono::high_resolution_clock::now();

  Video<uint8_t, 1> videoOcm[2];
  Video<uint8_t, 3> videoImg[2];
  Metric            metric( params.computeOm_, params.computeSsim_, params.computeIvssim_, params.computeQmiv_ );
  Rasterizer        rasterizer( params.useCPU_ ), rasterizerCpu( true ), rasterizerGpu( false );
  Timer             timer;
#if defined( USE_OPENMP )
  omp_set_dynamic(0);
  if ( params.numThreads_ > 0 ) {
    omp_set_num_threads( params.numThreads_ );
  } else {
    omp_set_num_threads(omp_get_num_procs());
  }
#endif
  for ( int32_t frameIndex = 0; frameIndex < params.frameCount_; frameIndex++ ) {
    
    // Load pointclouds
    Pointcloud pc[2];
    timer.tic( "Read" );
    if ( !pc[0].read( params.sourcePath_, params.startFrame_ + frameIndex, params.verbose_ ) ) return 1;
    if ( !pc[1].read( params.decodePath_, params.startFrame_ + frameIndex, params.verbose_ ) ) return 1;
    timer.toc( "Read", params.verbose_ );
    if ( params.verbose_ ){
      for ( int32_t i = 0; i < 2; i++ ) pc[i].trace();
    }

    // Create viewpoint list
    Viewpoints viewpoints;
    if ( !params.viewpointPath_.empty() ) { viewpoints.read( params.viewpointPath_ ); }
    if ( viewpoints.size() == 0 ) {
      if ( params.useCameraPosition_ ) {
        auto& cameraPosition0 = pc[0].getCameraPosition();
        auto& cameraPosition1 = pc[1].getCameraPosition();
        if ( cameraPosition0.size() == 0 && cameraPosition1.size() == 0 ) {
          printf( "Error: camera position is not correclty defined in the ply files \n" );
          return 1;
        }
        if ( cameraPosition0.size() == 0 ) cameraPosition0 = cameraPosition1;
        if ( cameraPosition1.size() == 0 ) cameraPosition1 = cameraPosition0;
        if ( cameraPosition0 != cameraPosition1 ) {
          printf( "Error: camera position of the two ply files are not equals\n" );
          printf( "cameraPosition0: \n" );
          for ( auto& el : cameraPosition0 ) el.log();
          printf( "cameraPosition1: \n" );
          for ( auto& el : cameraPosition1 ) el.log();
          fflush( stdout );
          return 1;
        }
        if (params.onlyViewpoint_ != -1 && params.onlyViewpoint_ < cameraPosition0.size()) {
          viewpoints.resize(1);
          const auto& el = cameraPosition0[params.onlyViewpoint_];
          viewpoints[0].set( el.pos(), el.view(), el.up(), el.focal_, el.width_, el.height_, el.center_ );
          if ( params.verbose_ ) { 
            printf("onlyViewpoint == %d, keep only one viewpoint \n", params.onlyViewpoint_);
            el.log();
          }
        } else {
          viewpoints.resize( cameraPosition0.size() );
          for ( size_t i = 0; i < cameraPosition0.size(); i++ ) {
            const auto& el = cameraPosition0[i];
            viewpoints[i].set( el.pos(), el.view(), el.up(), el.focal_, el.width_, el.height_, el.center_ );
            if ( params.verbose_ ) el.log();
         }
        }
        params.numPoints_ = viewpoints.size();
      } else {
        auto [minBound, maxBound] = pc[0].getBoundingBox();
        auto center               = ( maxBound + minBound ) * 0.5f;
        if ( params.verbose_ ) {
          logVec3( "minBound", minBound );
          logVec3( "maxBound", maxBound );
          logVec3( "center", center );
        }
        viewpoints.create( params.numPoints_, center );
        if ( params.saveViewpoints_ ) viewpoints.write( "viewpoint.txt" );
        if ( params.verbose_ ) {
          printf( "Create N random points N = %d \n", params.numPoints_ );
          logVec3( "minBound", minBound );
          logVec3( "maxBound", maxBound );
          viewpoints.trace();
        }
      }
    }
    if ( params.display_ ) {
      viewpoints.trace();
      viewpoints.display( pc[1], params.width_, params.height_ );
      return 0;
    }

    // Resize
    if ( params.save_ ) {
      auto numFrames = videoImg[0].size() + viewpoints.size();
      for ( int32_t i = 0; i < 2; i++ ) videoImg[i].resize( numFrames, params.width_, params.height_ );
      for ( int32_t i = 0; i < 2; i++ ) videoOcm[i].resize( numFrames, params.width_, params.height_ );
    }

    // Rendering
#pragma omp parallel if ( params.useCPU_ )
    {
#pragma omp for
      for ( int32_t viewpointIndex = 0; viewpointIndex < (int32_t)viewpoints.size(); viewpointIndex++ ) {
        auto  imageFloat = std::array<Image<float>, 2>{ Image<float, 3>( params.width_, params.height_ ),
                                                        Image<float, 3>( params.width_, params.height_ ) };
        auto  imageUint8 = std::array<Image<uint8_t>, 2>{ Image<uint8_t, 3>( params.width_, params.height_ ),
                                                          Image<uint8_t, 3>( params.width_, params.height_ ) };
        auto  imageOcm   = std::array<Image<uint8_t, 1>, 2>{ Image<uint8_t, 1>( params.width_, params.height_ ),
                                                          Image<uint8_t, 1>( params.width_, params.height_ ) };
        auto& viewpoint  = viewpoints[viewpointIndex];
        Image<uint8_t, 1> imageOcmUnion;
        if ( params.verbose_ ) {
          printf( "Frame %3d / %3d: viewpoints %3d / %3zu ", frameIndex, params.frameCount_, viewpointIndex,
                  viewpoints.size() );
#if defined( USE_OPENMP )
          printf( "Thread %2d / %2d (dynamic=%d) ", omp_get_thread_num(), omp_get_num_threads(), omp_get_dynamic() );
#endif
          printf( "\n" );
          fflush( stdout );
        }
        // Rasterize
        timer.tic( "Rasterize" );
        if ( !params.compareCpuGpu_ ) {
          for ( int32_t i = 0; i < 2; i++ )
            if ( params.useFloat_ )
              rasterizer.render( pc[i], viewpoint, imageFloat[i], imageOcm[i] );
            else
              rasterizer.render( pc[i], viewpoint, imageUint8[i], imageOcm[i] );
        } else {
          if ( params.useFloat_ ) {
            rasterizerCpu.render( pc[0], viewpoint, imageFloat[0], imageOcm[0] );
            rasterizerGpu.render( pc[1], viewpoint, imageFloat[1], imageOcm[1] );
          } else {
            rasterizerCpu.render( pc[0], viewpoint, imageUint8[0], imageOcm[0] );
            rasterizerGpu.render( pc[1], viewpoint, imageUint8[1], imageOcm[1] );
          }
        }
        timer.toc( "Rasterize", params.verbose_ );

        // Save
        if ( params.save_ ) {
          uint32_t frameIndex = videoImg[0].size() - viewpoints.size() + viewpointIndex;
          if ( params.useFloat_ ) {
            for ( int32_t i = 0; i < 2; i++ ) videoImg[i][frameIndex] = imageFloat[i];
          } else {
            for ( int32_t i = 0; i < 2; i++ ) videoImg[i][frameIndex] = imageUint8[i];
          }
          for ( int32_t i = 0; i < 2; i++ ) videoOcm[i][frameIndex] = imageOcm[i];
        }

        // Compute the union of two masks (ocm_union)
        if ( params.computeOm_ ) {
          timer.tic( "UnionMask" );
          imageOcmUnion.resize( params.width_, params.height_ );
          for ( int32_t y = 0; y < params.height_; y++ ) {
            for ( int32_t x = 0; x < params.width_; x++ ) {
              uint8_t val0 = imageOcm[0].plane( 0 ).get( x, y );
              uint8_t val1 = imageOcm[1].plane( 0 ).get( x, y );
              imageOcmUnion.plane( 0 ).set( x, y, ( val0 || val1 ) ? 1 : 0 );
            }
          }
          timer.toc( "UnionMask", params.verbose_ );
        }

        // Metric
        timer.tic( "Metric" );
        if ( params.useFloat_ )
          metric.compute( imageFloat[0], imageFloat[1], imageOcmUnion, frameIndex );
        else
          metric.compute( imageUint8[0], imageUint8[1], imageOcmUnion, frameIndex );
        timer.toc( "Metric", params.verbose_ );
      }
    }
    if ( params.verbose_ ) {
      metric.getLastFrame().log( stringFormat( getBasename( params.decodePath_ ).c_str(), frameIndex ) );
    }
  }

  // Show metric results
  Results avgResults = metric.getAverage();
  avgResults.log( stringFormat( "Average for %d frames", params.frameCount_ ) );

  // Save the results to a file
  if ( !params.outputPath_.empty() ) {
    avgResults.saveToFile( params.outputPath_, getLabel( params.decodePath_ ) );
    printf( "Metrics saved to %s\n", params.outputPath_.c_str() );
  }
  if ( params.verbose_ ) timer.trace();

  // Save
  if ( !params.useFloat_ && params.save_ ) {
    auto srcName = getOuputName( getBasename( params.sourcePath_ ) );
    auto decName = getOuputName( getBasename( params.decodePath_ ) );
    auto suffix =
        ( params.outputPath_.empty() ? getOuputName( params.decodePath_ ) : getRemoveExtension( params.outputPath_ ) ) +
        ( params.compareCpuGpu_ ? "_cpu_vs_gpu" : ( params.useCPU_ ? "_cpu" : "_gpu" ) );

    // Creation of butterfly and difference videos
    auto videoImgBty = Video<uint8_t>( videoImg[0], videoImg[1], true );
    auto videoImgDif = videoImg[0] - videoImg[1];
    auto videoOcmDif = videoOcm[0] - videoOcm[1];

    // Add overlay
    // videoImg[0].draw( "Source", 0, 8 );
    // videoImg[0].draw( srcName, 0, 24 );
    // videoImg[1].draw( "Decode", 0, 8 );
    // videoImg[1].draw( decName, 0, 24 );
    // videoImgBty.draw( "Source", 0, 8 );
    // videoImgBty.draw( srcName, 0, 24 );
    // videoImgBty.draw( "Decode", params.width_ - 8, 8, true );
    // videoImgBty.draw( decName, params.width_ - 8, 24, true );

    // Save video
    videoImg[0].save( suffix + "_img_src", true );
    videoImg[1].save( suffix + "_img_dec", true );
    // videoOcm[0].save( suffix + "_ocm_src", true );
    // videoOcm[1].save( suffix + "_ocm_dec", true );
    // videoImgDif.save( suffix + "_img_dif", true );
    // videoOcmDif.save( suffix + "_ocm_dif", true );
    // videoImgBty.save( suffix + "_img_bty", true );
  }
  std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
  printf( "Total time: %.3f seconds\n", elapsed.count() );
  fflush( stdout );
  return 0;
}