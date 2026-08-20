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
#include <filesystem> 
#include <fstream>

#if defined( USE_OPENMP )
#include <omp.h>
#endif
namespace fs = std::filesystem;

std::string getFileHash(const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) return "0000000000000000";
  const uint64_t FNV_prime = 1099511628211ULL;
  uint64_t hash = 14695981039346656037ULL;
  char buffer[8192]; 
  while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
    std::streamsize bytes = file.gcount();
    for (std::streamsize i = 0; i < bytes; ++i) {
      hash ^= static_cast<uint8_t>(buffer[i]);
      hash *= FNV_prime;
    }
  }
  std::stringstream ss;
  ss << std::hex << std::setw(16) << std::setfill('0') << hash;
  return ss.str();
}

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
  bool        useSrcImages_      = true;

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
        ("useSrcImages",      useSrcImages_,      useSrcImages_,      "Create source images")
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
      printf( "Warning: Floating point images are not compatible with the recording option, "
          "disabling the creation of output videos\n" );
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
  Timer      timer;
  Metric     metric( params.computeOm_, params.computeSsim_, params.computeIvssim_, params.computeQmiv_ );
  Rasterizer rasterizer( params.useCPU_ ), rasterizerCpu( true ), rasterizerGpu( false );
#if defined( USE_OPENMP )
  omp_set_dynamic(0);
  if ( params.numThreads_ > 0 ) {
    omp_set_num_threads( params.numThreads_ );
  } else {
    omp_set_num_threads(omp_get_num_procs());
  }
#endif
  // Create saved video files 
  auto srcName = getOuputName( getBasename( params.sourcePath_ ) );
  auto decName = getOuputName( getBasename( params.decodePath_ ) );
  auto suffix =
      ( params.outputPath_.empty() ? getOuputName( params.decodePath_ ) : getRemoveExtension( params.outputPath_ ) ) +
      ( params.compareCpuGpu_ ? "_cpu_vs_gpu" : ( params.useCPU_ ? "_cpu" : "_gpu" ) );
  std::string filenameSrc = suffix + "_img_src_" + std::to_string( params.width_ ) + "x" + std::to_string( params.height_ ) + "_" +
                            std::to_string( sizeof( uint8_t ) * 8 ) + "b_p444.rgb";
  std::string filenameDec = suffix + "_img_dec_" + std::to_string( params.width_ ) + "x" + std::to_string( params.height_ ) + "_" +
                            std::to_string( sizeof( uint8_t ) * 8 ) + "b_p444.rgb";
  std::ofstream fileSrc;
  std::ofstream fileDec;
  if ( !params.useFloat_ && params.save_ ) {
    createDirectory( filenameSrc );
    fileSrc.open( filenameSrc, std::ios::binary );
    if ( !fileSrc.is_open() ) {
      printf( "Error: Unable to open file %s for writing.\n", filenameSrc.c_str() );
      return 1;
    }
    createDirectory( filenameDec );
    fileDec.open( filenameDec, std::ios::binary );
    if ( !fileDec.is_open() ) {
      printf( "Error: Unable to open file %s for writing.\n", filenameDec.c_str() );
      return 1;
    }
  }
  std::filesystem::path srcFilePath(params.sourcePath_);
  std::filesystem::path srcDir = srcFilePath.parent_path();
  std::string objName = srcDir.filename().string();
  std::string fileHash;
  // Loop frames
  for ( int32_t frameIndex = 0; frameIndex < params.frameCount_; frameIndex++ ) {
    
    if (params.useSrcImages_){         
      auto filename = createFilename( params.sourcePath_, params.startFrame_ + frameIndex );
      if (params.verbose_) printf("Hashing source file %s ...\n", filename.c_str());
      fileHash = getFileHash(filename);
    }
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
          viewpoints[0].set( el.index_, el.pos(), el.view(), el.up(), el.focal_, el.width_, el.height_, el.center_ );
          if ( params.verbose_ ) { 
            printf("onlyViewpoint == %d, keep only one viewpoint \n", params.onlyViewpoint_);
            el.log();
          }
        } else {
          viewpoints.resize( cameraPosition0.size() );
          for ( size_t i = 0; i < cameraPosition0.size(); i++ ) {
            const auto& el = cameraPosition0[i];
            viewpoints[i].set( el.index_, el.pos(), el.view(), el.up(), el.focal_, el.width_, el.height_, el.center_ );
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
    // Rendering
    for ( int32_t viewpointIndex = 0; viewpointIndex < (int32_t)viewpoints.size(); viewpointIndex++ ) {
      std::array<Image<float>, 2>       imageFloat;
      std::array<Image<uint8_t>, 2>     imageUint8;
      std::array<Image<uint8_t, 1>, 2>  imageOcm;
      Image<uint8_t, 1>                 imageOcmUnion;
      if ( params.useFloat_ ) {
        imageFloat = { Image<float, 3>( params.width_, params.height_ ), Image<float, 3>( params.width_, params.height_ ) };
      } else {
        imageUint8 = { Image<uint8_t, 3>( params.width_, params.height_ ), Image<uint8_t, 3>( params.width_, params.height_ ) };
      }
      imageOcm = { Image<uint8_t, 1>( params.width_, params.height_ ), Image<uint8_t, 1>( params.width_, params.height_ ) };
      auto& viewpoint  = viewpoints[viewpointIndex];
      // Rasterize
      timer.tic( "Rasterize" );
      if ( !params.compareCpuGpu_ ) {
        if ( params.useFloat_ ){
          rasterizer.render( pc[0], viewpoint, imageFloat[0], imageOcm[0] );
          rasterizer.render( pc[1], viewpoint, imageFloat[1], imageOcm[1] );
        } else{
          bool srcImageLoaded = false;
          std::filesystem::path pngName; 
          if (params.useSrcImages_) {            
            pngName = srcDir / "rasterized_images" / (objName + stringFormat( "_F%04d_", frameIndex ) + fileHash) / 
              ( objName + stringFormat("_F%04d_V%04d_%dx%d.png", frameIndex, viewpoint.index(), params.width_, params.height_));   
            if (std::filesystem::exists(pngName)) {
              imageUint8[0].loadPNG(pngName.string());               
              srcImageLoaded = true;
              if (params.verbose_) printf("Load source image from cache.\n");
            }
          }
          if (!srcImageLoaded) {                 
            rasterizer.render( pc[0], viewpoint, imageUint8[0], imageOcm[0] );          
            if (params.useSrcImages_) {              
              std::filesystem::create_directories(pngName.parent_path()); 
              imageUint8[0].savePNG(pngName.string()); 
              if (params.verbose_) printf("Rasterize and save source image from cache.\n");
            }
          }
          rasterizer.render( pc[1], viewpoint, imageUint8[1], imageOcm[1] );
        }
      } else {
        if ( params.useFloat_ ) {
          rasterizerCpu.render( pc[0], viewpoint, imageFloat[0], imageOcm[0] );
          rasterizerGpu.render( pc[1], viewpoint, imageFloat[1], imageOcm[1] );
        } else {
          rasterizerCpu.render( pc[0], viewpoint, imageUint8[0], imageOcm[0] );
          rasterizerGpu.render( pc[1], viewpoint, imageUint8[1], imageOcm[1] );
        }
      }
      timer.toc( "Rasterize", false );
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
        timer.toc( "UnionMask", false );
      }
      // Metric
      if ( params.useFloat_ ){
        metric.compute( imageFloat[0], imageFloat[1], imageOcmUnion, frameIndex );
      } else {
        metric.compute( imageUint8[0], imageUint8[1], imageOcmUnion, frameIndex );
      }
      // Save
      if ( params.save_ && !params.useFloat_ ) {        
        imageUint8[0].save( fileSrc, true );
        imageUint8[1].save( fileDec, true );      
      }
      if ( params.verbose_ ) {
        printf( "Frame: %3d / %3d: View: %3d / %3zu Psnr: RGB = %8.4f YUV = %8.4f  Time: %s \n", frameIndex, params.frameCount_, viewpointIndex,
                viewpoints.size(), metric.getLastFrame().psnrRgbAvg_, metric.getLastFrame().psnrYuvAvg_, timer.getLastDurationsString().c_str()  );
        fflush( stdout );
      }
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

  std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
  printf( "Peak Memory: %.2f MB\n", static_cast<double>(getPeakMemoryUsage()) / (1024.0 * 1024.0) );
  printf( "Total time: %.3f seconds\n", elapsed.count() );
  fflush( stdout );
  return 0;
}
