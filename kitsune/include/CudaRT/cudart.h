/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under 
 * the LLVM license.
 */


#ifndef __KITSUNE_CUDART_H__
#define __KITSUNE_CUDART_H__

#include <stdint.h>
#include <cuda.h>

extern "C" 
{
  /// Initialize the cuda runtime layer for use by the compiler's 
  /// code generation.  This call will error internally (and fatally)
  /// if any errors occur.  Upon completion you can check the number 
  /// of devices (GPUs) available -- each device is recognized by a 
  /// unique integral value from [0...N-1], where N is the number of 
  /// GPUs discovered.  We use a similar cuda terminology here where 
  /// these values are referred as "device IDs". 
  extern void __kitsune_cudart_initialize();


  extern void __kitsune_cudart_malloc_float(float *dataPtr, int dataSize);
  /// Shutdown the cuda runtime layer freeing all the resources that 
  /// were acquired during execution.  This call will error internally 
  /// (and fatally) if any errors occur.  In particular, calling this 
  /// function when __kitsune_cudart_initialize() failed (or has not been
  /// called) with result in a fatal error. 
  extern void __kitsune_cudart_finalize(); 

  /// Return the number of GPUs "disovered" in the system. 
  extern int  __kitsune_cudart_ndevices();

  /// Return the number of kernels that are registered with the  
  /// runtime.  
  extern int  __kitsune_cudart_nkernels();

  typedef uint_fast8_t CudaRTArgKind;
  extern const CudaRTArgKind CUDART_ARG_TYPE_UNKNOWN;
  extern const CudaRTArgKind CUDART_ARG_TYPE_SCALAR;
  extern const CudaRTArgKind CUDART_ARG_TYPE_DATA_BLK;

  typedef uint_fast8_t CudaRTArgAccessMode;
  extern const CudaRTArgAccessMode CUDART_UNKNOWN_ACCESS;
  extern const CudaRTArgAccessMode CUDART_ARG_READ_ONLY;
  extern const CudaRTArgAccessMode CUDART_ARG_WRITE_ONLY;
  extern const CudaRTArgAccessMode CUDART_ARG_READ_WRITE;

  /// Create a new kernel for use on the given device. The source code
  /// for the kernel should be in PTX format and passed in as a buffer 
  /// (no file I/O support yet).  The name of the kernel located in 
  /// the PTX code must also be provided. 
  ///
  /// The call returns a unique kernel identifier.  Any errors encountered 
  /// by the runtime result in a hard stop.
  extern int  __kitsune_cudart_create_kernel(int devID, 
                                            const char *ptxSource, 
                                            const char *funcName);

  /// Add an argument (parameter) to the given kernel (kernID).  The 
  /// arguments should be provided in order as they appear in the PTX code: 
  /// call for argument 1, followed by 2, then 3, etc... 
  ///
  /// The host-side pointer and size in bytes should be provided for the 
  /// argument,  The kind of the parameter should specify if it is a data
  /// block (i.e., array) or a scalar.  Finally, the access mode of the 
  /// argument should be provide to indicate if it is read-only, write-only,
  /// or read-write.  Together the kind and access mode determine when 
  /// gpu-side memory allocation and data movement (host --> gpu and 
  /// gpu --> host) occur.
  ///
  /// Any errors encountered by the runtime result in a hard stop.  
  extern void __kitsune_cudart_add_arg(int kernID, 
                                       void *hostArgPtr, 
                                       size_t argSizeInBytes, 
                                       CudaRTArgKind kind, 
                                       CudaRTArgAccessMode modeFlag);

  // Set the kernel's grid dimensions for the launch.  This call follows the 
  // terminology established in the driver API's cuLaunchKernel() call.
  extern void __kitsune_cudart_set_grid_dims(int kernID, int dimX, int dimY, int dimZ);

  // Set the kernel's block dimensions for the launch.  This call follows the 
  // terminology established in the driver API's cuLaunchKernel() call. 
  extern void __kitsune_cudart_set_block_dims(int kernID, int dimX, int dimY, int dimZ);

  // Set the kernel's grid and block dimensions for the launch.  This call follows the
  // terminology established in the driver API's cuLaunchKernel() call.
  extern void __kitsune_cudart_set_launch_params(int kernID, int gridDims[3], int blockDims[3]);

  /// Launch the kernel.  This assumes the kernel has been created with
  /// __kitsune_cudart_create_kernel() and that all the kernel arguments
  /// are bound with __kitsune_cudart_add_arg().
  ///
  /// Any errors encountered by the runtime result in a hard stop. 
  extern void __kitsune_cudart_launch_kernel(int kernID);
} // extern "C" 

#endif
