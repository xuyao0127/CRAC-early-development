#ifndef LOWER_HALF_CUDA_IF_H
#define LOWER_HALF_CUDA_IF_H


#define FOREACH_FNC(MACRO) \
MACRO(cudaCreateTextureObject) ,\
MACRO(cudaDestroyTextureObject) ,\
MACRO(cudaCreateChannelDesc) ,\
MACRO(cudaEventCreate) ,\
MACRO(cudaEventCreateWithFlags) ,\
MACRO(cudaEventDestroy) ,\
MACRO(cudaEventElapsedTime) ,\
MACRO(cudaEventQuery) ,\
MACRO(cudaEventRecord) ,\
MACRO(cudaEventSynchronize) ,\
MACRO(cudaMalloc) ,\
MACRO(cudaFree) ,\
MACRO(cudaMallocArray) ,\
MACRO(cudaFreeArray) ,\
MACRO(cudaHostRegister) ,\
MACRO(cudaDeviceGetAttribute) ,\
MACRO(cudaMallocHost) ,\
MACRO(cudaFreeHost) ,\
MACRO(cudaHostAlloc) ,\
MACRO(cudaMallocPitch) ,\
MACRO(cudaGetDevice) ,\
MACRO(cudaSetDevice) ,\
MACRO(cudaDeviceGetLimit) ,\
MACRO(cudaDeviceSetLimit) ,\
MACRO(cudaGetDeviceCount) ,\
MACRO(cudaDeviceSetCacheConfig) ,\
MACRO(cudaGetDeviceProperties_v2) ,\
MACRO(cudaDeviceCanAccessPeer) ,\
MACRO(cudaDeviceGetPCIBusId) ,\
MACRO(cudaDeviceReset) ,\
MACRO(cudaDeviceSynchronize) ,\
MACRO(cudaLaunchKernel) ,\
MACRO(cudaMallocManaged) ,\
MACRO(cudaMemcpy) ,\
MACRO(cudaMemcpy2D) ,\
MACRO(cudaMemcpyToSymbol) ,\
MACRO(cudaMemcpyAsync) ,\
MACRO(cudaMemset) ,\
MACRO(cudaMemset2D) ,\
MACRO(cudaMemsetAsync) ,\
MACRO(cudaMemGetInfo) ,\
MACRO(cudaMemAdvise) ,\
MACRO(cudaMemPrefetchAsync) ,\
MACRO(cudaStreamCreate) ,\
MACRO(cudaStreamCreateWithPriority) ,\
MACRO(cudaStreamCreateWithFlags) ,\
MACRO(cudaStreamDestroy) ,\
MACRO(cudaStreamSynchronize) ,\
MACRO(cudaStreamWaitEvent) ,\
MACRO(cudaThreadSynchronize) ,\
MACRO(cudaThreadExit) ,\
MACRO(cudaPointerGetAttributes) ,\
MACRO(cudaGetErrorString) ,\
MACRO(cudaGetErrorName) ,\
MACRO(cudaGetLastError) ,\
MACRO(cudaPeekAtLastError) ,\
MACRO(cudaFuncSetCacheConfig) ,\
MACRO(__cudaInitModule) ,\
MACRO(__cudaPopCallConfiguration) ,\
MACRO(__cudaPushCallConfiguration) ,\
MACRO(__cudaRegisterFatBinary) ,\
MACRO(__cudaUnregisterFatBinary) ,\
MACRO(__cudaRegisterFunction) ,\
MACRO(__cudaRegisterManagedVar) ,\
MACRO(__cudaRegisterVar) ,\
MACRO(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) ,\
MACRO(cudaFuncGetAttributes) ,\
MACRO(cublasCreate_v2) ,\
MACRO(cublasSetStream_v2) ,\
MACRO(cublasDdot_v2) ,\
MACRO(cublasDestroy_v2) ,\
MACRO(cublasDaxpy_v2) ,\
MACRO(cublasDasum_v2) ,\
MACRO(cublasDgemm_v2) ,\
MACRO(cublasDgemv_v2) ,\
MACRO(cublasDnrm2_v2) ,\
MACRO(cublasDscal_v2) ,\
MACRO(cublasDswap_v2) ,\
MACRO(cublasIdamax_v2) ,\
MACRO(cusparseCreate) ,\
MACRO(cusparseSetStream) ,\
MACRO(cusparseCreateMatDescr) ,\
MACRO(cusparseSetMatType) ,\
MACRO(cusparseSetMatIndexBase) ,\
MACRO(cusparseDestroy) ,\
MACRO(cusparseDestroyMatDescr) ,\
MACRO(cusparseCreateCsric02Info) ,\
MACRO(cusparseDestroyCsric02Info) ,\
MACRO(cusparseCreateCsrilu02Info) ,\
MACRO(cusparseDestroyCsrilu02Info) ,\
MACRO(cusparseCreateBsrsv2Info) ,\
MACRO(cusparseDestroyBsrsv2Info) ,\
MACRO(cusparseCsr2cscEx2_bufferSize) ,\
MACRO(cusparseCsr2cscEx2) ,\
MACRO(cusparseDgemvi_bufferSize) ,\
MACRO(cusparseDgemvi) ,\
MACRO(cusparseDbsrsv2_analysis) ,\
MACRO(cusparseDbsrsv2_solve) ,\
MACRO(cusparseGetMatType) ,\
MACRO(cusparseSetMatFillMode) ,\
MACRO(cusparseGetMatFillMode) ,\
MACRO(cusparseSetMatDiagType) ,\
MACRO(cusparseGetMatDiagType) ,\
MACRO(cusparseGetMatIndexBase) ,\
MACRO(cusparseSetPointerMode) ,\
MACRO(cusolverDnCreate) ,\
MACRO(cusolverDnDestroy) ,\
MACRO(cusolverDnSetStream) ,\
MACRO(cusolverDnGetStream) ,\
MACRO(cusolverDnDgetrf_bufferSize) ,\
MACRO(cusolverDnDgetrf) ,\
MACRO(cusolverDnDgetrs) ,\
MACRO(cusolverDnDpotrf_bufferSize) ,\
MACRO(cusolverDnDpotrf) ,\
MACRO(cusolverDnDpotrs) ,\
MACRO(cuInit) ,\
MACRO(cuDriverGetVersion) ,\
MACRO(cuDeviceGet) ,\
MACRO(cuDeviceGetAttribute) ,\
MACRO(cuDeviceGetCount) ,\
MACRO(cuDeviceGetName) ,\
MACRO(cuDeviceGetUuid) ,\
MACRO(cuDeviceTotalMem_v2) ,\
MACRO(cuDeviceComputeCapability) ,\
MACRO(cuDeviceGetProperties) ,\
MACRO(cuDevicePrimaryCtxGetState) ,\
MACRO(cuDevicePrimaryCtxRelease_v2) ,\
MACRO(cuDevicePrimaryCtxReset_v2) ,\
MACRO(cuDevicePrimaryCtxRetain) ,\
MACRO(cuDevicePrimaryCtxSetFlags_v2) ,\
MACRO(cuCtxCreate_v2) ,\
MACRO(cuCtxDestroy_v2) ,\
MACRO(cuCtxGetApiVersion) ,\
MACRO(cuCtxGetCacheConfig) ,\
MACRO(cuCtxGetCurrent) ,\
MACRO(cuCtxGetDevice) ,\
MACRO(cuCtxGetFlags) ,\
MACRO(cuCtxGetLimit) ,\
MACRO(cuCtxGetSharedMemConfig) ,\
MACRO(cuCtxGetStreamPriorityRange) ,\
MACRO(cuCtxPopCurrent_v2) ,\
MACRO(cuCtxPushCurrent_v2) ,\
MACRO(cuCtxSetCacheConfig) ,\
MACRO(cuCtxSetCurrent) ,\
MACRO(cuCtxSetLimit) ,\
MACRO(cuCtxSetSharedMemConfig) ,\
MACRO(cuCtxSynchronize) ,\
MACRO(cuCtxAttach) ,\
MACRO(cuCtxDetach) ,\
MACRO(cuLinkAddData_v2) ,\
MACRO(cuLinkAddFile_v2) ,\
MACRO(cuLinkComplete) ,\
MACRO(cuLinkCreate_v2) ,\
MACRO(cuLinkDestroy) ,\
MACRO(cuModuleGetFunction) ,\
MACRO(cuModuleGetGlobal_v2) ,\
MACRO(cuModuleGetSurfRef) ,\
MACRO(cuModuleGetTexRef) ,\
MACRO(cuModuleLoad) ,\
MACRO(cuModuleLoadData) ,\
MACRO(cuModuleLoadDataEx) ,\
MACRO(cuModuleLoadFatBinary) ,\
MACRO(cuModuleUnload) ,\
MACRO(cuArray3DCreate_v2) ,\
MACRO(cuArray3DGetDescriptor_v2) ,\
MACRO(cuArrayCreate_v2) ,\
MACRO(cuArrayDestroy) ,\
MACRO(cuArrayGetDescriptor_v2) ,\
MACRO(cuDeviceGetByPCIBusId) ,\
MACRO(cuDeviceGetPCIBusId) ,\
MACRO(cuIpcCloseMemHandle) ,\
MACRO(cuIpcGetEventHandle) ,\
MACRO(cuIpcGetMemHandle) ,\
MACRO(cuIpcOpenEventHandle) ,\
MACRO(cuIpcOpenMemHandle_v2) ,\
MACRO(cuMemAlloc_v2) ,\
MACRO(cuMemAllocHost_v2) ,\
MACRO(cuMemAllocManaged) ,\
MACRO(cuMemAllocPitch_v2) ,\
MACRO(cuMemFree_v2) ,\
MACRO(cuMemFreeHost) ,\
MACRO(cuMemGetAddressRange_v2) ,\
MACRO(cuMemGetInfo_v2) ,\
MACRO(cuMemHostAlloc) ,\
MACRO(cuMemHostGetDevicePointer_v2) ,\
MACRO(cuMemHostGetFlags) ,\
MACRO(cuMemHostRegister_v2) ,\
MACRO(cuMemHostUnregister) ,\
MACRO(cuMemcpy) ,\
MACRO(cuMemcpy2D_v2) ,\
MACRO(cuMemcpy2DAsync_v2) ,\
MACRO(cuMemcpy2DUnaligned_v2) ,\
MACRO(cuMemcpy3D_v2) ,\
MACRO(cuMemcpy3DAsync_v2) ,\
MACRO(cuMemcpy3DPeer) ,\
MACRO(cuMemcpy3DPeerAsync) ,\
MACRO(cuMemcpyAsync) ,\
MACRO(cuMemcpyAtoA_v2) ,\
MACRO(cuMemcpyAtoD_v2) ,\
MACRO(cuMemcpyAtoH_v2) ,\
MACRO(cuMemcpyAtoHAsync_v2) ,\
MACRO(cuMemcpyDtoA_v2) ,\
MACRO(cuMemcpyDtoD_v2) ,\
MACRO(cuMemcpyDtoDAsync_v2) ,\
MACRO(cuMemcpyDtoH_v2) ,\
MACRO(cuMemcpyDtoHAsync_v2) ,\
MACRO(cuMemcpyHtoA_v2) ,\
MACRO(cuMemcpyHtoAAsync_v2) ,\
MACRO(cuMemcpyHtoD_v2) ,\
MACRO(cuMemcpyHtoDAsync_v2) ,\
MACRO(cuMemcpyPeer) ,\
MACRO(cuMemcpyPeerAsync) ,\
MACRO(cuMemsetD16_v2) ,\
MACRO(cuMemsetD16Async) ,\
MACRO(cuMemsetD2D16_v2) ,\
MACRO(cuMemsetD2D16Async) ,\
MACRO(cuMemsetD2D32_v2) ,\
MACRO(cuMemsetD2D32Async) ,\
MACRO(cuMemsetD2D8_v2) ,\
MACRO(cuMemsetD2D8Async) ,\
MACRO(cuMemsetD32_v2) ,\
MACRO(cuMemsetD32Async) ,\
MACRO(cuMemsetD8_v2) ,\
MACRO(cuMemsetD8Async) ,\
MACRO(cuMipmappedArrayCreate) ,\
MACRO(cuMipmappedArrayDestroy) ,\
MACRO(cuMipmappedArrayGetLevel) ,\
MACRO(cuMemAdvise) ,\
MACRO(cuMemPrefetchAsync) ,\
MACRO(cuMemRangeGetAttribute) ,\
MACRO(cuMemRangeGetAttributes) ,\
MACRO(cuPointerGetAttribute) ,\
MACRO(cuPointerGetAttributes) ,\
MACRO(cuPointerSetAttribute) ,\
MACRO(cuStreamAddCallback) ,\
MACRO(cuStreamAttachMemAsync) ,\
MACRO(cuStreamBeginCapture_v2) ,\
MACRO(cuStreamCreate) ,\
MACRO(cuStreamCreateWithPriority) ,\
MACRO(cuStreamDestroy_v2) ,\
MACRO(cuStreamEndCapture) ,\
MACRO(cuStreamGetCtx) ,\
MACRO(cuStreamGetFlags) ,\
MACRO(cuStreamGetPriority) ,\
MACRO(cuStreamIsCapturing) ,\
MACRO(cuStreamQuery) ,\
MACRO(cuStreamSynchronize) ,\
MACRO(cuStreamWaitEvent) ,\
MACRO(cuEventCreate) ,\
MACRO(cuEventDestroy_v2) ,\
MACRO(cuEventElapsedTime) ,\
MACRO(cuEventQuery) ,\
MACRO(cuEventRecord) ,\
MACRO(cuEventRecordWithFlags) ,\
MACRO(cuEventSynchronize) ,\
MACRO(cuDestroyExternalMemory) ,\
MACRO(cuDestroyExternalSemaphore) ,\
MACRO(cuExternalMemoryGetMappedBuffer) ,\
MACRO(cuExternalMemoryGetMappedMipmappedArray) ,\
MACRO(cuImportExternalMemory) ,\
MACRO(cuImportExternalSemaphore) ,\
MACRO(cuSignalExternalSemaphoresAsync) ,\
MACRO(cuWaitExternalSemaphoresAsync) ,\
MACRO(cuStreamBatchMemOp_v2) ,\
MACRO(cuStreamWaitValue32_v2) ,\
MACRO(cuStreamWaitValue64_v2) ,\
MACRO(cuStreamWriteValue32_v2) ,\
MACRO(cuStreamWriteValue64_v2) ,\
MACRO(cuFuncGetAttribute) ,\
MACRO(cuFuncSetAttribute) ,\
MACRO(cuFuncSetCacheConfig) ,\
MACRO(cuFuncSetSharedMemConfig) ,\
MACRO(cuLaunchCooperativeKernel) ,\
MACRO(cuLaunchCooperativeKernelMultiDevice) ,\
MACRO(cuLaunchHostFunc) ,\
MACRO(cuLaunchKernel) ,\
MACRO(cuFuncSetBlockShape) ,\
MACRO(cuFuncSetSharedSize) ,\
MACRO(cuLaunch) ,\
MACRO(cuLaunchGrid) ,\
MACRO(cuLaunchGridAsync) ,\
MACRO(cuParamSetSize) ,\
MACRO(cuParamSetTexRef) ,\
MACRO(cuParamSetf) ,\
MACRO(cuParamSeti) ,\
MACRO(cuParamSetv) ,\
MACRO(cuGraphCreate) ,\
MACRO(cuGraphDestroy) ,\
MACRO(cuGraphDestroyNode) ,\
MACRO(cuGraphExecDestroy) ,\
MACRO(cuGraphGetEdges) ,\
MACRO(cuGraphGetNodes) ,\
MACRO(cuGraphGetRootNodes) ,\
MACRO(cuGraphHostNodeGetParams) ,\
MACRO(cuGraphHostNodeSetParams) ,\
MACRO(cuGraphKernelNodeGetParams_v2) ,\
MACRO(cuGraphKernelNodeSetParams_v2) ,\
MACRO(cuGraphLaunch) ,\
MACRO(cuGraphMemcpyNodeGetParams) ,\
MACRO(cuGraphMemcpyNodeSetParams) ,\
MACRO(cuGraphMemsetNodeGetParams) ,\
MACRO(cuGraphMemsetNodeSetParams) ,\
MACRO(cuGraphNodeFindInClone) ,\
MACRO(cuGraphNodeGetDependencies) ,\
MACRO(cuGraphNodeGetDependentNodes) ,\
MACRO(cuGraphNodeGetType) ,\
MACRO(cuOccupancyMaxActiveBlocksPerMultiprocessor) ,\
MACRO(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) ,\
MACRO(cuOccupancyMaxPotentialBlockSize) ,\
MACRO(cuOccupancyMaxPotentialBlockSizeWithFlags) ,\
MACRO(cuTexRefCreate) ,\
MACRO(cuTexRefDestroy) ,\
MACRO(cuTexRefGetAddress_v2) ,\
MACRO(cuTexRefGetAddressMode) ,\
MACRO(cuTexRefGetArray) ,\
MACRO(cuTexRefGetBorderColor) ,\
MACRO(cuTexRefGetFilterMode) ,\
MACRO(cuTexRefGetFlags) ,\
MACRO(cuTexRefGetFormat) ,\
MACRO(cuTexRefGetMaxAnisotropy) ,\
MACRO(cuTexRefGetMipmapFilterMode) ,\
MACRO(cuTexRefGetMipmapLevelBias) ,\
MACRO(cuTexRefGetMipmapLevelClamp) ,\
MACRO(cuTexRefGetMipmappedArray) ,\
MACRO(cuTexRefSetAddress_v2) ,\
MACRO(cuTexRefSetAddress2D_v3) ,\
MACRO(cuTexRefSetAddressMode) ,\
MACRO(cuTexRefSetArray) ,\
MACRO(cuTexRefSetBorderColor) ,\
MACRO(cuTexRefSetFilterMode) ,\
MACRO(cuTexRefSetFlags) ,\
MACRO(cuTexRefSetFormat) ,\
MACRO(cuTexRefSetMaxAnisotropy) ,\
MACRO(cuTexRefSetMipmapFilterMode) ,\
MACRO(cuTexRefSetMipmapLevelBias) ,\
MACRO(cuTexRefSetMipmapLevelClamp) ,\
MACRO(cuTexRefSetMipmappedArray) ,\
MACRO(cuSurfRefGetArray) ,\
MACRO(cuSurfRefSetArray) ,\
MACRO(cuTexObjectCreate) ,\
MACRO(cuTexObjectDestroy) ,\
MACRO(cuTexObjectGetResourceDesc) ,\
MACRO(cuTexObjectGetResourceViewDesc) ,\
MACRO(cuTexObjectGetTextureDesc) ,\
MACRO(cuSurfObjectCreate) ,\
MACRO(cuSurfObjectDestroy) ,\
MACRO(cuSurfObjectGetResourceDesc) ,\
MACRO(cuCtxDisablePeerAccess) ,\
MACRO(cuCtxEnablePeerAccess) ,\
MACRO(cuDeviceCanAccessPeer) ,\
MACRO(cuDeviceGetP2PAttribute) ,\
MACRO(cuGraphicsMapResources) ,\
MACRO(cuGraphicsResourceGetMappedMipmappedArray) ,\
MACRO(cuGraphicsResourceGetMappedPointer_v2) ,\
MACRO(cuGraphicsResourceSetMapFlags_v2) ,\
MACRO(cuGraphicsSubResourceGetMappedArray) ,\
MACRO(cuGraphicsUnmapResources) ,\
MACRO(cuGraphicsUnregisterResource) ,\
MACRO(__cudaRegisterFatBinaryEnd) ,
#define GENERATE_ENUM(ENUM) Cuda_Fnc_##ENUM

#define GENERATE_FNC_PTR(FNC) ((void*)&FNC)

typedef enum __Cuda_Fncs {
  Cuda_Fnc_NULL,
  FOREACH_FNC(GENERATE_ENUM)
  Cuda_Fnc_Invalid,
} Cuda_Fncs_t;

static const char *cuda_Fnc_to_str[]  __attribute__((used)) =
{
  "Cuda_Fnc_NULL", 
  "cudaCreateTextureObject",
  "cudaDestroyTextureObject",
  "cudaCreateChannelDesc",
  "cudaEventCreate",
  "cudaEventCreateWithFlags",
  "cudaEventDestroy",
  "cudaEventElapsedTime",
  "cudaEventQuery",
  "cudaEventRecord",
  "cudaEventSynchronize",
  "cudaMalloc",
  "cudaFree",
  "cudaMallocArray",
  "cudaFreeArray",
  "cudaHostRegister",
  "cudaDeviceGetAttribute",
  "cudaMallocHost",
  "cudaFreeHost",
  "cudaHostAlloc",
  "cudaMallocPitch",
  "cudaGetDevice",
  "cudaSetDevice",
  "cudaDeviceGetLimit",
  "cudaDeviceSetLimit",
  "cudaGetDeviceCount",
  "cudaDeviceSetCacheConfig",
  "cudaGetDeviceProperties_v2",
  "cudaDeviceCanAccessPeer",
  "cudaDeviceGetPCIBusId",
  "cudaDeviceReset",
  "cudaDeviceSynchronize",
  "cudaLaunchKernel",
  "cudaMallocManaged",
  "cudaMemcpy",
  "cudaMemcpy2D",
  "cudaMemcpyToSymbol",
  "cudaMemcpyAsync",
  "cudaMemset",
  "cudaMemset2D",
  "cudaMemsetAsync",
  "cudaMemGetInfo",
  "cudaMemAdvise",
  "cudaMemPrefetchAsync",
  "cudaStreamCreate",
  "cudaStreamCreateWithPriority",
  "cudaStreamCreateWithFlags",
  "cudaStreamDestroy",
  "cudaStreamSynchronize",
  "cudaStreamWaitEvent",
  "cudaThreadSynchronize",
  "cudaThreadExit",
  "cudaPointerGetAttributes",
  "cudaGetErrorString",
  "cudaGetErrorName",
  "cudaGetLastError",
  "cudaPeekAtLastError",
  "cudaFuncSetCacheConfig",
  "__cudaInitModule",
  "__cudaPopCallConfiguration",
  "__cudaPushCallConfiguration",
  "__cudaRegisterFatBinary",
  "__cudaUnregisterFatBinary",
  "__cudaRegisterFunction",
  "__cudaRegisterManagedVar",
  "__cudaRegisterVar",
  "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
  "cudaFuncGetAttributes",
  "cublasCreate_v2",
  "cublasSetStream_v2",
  "cublasDdot_v2",
  "cublasDestroy_v2",
  "cublasDaxpy_v2",
  "cublasDasum_v2",
  "cublasDgemm_v2",
  "cublasDgemv_v2",
  "cublasDnrm2_v2",
  "cublasDscal_v2",
  "cublasDswap_v2",
  "cublasIdamax_v2",
  "cusparseCreate",
  "cusparseSetStream",
  "cusparseCreateMatDescr",
  "cusparseSetMatType",
  "cusparseSetMatIndexBase",
  "cusparseDestroy",
  "cusparseDestroyMatDescr",
  "cusparseCreateCsric02Info",
  "cusparseDestroyCsric02Info",
  "cusparseCreateCsrilu02Info",
  "cusparseDestroyCsrilu02Info",
  "cusparseCreateBsrsv2Info",
  "cusparseDestroyBsrsv2Info",
  "cusparseCsr2cscEx2_bufferSize",
  "cusparseCsr2cscEx2",
  "cusparseDgemvi_bufferSize",
  "cusparseDgemvi",
  "cusparseDbsrsv2_analysis",
  "cusparseDbsrsv2_solve",
  "cusparseGetMatType",
  "cusparseSetMatFillMode",
  "cusparseGetMatFillMode",
  "cusparseSetMatDiagType",
  "cusparseGetMatDiagType",
  "cusparseGetMatIndexBase",
  "cusparseSetPointerMode",
  "cusolverDnCreate",
  "cusolverDnDestroy",
  "cusolverDnSetStream",
  "cusolverDnGetStream",
  "cusolverDnDgetrf_bufferSize",
  "cusolverDnDgetrf",
  "cusolverDnDgetrs",
  "cusolverDnDpotrf_bufferSize",
  "cusolverDnDpotrf",
  "cusolverDnDpotrs",
  "cuInit",
  "cuDriverGetVersion",
  "cuDeviceGet",
  "cuDeviceGetAttribute",
  "cuDeviceGetCount",
  "cuDeviceGetName",
  "cuDeviceGetUuid",
  "cuDeviceTotalMem_v2",
  "cuDeviceComputeCapability",
  "cuDeviceGetProperties",
  "cuDevicePrimaryCtxGetState",
  "cuDevicePrimaryCtxRelease_v2",
  "cuDevicePrimaryCtxReset_v2",
  "cuDevicePrimaryCtxRetain",
  "cuDevicePrimaryCtxSetFlags_v2",
  "cuCtxCreate_v2",
  "cuCtxDestroy_v2",
  "cuCtxGetApiVersion",
  "cuCtxGetCacheConfig",
  "cuCtxGetCurrent",
  "cuCtxGetDevice",
  "cuCtxGetFlags",
  "cuCtxGetLimit",
  "cuCtxGetSharedMemConfig",
  "cuCtxGetStreamPriorityRange",
  "cuCtxPopCurrent_v2",
  "cuCtxPushCurrent_v2",
  "cuCtxSetCacheConfig",
  "cuCtxSetCurrent",
  "cuCtxSetLimit",
  "cuCtxSetSharedMemConfig",
  "cuCtxSynchronize",
  "cuCtxAttach",
  "cuCtxDetach",
  "cuLinkAddData_v2",
  "cuLinkAddFile_v2",
  "cuLinkComplete",
  "cuLinkCreate_v2",
  "cuLinkDestroy",
  "cuModuleGetFunction",
  "cuModuleGetGlobal_v2",
  "cuModuleGetSurfRef",
  "cuModuleGetTexRef",
  "cuModuleLoad",
  "cuModuleLoadData",
  "cuModuleLoadDataEx",
  "cuModuleLoadFatBinary",
  "cuModuleUnload",
  "cuArray3DCreate_v2",
  "cuArray3DGetDescriptor_v2",
  "cuArrayCreate_v2",
  "cuArrayDestroy",
  "cuArrayGetDescriptor_v2",
  "cuDeviceGetByPCIBusId",
  "cuDeviceGetPCIBusId",
  "cuIpcCloseMemHandle",
  "cuIpcGetEventHandle",
  "cuIpcGetMemHandle",
  "cuIpcOpenEventHandle",
  "cuIpcOpenMemHandle_v2",
  "cuMemAlloc_v2",
  "cuMemAllocHost_v2",
  "cuMemAllocManaged",
  "cuMemAllocPitch_v2",
  "cuMemFree_v2",
  "cuMemFreeHost",
  "cuMemGetAddressRange_v2",
  "cuMemGetInfo_v2",
  "cuMemHostAlloc",
  "cuMemHostGetDevicePointer_v2",
  "cuMemHostGetFlags",
  "cuMemHostRegister_v2",
  "cuMemHostUnregister",
  "cuMemcpy",
  "cuMemcpy2D_v2",
  "cuMemcpy2DAsync_v2",
  "cuMemcpy2DUnaligned_v2",
  "cuMemcpy3D_v2",
  "cuMemcpy3DAsync_v2",
  "cuMemcpy3DPeer",
  "cuMemcpy3DPeerAsync",
  "cuMemcpyAsync",
  "cuMemcpyAtoA_v2",
  "cuMemcpyAtoD_v2",
  "cuMemcpyAtoH_v2",
  "cuMemcpyAtoHAsync_v2",
  "cuMemcpyDtoA_v2",
  "cuMemcpyDtoD_v2",
  "cuMemcpyDtoDAsync_v2",
  "cuMemcpyDtoH_v2",
  "cuMemcpyDtoHAsync_v2",
  "cuMemcpyHtoA_v2",
  "cuMemcpyHtoAAsync_v2",
  "cuMemcpyHtoD_v2",
  "cuMemcpyHtoDAsync_v2",
  "cuMemcpyPeer",
  "cuMemcpyPeerAsync",
  "cuMemsetD16_v2",
  "cuMemsetD16Async",
  "cuMemsetD2D16_v2",
  "cuMemsetD2D16Async",
  "cuMemsetD2D32_v2",
  "cuMemsetD2D32Async",
  "cuMemsetD2D8_v2",
  "cuMemsetD2D8Async",
  "cuMemsetD32_v2",
  "cuMemsetD32Async",
  "cuMemsetD8_v2",
  "cuMemsetD8Async",
  "cuMipmappedArrayCreate",
  "cuMipmappedArrayDestroy",
  "cuMipmappedArrayGetLevel",
  "cuMemAdvise",
  "cuMemPrefetchAsync",
  "cuMemRangeGetAttribute",
  "cuMemRangeGetAttributes",
  "cuPointerGetAttribute",
  "cuPointerGetAttributes",
  "cuPointerSetAttribute",
  "cuStreamAddCallback",
  "cuStreamAttachMemAsync",
  "cuStreamBeginCapture_v2",
  "cuStreamCreate",
  "cuStreamCreateWithPriority",
  "cuStreamDestroy_v2",
  "cuStreamEndCapture",
  "cuStreamGetCtx",
  "cuStreamGetFlags",
  "cuStreamGetPriority",
  "cuStreamIsCapturing",
  "cuStreamQuery",
  "cuStreamSynchronize",
  "cuStreamWaitEvent",
  "cuEventCreate",
  "cuEventDestroy_v2",
  "cuEventElapsedTime",
  "cuEventQuery",
  "cuEventRecord",
  "cuEventRecordWithFlags",
  "cuEventSynchronize",
  "cuDestroyExternalMemory",
  "cuDestroyExternalSemaphore",
  "cuExternalMemoryGetMappedBuffer",
  "cuExternalMemoryGetMappedMipmappedArray",
  "cuImportExternalMemory",
  "cuImportExternalSemaphore",
  "cuSignalExternalSemaphoresAsync",
  "cuWaitExternalSemaphoresAsync",
  "cuStreamBatchMemOp_v2",
  "cuStreamWaitValue32_v2",
  "cuStreamWaitValue64_v2",
  "cuStreamWriteValue32_v2",
  "cuStreamWriteValue64_v2",
  "cuFuncGetAttribute",
  "cuFuncSetAttribute",
  "cuFuncSetCacheConfig",
  "cuFuncSetSharedMemConfig",
  "cuLaunchCooperativeKernel",
  "cuLaunchCooperativeKernelMultiDevice",
  "cuLaunchHostFunc",
  "cuLaunchKernel",
  "cuFuncSetBlockShape",
  "cuFuncSetSharedSize",
  "cuLaunch",
  "cuLaunchGrid",
  "cuLaunchGridAsync",
  "cuParamSetSize",
  "cuParamSetTexRef",
  "cuParamSetf",
  "cuParamSeti",
  "cuParamSetv",
  "cuGraphCreate",
  "cuGraphDestroy",
  "cuGraphDestroyNode",
  "cuGraphExecDestroy",
  "cuGraphGetEdges",
  "cuGraphGetNodes",
  "cuGraphGetRootNodes",
  "cuGraphHostNodeGetParams",
  "cuGraphHostNodeSetParams",
  "cuGraphKernelNodeGetParams_v2",
  "cuGraphKernelNodeSetParams_v2",
  "cuGraphLaunch",
  "cuGraphMemcpyNodeGetParams",
  "cuGraphMemcpyNodeSetParams",
  "cuGraphMemsetNodeGetParams",
  "cuGraphMemsetNodeSetParams",
  "cuGraphNodeFindInClone",
  "cuGraphNodeGetDependencies",
  "cuGraphNodeGetDependentNodes",
  "cuGraphNodeGetType",
  "cuOccupancyMaxActiveBlocksPerMultiprocessor",
  "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
  "cuOccupancyMaxPotentialBlockSize",
  "cuOccupancyMaxPotentialBlockSizeWithFlags",
  "cuTexRefCreate",
  "cuTexRefDestroy",
  "cuTexRefGetAddress_v2",
  "cuTexRefGetAddressMode",
  "cuTexRefGetArray",
  "cuTexRefGetBorderColor",
  "cuTexRefGetFilterMode",
  "cuTexRefGetFlags",
  "cuTexRefGetFormat",
  "cuTexRefGetMaxAnisotropy",
  "cuTexRefGetMipmapFilterMode",
  "cuTexRefGetMipmapLevelBias",
  "cuTexRefGetMipmapLevelClamp",
  "cuTexRefGetMipmappedArray",
  "cuTexRefSetAddress_v2",
  "cuTexRefSetAddress2D_v3",
  "cuTexRefSetAddressMode",
  "cuTexRefSetArray",
  "cuTexRefSetBorderColor",
  "cuTexRefSetFilterMode",
  "cuTexRefSetFlags",
  "cuTexRefSetFormat",
  "cuTexRefSetMaxAnisotropy",
  "cuTexRefSetMipmapFilterMode",
  "cuTexRefSetMipmapLevelBias",
  "cuTexRefSetMipmapLevelClamp",
  "cuTexRefSetMipmappedArray",
  "cuSurfRefGetArray",
  "cuSurfRefSetArray",
  "cuTexObjectCreate",
  "cuTexObjectDestroy",
  "cuTexObjectGetResourceDesc",
  "cuTexObjectGetResourceViewDesc",
  "cuTexObjectGetTextureDesc",
  "cuSurfObjectCreate",
  "cuSurfObjectDestroy",
  "cuSurfObjectGetResourceDesc",
  "cuCtxDisablePeerAccess",
  "cuCtxEnablePeerAccess",
  "cuDeviceCanAccessPeer",
  "cuDeviceGetP2PAttribute",
  "cuGraphicsMapResources",
  "cuGraphicsResourceGetMappedMipmappedArray",
  "cuGraphicsResourceGetMappedPointer_v2",
  "cuGraphicsResourceSetMapFlags_v2",
  "cuGraphicsSubResourceGetMappedArray",
  "cuGraphicsUnmapResources",
  "cuGraphicsUnregisterResource",
  "__cudaRegisterFatBinaryEnd",
  "Cuda_Fnc_Invalid"
};
#endif // LOWER_HALF_CUDA_IF_H