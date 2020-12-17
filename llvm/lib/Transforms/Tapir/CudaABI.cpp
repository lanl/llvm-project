//===- CudaABI.cpp - Lower Tapir loop to a CUDA kernel --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CUDA ABI to convert Tapir instructions to calls into
// GPU runtime system for NVIDIA GPU code.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/CudaABI.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Vectorize.h"

#include <iostream>

using namespace llvm;

#define DEBUG_TYPE "cudaabi"

// Copied from clang/lib/CodeGen/CGCUDANV.cpp
constexpr unsigned CudaFatMagic = 0x466243b1;

static cl::opt<std::string>
GPUArch("tapir-gpu-arch", cl::init("sm_70"),
        cl::desc("GPU architecture for Tapir-CUDA backend"));

static cl::opt<std::string>
GPUFeatures("tapir-gpu-features", cl::init("+ptx64"),
            cl::desc("GPU features for Tapir-CUDA backend"));

static cl::opt<std::string>
PTXASOptLevel("tapir-ptxas-opt", cl::init("-O2"),
            cl::desc("Optimization level for PTXAS"));

// const unsigned ThreadsPerBlock = 1024;
static cl::opt<unsigned> ThreadsPerBlock(
    "threads-per-block", cl::init(1024), cl::Hidden,
    cl::desc("Use this threads-per-block count"));
// TODO: Better ThreadsPerBlock value = 256?

static cl::opt<bool>
Verbose("tapir-cuda-verbose", cl::init(false),
        cl::desc("Verbose output for Tapir-CUDA backend"));

/// Get the compute_xx corresponding to a CUDA architecture, e.g., sm_yy.
static std::string VirtualArchForCudaArch(StringRef Arch) {
  return llvm::StringSwitch<std::string>(Arch)
      .Case("sm_20", "compute_20")
      .Case("sm_21", "compute_20")
      .Case("sm_30", "compute_30")
      .Case("sm_32", "compute_32")
      .Case("sm_35", "compute_35")
      .Case("sm_37", "compute_37")
      .Case("sm_50", "compute_50")
      .Case("sm_52", "compute_52")
      .Case("sm_53", "compute_53")
      .Case("sm_60", "compute_60")
      .Case("sm_61", "compute_61")
      .Case("sm_62", "compute_62")
      .Case("sm_70", "compute_70")
      .Case("sm_72", "compute_72")
      .Case("sm_75", "compute_75")
      .Case("gfx600", "compute_amdgcn")
      .Case("gfx601", "compute_amdgcn")
      .Case("gfx700", "compute_amdgcn")
      .Case("gfx701", "compute_amdgcn")
      .Case("gfx702", "compute_amdgcn")
      .Case("gfx703", "compute_amdgcn")
      .Case("gfx704", "compute_amdgcn")
      .Case("gfx801", "compute_amdgcn")
      .Case("gfx802", "compute_amdgcn")
      .Case("gfx803", "compute_amdgcn")
      .Case("gfx810", "compute_amdgcn")
      .Case("gfx900", "compute_amdgcn")
      .Case("gfx902", "compute_amdgcn")
      .Case("gfx904", "compute_amdgcn")
      .Case("gfx906", "compute_amdgcn")
      .Case("gfx908", "compute_amdgcn")
      .Case("gfx909", "compute_amdgcn")
      .Case("gfx1010", "compute_amdgcn")
      .Case("gfx1011", "compute_amdgcn")
      .Case("gfx1012", "compute_amdgcn")
      .Default("unknown");
}

FunctionType *CudaLoop::getRegisterGlobalsFnTy() const {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  return FunctionType::get(VoidTy, VoidPtrTy->getPointerTo(), false);
}

FunctionType *CudaLoop::getCallbackFnTy() const {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  return FunctionType::get(VoidTy, VoidPtrTy, false);
}

FunctionType *CudaLoop::getRegisterLinkedBinaryFnTy() const {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  auto CallbackFnTy = getCallbackFnTy();
  auto RegisterGlobalsFnTy = getRegisterGlobalsFnTy();
  Type *Params[] = {RegisterGlobalsFnTy->getPointerTo(), VoidPtrTy,
                    VoidPtrTy, CallbackFnTy->getPointerTo()};
  return FunctionType::get(VoidTy, Params, false);
}

/// Helper function that generates an empty dummy function returning void.
static Function *makeDummyFunction(Module &M, FunctionType *FnTy) {
  assert(FnTy->getReturnType()->isVoidTy() &&
         "Can only generate dummy functions returning void!");
  LLVMContext &Ctx = M.getContext();
  Function *DummyFunc = Function::Create(
      FnTy, GlobalValue::InternalLinkage, "dummy", &M);

  BasicBlock *DummyBlock = BasicBlock::Create(Ctx, "", DummyFunc);
  IRBuilder<> FuncBuilder(DummyBlock);
  FuncBuilder.CreateRetVoid();

  return DummyFunc;
}

/// Creates a function that sets up state on the host side for CUDA objects that
/// have a presence on both the host and device sides. Specifically, registers
/// the host side of kernel functions and device global variables with the CUDA
/// runtime.
/// \code
/// void __cuda_register_globals(void** GpuBinaryHandle) {
///    __cudaRegisterFunction(GpuBinaryHandle,Kernel0,...);
///    ...
///    __cudaRegisterFunction(GpuBinaryHandle,KernelM,...);
///    __cudaRegisterVar(GpuBinaryHandle, GlobalVar0, ...);
///    ...
///    __cudaRegisterVar(GpuBinaryHandle, GlobalVarN, ...);
/// }
/// \endcode
Function *CudaLoop::makeRegisterGlobalsFn() {
  // No need to register anything
  if (EmittedKernels.empty()/* && DeviceVars.empty()*/)
    return nullptr;

  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  PointerType *CharPtrTy = Type::getInt8Ty(Ctx)->getPointerTo();
  Type *IntTy = Type::getInt32Ty(Ctx);
  // Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();

  Function *RegisterKernelsFunc = Function::Create(
      getRegisterGlobalsFnTy(), GlobalValue::InternalLinkage,
      "__cuda_register_globals", &M);
  BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", RegisterKernelsFunc);
  IRBuilder<> Builder(EntryBB);

  // void __cudaRegisterFunction(void **, const char *, char *, const char *,
  //                             int, uint3*, uint3*, dim3*, dim3*, int*)
  Type *RegisterFuncParams[] = {
      VoidPtrPtrTy, CharPtrTy, CharPtrTy, CharPtrTy, IntTy,
      VoidPtrTy,    VoidPtrTy, VoidPtrTy, VoidPtrTy, IntTy->getPointerTo()};
  FunctionCallee RegisterFunc = M.getOrInsertFunction(
      "__cudaRegisterFunction",
      FunctionType::get(IntTy, RegisterFuncParams, false));

  // Extract GpuBinaryHandle passed as the first argument passed to
  // __cuda_register_globals() and generate __cudaRegisterFunction() call for
  // each emitted kernel.
  Argument &GpuBinaryHandlePtr = *RegisterKernelsFunc->arg_begin();
  for (auto &&I : EmittedKernels) {
    Constant *KernelNameCS =
        ConstantDataArray::getString(Ctx, I.DeviceFunc.str());
    GlobalVariable *KernelNameGV =
        new GlobalVariable(M, KernelNameCS->getType(), true,
                           GlobalValue::PrivateLinkage, KernelNameCS, ".str");
    KernelNameGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    Type *StrTy = KernelNameGV->getType();
    Constant *Zeros[] = { ConstantInt::get(DL.getIndexType(StrTy), 0),
                          ConstantInt::get(DL.getIndexType(StrTy), 0) };
    Constant *KernelName = ConstantExpr::getGetElementPtr(
        KernelNameGV->getValueType(), KernelNameGV, Zeros);
    Constant *NullPtr = ConstantPointerNull::get(VoidPtrTy);
    Value *Args[] = {
        &GpuBinaryHandlePtr,
        /*hostFun*/    Builder.CreateBitCast(I.Kernel, VoidPtrTy),
        /*deviceFun*/  KernelName,
        /*deviceName*/ KernelName,
        ConstantInt::get(IntTy, -1),
        NullPtr,
        NullPtr,
        NullPtr,
        NullPtr,
        ConstantPointerNull::get(IntTy->getPointerTo())};
    Builder.CreateCall(RegisterFunc, Args);
  }

  // // void __cudaRegisterVar(void **, char *, char *, const char *,
  // //                        int, int, int, int)
  // Type *RegisterVarParams[] = {VoidPtrPtrTy, CharPtrTy, CharPtrTy,
  //                              CharPtrTy,    IntTy,     IntTy,
  //                              IntTy,        IntTy};
  // FunctionCallee RegisterVar = CGM.CreateRuntimeFunction(
  //     FunctionType::get(IntTy, RegisterVarParams, false),
  //     addUnderscoredPrefixToName("RegisterVar"));
  // for (auto &&Info : DeviceVars) {
  //   GlobalVariable *Var = Info.Var;
  //   unsigned Flags = Info.Flag;
  //   Constant *VarName = makeConstantString(getDeviceSideName(Info.D));
  //   uint64_t VarSize =
  //       CGM.getDataLayout().getTypeAllocSize(Var->getValueType());
  //   Value *Args[] = {
  //       &GpuBinaryHandlePtr,
  //       Builder.CreateBitCast(Var, VoidPtrTy),
  //       VarName,
  //       VarName,
  //       ConstantInt::get(IntTy, (Flags & ExternDeviceVar) ? 1 : 0),
  //       ConstantInt::get(IntTy, VarSize),
  //       ConstantInt::get(IntTy, (Flags & ConstantDeviceVar) ? 1 : 0),
  //       ConstantInt::get(IntTy, 0)};
  //   Builder.CreateCall(RegisterVar, Args);
  // }

  Builder.CreateRetVoid();
  return RegisterKernelsFunc;
}

/// Creates a global constructor function for the module:
///
/// For CUDA:
/// \code
/// void __cuda_module_ctor(void*) {
///     Handle = __cudaRegisterFatBinary(GpuBinaryBlob);
///     __cuda_register_globals(Handle);
/// }
/// \endcode
// Based on makeModuleCtorFunction() in clang/lib/CodeGen/CGCUDANV.cpp.
Function *CudaLoop::makeModuleCtorFunction() {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *IntTy = Type::getInt32Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();

  // void __cuda_register_globals(void* handle);
  Function *RegisterGlobalsFunc = makeRegisterGlobalsFn();
  // We always need a function to pass in as callback. Create a dummy
  // implementation if we don't need to register anything.
  if (!RegisterGlobalsFunc)
    RegisterGlobalsFunc = makeDummyFunction(M, getRegisterGlobalsFnTy());

  // void ** __cudaRegisterFatBinary(void *);
  FunctionCallee RegisterFatbinFunc = M.getOrInsertFunction(
      "__cudaRegisterFatBinary",
      FunctionType::get(VoidPtrPtrTy, VoidPtrTy, false));

  // struct { int magic, int version, void * gpu_binary, void * dont_care };
  StructType *FatbinWrapperTy =
      StructType::get(IntTy, IntTy, VoidPtrTy, VoidPtrTy);

  Function *ModuleCtorFunc = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false),
      GlobalValue::InternalLinkage,
      "__cuda_module_ctor", &M);
  BasicBlock *CtorEntryBB =
      BasicBlock::Create(Ctx, "entry", ModuleCtorFunc);
  IRBuilder<> CtorBuilder(CtorEntryBB);

  // TODO: Differentiate the following code based on whether the kernels call
  // external functions, i.e., according to -fgpu-rdc definition.
  const char *FatbinConstantName = ".nv_fatbin";
  // const char *FatbinConstantName = "__nv_relfatbin";
  const char *FatbinSectionName = ".nvFatBinSegment";
  // const char *ModuleIDSectionName = "__nv_module_id";

  // PTXGlobal should be a string literal containing the fat binary of the
  // outlined kernels.
  PTXGlobal->setSection(FatbinConstantName);
  // Mark the address as used which make sure that this section isn't
  // merged and we will really have it in the object file.
  PTXGlobal->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
  PTXGlobal->setAlignment(Align(DL.getPrefTypeAlignment(PTXGlobal->getType())));

  unsigned FatbinVersion = 1;
  unsigned FatMagic = CudaFatMagic;

  Type *StrTy = PTXGlobal->getType();
  Constant *Zeros[] = { ConstantInt::get(DL.getIndexType(StrTy), 0),
                        ConstantInt::get(DL.getIndexType(StrTy), 0) };
  Constant *PTXGlobalPtr = ConstantExpr::getGetElementPtr(
      PTXGlobal->getValueType(), PTXGlobal, Zeros);
  Constant *FatbinWrapperVal =
      ConstantStruct::get(FatbinWrapperTy,
                          ConstantInt::get(IntTy, FatMagic),
                          ConstantInt::get(IntTy, FatbinVersion),
                          PTXGlobalPtr, ConstantPointerNull::get(VoidPtrTy));
  GlobalVariable *FatbinWrapper = new GlobalVariable(
      M, FatbinWrapperTy, /*isConstant*/ true, GlobalValue::InternalLinkage,
      FatbinWrapperVal, "__cuda_fatbin_wrapper");
  FatbinWrapper->setSection(FatbinSectionName);
  FatbinWrapper->setAlignment(
      Align(DL.getPrefTypeAlignment(FatbinWrapper->getType())));

  CallInst *RegisterFatbinCall = CtorBuilder.CreateCall(
      RegisterFatbinFunc,
      CtorBuilder.CreateBitCast(FatbinWrapper, VoidPtrTy));
  GpuBinaryHandle = new GlobalVariable(
      M, VoidPtrPtrTy, /*isConstant*/ false, GlobalValue::InternalLinkage,
      ConstantPointerNull::get(VoidPtrPtrTy), "__cuda_gpubin_handle");
  GpuBinaryHandle->setAlignment(Align(DL.getPointerABIAlignment(0)));
  GpuBinaryHandle->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
  CtorBuilder.CreateAlignedStore(RegisterFatbinCall, GpuBinaryHandle,
                                 DL.getPointerABIAlignment(0));

  // Call __cuda_register_globals(GpuBinaryHandle);
  if (RegisterGlobalsFunc)
    CtorBuilder.CreateCall(RegisterGlobalsFunc, RegisterFatbinCall);

  // // Call __cudaRegisterFatBinaryEnd(Handle) if this CUDA version needs it.
  // if (CudaFeatureEnabled(CGM.getTarget().getSDKVersion(),
  //                        CudaFeature::CUDA_USES_FATBIN_REGISTER_END)) {
  //   // void __cudaRegisterFatBinaryEnd(void **);
  //   llvm::FunctionCallee RegisterFatbinEndFunc = CGM.CreateRuntimeFunction(
  //       llvm::FunctionType::get(VoidTy, VoidPtrPtrTy, false),
  //       "__cudaRegisterFatBinaryEnd");
  //   CtorBuilder.CreateCall(RegisterFatbinEndFunc, RegisterFatbinCall);
  // }
  FunctionCallee RegisterFatbinEndFunc = M.getOrInsertFunction(
      "__cudaRegisterFatBinaryEnd",
      FunctionType::get(VoidTy, VoidPtrPtrTy, false));
  CtorBuilder.CreateCall(RegisterFatbinEndFunc, RegisterFatbinCall);

  // Create destructor and register it with atexit() the way NVCC does it. Doing
  // it during regular destructor phase worked in CUDA before 9.2 but results in
  // double-free in 9.2.
  if (Function *CleanupFn = makeModuleDtorFunction()) {
    // extern "C" int atexit(void (*f)(void));
    FunctionType *AtExitTy =
        FunctionType::get(IntTy, CleanupFn->getType(), false);
    FunctionCallee AtExitFunc =
        M.getOrInsertFunction("atexit", AtExitTy, AttributeList());
    CtorBuilder.CreateCall(AtExitFunc, CleanupFn);
  }

  CtorBuilder.CreateRetVoid();
  return ModuleCtorFunc;
}

Function *CudaLoop::makeModuleDtorFunction() {
  // No need for destructor if we don't have a handle to unregister.
  if (!GpuBinaryHandle)
    return nullptr;

  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  Type *VoidPtrPtrTy = VoidPtrTy->getPointerTo();

  // void __cudaUnregisterFatBinary(void ** handle);
  FunctionCallee UnregisterFatbinFunc = M.getOrInsertFunction(
      "__cudaUnregisterFatBinary",
      FunctionType::get(VoidTy, VoidPtrPtrTy, false));

  Function *ModuleDtorFunc = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false),
      GlobalValue::InternalLinkage,
      "__cuda_module_dtor", &M);

  BasicBlock *DtorEntryBB =
      BasicBlock::Create(Ctx, "entry", ModuleDtorFunc);
  IRBuilder<> DtorBuilder(DtorEntryBB);

  Value *HandleValue =
      DtorBuilder.CreateAlignedLoad(VoidPtrPtrTy, GpuBinaryHandle,
                                    DL.getPointerABIAlignment(0));
  DtorBuilder.CreateCall(UnregisterFatbinFunc, HandleValue);
  DtorBuilder.CreateRetVoid();
  return ModuleDtorFunc;
}

static void AddOptimizationPasses(legacy::PassManagerBase &MPM,
                                  legacy::FunctionPassManager &FPM,
                                  TargetMachine *TM, unsigned OptLevel,
                                  unsigned SizeLevel) {
  PassManagerBuilder Builder;
  Builder.OptLevel = OptLevel;
  Builder.SizeLevel = SizeLevel;

  if (TM)
    TM->adjustPassManager(Builder);

  Builder.populateFunctionPassManager(FPM);
  Builder.populateModulePassManager(MPM);

  // // Add in our optimization passes.
  // //
  // // TODO: Fix this set of optimization passes for PTX.

  // //MPM.add(createInstructionCombiningPass());
  // MPM.add(createReassociatePass());
  // MPM.add(createGVNPass());
  // MPM.add(createCFGSimplificationPass());
  // MPM.add(createSLPVectorizerPass());
  // //MPM.add(createBreakCriticalEdgesPass());
  // MPM.add(createConstantPropagationPass());
  // MPM.add(createDeadInstEliminationPass());
  // MPM.add(createDeadStoreEliminationPass());
  // //MPM.add(createInstructionCombiningPass());
  // MPM.add(createCFGSimplificationPass());
}

void PTXLoop::EmitPTX(raw_pwrite_stream *OS) {
  legacy::PassManager PM;
  legacy::FunctionPassManager FPM(&PTXM);

  // Add final module optimization passes and emit PTX.
  AddOptimizationPasses(PM, FPM, PTXTargetMachine, CodeGenOpt::Default, 0);

  bool Fail = PTXTargetMachine->addPassesToEmitFile(
      PM, *OS, nullptr,
      CodeGenFileType::CGFT_AssemblyFile, false);
  assert(!Fail && "Failed to emit PTX");

  // Add function optimization passes.
  FPM.doInitialization();
  for (Function &F : PTXM)
    FPM.run(F);
  FPM.doFinalization();

  PM.add(createVerifierPass());

  PM.run(PTXM);
}

void PTXLoop::makeFatBinaryString() {
  LLVM_DEBUG(dbgs() << "PTX Module: " << PTXM);

  // Get ptxas and fatbinary executables on the system.
  auto PTXASExec  = sys::findProgramByName("ptxas");
  auto FatBinExec = sys::findProgramByName("fatbinary");
  LLVM_DEBUG({
      if (PTXASExec)
        dbgs() << "Found " << *PTXASExec << "\n";
      if (FatBinExec)
        dbgs() << "Found " << *FatBinExec << "\n";
    });
  assert(PTXASExec && "Failed to find ptxas executable.");
  assert(FatBinExec && "Failed to find fatbinary executable.");

  // Output file storing PTX
  std::string PTXFile = M.getSourceFileName() + ".s";
  // Output file storing assembly generated from running ptxas
  std::string AsmFile = M.getSourceFileName() + ".o";
  // Output file storing fat binary generated from assembly
  std::string FatBinFile = M.getSourceFileName() + ".cubin";
  LLVM_DEBUG({
      dbgs() << "PTXFile = " << PTXFile << "\n";
      dbgs() << "AsmFile = " << AsmFile << "\n";
      dbgs() << "FatBinFile = " << FatBinFile << "\n";
    });

  // Emit PTX for the PTX module
  std::error_code EC;
  sys::fs::OpenFlags OpenFlags = sys::fs::F_None;
  std::unique_ptr<ToolOutputFile> FDOut =
      std::make_unique<ToolOutputFile>(PTXFile, EC, OpenFlags);
  raw_pwrite_stream *OS = &FDOut->os();

  EmitPTX(OS);

  FDOut->keep();

  opt::ArgStringList PTXASArgList, FatBinArgList;

  // Setup arguments to ptxas.
  PTXASArgList.push_back("-m64");
  if (!PTXM.getNamedMetadata("llvm.dbg.cu"))
    PTXASArgList.push_back(PTXASOptLevel.c_str());
  PTXASArgList.push_back("--gpu-name");
  PTXASArgList.push_back(GPUArch.c_str());
  PTXASArgList.push_back("--output-file");
  PTXASArgList.push_back(AsmFile.c_str());
  PTXASArgList.push_back(PTXFile.c_str());

  // Setup arguments to fatbinary.
  FatBinArgList.push_back("-64");
  FatBinArgList.push_back("--create");
  FatBinArgList.push_back(FatBinFile.c_str());
  std::string AsmInput =
      std::string("--image=profile=") + GPUArch + ",file=" +
      AsmFile;
  FatBinArgList.push_back(AsmInput.c_str());
  std::string PTXInput =
      std::string("--image=profile=") + VirtualArchForCudaArch(GPUArch) +
      ",file=" + PTXFile;
  FatBinArgList.push_back(PTXInput.c_str());

  // Run ptxas on the emitted PTX
  SmallVector<const char *, 128> PTXASArgv;
  PTXASArgv.push_back(PTXASExec->c_str());
  PTXASArgv.append(PTXASArgList.begin(), PTXASArgList.end());
  PTXASArgv.push_back(nullptr);
  auto PTXASArgs = toStringRefArray(PTXASArgv.data());
  LLVM_DEBUG({
      for (auto Str : PTXASArgs)
        dbgs() << Str << "\n";
    });
  sys::ExecuteAndWait(*PTXASExec, PTXASArgs);

  // Run fatbinary
  SmallVector<const char *, 128> FatBinArgv;
  FatBinArgv.push_back(FatBinExec->c_str());
  FatBinArgv.append(FatBinArgList.begin(), FatBinArgList.end());
  FatBinArgv.push_back(nullptr);
  auto FatBinArgs = toStringRefArray(FatBinArgv.data());
  LLVM_DEBUG({
      for (auto Str : FatBinArgs)
        dbgs() << Str << "\n";
    });
  sys::ExecuteAndWait(*FatBinExec, FatBinArgs);

  // Create a global string in the original module to hold the binary output of
  // running ptxas and fatbinary.
  ErrorOr<std::unique_ptr<MemoryBuffer>> FatBinBuf =
      MemoryBuffer::getFile(FatBinFile);
  Constant *PCS = ConstantDataArray::getString(M.getContext(),
                                               FatBinBuf.get()->getBuffer());
  PTXGlobal = new GlobalVariable(M, PCS->getType(), true,
                                 GlobalValue::PrivateLinkage, PCS,
                                 "ptx" + Twine(MyKernelID));
}

Value *CudaABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  Value *Grainsize = ConstantInt::get(GrainsizeCall->getType(), 8);

  // Replace uses of grainsize intrinsic call with this grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  return Grainsize;
}

void CudaABI::lowerSync(SyncInst &SI) {
  // currently a no-op...
}

void CudaABI::preProcessFunction(Function &F, TaskInfo &TI,
                                 bool OutliningTapirLoops) {
}

// Adapted from Transforms/Utils/ModuleUtils.cpp
static void appendToGlobalArray(const char *Array, Module &M, Constant *C,
                                int Priority, Constant *Data) {
  IRBuilder<> IRB(M.getContext());
  FunctionType *FnTy = FunctionType::get(IRB.getVoidTy(), false);

  // Get the current set of static global constructors and add the new ctor
  // to the list.
  SmallVector<Constant *, 16> CurrentCtors;
  StructType *EltTy = StructType::get(
      IRB.getInt32Ty(), PointerType::getUnqual(FnTy), IRB.getInt8PtrTy());
  if (GlobalVariable *GVCtor = M.getNamedGlobal(Array)) {
    if (Constant *Init = GVCtor->getInitializer()) {
      unsigned n = Init->getNumOperands();
      CurrentCtors.reserve(n + 1);
      for (unsigned i = 0; i != n; ++i)
        CurrentCtors.push_back(cast<Constant>(Init->getOperand(i)));
    }
    GVCtor->eraseFromParent();
  }

  // Build a 3 field global_ctor entry.  We don't take a comdat key.
  Constant *CSVals[3];
  CSVals[0] = IRB.getInt32(Priority);
  CSVals[1] = C;
  CSVals[2] = Data ? ConstantExpr::getPointerCast(Data, IRB.getInt8PtrTy())
                   : Constant::getNullValue(IRB.getInt8PtrTy());
  Constant *RuntimeCtorInit =
      ConstantStruct::get(EltTy, makeArrayRef(CSVals, EltTy->getNumElements()));

  CurrentCtors.push_back(RuntimeCtorInit);

  // Create a new initializer.
  ArrayType *AT = ArrayType::get(EltTy, CurrentCtors.size());
  Constant *NewInit = ConstantArray::get(AT, CurrentCtors);

  // Create the new global variable and replace all uses of
  // the old global variable with the new one.
  (void)new GlobalVariable(M, NewInit->getType(), false,
                           GlobalValue::AppendingLinkage, NewInit, Array);
}

void CudaABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
  if (!OutliningTapirLoops || !LOP)
    return;

  LOP->makeFatBinaryString();

  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  if (Function *CudaCtorFunction = LOP->makeModuleCtorFunction()) {
    // Ctor function type is void()*.
    FunctionType* CtorFTy = FunctionType::get(VoidTy, false);
    Type *CtorPFTy =
        PointerType::get(CtorFTy, M.getDataLayout().getProgramAddressSpace());
    appendToGlobalArray(
        "llvm.global_ctors", M,
        ConstantExpr::getBitCast(CudaCtorFunction, CtorPFTy), 65536, nullptr);
  }
}

void CudaABI::postProcessHelper(Function &F) {
}

void CudaABI::preProcessOutlinedTask(Function &F, Instruction *DetachPt,
                                     Instruction *TaskFrameCreate,
                                     bool IsSpawner) {
}

void CudaABI::postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                                      Instruction *TaskFrameCreate,
                                      bool IsSpawner) {
}

void CudaABI::preProcessRootSpawner(Function &F) {
}

void CudaABI::postProcessRootSpawner(Function &F) {
}

void CudaABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
}

LoopOutlineProcessor *CudaABI::getLoopOutlineProcessor(
    const TapirLoopInfo *TL) {
  if (!LOP)
    LOP = new CudaLoop(M);
  return LOP;
}

// Static counter for assigning IDs to kernels.
unsigned PTXLoop::NextKernelID = 0;

PTXLoop::PTXLoop(Module &M)
    : LoopOutlineProcessor(M, PTXM),
      PTXM("ptxModule", M.getContext()) {
  // Assign an ID to this kernel.
  MyKernelID = NextKernelID++;

  // Setup an NVPTX triple.
  Triple PTXTriple("nvptx64", "nvidia", "cuda");
  PTXM.setTargetTriple(PTXTriple.str());
  PTXM.setSDKVersion(VersionTuple(10, 1));
  if (M.getSDKVersion().empty())
    M.setSDKVersion(VersionTuple(10, 1));

  // Find the NVPTX module pass which will create the PTX code
  std::string error;
  const Target *PTXTarget = TargetRegistry::lookupTarget("", PTXTriple, error);
  LLVM_DEBUG({
      if (!PTXTarget)
        dbgs() << "ERROR: Failed to lookup NVPTX target: " << error << "\n";
    });
  assert(PTXTarget && "Failed to find NVPTX target");

  // TODO: Hard-coded machine configuration for Supercloud nodes with Voltas and
  // CUDA 10.1  Generalize this code.
  PTXTargetMachine =
      PTXTarget->createTargetMachine(PTXTriple.getTriple(), GPUArch,
                                     "+ptx64", TargetOptions(), Reloc::PIC_,
                                     CodeModel::Small, CodeGenOpt::Aggressive);
  PTXM.setDataLayout(PTXTargetMachine->createDataLayout());

  // Insert runtime-function declarations in PTX host modules.
  Type *PTXInt32Ty = Type::getInt32Ty(PTXM.getContext());
  GetThreadIdx = PTXM.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.x",
                                          PTXInt32Ty);
  GetBlockIdx = PTXM.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.ctaid.x",
                                         PTXInt32Ty);
  GetBlockDim = PTXM.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.ntid.x",
                                         PTXInt32Ty);
}

void PTXLoop::setupLoopOutlineArgs(
    Function &F, ValueSet &HelperArgs, SmallVectorImpl<Value *> &HelperInputs,
    ValueSet &InputSet, const SmallVectorImpl<Value *> &LCArgs,
    const SmallVectorImpl<Value *> &LCInputs, const ValueSet &TLInputsFixed) {
  // Add the loop control inputs.

  // The first parameter defines the extent of the index space, i.e., the number
  // of threads to launch.
  {
    Argument *EndArg = cast<Argument>(LCArgs[1]);
    EndArg->setName("runSize");
    HelperArgs.insert(EndArg);

    Value *InputVal = LCInputs[1];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }
  // The second parameter defines the start of the index space.
  {
    Argument *StartArg = cast<Argument>(LCArgs[0]);
    StartArg->setName("runStart");
    HelperArgs.insert(StartArg);

    Value *InputVal = LCInputs[0];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }
  // The third parameter defines the grainsize, if it is not constant.
  if (!isa<ConstantInt>(LCInputs[2])) {
    Argument *GrainsizeArg = cast<Argument>(LCArgs[2]);
    GrainsizeArg->setName("runStride");
    HelperArgs.insert(GrainsizeArg);

    Value *InputVal = LCInputs[2];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // Add the remaining inputs
  for (Value *V : TLInputsFixed) {
    assert(!HelperArgs.count(V));
    HelperArgs.insert(V);
    HelperInputs.push_back(V);
  }
}

unsigned PTXLoop::getIVArgIndex(const Function &F, const ValueSet &Args) const {
  // The argument for the primary induction variable is the second input.
  return 1;
}

unsigned PTXLoop::getLimitArgIndex(const Function &F, const ValueSet &Args)
  const {
  // The argument for the loop limit is the first input.
  return 0;
}

void PTXLoop::postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                                 ValueToValueMapTy &VMap) {
  Task *T = TL.getTask();
  Loop *L = TL.getLoop();

  Function *Helper = Out.Outline;
  BasicBlock *Entry = cast<BasicBlock>(VMap[L->getLoopPreheader()]);
  BasicBlock *Header = cast<BasicBlock>(VMap[L->getHeader()]);
  BasicBlock *Exit = cast<BasicBlock>(VMap[TL.getExitBlock()]);
  PHINode *PrimaryIV = cast<PHINode>(VMap[TL.getPrimaryInduction().first]);
  Value *PrimaryIVInput = PrimaryIV->getIncomingValueForBlock(Entry);
  Instruction *ClonedSyncReg = cast<Instruction>(
      VMap[T->getDetach()->getSyncRegion()]);

  // We no longer need the cloned sync region.
  ClonedSyncReg->eraseFromParent();

  // Set the helper function to have external linkage.
  Helper->setLinkage(Function::ExternalLinkage);
  // Set the target-cpu and target-features
  AttrBuilder Attrs;
  Attrs.addAttribute("target-cpu", GPUArch);
  Attrs.addAttribute("target-features", GPUFeatures + ",+" + GPUArch);
  Helper->removeFnAttr("target-cpu");
  Helper->removeFnAttr("target-features");
  Helper->addAttributes(AttributeList::FunctionIndex, Attrs);

  // Verify that the Thread ID corresponds to a valid iteration.  Because Tapir
  // loops use canonical induction variables, valid iterations range from 0 to
  // the loop limit with stride 1.  The End argument encodes the loop limit.
  // Get end and grainsize arguments
  Argument *End, *Start;
  Value *Grainsize;
  {
    auto OutlineArgsIter = Helper->arg_begin();
    // End argument is the first LC arg.
    End = &*OutlineArgsIter++;
    // Start argument is the second LC arg.
    Start = &*OutlineArgsIter++;

    // Get the grainsize value, which is either constant or the third LC arg.
    if (unsigned ConstGrainsize = TL.getGrainsize())
      Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
    else
      // Grainsize argument is the third LC arg.
      Grainsize = &*OutlineArgsIter;
  }

  // Get the thread ID for this invocation of Helper.
  IRBuilder<> B(Entry->getTerminator());
  Value *ThreadIdx = B.CreateCall(GetThreadIdx);
  Value *BlockIdx = B.CreateCall(GetBlockIdx);
  Value *BlockDim = B.CreateCall(GetBlockDim);
  Value *ThreadID = B.CreateIntCast(
      B.CreateAdd(ThreadIdx, B.CreateMul(BlockIdx, BlockDim), "threadId"),
      PrimaryIV->getType(), false);

  ThreadID = B.CreateMul(ThreadID, Grainsize);
  Value *ThreadEnd = B.CreateAdd(ThreadID, Grainsize);
  Value *Cond = B.CreateICmpUGE(ThreadID, End);

  ReplaceInstWithInst(Entry->getTerminator(), BranchInst::Create(Exit, Header,
                                                                 Cond));
  // Use the thread ID as the start iteration number for the primary IV.
  PrimaryIVInput->replaceAllUsesWith(ThreadID);

  // Update cloned loop condition to use the thread-end value.
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TL.getCondition()]);
  if (ClonedCond->getOperand(0) != End)
    ++TripCountIdx;
  assert(ClonedCond->getOperand(TripCountIdx) == End &&
         "End argument not used in condition");
  ClonedCond->setOperand(TripCountIdx, ThreadEnd);

  LLVMContext &Ctx = PTXM.getContext();
  // Add the necessary NVPTX to mark the global function
  NamedMDNode *Annotations =
    PTXM.getOrInsertNamedMetadata("nvvm.annotations");

  SmallVector<Metadata *, 3> AV;
  AV.push_back(ValueAsMetadata::get(Helper));
  AV.push_back(MDString::get(Ctx, "kernel"));
  AV.push_back(ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx),
                                                     1)));
  Annotations->addOperand(MDNode::get(Ctx, AV));
}


CudaLoop::CudaLoop(Module &M) : PTXLoop(M) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  CudaStreamTy = StructType::lookupOrCreate(Ctx, "struct.CUstream_st");
  Type *CudaStreamPtrTy = PointerType::getUnqual(CudaStreamTy);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *SizeTy = DL.getIntPtrType(Ctx);
  Dim3Ty = StructType::create("struct.dim3", Int32Ty, Int32Ty, Int32Ty);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);

  CudaPushCallConfig = M.getOrInsertFunction(
      "__cudaPushCallConfiguration", Int32Ty, Int64Ty, Int32Ty, Int64Ty,
      Int32Ty, SizeTy, VoidPtrTy);
  CudaPopCallConfig = M.getOrInsertFunction(
      "__cudaPopCallConfiguration", Int32Ty, PointerType::getUnqual(Dim3Ty),
      PointerType::getUnqual(Dim3Ty), PointerType::getUnqual(SizeTy),
      PointerType::getUnqual(VoidPtrTy));
  CudaLaunchKernel = M.getOrInsertFunction(
      "cudaLaunchKernel", Int32Ty, VoidPtrTy, Int64Ty, Int32Ty, Int64Ty,
      Int32Ty, PointerType::getUnqual(VoidPtrTy), SizeTy, CudaStreamPtrTy);

  /// KitsuneCuda Runtime Calls
  Type *VoidTy = Type::getVoidTy(M.getContext());
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  KitsuneCUDAInit = M.getOrInsertFunction("__kitsune_cudart_initialize", VoidTy);
  KitsuneCUDAMallocManaged = M.getOrInsertFunction(
      "__kitsune_cudart_malloc_managed", VoidTy, VoidPtrTy, Int64Ty, Int32Ty);

}

void CudaLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                       DominatorTree &DT) {
  Function *Outlined = TOI.Outline;
  Instruction *ReplStart = TOI.ReplStart;
  Instruction *ReplCall = TOI.ReplCall;
  CallSite CS(ReplCall);
  BasicBlock *CallCont = TOI.ReplRet;
  BasicBlock *UnwindDest = TOI.ReplUnwind;
  Function *Parent = ReplCall->getFunction();
  Value *TripCount = CS.getArgOperand(getLimitArgIndex(*Parent, TOI.InputSet));

  // Fixup name of outlined function, since PTX does not like '.' characters in
  // function names.
  SmallString<256> Buf;
  for (char C : Outlined->getName().bytes()) {
    if ('.' == C)
      Buf.push_back('_');
    else
      Buf.push_back(C);
  }
  Outlined->setName(Buf);

  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *SizeTy = DL.getIntPtrType(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *CudaStreamPtrTy = CudaStreamTy->getPointerTo();

  ///------------------------------------------------------------
  // Insert calls to KitsuneCUDAInit and KitsuneCUDAMallocManaged. 
  IRBuilder<> KitsuneCB(Ctx);
  BasicBlock *ParentEB = &Parent->front();
  Instruction *FirstInst = ParentEB->getFirstNonPHI();

  // Insert KitssuneCUDAInit and KitsuneCUDAMallocManaged 
  // at the beginning of the first BB of host function.
  KitsuneCB.SetInsertPoint(FirstInst->getNextNode());

  KitsuneCB.CreateCall(KitsuneCUDAInit, {});

  Value *DataPtr = nullptr;
  Value *DataSize = nullptr;
  Value *DataTypeSize = nullptr;
  Function::arg_iterator I;
  Function::arg_iterator E = Parent->arg_end();
  Type *FloatPtrTy = Type::getFloatPtrTy(Ctx);

  // Find the data size passed as an integer argument.
  for (I = Parent->arg_begin(); I != E; I++){
      if (I->getType()->getTypeID() == llvm::Type::IntegerTyID)
          DataSize = I;
  }

  // For each pointer type argument, call KitsuneCUDAMallocManaged.
  for (I = Parent->arg_begin(); I != E; I++){
        Type *ArgTy = I->getType();
        Type *ArgContainedTy = ArgTy->getContainedType(0);

        if (ArgTy->getTypeID() == llvm::Type::PointerTyID){
            DataTypeSize = ConstantInt::get(Int32Ty,
                    ArgContainedTy->getPrimitiveSizeInBits()/8);

            Value *AInst = KitsuneCB.CreateAlloca(ArgTy);
            StoreInst *SInst = KitsuneCB.CreateStore(I, AInst);
            Value* DataPtr = KitsuneCB.CreateBitCast(
                    SInst->getOperand(1), VoidPtrTy);
            CallInst *CInst = KitsuneCB.CreateCall(KitsuneCUDAMallocManaged,
                 {DataPtr, DataSize, DataTypeSize});
        }
  }
  ///-----------------------------------------------------------------
  // Split the basic block containing the detach replacement just before the
  // start of the detach-replacement instructions.
  BasicBlock *DetBlock = ReplStart->getParent();
  BasicBlock *CallBlock = SplitBlock(DetBlock, ReplStart);

  // Create a call to CudaPopCallConfiguration
  IRBuilder<> B(ReplCall);
  // Create local allocations for components of the configuration.
  AllocaInst *GridDim = B.CreateAlloca(Dim3Ty, nullptr, "grid_dim");
  AllocaInst *BlockDim = B.CreateAlloca(Dim3Ty, nullptr, "block_dim");
  AllocaInst *SHMemSize = B.CreateAlloca(SizeTy, nullptr, "shmem_size");
  AllocaInst *StreamPtr = B.CreateAlloca(VoidPtrTy, nullptr, "stream");
  // Call __cudaPopCallConfiguration
  B.CreateCall(CudaPopCallConfig, { GridDim, BlockDim, SHMemSize, StreamPtr });

  // Coerce dimensions into arguments for launch kernel
  Type *CoercedDim3Ty = StructType::get(Int64Ty, Int32Ty);
  AllocaInst *CoercedGridDim = B.CreateAlloca(CoercedDim3Ty);
  AllocaInst *CoercedBlockDim = B.CreateAlloca(CoercedDim3Ty);
  B.CreateMemCpy(CoercedGridDim, MaybeAlign(CoercedGridDim->getAlignment()),
                 GridDim, MaybeAlign(GridDim->getAlignment()),
                 ConstantInt::get(SizeTy, DL.getTypeAllocSize(Dim3Ty)));
  B.CreateMemCpy(CoercedBlockDim, MaybeAlign(CoercedBlockDim->getAlignment()),
                 BlockDim, MaybeAlign(BlockDim->getAlignment()),
                 ConstantInt::get(SizeTy, DL.getTypeAllocSize(Dim3Ty)));

  // Load coerced grid and block dimensions
  Value *GridDimStart =
      B.CreateLoad(Int64Ty, B.CreateConstInBoundsGEP2_32(CoercedDim3Ty,
                                                         CoercedGridDim, 0, 0));
  Value *GridDimEnd =
      B.CreateLoad(Int32Ty, B.CreateConstInBoundsGEP2_32(CoercedDim3Ty,
                                                         CoercedGridDim, 0, 1));
  Value *BlockDimStart =
      B.CreateLoad(Int64Ty, B.CreateConstInBoundsGEP2_32(CoercedDim3Ty,
                                                         CoercedBlockDim,
                                                         0, 0));
  Value *BlockDimEnd =
      B.CreateLoad(Int32Ty, B.CreateConstInBoundsGEP2_32(CoercedDim3Ty,
                                                         CoercedBlockDim,
                                                         0, 1));

  // Create an array of kernel arguments.
  AllocaInst *KernelArgs;
  // Calculate amount of space we will need for all arguments.  If we have no
  // args, allocate a single pointer so we still have a valid pointer to the
  // argument array that we can pass to runtime, even if it will be unused.
  if (CS.arg_empty())
    // CS.args() contains no arguments to pass to the kernel.
    KernelArgs = B.CreateAlloca(VoidPtrTy, nullptr, "kernel_args");
  else {
    KernelArgs = B.CreateAlloca(VoidPtrTy, B.getInt8(CS.arg_size()),
                                "kernel_args");
    // Store pointers to the arguments.
    unsigned Index = 0, ArgNum = 0;
    for (Value *Arg : CS.args()) {
      AllocaInst *ArgAlloc = B.CreateAlloca(Arg->getType());
      B.CreateStore(Arg, ArgAlloc);
      B.CreateStore(
          B.CreateBitCast(ArgAlloc, VoidPtrTy),
          B.CreateConstInBoundsGEP1_32(VoidPtrTy, KernelArgs, Index));
      ++Index; ++ArgNum;
    }
  }
  // Insert call to cudaLaunchKernel
  CallInst *Call = B.CreateCall(CudaLaunchKernel,
                                { ConstantPointerNull::get(VoidPtrTy),
                                  GridDimStart, GridDimEnd, BlockDimStart,
                                  BlockDimEnd, KernelArgs,
                                  B.CreateLoad(SHMemSize),
                                  B.CreateBitCast(B.CreateLoad(StreamPtr),
                                                  CudaStreamPtrTy) });

  LLVM_DEBUG(dbgs() << "CudaLoop: Adding helper for cudaLaunchKernel call "
                       "site.\n");

  // Update the value of ReplCall.
  ReplCall = TOI.ReplCall;

  // cudaLaunchKernel needs this function to have the same type as the kernel
  ValueSet SHInputs;
  for (Value *Arg : CS.args())
    SHInputs.insert(Arg);

  ValueSet Outputs;  // Should be empty.
  // Only one block needs to be cloned into the spawn helper
  std::vector<BasicBlock *> BlocksToClone;
  BlocksToClone.push_back(CallBlock);
  SmallVector<ReturnInst *, 1> Returns;  // Ignore returns cloned.
  ValueToValueMapTy VMap;
  Twine NameSuffix = ".stub";
  Function *CudaLoopHelper =
      CreateHelper(SHInputs, Outputs, BlocksToClone, CallBlock, DetBlock,
                   CallCont, VMap, &M, Parent->getSubprogram() != nullptr,
                   Returns, NameSuffix.str(), nullptr, nullptr, nullptr,
                   UnwindDest);

  assert(Returns.empty() && "Returns cloned when creating SpawnHelper.");

  // If there is no unwind destination, then the SpawnHelper cannot throw.
  if (!UnwindDest)
    CudaLoopHelper->setDoesNotThrow();

  // Add attributes to new helper function.
  CudaLoopHelper->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

  // cudaLaunchKernel needs this function to have the same name as the kernel
  CudaLoopHelper->setName(Outlined->getName());

  // Add alignment assumptions to arguments of helper, based on alignment of
  // values in old function.
  AddAlignmentAssumptions(Parent, SHInputs, VMap, ReplCall, nullptr, nullptr);

  // Move allocas in the newly cloned block to the entry block of the helper.
  {
    // Collect the end instructions of the task.
    SmallVector<Instruction *, 4> Ends;
    Ends.push_back(cast<BasicBlock>(VMap[CallCont])->getTerminator());
    if (isa<InvokeInst>(ReplCall))
      Ends.push_back(cast<BasicBlock>(VMap[UnwindDest])->getTerminator());

    // Move allocas in cloned detached block to entry of helper function.
    BasicBlock *ClonedBlock = cast<BasicBlock>(VMap[CallBlock]);
    MoveStaticAllocasInBlock(&CudaLoopHelper->getEntryBlock(), ClonedBlock,
                             Ends);

    // We do not need to add new llvm.stacksave/llvm.stackrestore intrinsics,
    // because calling and returning from the helper will automatically manage
    // the stack appropriately.
  }

  // Insert a call to the spawn helper.
  SmallVector<Value *, 8> SHInputVec;
  for (Value *V : SHInputs)
    SHInputVec.push_back(V);
  SplitEdge(DetBlock, CallBlock);
  B.SetInsertPoint(CallBlock->getTerminator());
  // Value *ModVal = B.CreateAnd(TripCount, ThreadsPerBlock, "xtraiter");
  Value *BranchVal = B.CreateICmpULT(TripCount,
                                     ConstantInt::get(TripCount->getType(),
                                                      ThreadsPerBlock));
  Value *BlockVal = B.CreateSelect(BranchVal, TripCount,
                                   ConstantInt::get(TripCount->getType(),
                                                    ThreadsPerBlock));
  // Value *Threads =
  //     B.CreateSelect(BranchVal, ConstantInt::get(TripCount->getType(), 1),
  //                    B.CreateAdd(
  //                        ConstantInt::get(TripCount->getType(), 1),
  //                        B.CreateUDiv(TripCount,
  //                                     ConstantInt::get(TripCount->getType(),
  //                                                      ThreadsPerBlock))));
  Value *Threads = B.CreateAdd(
      ConstantInt::get(TripCount->getType(), 1),
      B.CreateUDiv(TripCount, ConstantInt::get(TripCount->getType(),
                                               ThreadsPerBlock)));

  // Instruction *ThenTerm, *ElseTerm;
  // SplitBlockAndInsertIfThenElse(BranchVal, CallBlock->getTerminator(),
  //                               &ThenTerm, &ElseTerm);
  // {
  //   B.SetInsertPoint(ThenTerm);
  //   // Insert call to __cudaPushCallConfiguration.
  //   Value *Threads = ModVal;
  //   Value *CoercedThreads = B.CreateOr(B.getInt64(1UL << 32), Threads);
  //   B.CreateCall(CudaPushCallConfig,
  //                { /*gridDim*/   B.getInt64(1UL << 32 | 1UL), B.getInt32(1),
  //                  /*blockDim*/  CoercedThreads, B.getInt32(1),
  //                  /*sharedMem*/ ConstantInt::get(SizeTy, 0),
  //                  /*stream*/    ConstantPointerNull::get(VoidPtrTy) });

  //   CallInst *HelperCall = B.CreateCall(CudaLoopHelper, SHInputVec);
  //   HelperCall->setDebugLoc(ReplCall->getDebugLoc());
  //   HelperCall->setCallingConv(CudaLoopHelper->getCallingConv());
  //   HelperCall->setDoesNotThrow();
  // }

  // {
  //   B.SetInsertPoint(ElseTerm);
  //   // Insert call to __cudaPushCallConfiguration.
  //   Value *Threads = B.CreateAdd(
  //       ConstantInt::get(TripCount->getType(), 1),
  //       B.CreateUDiv(TripCount, ConstantInt::get(TripCount->getType(),
  //                                                ThreadsPerBlock)));
  //   Value *CoercedThreads = B.CreateOr(B.getInt64(1UL << 32), Threads);
  //   B.CreateCall(CudaPushCallConfig,
  //                { /*gridDim*/   CoercedThreads, B.getInt32(1),
  //                  /*blockDim*/  B.getInt64(
  //                      1UL << 32 | (uint64_t)ThreadsPerBlock), B.getInt32(1),
  //                  /*sharedMem*/ ConstantInt::get(SizeTy, 0),
  //                  /*stream*/    ConstantPointerNull::get(VoidPtrTy) });

  //   CallInst *HelperCall = B.CreateCall(CudaLoopHelper, SHInputVec);
  //   HelperCall->setDebugLoc(ReplCall->getDebugLoc());
  //   HelperCall->setCallingConv(CudaLoopHelper->getCallingConv());
  //   HelperCall->setDoesNotThrow();
  // }

  // First insert call to __cudaPushCallConfiguration.
  // Value *Threads = TripCount;
  // Value *CoercedThreads = B.CreateOr(B.getInt64(1UL << 32), Threads);
  // B.CreateCall(CudaPushCallConfig,
  //              { /*gridDim*/   CoercedThreads, B.getInt32(1),
  //                /*blockDim*/  B.getInt64(1UL << 32 | 1UL), B.getInt32(1),
  //                /*sharedMem*/ ConstantInt::get(SizeTy, 0),
  //                /*stream*/    ConstantPointerNull::get(VoidPtrTy) });

  // TODO: Use LoopStripMine to generate specialized versions of the kernel for
  // different iteration counts, rather than use a complex call configuration.
  Value *CoercedThreads = B.CreateOr(B.getInt64(1UL << 32), Threads);
  Value *CoercedBlockVal = B.CreateOr(B.getInt64(1UL << 32), BlockVal);
  B.CreateCall(CudaPushCallConfig,
               { /*gridDim*/   CoercedThreads, B.getInt32(1),
                 /*blockDim*/  CoercedBlockVal, B.getInt32(1),
                 /*sharedMem*/ ConstantInt::get(SizeTy, 0),
                 /*stream*/    ConstantPointerNull::get(VoidPtrTy) });
  if (isa<InvokeInst>(ReplCall)) {
    InvokeInst *HelperCall = InvokeInst::Create(CudaLoopHelper, CallCont,
                                                UnwindDest, SHInputVec);
    HelperCall->setDebugLoc(ReplCall->getDebugLoc());
    HelperCall->setCallingConv(CudaLoopHelper->getCallingConv());
    ReplaceInstWithInst(CallBlock->getTerminator(), HelperCall);
  } else {
    CallInst *HelperCall = B.CreateCall(CudaLoopHelper, SHInputVec);
    HelperCall->setDebugLoc(ReplCall->getDebugLoc());
    HelperCall->setCallingConv(CudaLoopHelper->getCallingConv());
    HelperCall->setDoesNotThrow();
    // Branch around CallBlock.  Its contents are now dead.
    ReplaceInstWithInst(CallBlock->getTerminator(),
                        BranchInst::Create(CallCont));
  }

  // Replace the first argument of cudaLaunchKernel to point to the helper.
  CallInst *CallInHelper = cast<CallInst>(VMap[Call]);
  B.SetInsertPoint(CallInHelper);
  CallInHelper->setArgOperand(0, B.CreateBitCast(CudaLoopHelper, VoidPtrTy));

  // Record CudaLoopHelper as an emitted kernel
  EmittedKernels.push_back({ CudaLoopHelper, Outlined->getName() });

  // Erase extraneous call instructions
  cast<Instruction>(VMap[ReplCall])->eraseFromParent();
  ReplCall->eraseFromParent();
}


//KitsuneCudaLoop::KitsuneCudaLoop(Module &M) : PTXLoop(M) {
//  Type *VoidTy = Type::getVoidTy(M.getContext());
//  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
//  Type *Int8Ty = Type::getInt8Ty(M.getContext());
//  Type *Int32Ty = Type::getInt32Ty(M.getContext());
//  Type *Int64Ty = Type::getInt64Ty(M.getContext());
//  KitsuneCUDAInit = M.getOrInsertFunction("__kitsune_cuda_init", VoidTy);
//  KitsuneGPUInitKernel = M.getOrInsertFunction("__kitsune_gpu_init_kernel",
//                                               VoidTy, Int32Ty, VoidPtrTy);
//  KitsuneGPUInitField = M.getOrInsertFunction("__kitsune_gpu_init_field",
//                                              VoidTy, Int32Ty, VoidPtrTy,
//                                              VoidPtrTy, Int32Ty, Int64Ty,
//                                              Int8Ty);
//  KitsuneGPUSetRunSize = M.getOrInsertFunction("__kitsune_gpu_set_run_size",
//                                               VoidTy, Int32Ty, Int64Ty,
//                                               Int64Ty, Int64Ty);
//  KitsuneGPURunKernel = M.getOrInsertFunction("__kitsune_gpu_run_kernel",
//                                              VoidTy, Int32Ty);
//  KitsuneGPUFinish = M.getOrInsertFunction("__kitsune_gpu_finish", VoidTy);
//
//}
//
//void KitsuneCudaLoop::processOutlinedLoopCall(TapirLoopInfo &TL,
//                                              TaskOutlineInfo &TOI,
//                                              DominatorTree &DT) {
//  LLVMContext &Ctx = M.getContext();
//  Type *Int8Ty = Type::getInt8Ty(Ctx);
//  Type *Int32Ty = Type::getInt32Ty(Ctx);
//  Type *Int64Ty = Type::getInt64Ty(Ctx);
//  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
//
//  Task *T = TL.getTask();
//  CallBase *ReplCall = cast<CallBase>(TOI.ReplCall);
//  Function *Parent = ReplCall->getFunction();
//  Value *RunStart = ReplCall->getArgOperand(getIVArgIndex(*Parent,
//                                                          TOI.InputSet));
//  Value *TripCount = ReplCall->getArgOperand(getLimitArgIndex(*Parent,
//                                                              TOI.InputSet));
//  IRBuilder<> B(ReplCall);
//
//  Value *KernelID = ConstantInt::get(Int32Ty, MyKernelID);
//  Value *PTXStr = B.CreateBitCast(PTXGlobal, VoidPtrTy);
//
//  B.CreateCall(KitsuneCUDAInit, {});
//  B.CreateCall(KitsuneGPUInitKernel, { KernelID, PTXStr });
//
//  for (Value *V : TOI.InputSet) {
//    Value *ElementSize = nullptr;
//    Value *VPtr;
//    Value *FieldName;
//    Value *Size = nullptr;
//
//    // TODO: fix
//    // this is a temporary hack to get the size of the field
//    // it will currently only work for a limited case
//
//    if (BitCastInst *BC = dyn_cast<BitCastInst>(V)) {
//      CallInst *CI = dyn_cast<CallInst>(BC->getOperand(0));
//      assert(CI && "Unable to detect field size");
//
//      Value *Bytes = CI->getOperand(0);
//      assert(Bytes->getType()->isIntegerTy(64));
//
//      PointerType *PT = dyn_cast<PointerType>(V->getType());
//      IntegerType *IntT = dyn_cast<IntegerType>(PT->getElementType());
//      assert(IntT && "Expected integer type");
//
//      Constant *Fn = ConstantDataArray::getString(Ctx, CI->getName());
//      GlobalVariable *FieldNameGlobal =
//          new GlobalVariable(M, Fn->getType(), true,
//                             GlobalValue::PrivateLinkage, Fn, "field.name");
//      FieldName = B.CreateBitCast(FieldNameGlobal, VoidPtrTy);
//      VPtr = B.CreateBitCast(V, VoidPtrTy);
//      ElementSize = ConstantInt::get(Int32Ty, IntT->getBitWidth()/8);
//      Size = B.CreateUDiv(Bytes, ConstantInt::get(Int64Ty,
//                                                  IntT->getBitWidth()/8));
//    } else if (AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
//      Constant *Fn = ConstantDataArray::getString(Ctx, AI->getName());
//      GlobalVariable *FieldNameGlobal =
//          new GlobalVariable(M, Fn->getType(), true,
//                             GlobalValue::PrivateLinkage, Fn, "field.name");
//      FieldName = B.CreateBitCast(FieldNameGlobal, VoidPtrTy);
//      VPtr = B.CreateBitCast(V, VoidPtrTy);
//      ArrayType *AT = dyn_cast<ArrayType>(AI->getAllocatedType());
//      assert(AT && "Expected array type");
//      ElementSize =
//          ConstantInt::get(Int32Ty,
//                           AT->getElementType()->getPrimitiveSizeInBits()/8);
//      Size = ConstantInt::get(Int64Ty, AT->getNumElements());
//    }
//
//    unsigned m = 0;
//    for (const User *U : V->users()) {
//      if (const Instruction *I = dyn_cast<Instruction>(U)) {
//        // TODO: Properly restrict this check to users within the cloned loop
//        // body.  Checking the dominator tree doesn't properly check
//        // exception-handling code, although it's not clear we should see such
//        // code in these loops.
//        if (!DT.dominates(T->getEntry(), I->getParent()))
//          continue;
//
//        if (isa<LoadInst>(U))
//          m |= 1;
//        else if (isa<StoreInst>(U))
//          m |= 2;
//      }
//    }
//    Value *Mode = ConstantInt::get(Int8Ty, m);
//    if (ElementSize && Size)
//      B.CreateCall(KitsuneGPUInitField, { KernelID, FieldName, VPtr,
//                                          ElementSize, Size, Mode });
//  }
//
//  Value *RunSize = B.CreateSub(TripCount, ConstantInt::get(TripCount->getType(),
//                                                           1));
//  B.CreateCall(KitsuneGPUSetRunSize, { KernelID, RunSize, RunStart, RunStart });
//
//  B.CreateCall(KitsuneGPURunKernel, { KernelID });
//
//  B.CreateCall(KitsuneGPUFinish, {});
//
//  ReplCall->eraseFromParent();
//}
