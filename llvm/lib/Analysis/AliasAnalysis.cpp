//==- AliasAnalysis.cpp - Generic Alias Analysis Interface Implementation --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the generic AliasAnalysis interface which is used as the
// common interface used by all clients and implementations of alias analysis.
//
// This file also implements the default version of the AliasAnalysis interface
// that is to be used when no other implementation is specified.  This does some
// simple tests that detect obvious cases: two different global pointers cannot
// alias, a global cannot alias a malloc, two different mallocs cannot alias,
// etc.
//
// This alias analysis implementation really isn't very good for anything, but
// it is very fast, and makes a nice clean default implementation.  Because it
// handles lots of little corner cases, other, more complex, alias analysis
// implementations may choose to rely on this pass to resolve these simple and
// easy cases.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CFLAndersAliasAnalysis.h"
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/DataRaceFreeAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/ObjCARCAliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>

using namespace llvm;

/// Allow disabling BasicAA from the AA results. This is particularly useful
/// when testing to isolate a single AA implementation.
static cl::opt<bool> DisableBasicAA("disable-basic-aa", cl::Hidden,
                                    cl::init(false));

AAResults::AAResults(AAResults &&Arg)
    : TLI(Arg.TLI), AAs(std::move(Arg.AAs)), AADeps(std::move(Arg.AADeps)) {
  for (auto &AA : AAs)
    AA->setAAResults(this);
}

AAResults::~AAResults() {
// FIXME; It would be nice to at least clear out the pointers back to this
// aggregation here, but we end up with non-nesting lifetimes in the legacy
// pass manager that prevent this from working. In the legacy pass manager
// we'll end up with dangling references here in some cases.
#if 0
  for (auto &AA : AAs)
    AA->setAAResults(nullptr);
#endif
}

bool AAResults::invalidate(Function &F, const PreservedAnalyses &PA,
                           FunctionAnalysisManager::Invalidator &Inv) {
  // AAResults preserves the AAManager by default, due to the stateless nature
  // of AliasAnalysis. There is no need to check whether it has been preserved
  // explicitly. Check if any module dependency was invalidated and caused the
  // AAManager to be invalidated. Invalidate ourselves in that case.
  auto PAC = PA.getChecker<AAManager>();
  if (!PAC.preservedWhenStateless())
    return true;

  // Check if any of the function dependencies were invalidated, and invalidate
  // ourselves in that case.
  for (AnalysisKey *ID : AADeps)
    if (Inv.invalidate(ID, F, PA))
      return true;

  // Everything we depend on is still fine, so are we. Nothing to invalidate.
  return false;
}

//===----------------------------------------------------------------------===//
// Default chaining methods
//===----------------------------------------------------------------------===//

AliasResult AAResults::alias(const MemoryLocation &LocA,
                             const MemoryLocation &LocB) {
  AAQueryInfo AAQIP;
  return alias(LocA, LocB, AAQIP);
}

AliasResult AAResults::alias(const MemoryLocation &LocA,
                             const MemoryLocation &LocB,
                             bool AssumeSameSpindle) {
  AAQueryInfo AAQIP;
  AAQIP.AssumeSameSpindle = AssumeSameSpindle;
  return alias(LocA, LocB, AAQIP);
}

AliasResult AAResults::alias(const MemoryLocation &LocA,
                             const MemoryLocation &LocB, AAQueryInfo &AAQI) {
  for (const auto &AA : AAs) {
    auto Result = AA->alias(LocA, LocB, AAQI);
    if (Result != MayAlias)
      return Result;
  }
  return MayAlias;
}

bool AAResults::pointsToConstantMemory(const MemoryLocation &Loc,
                                       bool OrLocal) {
  AAQueryInfo AAQIP;
  return pointsToConstantMemory(Loc, AAQIP, OrLocal);
}

bool AAResults::pointsToConstantMemory(const MemoryLocation &Loc,
                                       AAQueryInfo &AAQI, bool OrLocal) {
  for (const auto &AA : AAs)
    if (AA->pointsToConstantMemory(Loc, AAQI, OrLocal))
      return true;

  return false;
}

ModRefInfo AAResults::getArgModRefInfo(const CallBase *Call, unsigned ArgIdx) {
  ModRefInfo Result = ModRefInfo::ModRef;

  for (const auto &AA : AAs) {
    Result = intersectModRef(Result, AA->getArgModRefInfo(Call, ArgIdx));

    // Early-exit the moment we reach the bottom of the lattice.
    if (isNoModRef(Result))
      return ModRefInfo::NoModRef;
  }

  return Result;
}

ModRefInfo AAResults::getModRefInfo(Instruction *I, const CallBase *Call2,
                                    bool AssumeSameSpindle) {
  AAQueryInfo AAQIP;
  AAQIP.AssumeSameSpindle = AssumeSameSpindle;
  return getModRefInfo(I, Call2, AAQIP);
}

/// Returns true if the given instruction performs a detached rethrow, false
/// otherwise.
static bool isDetachedRethrow(const Instruction *I,
                              const Value *SyncRegion = nullptr) {
  if (const InvokeInst *II = dyn_cast<InvokeInst>(I))
    if (const Function *Called = II->getCalledFunction())
      if (Intrinsic::detached_rethrow == Called->getIntrinsicID())
        if (!SyncRegion || (SyncRegion == II->getArgOperand(0)))
          return true;
  return false;
}

static bool taskTerminator(const Instruction *T, const Value *SyncRegion) {
  if (const ReattachInst *RI = dyn_cast<ReattachInst>(T))
    if (SyncRegion == RI->getSyncRegion())
      return true;

  if (isDetachedRethrow(T, SyncRegion))
    return true;

  return false;
}

ModRefInfo AAResults::getModRefInfo(Instruction *I, const CallBase *Call2,
                                    AAQueryInfo &AAQI) {
  // We may have two calls.
  if (const auto *Call1 = dyn_cast<CallBase>(I)) {
    // Check if the two calls modify the same memory.
    return getModRefInfo(Call1, Call2, AAQI);
  } else if (I->isFenceLike()) {
    // If this is a fence, just return ModRef.
    return ModRefInfo::ModRef;
  } else if (auto D = dyn_cast<DetachInst>(I)) {
    ModRefInfo Result = ModRefInfo::NoModRef;
    SmallPtrSet<const BasicBlock *, 32> Visited;
    SmallVector<BasicBlock *, 32> WorkList;
    WorkList.push_back(D->getDetached());
    while (!WorkList.empty()) {
      BasicBlock *BB = WorkList.pop_back_val();
      if (!Visited.insert(BB).second)
        continue;

      for (Instruction &DI : BB->instructionsWithoutDebug()) {
        // Fail fast if we encounter an invalid CFG.
        assert(!(D == &DI) &&
               "Detached CFG reaches its own Detach instruction.");

        if (&DI == Call2)
          return ModRefInfo::NoModRef;

        // No need to recursively check nested syncs or detaches, as nested
        // tasks are wholly contained in the detached sub-CFG we're iterating
        // through.
        if (isa<SyncInst>(DI) || isa<DetachInst>(DI))
          continue;

        if (isa<VAArgInst>(DI) || isa<LoadInst>(DI) || isa<StoreInst>(DI) ||
            isa<AtomicCmpXchgInst>(DI) || isa<AtomicRMWInst>(DI) ||
            isa<CatchPadInst>(DI) || isa<CatchReturnInst>(DI) ||
            DI.isFenceLike() || isa<CallBase>(DI))
          Result = unionModRef(Result, getModRefInfo(&DI, Call2, AAQI));
      }

      // Add successors
      const Instruction *T = BB->getTerminator();
      if (taskTerminator(T, D->getSyncRegion()))
        continue;
      for (unsigned idx = 0, max = T->getNumSuccessors(); idx < max; ++idx)
        WorkList.push_back(T->getSuccessor(idx));
    }
    return Result;
  } else {
    // Otherwise, check if the call modifies or references the
    // location this memory access defines.  The best we can say
    // is that if the call references what this instruction
    // defines, it must be clobbered by this location.
    const MemoryLocation DefLoc = MemoryLocation::get(I);
    ModRefInfo MR = getModRefInfo(Call2, DefLoc, AAQI);
    if (isModOrRefSet(MR))
      return setModAndRef(MR);
  }
  return ModRefInfo::NoModRef;
}

ModRefInfo AAResults::getModRefInfo(const CallBase *Call,
                                    const MemoryLocation &Loc) {
  AAQueryInfo AAQIP;
  return getModRefInfo(Call, Loc, AAQIP);
}

ModRefInfo AAResults::getModRefInfo(const CallBase *Call,
                                    const MemoryLocation &Loc,
                                    bool SameSpindle) {
  AAQueryInfo AAQIP;
  AAQIP.AssumeSameSpindle = SameSpindle;
  return getModRefInfo(Call, Loc, AAQIP);
}

static bool effectivelyArgMemOnly(const CallBase *Call, AAQueryInfo &AAQI) {
  return Call->isStrandPure() && AAQI.AssumeSameSpindle;
}

ModRefInfo AAResults::getModRefInfo(const CallBase *Call,
                                    const MemoryLocation &Loc,
                                    AAQueryInfo &AAQI) {
  ModRefInfo Result = ModRefInfo::ModRef;

  for (const auto &AA : AAs) {
    Result = intersectModRef(Result, AA->getModRefInfo(Call, Loc, AAQI));

    // Early-exit the moment we reach the bottom of the lattice.
    if (isNoModRef(Result))
      return ModRefInfo::NoModRef;
  }

  // Try to refine the mod-ref info further using other API entry points to the
  // aggregate set of AA results.
  auto MRB = getModRefBehavior(Call);
  if (onlyAccessesInaccessibleMem(MRB))
    return ModRefInfo::NoModRef;

  if (onlyReadsMemory(MRB))
    Result = clearMod(Result);
  else if (doesNotReadMemory(MRB))
    Result = clearRef(Result);

  if (onlyAccessesArgPointees(MRB) || onlyAccessesInaccessibleOrArgMem(MRB) ||
      effectivelyArgMemOnly(Call, AAQI)) {
    bool IsMustAlias = true;
    ModRefInfo AllArgsMask = ModRefInfo::NoModRef;
    if (doesAccessArgPointees(MRB)) {
      for (auto AI = Call->arg_begin(), AE = Call->arg_end(); AI != AE; ++AI) {
        const Value *Arg = *AI;
        if (!Arg->getType()->isPointerTy())
          continue;
        unsigned ArgIdx = std::distance(Call->arg_begin(), AI);
        MemoryLocation ArgLoc =
            MemoryLocation::getForArgument(Call, ArgIdx, TLI);
        AliasResult ArgAlias = alias(ArgLoc, Loc, AAQI.AssumeSameSpindle);
        if (ArgAlias != NoAlias) {
          ModRefInfo ArgMask = getArgModRefInfo(Call, ArgIdx);
          AllArgsMask = unionModRef(AllArgsMask, ArgMask);
        }
        // Conservatively clear IsMustAlias unless only MustAlias is found.
        IsMustAlias &= (ArgAlias == MustAlias);
      }
    }
    // Return NoModRef if no alias found with any argument.
    if (isNoModRef(AllArgsMask))
      return ModRefInfo::NoModRef;
    // Logical & between other AA analyses and argument analysis.
    Result = intersectModRef(Result, AllArgsMask);
    // If only MustAlias found above, set Must bit.
    Result = IsMustAlias ? setMust(Result) : clearMust(Result);
  }

  // If Loc is a constant memory location, the call definitely could not
  // modify the memory location.
  if (isModSet(Result) && pointsToConstantMemory(Loc, /*OrLocal*/ false))
    Result = clearMod(Result);

  return Result;
}

ModRefInfo AAResults::getModRefInfo(const CallBase *Call1,
                                    const CallBase *Call2,
                                    bool AssumeSameSpindle) {
  AAQueryInfo AAQIP;
  AAQIP.AssumeSameSpindle = AssumeSameSpindle;
  return getModRefInfo(Call1, Call2, AAQIP);
}

ModRefInfo AAResults::getModRefInfo(const CallBase *Call1,
                                    const CallBase *Call2, AAQueryInfo &AAQI) {
  ModRefInfo Result = ModRefInfo::ModRef;

  for (const auto &AA : AAs) {
    Result = intersectModRef(Result, AA->getModRefInfo(Call1, Call2, AAQI));

    // Early-exit the moment we reach the bottom of the lattice.
    if (isNoModRef(Result))
      return ModRefInfo::NoModRef;
  }

  // Try to refine the mod-ref info further using other API entry points to the
  // aggregate set of AA results.

  // If Call1 or Call2 are readnone, they don't interact.
  auto Call1B = getModRefBehavior(Call1);
  if (Call1B == FMRB_DoesNotAccessMemory)
    return ModRefInfo::NoModRef;

  auto Call2B = getModRefBehavior(Call2);
  if (Call2B == FMRB_DoesNotAccessMemory)
    return ModRefInfo::NoModRef;

  // If they both only read from memory, there is no dependence.
  if (onlyReadsMemory(Call1B) && onlyReadsMemory(Call2B))
    return ModRefInfo::NoModRef;

  // If Call1 only reads memory, the only dependence on Call2 can be
  // from Call1 reading memory written by Call2.
  if (onlyReadsMemory(Call1B))
    Result = clearMod(Result);
  else if (doesNotReadMemory(Call1B))
    Result = clearRef(Result);

  // If Call2 only access memory through arguments, accumulate the mod/ref
  // information from Call1's references to the memory referenced by
  // Call2's arguments.
  if (onlyAccessesArgPointees(Call2B) || effectivelyArgMemOnly(Call2, AAQI)) {
    if (!doesAccessArgPointees(Call2B))
      return ModRefInfo::NoModRef;
    ModRefInfo R = ModRefInfo::NoModRef;
    bool IsMustAlias = true;
    for (auto I = Call2->arg_begin(), E = Call2->arg_end(); I != E; ++I) {
      const Value *Arg = *I;
      if (!Arg->getType()->isPointerTy())
        continue;
      unsigned Call2ArgIdx = std::distance(Call2->arg_begin(), I);
      auto Call2ArgLoc =
          MemoryLocation::getForArgument(Call2, Call2ArgIdx, TLI);

      // ArgModRefC2 indicates what Call2 might do to Call2ArgLoc, and the
      // dependence of Call1 on that location is the inverse:
      // - If Call2 modifies location, dependence exists if Call1 reads or
      //   writes.
      // - If Call2 only reads location, dependence exists if Call1 writes.
      ModRefInfo ArgModRefC2 = getArgModRefInfo(Call2, Call2ArgIdx);
      ModRefInfo ArgMask = ModRefInfo::NoModRef;
      if (isModSet(ArgModRefC2))
        ArgMask = ModRefInfo::ModRef;
      else if (isRefSet(ArgModRefC2))
        ArgMask = ModRefInfo::Mod;

      // ModRefC1 indicates what Call1 might do to Call2ArgLoc, and we use
      // above ArgMask to update dependence info.
      ModRefInfo ModRefC1 = getModRefInfo(Call1, Call2ArgLoc);
      ArgMask = intersectModRef(ArgMask, ModRefC1);

      // Conservatively clear IsMustAlias unless only MustAlias is found.
      IsMustAlias &= isMustSet(ModRefC1);

      R = intersectModRef(unionModRef(R, ArgMask), Result);
      if (R == Result) {
        // On early exit, not all args were checked, cannot set Must.
        if (I + 1 != E)
          IsMustAlias = false;
        break;
      }
    }

    if (isNoModRef(R))
      return ModRefInfo::NoModRef;

    // If MustAlias found above, set Must bit.
    return IsMustAlias ? setMust(R) : clearMust(R);
  }

  // If Call1 only accesses memory through arguments, check if Call2 references
  // any of the memory referenced by Call1's arguments. If not, return NoModRef.
  if (onlyAccessesArgPointees(Call1B) || effectivelyArgMemOnly(Call1, AAQI)) {
    if (!doesAccessArgPointees(Call1B))
      return ModRefInfo::NoModRef;
    ModRefInfo R = ModRefInfo::NoModRef;
    bool IsMustAlias = true;
    for (auto I = Call1->arg_begin(), E = Call1->arg_end(); I != E; ++I) {
      const Value *Arg = *I;
      if (!Arg->getType()->isPointerTy())
        continue;
      unsigned Call1ArgIdx = std::distance(Call1->arg_begin(), I);
      auto Call1ArgLoc =
          MemoryLocation::getForArgument(Call1, Call1ArgIdx, TLI);

      // ArgModRefC1 indicates what Call1 might do to Call1ArgLoc; if Call1
      // might Mod Call1ArgLoc, then we care about either a Mod or a Ref by
      // Call2. If Call1 might Ref, then we care only about a Mod by Call2.
      ModRefInfo ArgModRefC1 = getArgModRefInfo(Call1, Call1ArgIdx);
      ModRefInfo ModRefC2 = getModRefInfo(Call2, Call1ArgLoc);
      if ((isModSet(ArgModRefC1) && isModOrRefSet(ModRefC2)) ||
          (isRefSet(ArgModRefC1) && isModSet(ModRefC2)))
        R = intersectModRef(unionModRef(R, ArgModRefC1), Result);

      // Conservatively clear IsMustAlias unless only MustAlias is found.
      IsMustAlias &= isMustSet(ModRefC2);

      if (R == Result) {
        // On early exit, not all args were checked, cannot set Must.
        if (I + 1 != E)
          IsMustAlias = false;
        break;
      }
    }

    if (isNoModRef(R))
      return ModRefInfo::NoModRef;

    // If MustAlias found above, set Must bit.
    return IsMustAlias ? setMust(R) : clearMust(R);
  }

  return Result;
}

FunctionModRefBehavior AAResults::getModRefBehavior(const CallBase *Call) {
  FunctionModRefBehavior Result = FMRB_UnknownModRefBehavior;

  for (const auto &AA : AAs) {
    Result = FunctionModRefBehavior(Result & AA->getModRefBehavior(Call));

    // Early-exit the moment we reach the bottom of the lattice.
    if (Result == FMRB_DoesNotAccessMemory)
      return Result;
  }

  return Result;
}

FunctionModRefBehavior AAResults::getModRefBehavior(const Function *F) {
  FunctionModRefBehavior Result = FMRB_UnknownModRefBehavior;

  for (const auto &AA : AAs) {
    Result = FunctionModRefBehavior(Result & AA->getModRefBehavior(F));

    // Early-exit the moment we reach the bottom of the lattice.
    if (Result == FMRB_DoesNotAccessMemory)
      return Result;
  }

  return Result;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, AliasResult AR) {
  switch (AR) {
  case NoAlias:
    OS << "NoAlias";
    break;
  case MustAlias:
    OS << "MustAlias";
    break;
  case MayAlias:
    OS << "MayAlias";
    break;
  case PartialAlias:
    OS << "PartialAlias";
    break;
  }
  return OS;
}

FunctionModRefBehavior AAResults::getModRefBehavior(const DetachInst *D) {
  FunctionModRefBehavior Result = FMRB_DoesNotAccessMemory;
  SmallPtrSet<const BasicBlock *, 32> Visited;
  SmallVector<const BasicBlock *, 32> WorkList;
  WorkList.push_back(D->getDetached());
  while (!WorkList.empty()) {
    const BasicBlock *BB = WorkList.pop_back_val();
    if (!Visited.insert(BB).second)
      continue;

    for (const Instruction &I : *BB) {
      // Fail fast if we encounter an invalid CFG.
      assert(!(D == &I) &&
             "Invalid CFG found: Detached CFG reaches its own Detach.");

      if (const auto *CS = dyn_cast<CallBase>(&I))
        Result = FunctionModRefBehavior(Result | getModRefBehavior(CS));

      // Early-exit the moment we reach the top of the lattice.
      if (Result == FMRB_UnknownModRefBehavior)
        return Result;
    }

    // Add successors
    const Instruction *T = BB->getTerminator();
    if (taskTerminator(T, D->getSyncRegion()))
      continue;
    for (unsigned idx = 0, max = T->getNumSuccessors(); idx < max; ++idx)
      WorkList.push_back(T->getSuccessor(idx));
  }

  return Result;
}

FunctionModRefBehavior AAResults::getModRefBehavior(const SyncInst *S) {
  FunctionModRefBehavior Result = FMRB_DoesNotAccessMemory;
  SmallPtrSet<const BasicBlock *, 32> Visited;
  SmallVector<const BasicBlock *, 32> WorkList;
  WorkList.push_back(S->getParent());
  while (!WorkList.empty()) {
    const BasicBlock *BB = WorkList.pop_back_val();
    if (!Visited.insert(BB).second)
      continue;

    if (const DetachInst *D = dyn_cast<DetachInst>(BB->getTerminator()))
      if (D->getSyncRegion() == S->getSyncRegion())
        Result = FunctionModRefBehavior(Result | getModRefBehavior(D));

    // Early-exit the moment we reach the top of the lattice.
    if (Result == FMRB_UnknownModRefBehavior)
      return Result;

    // Add predecessors
    for (const BasicBlock *Pred : predecessors(BB)) {
      const Instruction *PT = Pred->getTerminator();
      // Ignore reattached predecessors and predecessors that end in syncs,
      // because this sync does not wait on those predecessors.
      if (isa<ReattachInst>(PT) || isa<SyncInst>(PT) || isDetachedRethrow(PT))
        continue;

      // If this block is detached, ignore the predecessor that detaches it.
      if (const DetachInst *Det = dyn_cast<DetachInst>(PT))
        if (Det->getDetached() == BB)
          continue;

      WorkList.push_back(Pred);
    }
  }

  return Result;
}

//===----------------------------------------------------------------------===//
// Helper method implementation
//===----------------------------------------------------------------------===//

ModRefInfo AAResults::getModRefInfo(const LoadInst *L,
                                    const MemoryLocation &Loc) {
  AAQueryInfo AAQIP;
  return getModRefInfo(L, Loc, AAQIP);
}
ModRefInfo AAResults::getModRefInfo(const LoadInst *L,
                                    const MemoryLocation &Loc,
                                    AAQueryInfo &AAQI) {
  // Be conservative in the face of atomic.
  if (isStrongerThan(L->getOrdering(), AtomicOrdering::Unordered))
    return ModRefInfo::ModRef;

  // If the load address doesn't alias the given address, it doesn't read
  // or write the specified memory.
  if (Loc.Ptr) {
    AliasResult AR = alias(MemoryLocation::get(L), Loc, AAQI);
    if (AR == NoAlias)
      return ModRefInfo::NoModRef;
    if (AR == MustAlias)
      return ModRefInfo::MustRef;
  }
  // Otherwise, a load just reads.
  return ModRefInfo::Ref;
}

ModRefInfo AAResults::getModRefInfo(const StoreInst *S,
                                    const MemoryLocation &Loc) {
  AAQueryInfo AAQIP;
  return getModRefInfo(S, Loc, AAQIP);
}
ModRefInfo AAResults::getModRefInfo(const StoreInst *S,
                                    const MemoryLocation &Loc,
                                    AAQueryInfo &AAQI) {
  // Be conservative in the face of atomic.
  if (isStrongerThan(S->getOrdering(), AtomicOrdering::Unordered))
    return ModRefInfo::ModRef;

  if (Loc.Ptr) {
    AliasResult AR = alias(MemoryLocation::get(S), Loc, AAQI);
    // If the store address cannot alias the pointer in question, then the
    // specified memory cannot be modified by the store.
    if (AR == NoAlias)
      return ModRefInfo::NoModRef;

    // If the pointer is a pointer to constant memory, then it could not have
    // been modified by this store.
    if (pointsToConstantMemory(Loc, AAQI))
      return ModRefInfo::NoModRef;

    // If the store address aliases the pointer as must alias, set Must.
    if (AR == MustAlias)
      return ModRefInfo::MustMod;
  }

  // Otherwise, a store just writes.
  return ModRefInfo::Mod;
}

ModRefInfo AAResults::getModRefInfo(const FenceInst *S, const MemoryLocation &Loc) {
  AAQueryInfo AAQIP;
  return getModRefInfo(S, Loc, AAQIP);
}

ModRefInfo AAResults::getModRefInfo(const FenceInst *S,
                                    const MemoryLocation &Loc,
                                    AAQueryInfo &AAQI) {
  // If we know that the location is a constant memory location, the fence
  // cannot modify this location.
  if (Loc.Ptr && pointsToConstantMemory(Loc, AAQI))
    return ModRefInfo::Ref;
  return ModRefInfo::ModRef;
}

ModRefInfo AAResults::getModRefInfo(const VAArgInst *V,
                                    const MemoryLocation &Loc) {
  AAQueryInfo AAQIP;
  return getModRefInfo(V, Loc, AAQIP);
}

ModRefInfo AAResults::getModRefInfo(const VAArgInst *V,
                                    const MemoryLocation &Loc,
                                    AAQueryInfo &AAQI) {
  if (Loc.Ptr) {
    AliasResult AR = alias(MemoryLocation::get(V), Loc, AAQI);
    // If the va_arg address cannot alias the pointer in question, then the
    // specified memory cannot be accessed by the va_arg.
    if (AR == NoAlias)
      return ModRefInfo::NoModRef;

    // If the pointer is a pointer to constant memory, then it could not have
    // been modified by this va_arg.
    if (pointsToConstantMemory(Loc, AAQI))
      return ModRefInfo::NoModRef;

    // If the va_arg aliases the pointer as must alias, set Must.
    if (AR == MustAlias)
      return ModRefInfo::MustModRef;
  }

  // Otherwise, a va_arg reads and writes.
  return ModRefInfo::ModRef;
}

ModRefInfo AAResults::getModRefInfo(const CatchPadInst *CatchPad,
                                    const MemoryLocation &Loc) {
  AAQueryInfo AAQIP;
  return getModRefInfo(CatchPad, Loc, AAQIP);
}

ModRefInfo AAResults::getModRefInfo(const CatchPadInst *CatchPad,
                                    const MemoryLocation &Loc,
                                    AAQueryInfo &AAQI) {
  if (Loc.Ptr) {
    // If the pointer is a pointer to constant memory,
    // then it could not have been modified by this catchpad.
    if (pointsToConstantMemory(Loc, AAQI))
      return ModRefInfo::NoModRef;
  }

  // Otherwise, a catchpad reads and writes.
  return ModRefInfo::ModRef;
}

ModRefInfo AAResults::getModRefInfo(const CatchReturnInst *CatchRet,
                                    const MemoryLocation &Loc) {
  AAQueryInfo AAQIP;
  return getModRefInfo(CatchRet, Loc, AAQIP);
}

ModRefInfo AAResults::getModRefInfo(const CatchReturnInst *CatchRet,
                                    const MemoryLocation &Loc,
                                    AAQueryInfo &AAQI) {
  if (Loc.Ptr) {
    // If the pointer is a pointer to constant memory,
    // then it could not have been modified by this catchpad.
    if (pointsToConstantMemory(Loc, AAQI))
      return ModRefInfo::NoModRef;
  }

  // Otherwise, a catchret reads and writes.
  return ModRefInfo::ModRef;
}

ModRefInfo AAResults::getModRefInfo(const AtomicCmpXchgInst *CX,
                                    const MemoryLocation &Loc) {
  AAQueryInfo AAQIP;
  return getModRefInfo(CX, Loc, AAQIP);
}

ModRefInfo AAResults::getModRefInfo(const AtomicCmpXchgInst *CX,
                                    const MemoryLocation &Loc,
                                    AAQueryInfo &AAQI) {
  // Acquire/Release cmpxchg has properties that matter for arbitrary addresses.
  if (isStrongerThanMonotonic(CX->getSuccessOrdering()))
    return ModRefInfo::ModRef;

  if (Loc.Ptr) {
    AliasResult AR = alias(MemoryLocation::get(CX), Loc, AAQI);
    // If the cmpxchg address does not alias the location, it does not access
    // it.
    if (AR == NoAlias)
      return ModRefInfo::NoModRef;

    // If the cmpxchg address aliases the pointer as must alias, set Must.
    if (AR == MustAlias)
      return ModRefInfo::MustModRef;
  }

  return ModRefInfo::ModRef;
}

ModRefInfo AAResults::getModRefInfo(const AtomicRMWInst *RMW,
                                    const MemoryLocation &Loc) {
  AAQueryInfo AAQIP;
  return getModRefInfo(RMW, Loc, AAQIP);
}

ModRefInfo AAResults::getModRefInfo(const AtomicRMWInst *RMW,
                                    const MemoryLocation &Loc,
                                    AAQueryInfo &AAQI) {
  // Acquire/Release atomicrmw has properties that matter for arbitrary addresses.
  if (isStrongerThanMonotonic(RMW->getOrdering()))
    return ModRefInfo::ModRef;

  if (Loc.Ptr) {
    AliasResult AR = alias(MemoryLocation::get(RMW), Loc, AAQI);
    // If the atomicrmw address does not alias the location, it does not access
    // it.
    if (AR == NoAlias)
      return ModRefInfo::NoModRef;

    // If the atomicrmw address aliases the pointer as must alias, set Must.
    if (AR == MustAlias)
      return ModRefInfo::MustModRef;
  }

  return ModRefInfo::ModRef;
}

ModRefInfo AAResults::getModRefInfo(const DetachInst *D,
                                    const MemoryLocation &Loc) {
  AAQueryInfo AAQIP;
  return getModRefInfo(D, Loc, AAQIP);
}

ModRefInfo AAResults::getModRefInfo(const DetachInst *D,
                                    const MemoryLocation &Loc,
                                    AAQueryInfo &AAQI) {
  ModRefInfo Result = ModRefInfo::NoModRef;
  SmallPtrSet<const BasicBlock *, 32> Visited;
  SmallVector<const BasicBlock *, 32> WorkList;
  WorkList.push_back(D->getDetached());
  while (!WorkList.empty()) {
    const BasicBlock *BB = WorkList.pop_back_val();
    if (!Visited.insert(BB).second)
      continue;

    for (const Instruction &I : BB->instructionsWithoutDebug()) {
      // Fail fast if we encounter an invalid CFG.
      assert(!(D == &I) &&
             "Invalid CFG found: Detached CFG reaches its own Detach.");

      // No need to recursively check nested syncs or detaches, as nested tasks
      // are wholly contained in the detached sub-CFG we're iterating through.
      if (isa<SyncInst>(I) || isa<DetachInst>(I))
        continue;

      Result = unionModRef(Result, getModRefInfo(&I, Loc, AAQI));

      // Early-exit the moment we reach the top of the lattice.
      if (isModAndRefSet(Result))
	return Result;
    }

    // Add successors
    const Instruction *T = BB->getTerminator();
    if (taskTerminator(T, D->getSyncRegion()))
      continue;
    for (const BasicBlock *Successor : successors(BB))
      WorkList.push_back(Successor);
  }

  return Result;
}

ModRefInfo AAResults::getModRefInfo(const SyncInst *S,
                                    const MemoryLocation &Loc) {
  AAQueryInfo AAQIP;
  return getModRefInfo(S, Loc, AAQIP);
}

ModRefInfo AAResults::getModRefInfo(const SyncInst *S,
                                    const MemoryLocation &Loc,
                                    AAQueryInfo &AAQI) {
  // If no memory location pointer is given, treat the sync like a fence.
  if (!Loc.Ptr)
    return ModRefInfo::ModRef;

  ModRefInfo Result = ModRefInfo::NoModRef;
  SmallPtrSet<const BasicBlock *, 32> Visited;
  SmallVector<const BasicBlock *, 32> WorkList;
  WorkList.push_back(S->getParent());
  while(!WorkList.empty()) {
    const BasicBlock *BB = WorkList.pop_back_val();
    if (!Visited.insert(BB).second)
      continue;

    if (const DetachInst *D = dyn_cast<DetachInst>(BB->getTerminator())) {
      Result = unionModRef(Result, getModRefInfo(D, Loc, AAQI));

      // Early-exit the moment we reach the top of the lattice.
      if (isModAndRefSet(Result))
	return Result;
    }

    // Add predecessors
    for (const BasicBlock *Pred : predecessors(BB)) {
      const Instruction *PT = Pred->getTerminator();
      // Ignore reattached predecessors and predecessors that end in syncs,
      // because this sync does not wait on those predecessors.
      if (isa<ReattachInst>(PT) || isa<SyncInst>(PT) || isDetachedRethrow(PT))
	continue;
      // If this block is detached, ignore the predecessor that detaches it.
      if (const DetachInst *Det = dyn_cast<DetachInst>(PT))
        if (Det->getDetached() == BB)
          continue;

      WorkList.push_back(Pred);
    }
  }

  return Result;
}

/// Return information about whether a particular call site modifies
/// or reads the specified memory location \p MemLoc before instruction \p I
/// in a BasicBlock.
/// FIXME: this is really just shoring-up a deficiency in alias analysis.
/// BasicAA isn't willing to spend linear time determining whether an alloca
/// was captured before or after this particular call, while we are. However,
/// with a smarter AA in place, this test is just wasting compile time.
ModRefInfo AAResults::callCapturesBefore(const Instruction *I,
                                         const MemoryLocation &MemLoc,
                                         DominatorTree *DT) {
  if (!DT)
    return ModRefInfo::ModRef;

  const Value *Object =
      GetUnderlyingObject(MemLoc.Ptr, I->getModule()->getDataLayout());
  if (!isIdentifiedObject(Object) || isa<GlobalValue>(Object) ||
      isa<Constant>(Object))
    return ModRefInfo::ModRef;

  const auto *Call = dyn_cast<CallBase>(I);
  if (!Call || Call == Object)
    return ModRefInfo::ModRef;

  if (PointerMayBeCapturedBefore(Object, /* ReturnCaptures */ true,
                                 /* StoreCaptures */ true, I, DT,
                                 /* include Object */ true))
    return ModRefInfo::ModRef;

  unsigned ArgNo = 0;
  ModRefInfo R = ModRefInfo::NoModRef;
  bool IsMustAlias = true;
  // Set flag only if no May found and all operands processed.
  for (auto CI = Call->data_operands_begin(), CE = Call->data_operands_end();
       CI != CE; ++CI, ++ArgNo) {
    // Only look at the no-capture or byval pointer arguments.  If this
    // pointer were passed to arguments that were neither of these, then it
    // couldn't be no-capture.
    if (!(*CI)->getType()->isPointerTy() ||
        (!Call->doesNotCapture(ArgNo) && ArgNo < Call->getNumArgOperands() &&
         !Call->isByValArgument(ArgNo)))
      continue;

    AliasResult AR = alias(MemoryLocation(*CI), MemoryLocation(Object));
    // If this is a no-capture pointer argument, see if we can tell that it
    // is impossible to alias the pointer we're checking.  If not, we have to
    // assume that the call could touch the pointer, even though it doesn't
    // escape.
    if (AR != MustAlias)
      IsMustAlias = false;
    if (AR == NoAlias)
      continue;
    if (Call->doesNotAccessMemory(ArgNo))
      continue;
    if (Call->onlyReadsMemory(ArgNo)) {
      R = ModRefInfo::Ref;
      continue;
    }
    // Not returning MustModRef since we have not seen all the arguments.
    return ModRefInfo::ModRef;
  }
  return IsMustAlias ? setMust(R) : clearMust(R);
}

/// canBasicBlockModify - Return true if it is possible for execution of the
/// specified basic block to modify the location Loc.
///
bool AAResults::canBasicBlockModify(const BasicBlock &BB,
                                    const MemoryLocation &Loc) {
  return canInstructionRangeModRef(BB.front(), BB.back(), Loc, ModRefInfo::Mod);
}

/// canInstructionRangeModRef - Return true if it is possible for the
/// execution of the specified instructions to mod\ref (according to the
/// mode) the location Loc. The instructions to consider are all
/// of the instructions in the range of [I1,I2] INCLUSIVE.
/// I1 and I2 must be in the same basic block.
bool AAResults::canInstructionRangeModRef(const Instruction &I1,
                                          const Instruction &I2,
                                          const MemoryLocation &Loc,
                                          const ModRefInfo Mode) {
  assert(I1.getParent() == I2.getParent() &&
         "Instructions not in same basic block!");
  BasicBlock::const_iterator I = I1.getIterator();
  BasicBlock::const_iterator E = I2.getIterator();
  ++E;  // Convert from inclusive to exclusive range.

  for (; I != E; ++I) // Check every instruction in range
    if (isModOrRefSet(intersectModRef(getModRefInfo(&*I, Loc), Mode)))
      return true;
  return false;
}

// Provide a definition for the root virtual destructor.
AAResults::Concept::~Concept() = default;

// Provide a definition for the static object used to identify passes.
AnalysisKey AAManager::Key;

namespace {


} // end anonymous namespace

ExternalAAWrapperPass::ExternalAAWrapperPass() : ImmutablePass(ID) {
  initializeExternalAAWrapperPassPass(*PassRegistry::getPassRegistry());
}

ExternalAAWrapperPass::ExternalAAWrapperPass(CallbackT CB)
    : ImmutablePass(ID), CB(std::move(CB)) {
  initializeExternalAAWrapperPassPass(*PassRegistry::getPassRegistry());
}

char ExternalAAWrapperPass::ID = 0;

INITIALIZE_PASS(ExternalAAWrapperPass, "external-aa", "External Alias Analysis",
                false, true)

ImmutablePass *
llvm::createExternalAAWrapperPass(ExternalAAWrapperPass::CallbackT Callback) {
  return new ExternalAAWrapperPass(std::move(Callback));
}

AAResultsWrapperPass::AAResultsWrapperPass() : FunctionPass(ID) {
  initializeAAResultsWrapperPassPass(*PassRegistry::getPassRegistry());
}

char AAResultsWrapperPass::ID = 0;

INITIALIZE_PASS_BEGIN(AAResultsWrapperPass, "aa",
                      "Function Alias Analysis Results", false, true)
INITIALIZE_PASS_DEPENDENCY(BasicAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(CFLAndersAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(CFLSteensAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DRFAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ExternalAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ObjCARCAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SCEVAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScopedNoAliasAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TypeBasedAAWrapperPass)
INITIALIZE_PASS_END(AAResultsWrapperPass, "aa",
                    "Function Alias Analysis Results", false, true)

FunctionPass *llvm::createAAResultsWrapperPass() {
  return new AAResultsWrapperPass();
}

/// Run the wrapper pass to rebuild an aggregation over known AA passes.
///
/// This is the legacy pass manager's interface to the new-style AA results
/// aggregation object. Because this is somewhat shoe-horned into the legacy
/// pass manager, we hard code all the specific alias analyses available into
/// it. While the particular set enabled is configured via commandline flags,
/// adding a new alias analysis to LLVM will require adding support for it to
/// this list.
bool AAResultsWrapperPass::runOnFunction(Function &F) {
  // NB! This *must* be reset before adding new AA results to the new
  // AAResults object because in the legacy pass manager, each instance
  // of these will refer to the *same* immutable analyses, registering and
  // unregistering themselves with them. We need to carefully tear down the
  // previous object first, in this case replacing it with an empty one, before
  // registering new results.
  AAR.reset(
      new AAResults(getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F)));

  // BasicAA is always available for function analyses. Also, we add it first
  // so that it can trump TBAA results when it proves MustAlias.
  // FIXME: TBAA should have an explicit mode to support this and then we
  // should reconsider the ordering here.
  if (!DisableBasicAA)
    AAR->addAAResult(getAnalysis<BasicAAWrapperPass>().getResult());

  // Populate the results with the currently available AAs.
  if (auto *WrapperPass = getAnalysisIfAvailable<ScopedNoAliasAAWrapperPass>())
    AAR->addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = getAnalysisIfAvailable<TypeBasedAAWrapperPass>())
    AAR->addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass =
          getAnalysisIfAvailable<objcarc::ObjCARCAAWrapperPass>())
    AAR->addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = getAnalysisIfAvailable<GlobalsAAWrapperPass>())
    AAR->addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = getAnalysisIfAvailable<SCEVAAWrapperPass>())
    AAR->addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = getAnalysisIfAvailable<CFLAndersAAWrapperPass>())
    AAR->addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = getAnalysisIfAvailable<CFLSteensAAWrapperPass>())
    AAR->addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = getAnalysisIfAvailable<DRFAAWrapperPass>())
    AAR->addAAResult(WrapperPass->getResult());

  // If available, run an external AA providing callback over the results as
  // well.
  if (auto *WrapperPass = getAnalysisIfAvailable<ExternalAAWrapperPass>())
    if (WrapperPass->CB)
      WrapperPass->CB(*this, F, *AAR);

  // Analyses don't mutate the IR, so return false.
  return false;
}

void AAResultsWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<BasicAAWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();

  // We also need to mark all the alias analysis passes we will potentially
  // probe in runOnFunction as used here to ensure the legacy pass manager
  // preserves them. This hard coding of lists of alias analyses is specific to
  // the legacy pass manager.
  AU.addUsedIfAvailable<ScopedNoAliasAAWrapperPass>();
  AU.addUsedIfAvailable<TypeBasedAAWrapperPass>();
  AU.addUsedIfAvailable<objcarc::ObjCARCAAWrapperPass>();
  AU.addUsedIfAvailable<GlobalsAAWrapperPass>();
  AU.addUsedIfAvailable<SCEVAAWrapperPass>();
  AU.addUsedIfAvailable<CFLAndersAAWrapperPass>();
  AU.addUsedIfAvailable<CFLSteensAAWrapperPass>();
  AU.addUsedIfAvailable<DRFAAWrapperPass>();
  AU.addUsedIfAvailable<ExternalAAWrapperPass>();
}

AAResults llvm::createLegacyPMAAResults(Pass &P, Function &F,
                                        BasicAAResult &BAR) {
  AAResults AAR(P.getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F));

  // Add in our explicitly constructed BasicAA results.
  if (!DisableBasicAA)
    AAR.addAAResult(BAR);

  // Populate the results with the other currently available AAs.
  if (auto *WrapperPass =
          P.getAnalysisIfAvailable<ScopedNoAliasAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = P.getAnalysisIfAvailable<TypeBasedAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass =
          P.getAnalysisIfAvailable<objcarc::ObjCARCAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = P.getAnalysisIfAvailable<GlobalsAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = P.getAnalysisIfAvailable<CFLAndersAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = P.getAnalysisIfAvailable<CFLSteensAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = P.getAnalysisIfAvailable<DRFAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = P.getAnalysisIfAvailable<ExternalAAWrapperPass>())
    if (WrapperPass->CB)
      WrapperPass->CB(P, F, AAR);

  return AAR;
}

bool llvm::isNoAliasCall(const Value *V) {
  if (const auto *Call = dyn_cast<CallBase>(V))
    return Call->hasRetAttr(Attribute::NoAlias);
  return false;
}

bool llvm::isNoAliasCallIfInSameSpindle(const Value *V) {
  if (const auto *Call = dyn_cast<CallBase>(V))
    return Call->hasRetAttr(Attribute::StrandNoAlias);
  return isNoAliasCall(V);
}

bool llvm::isNoAliasArgument(const Value *V) {
  if (const Argument *A = dyn_cast<Argument>(V))
    return A->hasNoAliasAttr();
  return false;
}

bool llvm::isIdentifiedObject(const Value *V) {
  if (isa<AllocaInst>(V))
    return true;
  if (isa<GlobalValue>(V) && !isa<GlobalAlias>(V))
    return true;
  if (isNoAliasCall(V))
    return true;
  if (const Argument *A = dyn_cast<Argument>(V))
    return A->hasNoAliasAttr() || A->hasByValAttr();
  return false;
}

bool llvm::isIdentifiedObjectIfInSameSpindle(const Value *V) {
  if (isIdentifiedObject(V))
    return true;
  if (isNoAliasCallIfInSameSpindle(V))
    return true;
  return false;
}

bool llvm::isIdentifiedFunctionLocal(const Value *V) {
  return isa<AllocaInst>(V) || isNoAliasCall(V) || isNoAliasArgument(V);
}

void llvm::getAAResultsAnalysisUsage(AnalysisUsage &AU) {
  // This function needs to be in sync with llvm::createLegacyPMAAResults -- if
  // more alias analyses are added to llvm::createLegacyPMAAResults, they need
  // to be added here also.
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addUsedIfAvailable<ScopedNoAliasAAWrapperPass>();
  AU.addUsedIfAvailable<TypeBasedAAWrapperPass>();
  AU.addUsedIfAvailable<objcarc::ObjCARCAAWrapperPass>();
  AU.addUsedIfAvailable<GlobalsAAWrapperPass>();
  AU.addUsedIfAvailable<CFLAndersAAWrapperPass>();
  AU.addUsedIfAvailable<CFLSteensAAWrapperPass>();
  AU.addUsedIfAvailable<DRFAAWrapperPass>();
  AU.addUsedIfAvailable<ExternalAAWrapperPass>();
}
