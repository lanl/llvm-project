/**
 ***************************************************************************
 * TODO: Need to update LANL/Triad Copyright notice... 
 * 
 * Copyright (c) 2017, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 *
 ***************************************************************************/
#include "clang/AST/Attr.h"
#include "clang/Basic/Attributes.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "CodeGenFunction.h"
#include "CGCleanup.h"
#include "clang/AST/StmtKitsune.h"
#include "llvm/IR/ValueMap.h"


using namespace clang;
using namespace CodeGen;


LoopAttributes::LSStrategy CodeGenFunction::GetTapirStrategyAttr(
       ArrayRef<const Attr *> Attrs) {


//    LoopAttributes::LSStrategy Strategy = LoopAttributes::SEQ;
  LoopAttributes::LSStrategy Strategy = LoopAttributes::DAC;
  
  auto curAttr = Attrs.begin();

  while(curAttr != Attrs.end()) {
    
    const attr::Kind AttrKind = (*curAttr)->getKind();

    if (AttrKind == attr::TapirStrategy) {
      const auto *SAttr = cast<const TapirStrategyAttr>(*curAttr);
      
      switch(SAttr->getTapirStrategyType()) {

      case TapirStrategyAttr::SEQ:
        Strategy = LoopAttributes::SEQ;
        break;

      case TapirStrategyAttr::DAC:
        Strategy = LoopAttributes::DAC;
        break;

      case TapirStrategyAttr::GPU:
        Strategy = LoopAttributes::GPU;
        break;

      default:
        llvm_unreachable("all strategies should be handled before this!");
        break;
      }
    }
  }
  return Strategy;
}

LoopAttributes::LTarget CodeGenFunction::GetTapirRTTargetAttr(
      ArrayRef<const Attr *> Attrs) {

  auto curAttr = Attrs.begin();

  // TODO: Need to make sure the default matches build parameters! 
  LoopAttributes::LTarget Target = LoopAttributes::CilkRT; 

  while(curAttr != Attrs.end()) {
    
    const attr::Kind AttrKind = (*curAttr)->getKind();

    if (AttrKind == attr::TapirRTTarget) {
      const auto *TAttr = cast<const TapirRTTargetAttr>(*curAttr);
      
      switch(TAttr->getTapirRTTargetType()) {

      case TapirRTTargetAttr::CheetahRT:
        Target = LoopAttributes::CheetahRT;
        break;
      case TapirRTTargetAttr::CilkRT: 
        Target = LoopAttributes::CilkRT;
        break;        
      case TapirRTTargetAttr::CudaRT:
        Target = LoopAttributes::CudaRT;
        break;
      case TapirRTTargetAttr::KitCudaRT:
        Target = LoopAttributes::KitCudaRT;
        break;
      case TapirRTTargetAttr::HipRT:
        Target = LoopAttributes::HipRT;
        break;
      case TapirRTTargetAttr::OmpRT:
        Target = LoopAttributes::OmpRT;
        break;
      case TapirRTTargetAttr::QthreadsRT:
        Target = LoopAttributes::QthreadsRT;
        break;
      case TapirRTTargetAttr::RealmRT:
        Target = LoopAttributes::RealmRT;
        break;
      case TapirRTTargetAttr::RocmRT:
        Target = LoopAttributes::RocmRT;
        break;
      case TapirRTTargetAttr::SequentialRT:
        Target = LoopAttributes::SequentialRT;
        break;
      case TapirRTTargetAttr::ZeroRT:
        Target = LoopAttributes::ZeroRT;
        break;
      default:
        llvm_unreachable("All target attributes should be handled here!");
        break;
      }
    }
    curAttr++;
  }
  return Target;
}


// Stolen from CodeGenFunction.cpp
static void EmitIfUsed(CodeGenFunction &CGF, llvm::BasicBlock *BB) {
  if (!BB) return;
  if (!BB->use_empty())
    return CGF.CurFn->getBasicBlockList().push_back(BB);
  delete BB;
}

llvm::Instruction *CodeGenFunction::EmitLabeledSyncRegionStart(StringRef SV) {
  // Start the sync region.  To ensure the syncregion.start call dominates all
  // uses of the generated token, we insert this call at the alloca insertion
  // point.
  llvm::Instruction *SRStart = llvm::CallInst::Create(
      CGM.getIntrinsic(llvm::Intrinsic::syncregion_start), SV, AllocaInsertPt);
  return SRStart;
}

/// EmitSyncStmt - Emit a sync node.
void CodeGenFunction::EmitSyncStmt(const SyncStmt &S) {
  llvm::BasicBlock *ContinueBlock = createBasicBlock("sync.continue");

  // If this code is reachable then emit a stop point (if generating
  // debug info). We have to do this ourselves because we are on the
  // "simple" statement path.
  if (HaveInsertPoint())
    EmitStopPoint(&S);

  Builder.CreateSync(ContinueBlock, 
    getOrCreateLabeledSyncRegion(S.getSyncVar())->getSyncRegionStart());
  EmitBlock(ContinueBlock);
}

void CodeGenFunction::EmitSpawnStmt(const SpawnStmt &S) {
  // Set up to perform a detach.
  // PushDetachScope();
  SyncRegion* SR = getOrCreateLabeledSyncRegion(S.getSyncVar());
  //StartLabeledDetach(SR);

  llvm::BasicBlock* DetachedBlock = createBasicBlock("det.achd");
  llvm::BasicBlock* ContinueBlock = createBasicBlock("det.cont");

  auto OldAllocaInsertPt = AllocaInsertPt; 
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "",
                                             DetachedBlock);

  Builder.CreateDetach(DetachedBlock, ContinueBlock,
                           SR->getSyncRegionStart());


  EmitBlock(DetachedBlock);
  EmitStmt(S.getSpawnedStmt());

  Builder.CreateReattach(ContinueBlock,
                             SR->getSyncRegionStart());

  llvm::Instruction* ptr = AllocaInsertPt; 
  AllocaInsertPt = OldAllocaInsertPt; 
  ptr->eraseFromParent(); 

  EmitBlock(ContinueBlock);
}

void CodeGenFunction::EmitForallStmt(const ForallStmt &S,
                                  ArrayRef<const Attr *> ForAttrs) {
  JumpDest LoopExit = getJumpDestInCurrentScope("forall.end");
  JumpDest Sync = getJumpDestInCurrentScope("forall.sync");

  LexicalScope ForScope(*this, S.getSourceRange());

  // Evaluate the first part before the loop.
  if (S.getInit())
    EmitStmt(S.getInit());

  // create the sync region
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);

  // TODO: Need to check attributes for spawning strategy. 
  //LoopStack.setSpawnStrategy(LoopAttributes::DAC);
  LoopStack.setSpawnStrategy(GetTapirStrategyAttr(ForAttrs));

  // Start the loop with a block that tests the condition.
  // If there's an increment, the continue scope will be overwritten
  // later.
  JumpDest Continue = getJumpDestInCurrentScope("forall.cond");
  llvm::BasicBlock *CondBlock = Continue.getBlock();
  EmitBlock(CondBlock);

  const SourceRange &R = S.getSourceRange();
  LoopStack.push(CondBlock, CGM.getContext(), ForAttrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // If the for loop doesn't have an increment we can just use the
  // condition as the continue block.  Otherwise we'll need to create
  // a block for it (in the current scope, i.e. in the scope of the
  // condition), and that we will become our continue block.
  if (S.getInc())
    Continue = getJumpDestInCurrentScope("forall.inc");

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, Continue));

  // Create a cleanup scope for the condition variable cleanups.
  LexicalScope ConditionScope(*this, S.getSourceRange());

  auto OldAllocaInsertPt = AllocaInsertPt;
  llvm::SmallVector<std::pair<VarDecl*, RValue>, 4> ivs;  
  DeclMapTy IVDeclMap; 
  if (S.getCond()) {
    // If the for statement has a condition scope, emit the local variable
    // declaration.
    if (S.getConditionVariable()) {
      EmitDecl(*S.getConditionVariable());
    }

    llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
    // If there are any cleanups between here and the loop-exit scope,
    // create a block to stage a loop exit along.
    if (ForScope.requiresCleanups())
      ExitBlock = createBasicBlock("forall.cond.cleanup");

    // As long as the condition is true, iterate the loop.
    llvm::BasicBlock* Detach = createBasicBlock("forall.detach");
    llvm::BasicBlock *ForBody = createBasicBlock("forall.body");

    // C99 6.8.5p2/p4: The first substatement is executed if the expression
    // compares unequal to 0.  The condition must be a scalar type.
    llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
    Builder.CreateCondBr(
        BoolCondVal, Detach, Sync.getBlock(),
        createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody())));

    if (ExitBlock != LoopExit.getBlock()) {
      EmitBlock(ExitBlock);
      EmitBranchThroughCleanup(Sync);
    }

    EmitBlock(Detach);
    // Extract the DeclStmt from the statement init
    const DeclStmt *DS = cast<DeclStmt>(S.getInit());
   
    //Load induction variables as values before the detach 
    for (auto *DI : DS->decls()){
      auto *LoopVar = dyn_cast<VarDecl>(DI);
      Address OuterLoc = LocalDeclMap.find(LoopVar)->second; 
      LocalDeclMap.erase(LoopVar);
      IVDeclMap.insert({LoopVar, OuterLoc}); 
      QualType type = LoopVar->getType();
      LValue OuterLV = MakeAddrLValue(OuterLoc, type);
      RValue OuterRV = EmitLoadOfLValue(OuterLV, DI->getBeginLoc()); 
      ivs.push_back({LoopVar, OuterRV}); 
    }

    // create the detach terminator
    Builder.CreateDetach(ForBody, Continue.getBlock(), SRStart);  

    // Set up IVs to be copied as firstprivate 
    llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
    AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "",
                                               ForBody);
    EmitBlock(ForBody);
  } else {
    // Treat it as a non-zero constant.  Don't even create a new block for the
    // body, just fall into it.
  }


  incrementProfileCounter(&S);

  {
    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.
    RunCleanupsScope BodyScope(*this);
    if(S.getCond()){
      for (auto &ivp : ivs){
        auto* LoopVar = ivp.first; 
        auto OuterRV = ivp.second; 
        AutoVarEmission LVEmission = EmitAutoVarAlloca(*LoopVar);
        EmitAutoVarCleanups(LVEmission);
        QualType type = LoopVar->getType();
        Address Loc = LVEmission.getObjectAddress(*this);
        LValue LV = MakeAddrLValue(Loc, type);	
        LV.setNonGC(true);

        EmitStoreThroughLValue(OuterRV, LV, true);
      }
      EmitStmt(S.getBody());
    }
  }

  if(S.getCond()){
    auto tmp = AllocaInsertPt; 
    AllocaInsertPt = OldAllocaInsertPt; 
    tmp->removeFromParent(); 

    // Restore IVs after emitting body, and set lifetime ends
    for (const auto &p : IVDeclMap){
      LocalDeclMap.erase(p.first);
      LocalDeclMap.insert(p); 
    }
  
    JumpDest Reattach = getJumpDestInCurrentScope("forall.reattach");
    EmitBlock(Reattach.getBlock());
    Builder.CreateReattach(Continue.getBlock(), SRStart);
  }

  // If there is an increment, emit it next.
  if (S.getInc()) {
    EmitBlock(Continue.getBlock());
    EmitStmt(S.getInc());
  }

  BreakContinueStack.pop_back();

  ConditionScope.ForceCleanup();

  EmitStopPoint(&S);
  EmitBranch(CondBlock);

  ForScope.ForceCleanup();

  LoopStack.pop();

  EmitBlock(Sync.getBlock());
  Builder.CreateSync(LoopExit.getBlock(), SRStart);

  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock(), true);
}

void CodeGenFunction::EmitCXXForallRangeStmt(const CXXForallRangeStmt &S,
                                             ArrayRef<const Attr *> ForAttrs) {
  // Create all jump destinations and blocks in the order they appear in the IR
  // some are jump destinations, some are basic blocks
  JumpDest Condition = getJumpDestInCurrentScope("forall.cond");
  llvm::BasicBlock *Detach = createBasicBlock("forall.detach");
  llvm::BasicBlock *ForBody = createBasicBlock("forall.body");
  JumpDest Reattach = getJumpDestInCurrentScope("forall.reattach");
  llvm::BasicBlock *Increment = createBasicBlock("forall.inc");
  JumpDest Cleanup = getJumpDestInCurrentScope("forall.cond.cleanup");
  JumpDest Sync = getJumpDestInCurrentScope("forall.sync");
  llvm::BasicBlock *End = createBasicBlock("forall.end");

  // Extract a convenience block
  llvm::BasicBlock *ConditionBlock = Condition.getBlock();

  const SourceRange &R = S.getSourceRange();
  LexicalScope ForScope(*this, S.getSourceRange());

  // Evaluate the first pieces before the loop.
  if (S.getInit())
    EmitStmt(S.getInit());
  EmitStmt(S.getRangeStmt());
  EmitStmt(S.getBeginStmt());
  EmitStmt(S.getEndStmt());
  EmitStmt(S.getIndexStmt());
  EmitStmt(S.getIndexEndStmt());

  // create the sync region
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);

  // FIXME: Need to get attributes for spawning strategy from
  // code versus this hard-coded route...
//  LoopStack.setSpawnStrategy(LoopAttributes::DAC);
  LoopStack.setSpawnStrategy(GetTapirStrategyAttr(ForAttrs));

  EmitBlock(ConditionBlock);

  LoopStack.push(ConditionBlock, CGM.getContext(), ForAttrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // Store the blocks to use for break and continue.
   BreakContinueStack.push_back(BreakContinue(Reattach, Reattach));

  // The body is executed if the expression, contextually converted
  // to bool, is true.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
  Builder.CreateCondBr(
      BoolCondVal, Detach, Sync.getBlock(),
      createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody())));

  if (ForScope.requiresCleanups()) {
    EmitBlock(Cleanup.getBlock());
    EmitBranchThroughCleanup(Sync);
  }

  /////////////////////////////////
  // Create the detach block
  /////////////////////////////////

  // Emit the (currently empty) detach block
  EmitBlock(Detach);

  // Extract the DeclStmt from the statement init
  const DeclStmt *DS = cast<DeclStmt>(S.getIndexStmt());
  
  // Set up IVs to be copied as firstprivate 
  auto OldAllocaInsertPt = AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "",
                                             ForBody);
  
  DeclMapTy IVDeclMap; 
  llvm::SmallVector<
      std::pair<VarDecl *,
                std::unique_ptr<llvm::SmallVector<llvm::Value *, 4>>>,
      4>
      ivs;
  for (auto *DI : DS->decls()) {
    auto *LoopVar = dyn_cast<VarDecl>(DI);
    Address OuterLoc = LocalDeclMap.find(LoopVar)->second; 
    LocalDeclMap.erase(LoopVar);
    IVDeclMap.insert({LoopVar, OuterLoc}); 
    QualType type = LoopVar->getType();
    ivs.push_back(
          {LoopVar, std::make_unique<llvm::SmallVector<llvm::Value *, 4>>()});
    switch (getEvaluationKind(type)) {
      case TEK_Scalar: {
        LValue OuterLV = MakeAddrLValue(OuterLoc, type);
        RValue OuterRV = EmitLoadOfLValue(OuterLV, DI->getBeginLoc());
        ivs.back().second->push_back(OuterRV.getScalarVal());
        break;
      }
      case TEK_Complex: {
        ComplexPairTy Val =
          EmitLoadOfComplex(MakeAddrLValue(OuterLoc, type), DI->getBeginLoc());
        ivs.back().second->push_back(Val.first);
        ivs.back().second->push_back(Val.second);
        break;
      }
      case TEK_Aggregate: {
        if (const llvm::StructType *STy =
          dyn_cast<llvm::StructType>(OuterLoc.getElementType())) {
          // Load each element of the structure
          for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
            Address EltPtr = Builder.CreateStructGEP(OuterLoc, i);
            llvm::Value *Elt = Builder.CreateLoad(EltPtr);
            ivs.back().second->push_back(Elt);
          }
        } else {
          LValue OuterLV = MakeAddrLValue(OuterLoc, type);
          RValue OuterRV = EmitLoadOfLValue(OuterLV, DI->getBeginLoc());
          ivs.back().second->push_back(OuterRV.getScalarVal());
        }
        break;
      }
    }
  }
  /*
  for (auto *DI : DS->decls()){
    auto *LoopVar = dyn_cast<VarDecl>(DI);
    Address OuterLoc = LocalDeclMap.find(LoopVar)->second; 
    IVDeclMap.insert({LoopVar, OuterLoc}); 
    LocalDeclMap.erase(LoopVar);
    AutoVarEmission LVEmission = EmitAutoVarAlloca(*LoopVar);
    QualType type = LoopVar->getType();
    Address Loc = LVEmission.getObjectAddress(*this);
    LValue LV = MakeAddrLValue(Loc, type);
    LValue OuterLV = MakeAddrLValue(OuterLoc, type); 
    RValue OuterRV = EmitLoadOfLValue(OuterLV, DI->getBeginLoc()); 
    LV.setNonGC(true);
    EmitStoreThroughLValue(OuterRV, LV, true);
    EmitAutoVarCleanups(LVEmission);
  }
  */

  // create the detach terminator
  Builder.CreateDetach(ForBody, Increment, SRStart);

  
  EmitBlock(ForBody);
  incrementProfileCounter(&S);

  {
    // Create a separate cleanup scope for the loop variable and body.
    RunCleanupsScope BodyScope(*this);
    for (auto &ivp : ivs) {
      auto *LoopVar = ivp.first;
      auto &LoadedVals = ivp.second;
      AutoVarEmission LVEmission = EmitAutoVarAlloca(*LoopVar);
      EmitAutoVarCleanups(LVEmission);
      QualType type = LoopVar->getType();
      Address Loc = LVEmission.getObjectAddress(*this);
      LValue LV = MakeAddrLValue(Loc, type);
      LV.setNonGC(true);
      switch (getEvaluationKind(type)) {
        case TEK_Scalar: {
          EmitStoreOfScalar(LoadedVals->back(), LV, /*isInit*/ true);
          break;
        }
        case TEK_Complex: {
          ComplexPairTy Val = {(*LoadedVals)[0], (*LoadedVals)[1]};
          EmitStoreOfComplex(Val, LV, /*isInit*/ true);
          break;
        }
        case TEK_Aggregate: {
          if (const llvm::StructType *STy =
            dyn_cast<llvm::StructType>(Loc.getElementType())) {
            // Store the previously-loaded value into the new structure
            for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
              Address EltPtr = Builder.CreateStructGEP(Loc, i);
              llvm::Value *Elt = (*LoadedVals)[i];
              Builder.CreateStore(Elt, EltPtr);
            }
          } else {
            EmitStoreOfScalar(LoadedVals->back(), LV, /*isInit*/ true);
          }
          break;
        }
      }
    }
    EmitStmt(S.getLoopVarStmt());
    EmitStmt(S.getBody());
  }

  auto tmp = AllocaInsertPt; 
  AllocaInsertPt = OldAllocaInsertPt; 
  tmp->removeFromParent(); 


  // Restore IVs after emitting body
  for (const auto &p : IVDeclMap){
    LocalDeclMap.erase(p.first);
    LocalDeclMap.insert(p); 
  }

  /////////////////////////////////////////////////////////////////
  // Modify the body block to use the detach block variable mirror.
  // At this point in the codegen, the body block has been emitted
  // and we can safely replace the induction variable with the detach
  // block mirror in the entire function, since the increment block
  // (a valid use of the induction variable) has not been emitted yet.
  /////////////////////////////////////////////////////////////////

  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment, SRStart);

  EmitBlock(Increment);

  EmitStmt(S.getInc());

  BreakContinueStack.pop_back();

  EmitStopPoint(&S);
  
  EmitBranch(ConditionBlock);

  ForScope.ForceCleanup();

  LoopStack.pop();

  EmitBlock(Sync.getBlock());
  Builder.CreateSync(End, SRStart);

  EmitBlock(End, true);
}

