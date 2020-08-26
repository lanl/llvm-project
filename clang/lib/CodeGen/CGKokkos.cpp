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
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Pass.h"
#include "llvm/IR/Dominators.h"
#include "CodeGenFunction.h"

using namespace clang;
using namespace CodeGen;

namespace {

  // Part of the challenge in unwinding all the Kokkos
  // C++ details is getting through a bunch of implicit 
  // goop in the AST. Fortunately Clang can help here 
  // a lot as it has mechanisms to toss out implicit 
  // constructs and get us closer to the underlying 
  // expression.  Pulled out of inline code to allow 
  // for more agressive simplications/modifications 
  // if needed (and reuse across different constructs).
  static const Expr *SimplifyExpr(const Expr *E) {
    return E->IgnoreImplicit()->IgnoreImpCasts();
  }

  // Break apart the various components of a Kokkos 
  // parallel_for.  This boils down to tearing apart 
  // a CallExpr.  It helps that at this point we've 
  // already passed all the "goodness" in semantic 
  // analysis and other frontend stages.  However, 
  // there are aspects of what you can do generally 
  // in Kokkos and what we can tear apart here (both
  // big picture and in terms of the state of the 
  // current implementation).  A current set of 
  // limitations are:
  // 
  //   - Functors are not supported (primarily given
  //     challenges around separate compilation units)
  //   - 'Named constructs', which Kokkos uses for profiling,
  //     are extracted but unused. 
  //   - There are likely many holes in types that are legal 
  //     constructs but not yet captured/covered in the 
  //     current set of transformations currently implemented.
  //     The code tries to avoid crashing/asserting and instead
  //     falls back to the standard C++ mechanisms. We will start
  //     ramp up testing as we get more exposure to use cases. 
  // 
  static void 
  ExtractParallelForComponents(const CallExpr* CE,
					   std::string &CN, const Expr *& BE, 
					   const LambdaExpr *& LE)
  {
    // Recongized constructs: 
    // 
    //   1. parallel_for(N, lambda_expr...);
    //
    //   2. parallel_for("name", N, lambda_expr...);

    unsigned int curArgIndex = 0;

    // Check for a 'named' construct.  NOTE: we assume 
    // this only comes in the form of a string literal. 
    const Expr *OE = CE->getArg(curArgIndex); // Original expression 
    const Expr *SE = SimplifyExpr(OE);        // Simplified expression. 

    if (SE->getStmtClass() == Expr::CXXConstructExprClass) {
      const CXXConstructExpr *CXXCE = dyn_cast<CXXConstructExpr>(SE);
      SE = CXXCE->getArg(0)->IgnoreImplicit();
      if (SE->getStmtClass() == Expr::StringLiteralClass) {
        CN = dyn_cast<StringLiteral>(SE)->getString().str();
        curArgIndex++;
        OE = CE->getArg(curArgIndex);
        SE = SimplifyExpr(OE);
      } 
    } 

    // Check details of the bounds portion of the parallel_for. 
    // Note: This can take several forms and it is likely we've 
    // missed some cases... 
    if (SE->getStmtClass() == Expr::IntegerLiteralClass) {
      BE = OE;
      curArgIndex++;
      OE = CE->getArg(curArgIndex);
      SE = SimplifyExpr(OE);
    } else if (SE->getStmtClass() == Expr::BinaryOperatorClass) {
      BE = OE;
      curArgIndex++;
      OE = CE->getArg(curArgIndex);
      SE = SimplifyExpr(OE);
    } else if (SE->getStmtClass() == Expr::DeclRefExprClass) {
      BE = OE;
      curArgIndex++;
      OE = CE->getArg(curArgIndex);
      SE = SimplifyExpr(OE);
    } else if (SE->getStmtClass() == Expr::CallExprClass) { 
      BE = OE;
      curArgIndex++;
      OE = CE->getArg(curArgIndex);
      SE = SimplifyExpr(OE);
    } else {
      SE->dump();
      BE = nullptr;
      LE = nullptr;
      return;
    }

    if (SE->getStmtClass() == Expr::LambdaExprClass) {
      LE = dyn_cast<LambdaExpr>(SE);
    } else {
      LE = nullptr;
      return;
    }
  }

  // helpful for debugging predcessor code details... 
  static unsigned int numPredecessors(llvm::BasicBlock *BB) {
    unsigned int count = 8; 
    while(count > 0) {
      if (BB->hasNPredecessorsOrMore(count))
        break;
      count--;
    }

    return count;
  }

  using BBlockSet = llvm::SmallPtrSet<llvm::BasicBlock*, 8>;

  // Is the given basic block ('Target') "bound" between the given two
  // basic blocks?  In more traditional (LLVM) terminiology the question 
  // does the 'Dominant' BB dominate 'Target' dominate 
  // dominate the other).  This is currently very centric to the 
  // Tapir detach and reattach constructs around our code 
  // generation transformations for Kokkos. 
  // 
  // Return 
  static bool IsWithinDominanceRange(llvm::BasicBlock *BB,     // The block to begin the search from. 
                                     llvm::BasicBlock *Detach, // Dominant BB. 
                                     llvm::BasicBlock *Target, // The BB to search for. 
                                     BBlockSet &Visited) {       // For avoiding cycles... 

    assert(BB && "not expecting null basic block");
    assert(Detach && "vaild, non-null, detach (basic) block must be provided.");
    assert(Target && "valid, no-null, target (basic) block must be provided.");

    // The ordering of our checks are important here.  Specifically, we 
    // don't want to process the detach block so we use it for a termination 
    // point (of the recursive walk). 
    if (BB == Detach)
      return false;  // BB was not found -- we've walked all the way to the detach. 

    if (BB == Target)
      return true;  // Found the block. 

    // Check to see if we've visited BB.  If not, it will be inserted into 
    // the visited set and we will then search the predecessors list for BB... 
    if (Visited.insert(BB).second) {

      bool found = false;
      for(llvm::BasicBlock *P : predecessors(BB)) {
        found = IsWithinDominanceRange(P, Detach, Target, Visited);
        // A full walk can't terminate here unless we have successfully 
        // found the Target block -- i.e., we must continue to walk the 
        // full predecessor list for BB. 
        if (found) 
          return true; 
      }
      return false; // Target block not found... 
    } else
      return false; // We have already walked a previous path to BB -- stop search... 
  }

  // Convenenice interface to avoid creating and managing the visited set... 
  static bool IsWithinDominanceRange(llvm::BasicBlock *BB,       // The block to begin the search from. 
                                     llvm::BasicBlock *Detach,   // Dominant BB. 
                                     llvm::BasicBlock *Target) { // The BB to search for. 

    static BBlockSet Visited;
    Visited.clear();
    return IsWithinDominanceRange(BB, Detach, Target, Visited);
  }
  
}

// Sort through what sort of Kokkos construct we're looking at 
// and work on transforming it into a Tapir-centric lowering.  
// 
bool CodeGenFunction::EmitKokkosConstruct(const CallExpr *CE,
                ArrayRef<const Attr *> Attrs) {
  assert(CE != 0 && "CodeGenFunction::EmitKokkosConstruct: null callexpr passed!");

  const FunctionDecl *Func = CE->getDirectCallee();
  assert(Func != 0 && "Kokkos construct doesn't have a function declaration!");

  if (Func->getQualifiedNameAsString() == "Kokkos::parallel_for") {
    return EmitKokkosParallelFor(CE, Attrs);
  } else if (Func->getQualifiedNameAsString() == "Kokkos::parallel_reduce") {
    return EmitKokkosParallelReduce(CE, Attrs);
  } else {
    return false;
  }
}


const ParmVarDecl*
CodeGenFunction::EmitKokkosParallelForInductionVar(const LambdaExpr *Lambda) {
  const CXXMethodDecl *MD = Lambda->getCallOperator();
  assert(MD && "EmitKokkosParallelFor() -- bad method decl from labmda call.");
  const ParmVarDecl *InductionVarDecl = MD->getParamDecl(0);
  assert(InductionVarDecl && "EmitKokkosParallelFor() -- bad loop variable decl!");

  EmitVarDecl(*InductionVarDecl);
  Address Addr = GetAddrOfLocalVar(InductionVarDecl);
  llvm::Value *Zero = llvm::ConstantInt::get(ConvertType(InductionVarDecl->getType()), 0);
  Builder.CreateStore(Zero, Addr);

  return InductionVarDecl;
}

void CodeGenFunction::EmitKokkosParallelForCond(const Expr *BoundsExpr, 
     const ParmVarDecl *InductionVarDecl, 
     llvm::BasicBlock *DetachBlock,
     llvm::BasicBlock *ExitBlock,
     JumpDest &Sync) {

  llvm::Value *LoopEnd = nullptr;
  if (BoundsExpr->getStmtClass() == Expr::BinaryOperatorClass) {
    RValue RV = EmitAnyExpr(BoundsExpr);
    LoopEnd = RV.getScalarVal();
  } else { 
    LoopEnd = EmitScalarExpr(BoundsExpr);
  }

  llvm::Type  *InductionVarTy = ConvertType(InductionVarDecl->getType());
  unsigned NBits  = LoopEnd->getType()->getPrimitiveSizeInBits();
  unsigned LVBits = InductionVarTy->getPrimitiveSizeInBits();

  // We may need to truncate/extend the range to get it to match 
  // the type of loop variable. 
  if (NBits > LVBits) {
    LoopEnd = Builder.CreateTrunc(LoopEnd, InductionVarTy);
  } else if (NBits < LVBits) {
    LoopEnd = Builder.CreateZExt(LoopEnd, InductionVarTy);
  } else {
    // bit count matches, nothing to do...
  }

  llvm::Value *InductionVal = Builder.CreateLoad(GetAddrOfLocalVar(InductionVarDecl));
  llvm::Value *CondVal = Builder.CreateICmpULT(InductionVal, LoopEnd);
  Builder.CreateCondBr(CondVal, DetachBlock, Sync.getBlock());
}

bool CodeGenFunction::EmitKokkosParallelFor(const CallExpr *CE,
              ArrayRef<const Attr *> ForallAttrs) {
    
  // TODO: Need to add code to process any attributes (ForallAttrs).

  // Tease apart the parallel_for into its various components. 
  std::string      PFName; // construct name (for kokkos profiling)  
  const Expr       *BE = nullptr; // "bounds" expression
  const LambdaExpr *Lambda = nullptr; // the lambda  
  ExtractParallelForComponents(CE, PFName, BE, Lambda);
  
  if (Lambda == nullptr) { 
    // The parallel_for doesn't have a (recognizable) lambda expression.  
    // 
    // Functor support is problematic as it can live in a different 
    // compilaton unit.  We always punt and go the pure C++ route in
    // these cases. 
    DiagnosticsEngine &Diags = CGM.getDiags();
    // TODO: should reword this warning terminology. 
    Diags.Report(CE->getExprLoc(), diag::warn_kokkos_no_functor);
    return false;
  }

  if (BE == nullptr) {
    // We didn't get a known bounds expression back -- this is most likely 
    // due to some type of expression that we have yet to deal with.  We're
    // going to have to uncover these on a case-by-cases basis.
    DiagnosticsEngine &Diags = CGM.getDiags();
    Diags.Report(CE->getExprLoc(), diag::warn_kokkos_unknown_bounds_expr);
    return false;
  }

  // Create all jump destinations and basic blocks in the order they 
  // appear in the IR. 
  JumpDest Condition = getJumpDestInCurrentScope("kokkos.forall.cond");
  llvm::BasicBlock *Detach = createBasicBlock("kokkos.forall.detach");
  llvm::BasicBlock *PForBody = createBasicBlock("kokkos.forall.body");
  JumpDest Reattach = getJumpDestInCurrentScope("kokkos.forall.reattach");
  llvm::BasicBlock *Increment = createBasicBlock("kokkos.forall.inc");
  JumpDest Cleanup = getJumpDestInCurrentScope("kokkos.forall.cond.cleanup");
  JumpDest Sync = getJumpDestInCurrentScope("kokkos.forall.sync");
  llvm::BasicBlock *End = createBasicBlock("kokkos.forall.end");

  // Extract a conveince block and setup the lexical scope based on 
  // the lambda's source range. 
  llvm::BasicBlock *ConditionBlock = Condition.getBlock();
  
  const SourceRange &R = CE->getSourceRange();
  LexicalScope PForScope(*this, R);

  // Now we can start the dirty work of transforming the lambda into a 
  // for loop.  


  // The first step is to extract the argument to the lambda and transform it into 
  // the loop induction variable.  As part of this we assume the following are true
  // about the parallel_for:
  //    1. The iterator can be assigned a value of zero. 
  //    2. We ignore the details of what is captured by the lambda.
  // 
  // TODO: Do we need to "relax" these assumptions to support broader code coverage?
  // This is 'equivalent' to the Init statement in a tranditional for loop (e.g. int i = 0). 
  const ParmVarDecl *InductionVarDecl; 
  InductionVarDecl = EmitKokkosParallelForInductionVar(Lambda);

   // Create the sync region. 
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);

  // TODO: Need to check attributes for spawning strategy. 
  LoopStack.setSpawnStrategy(LoopAttributes::DAC);
  
  EmitBlock(ConditionBlock);
  
  LoopStack.push(ConditionBlock, CGM.getContext(), ForallAttrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // Store the blocks to use for break and continue. 
  BreakContinueStack.push_back(BreakContinue(Reattach, Reattach));

  // Create a scope for the condition variable cleanup. 
  LexicalScope ConditionScope(*this, R);

  // Create the conditional.
  EmitKokkosParallelForCond(BE, InductionVarDecl, Detach, End, Sync);

  if (PForScope.requiresCleanups()) {
    EmitBlock(Cleanup.getBlock());
    EmitBranchThroughCleanup(Sync);
  }

  // Handle the detach block...
  EmitBlock(Detach);

  llvm::Value *GInductionVar = GetAddrOfLocalVar(InductionVarDecl).getPointer();
  llvm::Value *GInductionVal = Builder.CreateLoad(GetAddrOfLocalVar(InductionVarDecl));

  QualType RefType = InductionVarDecl->getType();
  llvm::AllocaInst *TLInductionVar = Builder.CreateAlloca(
    getTypes().ConvertType(RefType), nullptr, InductionVarDecl->getName() + ".detach");
  Builder.CreateAlignedStore(GInductionVal, TLInductionVar, 
                getContext().getTypeAlignInChars(RefType));
  
  // Create the detach terminator 
  Builder.CreateDetach(PForBody, Increment, SRStart);

  EmitBlock(PForBody);
  incrementProfileCounter(CE);
  {
    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.
    InKokkosConstruct = true;
    RunCleanupsScope BodyScope(*this);
    EmitStmt(Lambda->getBody());
    InKokkosConstruct = false;
  }

  
  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment, SRStart);

  EmitBlock(Increment);
  llvm::Value *IncVal = Builder.CreateLoad(GetAddrOfLocalVar(InductionVarDecl));
  llvm::Value *One = llvm::ConstantInt::get(ConvertType(InductionVarDecl->getType()), 1);
  IncVal = Builder.CreateAdd(IncVal, One);
  Builder.CreateStore(IncVal, GetAddrOfLocalVar(InductionVarDecl));

  BreakContinueStack.pop_back();
  ConditionScope.ForceCleanup();
  EmitStopPoint(CE);

  EmitBranch(ConditionBlock);
  PForScope.ForceCleanup();
  LoopStack.pop();

  EmitBlock(Sync.getBlock());
  Builder.CreateSync(End, SRStart);
  EmitBlock(End, true);


  // Modify the body to use the ''detach''-local induction variable.
  // At this point in the codegen, the body block has been emitted 
  // and we can safely replace the ''sequential`` induction variable 
  // within the detach basic block.
  int count = 0;
  BBlockSet Visited;
  for(llvm::Value::use_iterator UI = GInductionVar->use_begin(), UE = GInductionVar->use_end(); 
      UI != UE; ) {
    llvm::Use &U = *UI++;
    llvm::Instruction *I = cast<llvm::Instruction>(U.getUser());
    count++;
    if (IsWithinDominanceRange(Reattach.getBlock(), Detach, I->getParent()))
      U.set(TLInductionVar);
  }

  return true;
}

bool CodeGenFunction::EmitKokkosParallelReduce(const CallExpr *CE,
                    ArrayRef<const Attr *> Attrs) {
  DiagnosticsEngine &Diags = CGM.getDiags();
  Diags.Report(CE->getExprLoc(), diag::warn_kokkos_reduce_unsupported);
  return false;
}

