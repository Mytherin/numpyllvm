
#include "optimizer.hpp"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Analysis/TargetTransformInfo.h"

using namespace llvm;

std::unique_ptr<legacy::FunctionPassManager>
CreatePassManager(Module *module, LLVMJIT* jit) {
	auto passmanager = llvm::make_unique<legacy::FunctionPassManager>(module);

#ifdef __has_feature
#   if __has_feature(address_sanitizer)
#   if defined(LLVM37) && !defined(LLVM38)
        // LLVM 3.7 BUG: ASAN pass doesn't properly initialize its dependencies
    initializeTargetLibraryInfoWrapperPassPass(*PassRegistry::getPassRegistry());
#   endif
        passmanager->add(createAddressSanitizerFunctionPass());
#   endif
#   if __has_feature(memory_sanitizer)
        passmanager->add(llvm::createMemorySanitizerPass(true));
#   endif
#endif

    passmanager->add(createTargetTransformInfoWrapperPass(jit->getTargetMachine().getTargetIRAnalysis()));
    //jit->getTargetMachine().addAnalysisPasses(*passmanager);

    // list of passes from vmkit
    passmanager->add(createCFGSimplificationPass()); // Clean up disgusting code
    passmanager->add(createPromoteMemoryToRegisterPass());// Kill useless allocas

#ifndef INSTCOMBINE_BUG
        passmanager->add(createInstructionCombiningPass()); // Cleanup for scalarrepl.
#endif
    passmanager->add(createSROAPass());                 // Break up aggregate allocas
#ifndef INSTCOMBINE_BUG
        passmanager->add(createInstructionCombiningPass()); // Cleanup for scalarrepl.
#endif
    passmanager->add(createJumpThreadingPass());        // Thread jumps.
    // NOTE: CFG simp passes after this point seem to hurt native codegen.
    // See issue #6112. Should be re-evaluated when we switch to MCJIT.
    //passmanager->add(createCFGSimplificationPass());    // Merge & remove BBs
#ifndef INSTCOMBINE_BUG
        passmanager->add(createInstructionCombiningPass()); // Combine silly seq's
#endif

    //passmanager->add(createCFGSimplificationPass());    // Merge & remove BBs
    passmanager->add(createReassociatePass());          // Reassociate expressions

    // this has the potential to make some things a bit slower
    //passmanager->add(createBBVectorizePass());

    passmanager->add(createEarlyCSEPass()); //// ****

    passmanager->add(createLoopIdiomPass()); //// ****
    passmanager->add(createLoopRotatePass());           // Rotate loops.
    // LoopRotate strips metadata from terminator, so run LowerSIMD afterwards
    passmanager->add(createLICMPass());                 // Hoist loop invariants
    passmanager->add(createLoopUnswitchPass());         // Unswitch loops.
    // Subsequent passes not stripping metadata from terminator
#ifndef INSTCOMBINE_BUG
        passmanager->add(createInstructionCombiningPass());
#endif
    passmanager->add(createIndVarSimplifyPass());       // Canonicalize indvars
    passmanager->add(createLoopDeletionPass());         // Delete dead loops
#if defined(LLVM35)
        passmanager->add(createSimpleLoopUnrollPass());     // Unroll small loops
#else
    passmanager->add(createLoopUnrollPass());           // Unroll small loops
#endif
#if !defined(LLVM35) && !defined(INSTCOMBINE_BUG)
        passmanager->add(createLoopVectorizePass());        // Vectorize loops
#endif
    //passmanager->add(createLoopStrengthReducePass());   // (jwb added)

#ifndef INSTCOMBINE_BUG
        passmanager->add(createInstructionCombiningPass()); // Clean up after the unroller
#endif
    passmanager->add(createGVNPass());                  // Remove redundancies
    passmanager->add(createMemCpyOptPass());            // Remove memcpy / form memset
    passmanager->add(createSCCPPass());                 // Constant prop with SCCP

    // Run instcombine after redundancy elimination to exploit opportunities
    // opened up by them.
    passmanager->add(createSinkingPass()); ////////////// ****
    passmanager->add(createInstructionSimplifierPass());///////// ****
#ifndef INSTCOMBINE_BUG
        passmanager->add(createInstructionCombiningPass());
#endif
    passmanager->add(createJumpThreadingPass());         // Thread jumps
    passmanager->add(createDeadStoreEliminationPass());  // Delete dead stores
#if !defined(INSTCOMBINE_BUG)
        passmanager->add(createSLPVectorizerPass());     // Vectorize straight-line code
#endif

    passmanager->add(createAggressiveDCEPass());         // Delete dead instructions
#if !defined(INSTCOMBINE_BUG)
        passmanager->add(createInstructionCombiningPass());   // Clean up after SLP loop vectorizer
#endif
#if defined(LLVM35)
        passmanager->add(createLoopVectorizePass());         // Vectorize loops
    passmanager->add(createInstructionCombiningPass());  // Clean up after loop vectorizer
#endif
    //passmanager->add(createCFGSimplificationPass());     // Merge & remove BBs
    passmanager->doInitialization();
    return passmanager;
}