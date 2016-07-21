
#ifndef Py_LLVMJIT_H
#define Py_LLVMJIT_H

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/DynamicLibrary.h"

typedef llvm::orc::ObjectLinkingLayer<> ObjLayerT;
typedef llvm::orc::IRCompileLayer<ObjLayerT> CompileLayerT;
typedef CompileLayerT::ModuleSetHandleT ModuleHandleT;

class LLVMJIT {
public:
	LLVMJIT();
	llvm::TargetMachine &getTargetMachine() { return *target_machine; }
	const llvm::DataLayout &getDataLayout() { return data_layout; }

	ModuleHandleT addModule(std::unique_ptr<llvm::Module> module);
	void removeModule(ModuleHandleT handle);

	llvm::orc::JITSymbol findSymbol(const std::string name);
private:
	std::string mangle(const std::string &name);
	llvm::orc::JITSymbol findMangledSymbol(const std::string &name);

	std::unique_ptr<llvm::TargetMachine> target_machine;
	const llvm::DataLayout data_layout;
	ObjLayerT object_layer;
	CompileLayerT compile_layer;
	std::vector<ModuleHandleT> module_handles;
};


#endif /* Py_LLVMJIT_H */
