
#include "llvmjit.hpp"

using namespace llvm;
using namespace orc;

template <typename T> 
static std::vector<T> 
singletonSet(T t) {
	std::vector<T> vector;
	vector.push_back(std::move(t));
	return vector;
}

LLVMJIT::LLVMJIT()
	:  target_machine(EngineBuilder().selectTarget()), data_layout(target_machine->createDataLayout()),
        compile_layer(object_layer, SimpleCompiler(*target_machine)) {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
}

ModuleHandleT 
LLVMJIT::addModule(std::unique_ptr<Module> module) {
	// We need a memory manager to allocate memory and resolve symbols for this
    // new module. Create one that resolves symbols by looking back into the
    // JIT.
    auto resolver = createLambdaResolver(
        [&](const std::string &name) {
        	if (auto symbol = findMangledSymbol(name))
        		return RuntimeDyld::SymbolInfo(symbol.getAddress(), symbol.getFlags());
        	return RuntimeDyld::SymbolInfo(nullptr);
        },
        [](const std::string &s) { return nullptr; });
    auto handle = compile_layer.addModuleSet(singletonSet(std::move(module)),
                                       make_unique<SectionMemoryManager>(),
                                       std::move(resolver));

    module_handles.push_back(handle);
    return handle;
}

void 
LLVMJIT::removeModule(ModuleHandleT handle) {
    module_handles.erase(
        std::find(module_handles.begin(), module_handles.end(), handle));
    compile_layer.removeModuleSet(handle);
}


JITSymbol 
LLVMJIT::findSymbol(const std::string name) {
	return findMangledSymbol(mangle(name));
}

std::string 
LLVMJIT::mangle(const std::string &name) {
    std::string mangled_name;
    {
    	raw_string_ostream mangled_name_stream(mangled_name);
		Mangler::getNameWithPrefix(mangled_name_stream, name, data_layout);
    }
    return mangled_name;
}

JITSymbol 
LLVMJIT::findMangledSymbol(const std::string &name) {
    // Search modules in reverse order: from last added to first added.
    // This is the opposite of the usual search order for dlsym, but makes more
    // sense in a REPL where we want to bind to the newest available definition.
    for (auto handle : make_range(module_handles.rbegin(), module_handles.rend()))
      if (auto symbol = compile_layer.findSymbolIn(handle, name, true))
        return symbol;

    // If we can't find the symbol in the JIT, try looking in the host process.
    if (auto symbol_address = RTDyldMemoryManager::getSymbolAddressInProcess(name))
      return JITSymbol(symbol_address, JITSymbolFlags::Exported);

    return nullptr;
}
