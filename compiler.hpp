



#ifndef Py_Compiler_H
#define Py_Compiler_H

#include "parser.hpp"
#include "scheduler.hpp"
#include "thread.hpp"

class JITFunction;

JITFunction* CompilePipeline(Pipeline *pipeline, Thread *thread);
void JITFunctionDECREF(JITFunction *f);
void ExecuteFunction(JITFunction *f, size_t start, size_t end);

#endif
