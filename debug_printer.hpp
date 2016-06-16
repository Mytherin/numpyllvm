
#ifndef Py_DBGPRINTER_H
#define Py_DBGPRINTER_H

#include "parser.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>


void PrintThunk(PyThunkObject *thunk);
void PrintPipeline(Pipeline *pipeline);

char *getname(const char *base);

#endif