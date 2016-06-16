
#ifndef Py_DBGPRINTER_H
#define Py_DBGPRINTER_H

#include "parser.hpp"

void PrintThunk(PyThunkObject *thunk);
void PrintPipeline(Pipeline *pipeline);

#endif