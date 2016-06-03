
#ifndef Py_PARSER_H
#define Py_PARSER_H

#include "gencode.hpp"
#include "thunk.hpp"

struct Pipeline;
struct PipelineNode;

struct PipelineNode {
    Pipeline *child;
    PipelineNode *next;
};

struct Pipeline {
    PyThunkObject *thunk;
    PipelineNode *children;
};

Pipeline *ParsePipeline(PyThunkObject *thunk);

#endif /*Py_PARSER_H*/
