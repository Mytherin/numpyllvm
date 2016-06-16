
#ifndef Py_PARSER_H
#define Py_PARSER_H

#include "gencode.hpp"
#include "thunk.hpp"

struct Pipeline;
struct PipelineNode;

enum operation_type {
    OPTYPE_nullop = 0, 
    OPTYPE_unop = 1,
    OPTYPE_binop = 2,
    OPTYPE_pipeline = 100,
    OPTYPE_obj = 101,
    OPTYPE_unknown = 255
};


#define OPTYPE_MAX_OPERATIONS 3

class Operation {
public:
    virtual ~Operation() {}
    virtual operation_type Type() { return OPTYPE_unknown; }
};

class ObjectOperation : public Operation {
public:    
    PyThunkObject *thunk;
    ObjectOperation(PyThunkObject *thunk) : thunk(thunk) {}
    virtual operation_type Type() { return OPTYPE_obj; }
};

class PipelineOperation : public Operation {
public:    
    Pipeline *pipeline;
    PipelineOperation(Pipeline *pipeline) : pipeline(pipeline) {}
    virtual operation_type Type() { return OPTYPE_pipeline; }
};

class NullaryOperation : public Operation {
public:
    ThunkOperation *operation;
    NullaryOperation(ThunkOperation *operation) : operation(operation) {} 
    virtual operation_type Type() { return OPTYPE_nullop; }
};

class UnaryOperation : public Operation {
public:
    ThunkOperation *operation;
    Operation* LHS;
    UnaryOperation(ThunkOperation *operation, Operation *LHS) : operation(operation), LHS(LHS) {} 
    virtual operation_type Type() { return OPTYPE_unop; }
};

class BinaryOperation : public Operation {
public:
    ThunkOperation *operation;
    Operation* LHS, *RHS;
    BinaryOperation(ThunkOperation *operation, Operation *LHS, Operation *RHS) : operation(operation), LHS(LHS), RHS(RHS) {} 
    virtual operation_type Type() { return OPTYPE_binop; }
};

Pipeline *ParsePipeline(PyThunkObject *thunk);

struct PipelineNode {
    Pipeline *child;
    PipelineNode *next;
    bool evaluated;
};

struct Pipeline {
    Operation *operation;
    Pipeline *parent;
    PipelineNode *children;
};

#endif /*Py_PARSER_H*/
