
#ifndef Py_PARSER_H
#define Py_PARSER_H

#include "gencode.hpp"
#include "thunk.hpp"
#include <vector>

class Pipeline;
class PipelineNode;

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
    /* Where the result of this operation will be stored. This is equal to NULL if there is no result location (i.e. unnecessary intermediate).
     * The final operation of a chain of pipelines will always have a result_object, but intermediate operations can also have result_objects 
     * Consider e.g. the following scenario:
     *   a = 1:100
     *   b = 1:100
     *   c = a * a * a
     *   d = c * b * b
     *   d.evaluate()
     * In this scenario the two result_objects will be "c" and "d"; with "d" being the final result_object
     * Unnecessary intermediates (e.g. [a * a] or [b * b] in this scenario) will have NULL as result_object*/
    PyThunkObject *result_object; 
    Operation(PyThunkObject *result) : result_object(result) { }
    ~Operation() {}
    virtual operation_type Type() { return OPTYPE_unknown; }
};

/* Stores Input Data from a Thunk object */
class ObjectOperation : public Operation {
public:    
    PyThunkObject *thunk;
    ObjectOperation(PyThunkObject *thunk) : Operation(NULL), thunk(thunk) {}
    virtual operation_type Type() { return OPTYPE_obj; }
};

/* Stores Input Data from a Pipeline (i.e. the input is the result of the given pipeline)*/
class PipelineOperation : public Operation {
public:    
    Pipeline *pipeline;
    PyThunkObject *thunk;
    PipelineOperation(Pipeline *pipeline, PyThunkObject *thunk) : Operation(NULL), pipeline(pipeline), thunk(thunk) {}
    virtual operation_type Type() { return OPTYPE_pipeline; }
};

/* Nullary operation -> operation without arguments */
class NullaryOperation : public Operation {
public:
    ThunkOperation *operation;
    NullaryOperation(PyThunkObject *result_object, ThunkOperation *operation) : Operation(result_object), operation(operation) {} 
    virtual operation_type Type() { return OPTYPE_nullop; }
};

/* Unary operation -> operation with 1 argument */
class UnaryOperation : public Operation {
public:
    ThunkOperation *operation;
    Operation* LHS;
    UnaryOperation(PyThunkObject *result_object, ThunkOperation *operation, Operation *LHS) : Operation(result_object), operation(operation), LHS(LHS) {} 
    virtual operation_type Type() { return OPTYPE_unop; }
};

/* Binary operation -> operation with 2 arguments */
class BinaryOperation : public Operation {
public:
    ThunkOperation *operation;
    Operation* LHS, *RHS;
    BinaryOperation(PyThunkObject *result_object, ThunkOperation *operation, Operation *LHS, Operation *RHS) : Operation(result_object), operation(operation), LHS(LHS), RHS(RHS) {} 
    virtual operation_type Type() { return OPTYPE_binop; }
};


Pipeline *ParsePipeline(PyThunkObject *thunk);
void DestroyPipeline(Pipeline *pipeline);

class DataElement {
public:
    PyThunkObject *object;
    Operation *operation;
    void *data;
    size_t size;
    int type;
    void *alloca_address;

    DataElement(PyThunkObject *object, Operation *op, void *data, size_t size, int type) : object(object), operation(op), data(data), size(size), type(type) { }
};

class DataBlock {
public:
    std::vector<DataElement> objects;

    DataBlock() : objects() { }
};

class PipelineNode {
public:
    Pipeline *child;
    PipelineNode *next;
    bool evaluated;
};

class Pipeline {
public:
    Operation *operation;
    Pipeline *parent;
    PipelineNode *children;
    DataBlock *inputData;
    DataBlock *outputData;
    char *name;
    size_t size;
};

#endif /*Py_PARSER_H*/
