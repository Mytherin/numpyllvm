
#ifndef Py_PARSER_H
#define Py_PARSER_H

#include "gencode.hpp"
#include "thunk.hpp"
#include "mutex.hpp"
#include <vector>

class Pipeline;
class PipelineNode;
class JITFunction;

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
    bool materialize;
    Operation(PyThunkObject *result) : result_object(result), materialize(false) { }
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
    void *extra;
    UnaryOperation(PyThunkObject *result_object, ThunkOperation *operation, Operation *LHS) : Operation(result_object), operation(operation), LHS(LHS), extra(NULL) {} 
    virtual operation_type Type() { return OPTYPE_unop; }
};

/* Binary operation -> operation with 2 arguments */
class BinaryOperation : public Operation {
public:
    ThunkOperation *operation;
    Operation* LHS, *RHS;
    void *extra;
    BinaryOperation(PyThunkObject *result_object, ThunkOperation *operation, Operation *LHS, Operation *RHS) : Operation(result_object), operation(operation), LHS(LHS), RHS(RHS), extra(NULL) {} 
    virtual operation_type Type() { return OPTYPE_binop; }
};


Pipeline *ParsePipeline(PyThunkObject *thunk);
void DestroyPipeline(Pipeline *pipeline);

class DataSource {
public:
    PyThunkObject *object;
    void *data;
    size_t size;
    int type;

    DataSource(PyThunkObject *object, void *data, size_t size, int type) : object(object), data(data), size(size), type(type) { }
};

class DataElement {
public:
    Operation *operation;
    DataSource *source;
    void *alloca_address;
    void *index_addr;

    DataElement(Operation *op, DataSource *source) : operation(op), source(source) { }
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
};

class Pipeline {
public:
    Operation *operation;
    Pipeline *parent;
    PipelineNode *children;
    DataBlock *inputData;
    DataBlock *outputData;
    char *name;
    bool evaluated;
    JITFunction *function;
    bool scheduled_for_execution;
    bool scheduled_for_compilation;
    semaphore_struct semaphore;
    semaphore_struct lock;
};

#endif /*Py_PARSER_H*/
