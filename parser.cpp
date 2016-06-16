

#include "parser.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

static Pipeline* CreatePipeline() {
    Pipeline *pipeline = (Pipeline*) malloc(sizeof(Pipeline));
    pipeline->operation = NULL;
    pipeline->children = NULL;
    pipeline->parent = NULL;
    return pipeline;
}

static void 
AddChild(Pipeline *parent, Pipeline *child) {
    PipelineNode *node = (PipelineNode*) malloc(sizeof(PipelineNode));
    node->child = child;
    node->next = NULL;
    node->evaluated = false;
    if (parent->children == NULL) {
        parent->children = node;
    } else {
        node->next = parent->children;
        parent->children = node;
    }
}

static Operation*
OperationFromArgs(ThunkOperation *operation, Operation** operations, int op_count) {
    switch(op_count) {
        case 0: 
            return new NullaryOperation(operation);
        case 1:
            return new UnaryOperation(operation, operations[0]);
        case 2:
            return new BinaryOperation(operation, operations[0], operations[1]);
    }
    return NULL;
}

static Operation* 
ParseOperationRecursive(PyThunkObject *thunk, Pipeline *current) {
    assert(PyThunk_CheckExact(thunk));
    if (thunk->evaluated) {
        // thunk is already evaluated, this is an 'endpoint'
        return new ObjectOperation(thunk);
    } else {
        assert(thunk->operation != NULL);
        assert(thunk->operation->gencode.type <= 2);
        Operation *operations[OPTYPE_MAX_OPERATIONS];
        bool breaker = false;
        // thunk is not evaluated, branch out into the children
        for(int i = 0; i < thunk->operation->gencode.type; i++) {
            PyObject *param = thunk->operation->gencode.parameter[i];
            if (PyThunk_CheckExact(param) && 
                ((PyThunkObject*)param)->operation != NULL && 
                (((PyThunkObject*)param)->operation)->type == OPTYPE_FULLBREAKER) {
                breaker = true;
                break;
            }
        }
        for(int i = 0; i < thunk->operation->gencode.type; i++) {
            if (breaker) {
                Pipeline *child = CreatePipeline();
                operations[i] = new PipelineOperation(child);
                child->operation = ParseOperationRecursive((PyThunkObject*) thunk->operation->gencode.parameter[i], child);
                AddChild(current, child);
            } else {
                operations[i] = ParseOperationRecursive((PyThunkObject*) thunk->operation->gencode.parameter[i], current);
            }
        }
        return OperationFromArgs(thunk->operation, operations, thunk->operation->gencode.type);
    }
}

Pipeline*
ParsePipeline(PyThunkObject *thunk) {
    Pipeline *pipeline = CreatePipeline();
    pipeline->operation = ParseOperationRecursive(thunk, pipeline);
    return pipeline;
}

void initialize_parser(void) {
    import_array();
    import_umath();
}