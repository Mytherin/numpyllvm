

#include "parser.hpp"
#include "debug_printer.hpp"

static Pipeline* CreatePipeline() {
    Pipeline *pipeline = (Pipeline*) malloc(sizeof(Pipeline));
    pipeline->operation = NULL;
    pipeline->children = NULL;
    pipeline->parent = NULL;
    pipeline->inputData = new DataBlock();
    pipeline->outputData = new DataBlock();
    pipeline->name = getname("Pipe");
    pipeline->size = 0;
    return pipeline;
}

void 
DestroyPipeline(Pipeline *pipeline) {
    //DestroyOperation(pipeline->operation);
    //DestroyChildren(pipeline->children);
    free(pipeline->name);
    free(pipeline);
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
OperationFromArgs(PyThunkObject *thunk, ThunkOperation *operation, Operation** operations, int op_count) {
    PyThunkObject *result_object = NULL;
    if (thunk->ob_refcnt > 1) {
        /* The thunk is not only used in this calculation, but in others as well, so we store the result */
        result_object = thunk;
    }
    switch(op_count) {
        case 0:
            return new NullaryOperation(result_object, operation);
        case 1:
            return new UnaryOperation(result_object, operation, operations[0]);
        case 2:
            return new BinaryOperation(result_object, operation, operations[0], operations[1]);
    }
    return NULL;
}

static Operation* 
ParseOperationRecursive(PyThunkObject *thunk, Pipeline *current) {
    assert(PyThunk_CheckExact(thunk));
    if (thunk->evaluated) {
        Operation *op = new ObjectOperation(thunk);
        if (current->size == 0) {
            current->size = PyArray_SIZE(thunk->storage);
        }
        assert(current->size == PyArray_SIZE(thunk->storage));
        current->inputData->objects.push_back(DataElement(thunk, op, PyArray_DATA(thunk->storage), PyArray_SIZE(thunk->storage), PyArray_TYPE(thunk->storage)));
        // thunk is already evaluated, this is an 'endpoint'
        return op;
    } else {
        // thunk is not evaluated, there must be an operation associated with the thunk (if there is no operation, we don't know how to evaluate it)
        assert(thunk->operation != NULL);
        assert(thunk->operation->gencode.type <= 2);
        // branch out into the children of the operation (if any)
        Operation *operations[OPTYPE_MAX_OPERATIONS];
        bool breaker = false;
        // if any of the children are pipeline breakers, we make each of the children a separate pipeline
        // example of this behavior:
        //    sort(a * b * c) * (d * e * f)
        // even though only 'sort' is a pipeline breaker, we create three separate pipelines here: 
        //   1: sort(a * b * c)
        //   2: (d * e * f)
        //   3: (Pipeline 1) * (Pipeline 2)
        // this is to allow for better parallelism, however, it might be unnecessary or a bad idea? haven't given it too much thought yet
        // the alternative is to only make the child that is a breaker a separate pipeline, this will result in less pipelines
        // example of that behavior:
        //    sort(a * b * c) * (d * e * f) 
        // Create two pipelines:
        //   1: sort(a * b * c)
        //   2: (Pipeline 1) * (d * e * f)
        // In this case, we cannot start with evaluating Pipeline 2 until Pipeline 1 is finished
        // this is probably worse, because pipeline breakers tend to not be fully parallelizable, thus Pipeline 1 likely does not use all the cores [effectively]
        // which means we have idle cores while waiting for Pipeline 1 -> these can be used to already compute Pipeline 2
        for(int i = 0; i < thunk->operation->gencode.type; i++) {
            PyObject *param = thunk->operation->gencode.parameter[i];
            if (PyThunk_CheckExact(param) && 
                ((PyThunkObject*)param)->operation != NULL && 
                (((PyThunkObject*)param)->operation)->type == OPTYPE_FULLBREAKER) {
                breaker = true;
                break;
            }
        }
        // now that we know if any of the child operations are breaking operations, we recursively create the operation tree
        for(int i = 0; i < thunk->operation->gencode.type; i++) {
            if (breaker) {
                // if any child operation is a breaking operation, we create a separate pipeline for each child
                Pipeline *child = CreatePipeline();
                child->operation = ParseOperationRecursive((PyThunkObject*) thunk->operation->gencode.parameter[i], child);
                // As input to the current operation, we create a "PipelineOperation". This is a placeholder operation that expects to receive input as result of the pipeline operation
                operations[i] = new PipelineOperation(child, (PyThunkObject*) thunk->operation->gencode.parameter[i]);
                // as input data we add the child thunk, and set the data pointers to NULL for now (because they are not present yet, as the result is not yet computed)
                current->inputData->objects.push_back(DataElement((PyThunkObject*) thunk->operation->gencode.parameter[i], operations[i], NULL, current->size, NPY_INT64));
                AddChild(current, child);
            } else {
                // if there is no breaking operation, we simply continue with parsing the current pipeline recursively
                operations[i] = ParseOperationRecursive((PyThunkObject*) thunk->operation->gencode.parameter[i], current);
            }
        }
        Operation *op = OperationFromArgs(thunk, thunk->operation, operations, thunk->operation->gencode.type);
        if (op->result_object) {
            // if the current operation has a result_object, the result of this operation must be stored thus we 
            current->outputData->objects.push_back(DataElement(thunk, op, NULL, current->size, NPY_INT64));
        }
        return op;
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
