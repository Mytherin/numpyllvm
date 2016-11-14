

#include "parser.hpp"
#include "debug_printer.hpp"

static Pipeline* 
CreatePipeline() {
    Pipeline *pipeline = (Pipeline*) calloc(1, sizeof(Pipeline));
    pipeline->operation = NULL;
    pipeline->children = NULL;
    pipeline->parent = NULL;
    pipeline->inputData = new DataBlock();
    pipeline->outputData = new DataBlock();
    pipeline->name = getname("Pipe");
    pipeline->evaluated = false;
    pipeline->semaphore = NULL;
    semaphore_init(&pipeline->lock, 1);
    return pipeline;
}

void 
DestroyPipeline(Pipeline *pipeline) {
    //DestroyOperation(pipeline->operation);
    //DestroyChildren(pipeline->children);
    if (pipeline->semaphore) {
        semaphore_destroy(&pipeline->semaphore);
    }
    if (pipeline->lock) {
        semaphore_destroy(&pipeline->lock);
    }
    free(pipeline->name);
    free(pipeline);
}

static void 
AddChild(Pipeline *parent, Pipeline *child) {
    assert(!child->parent);
    child->parent = parent;
    PipelineNode *node = (PipelineNode*) calloc(1, sizeof(PipelineNode));
    node->child = child;
    node->next = NULL;
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
ParseOperationRecursive(PyThunkObject *thunk, Pipeline *current, DataSource *output_source) {
    assert(PyThunk_CheckExact(thunk));
    if (thunk->evaluated) {
        Operation *op = new ObjectOperation(thunk);
        assert(thunk->cardinality == PyArray_SIZE(thunk->storage));
        DataSource *source = new DataSource(thunk, PyArray_DATA(thunk->storage), PyArray_SIZE(thunk->storage), PyArray_TYPE(thunk->storage));
        current->inputData->objects.push_back(DataElement(op, source));
        // thunk is already evaluated, this is an 'endpoint'
        return op;
    } else {
        // thunk is not evaluated, there must be an operation associated with the thunk (if there is no operation, we don't know how to evaluate it)
        assert(thunk->operation != NULL);
        assert(thunk->operation->gencode.type <= 2);
        // branch out into the children of the operation (if any)
        Operation *operations[OPTYPE_MAX_OPERATIONS];
        ssize_t child_cardinalities[OPTYPE_MAX_OPERATIONS];
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
        if (thunk->operation->type == optype_fullbreaker) {
            breaker = true;
        } else {
            for(int i = 0; i < thunk->operation->gencode.type; i++) {
                PyObject *param = thunk->operation->gencode.parameter[i];
                if (PyThunk_CheckExact(param) && 
                    ((PyThunkObject*)param)->operation != NULL && 
                    ((((PyThunkObject*)param)->operation)->type == optype_fullbreaker || 
                    (((PyThunkObject*)param)->operation)->type == optype_parallelbreaker)) {
                    breaker = true;
                    break;
                }
            }
        }
        // now that we know if any of the child operations are breaking operations, we recursively create the operation tree
        for(int i = 0; i < thunk->operation->gencode.type; i++) {
            if (breaker) {
                // if any child operation is a breaking operation, we create a separate pipeline for each child
                Pipeline *child = CreatePipeline();
                DataSource *source = new DataSource(NULL, NULL, 0, 0);
                // we materialize after a breaking operation
                child->operation = ParseOperationRecursive((PyThunkObject*) thunk->operation->gencode.parameter[i], child, source);
                // As input to the current operation, we create a "PipelineOperation". This is a placeholder operation that expects to receive input as result of the pipeline operation
                operations[i] = new PipelineOperation(child, (PyThunkObject*) thunk->operation->gencode.parameter[i]);
                // as input data we add the child thunk, and set the data pointers to NULL for now (because they are not present yet, as the result is not yet computed)
                current->inputData->objects.push_back(DataElement(operations[i], source));
                AddChild(current, child);
            } else {
                // if there is no breaking operation, we simply continue with parsing the current pipeline recursively
                operations[i] = ParseOperationRecursive((PyThunkObject*) thunk->operation->gencode.parameter[i], current, NULL);
            }
            child_cardinalities[i] = ((PyThunkObject*) thunk->operation->gencode.parameter[i])->cardinality;
        }
        thunk->cardinality = thunk->cardinality_function(child_cardinalities);

        Operation *op = OperationFromArgs(thunk, thunk->operation, operations, thunk->operation->gencode.type);
        if (op->result_object || output_source) {
            op->materialize = true;

            if (output_source) {
                output_source->object = thunk;
                output_source->data = NULL;
                output_source->size = thunk->cardinality;
                output_source->type = NPY_INT64;
                Py_INCREF(thunk);
            } else {
                output_source = new DataSource(thunk, NULL, thunk->cardinality, NPY_INT64);
                Py_INCREF(thunk);
            }
            // if the current operation has a result_object, the result of this operation must be stored
            // alternatively, if we are forced to materialize because of a pipeline breaker, we also materialize
            current->outputData->objects.push_back(DataElement(op, output_source));
        }
        return op;
    }
}

Pipeline*
ParsePipeline(PyThunkObject *thunk) {
    Pipeline *pipeline = CreatePipeline();
    pipeline->operation = ParseOperationRecursive(thunk, pipeline, NULL);
    return pipeline;
}

void initialize_parser(void) {
    import_array();
    import_umath();
}
