

#include "parser.hpp"

static Pipeline* CreatePipeline(PyThunkObject *thunk) {
    Pipeline *pipeline = (Pipeline*) malloc(sizeof(Pipeline));
    pipeline->thunk = thunk;
    pipeline->children = NULL;
    return pipeline;
}

static 
Pipeline* ParsePipelineRecursive(PyThunkObject *thunk, Pipeline *current) {
    if (thunk->evaluated) {
        // thunk is already evaluated, this is an 'endpoint'
        return current;
    } else {
        assert(thunk->operation != NULL);
        assert(thunk->operation->gencode < 3);
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
            PyObject *param = thunk->operation->gencode.parameter[i];
            if (PyThunk_CheckExact(param)) {
                if (breaker) {
                    Pipeline *pipeline = CreatePipeline((PyThunkObject*) param);
                    ParsePipelineRecursive((PyThunkObject*) param, pipeline);

                    PipelineNode *node = (PipelineNode*) malloc(sizeof(PipelineNode));
                    node->child = pipeline;
                    node->next = current->children;
                    current->children = node;
                } else {
                    ParsePipelineRecursive((PyThunkObject*) param, current);
                }
            }
        }
        return current;
    }
}

Pipeline*
ParsePipeline(PyThunkObject *thunk) {
    Pipeline *pipeline = CreatePipeline(thunk);
    return ParsePipelineRecursive(thunk, pipeline);
}

void initialize_parser(void) {
    import_array();
    import_umath();
}