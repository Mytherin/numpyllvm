
#include "initializers.hpp"
#include "scheduler.hpp"
#include "thread.hpp"
#include "Include/blockingconcurrentqueue.h"
#include "compiler.hpp"

moodycamel::BlockingConcurrentQueue<Task*> queue;
bool running = true;
static ssize_t thread_index = 0;

Thread::Thread(void) : context(), builder(context) {
    index = thread_index++;
    functions = 0;
}

Thread* CreateThread() {
    return new Thread();
}

void DestroyThread(Thread *thread) {
    delete thread;
}

void 
ScheduleTask(Task *task) {
    queue.enqueue(task);
}

void
SchedulePipeline(Pipeline *pipeline) {
    CompileTask* task = (CompileTask*) malloc(sizeof(CompileTask));
    task->type = TASKTYPE_COMPILE;
    task->pipeline = pipeline;
    ScheduleTask((Task*) task);
}

static void
ScheduleFunction(JITFunction *jf) {
    // create inputs/outputs for the function
    jf->inputs = (void**) malloc(sizeof(void*) * jf->pipeline->inputData->objects.size());
    jf->outputs = (void**) malloc(sizeof(void*) * jf->pipeline->outputData->objects.size());
    // set inputs
    size_t i = 0;
    for(auto it = jf->pipeline->inputData->objects.begin(); it != jf->pipeline->inputData->objects.end(); it++, i++) {
        jf->inputs[i] = it->data;
    }
    i = 0;
    // allocate space for the results (if the space has not yet been allocated) and setup outputs
    for (auto it = jf->pipeline->outputData->objects.begin(); it != jf->pipeline->outputData->objects.end(); it++) {
        if (it->object->storage == NULL) {
            npy_intp elements[] = { (npy_intp) it->size };
            // todo: set to PyArray_EMPTY
            it->object->storage = (PyArrayObject*) PyArray_ZEROS(1, elements, it->type, 0);
            it->data = PyArray_DATA(it->object->storage);
        }
        jf->outputs[i] = it->data;
    }
    ExecuteTask *task = (ExecuteTask*) malloc(sizeof(ExecuteTask));
    task->type = TASKTYPE_EXECUTE;
    task->start = 0;
    task->end = jf->size;
    task->function = jf;
    jf->references++;
    ScheduleTask((Task*) task);
}

void inline 
DestroyTask(Task *task) {
    free(task);
}

void
RunThread(Thread *thread) {
    while (running) {
        Task *task;
        if (!queue.try_dequeue(task)) return;
        //queue.wait_dequeue(task);
        if (task == NULL) {
            printf("Null.\n");
            continue;
        }
        //execute task
        switch(task->type) {
            case TASKTYPE_COMPILE:
            {
                printf("Compile pipeline %s\n", ((CompileTask*)task)->pipeline->name);
                JITFunction *jf = CompilePipeline(((CompileTask*) task)->pipeline, thread);
                if (jf != NULL) {
                    ScheduleFunction(jf);
                }
                break;
            }
            case TASKTYPE_EXECUTE:
            {
                ExecuteTask *ex = (ExecuteTask*) task;
                ExecuteFunction(ex->function, ex->start, ex->end);
                JITFunctionDECREF(ex->function);
                break;
            }
            default:
                printf("Unrecognized task type.\n");
        }
        DestroyTask(task);
    }
}

void 
initialize_scheduler(void) {
    import_array();
    import_umath();
}
