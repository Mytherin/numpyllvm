
#include "initializers.hpp"
#include "scheduler.hpp"
#include "Include/blockingconcurrentqueue.h"


moodycamel::BlockingConcurrentQueue<Task*> queue;
bool running = true;

void 
ScheduleTask(Task *task) {
    queue.enqueue(task);
}

void
SchedulePipeline(Pipeline *pipeline) {
    PipelineNode *node = pipeline->children;
    while (node) {
        SchedulePipeline(node->child);
        node = node->next;
    }
    CompileTask* task = (CompileTask*) malloc(sizeof(CompileTask));
    task->type = TASKTYPE_COMPILE;
    task->pipeline = pipeline;
    ScheduleTask((Task*) task);
}

void
RunThread(void) {
    while (running) {
        Task *task;
        queue.wait_dequeue(task);
        if (task == NULL) {
            printf("Null.\n");
            continue;
        }
        //execute task
        switch(task->type) {
            case TASKTYPE_COMPILE:
                printf("Compile pipeline %s\n", ((CompileTask*)task)->pipeline->name);
                break;
            case TASKTYPE_EXECUTE:
                break;
            default:
                printf("Unrecognized task type.\n");
        }
        free(task);
    }
}

void 
initialize_scheduler(void) {
    import_array();
    import_umath();
}