
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
RunThread(void) {
    while (running) {
        Task *task;
        queue.wait_dequeue(task);
        if (task == NULL) {
            printf("Null.");
            continue;
        }
        //execute task
        switch(task->type) {
            case TASKTYPE_COMPILE:
                break;
            case TASKTYPE_EXECUTE:
                break;
            default:
                printf("Unrecognized task type.");
        }
        free(task);
    }
}

void 
initialize_scheduler(void) {
    import_array();
    import_umath();
}