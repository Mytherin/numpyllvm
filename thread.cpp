

#include "parser.hpp"
#include "scheduler.hpp"
#include "thread.hpp"
#include "llvmjit.hpp"

static ssize_t thread_index = 0;

Thread::Thread(void) : context(), builder(context) {
    jit = llvm::make_unique<LLVMJIT>();
    jit->getTargetMachine().setOptLevel(llvm::CodeGenOpt::Aggressive);

    index = thread_index++;
    functions = 0;
    thread = NULL;
}

Thread* CreateThread() {
    Thread *thread = new Thread();
    int res = pthread_create(&thread->thread, NULL, (void * (*)(void*))&RunThread, thread);
    if (res != 0) {
    	delete thread;
    	return NULL;
    }
    return thread;
}

void DestroyThread(Thread *thread) {
    delete thread;
}