#include <mutex>

#include "thread/thread_pool.h"

using std::unique_lock;
using std::lock_guard;

Thread_Pool::Thread_Pool(size_t n) {
    for (size_t i = 0; i < n; i++) {
        _workers.emplace_back(
            [this] {
                while (true) {
                    function<void()> task;
                    {
                        unique_lock lock(_mtx);
                        _cv.wait(
                            lock, 
                            [this] {
                                return _stop || !_tasks.empty();
                            }
                        );

                        if (_stop && _tasks.empty()) return;

                        task = std::move(_tasks.front());
                        _tasks.pop();
                    }
                    task();
                }
            }
        );
    }
}

Thread_Pool::~Thread_Pool() {
    {
        lock_guard lock(_mtx);
        _stop = true;
    }
    _cv.notify_all();
    
    for (auto& worker : _workers)
        worker.join();
}

void Thread_Pool::enqueue(function<void()> task) {
    {
        lock_guard lock(_mtx);
        _tasks.push(std::move(task));
    }
    _cv.notify_one();
}
