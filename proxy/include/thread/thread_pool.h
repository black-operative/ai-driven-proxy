#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

using std::vector;
using std::thread;
using std::queue;
using std::function;
using std::mutex;
using std::condition_variable;

class Thread_Pool {
    private:
        vector<thread>          _workers;
        queue<function<void()>> _tasks;
        mutex                   _mtx;
        condition_variable      _cv;
        bool                    _stop = false;

    public:
        explicit Thread_Pool(size_t n);
        ~Thread_Pool();

        void enqueue(function<void()> task);
};