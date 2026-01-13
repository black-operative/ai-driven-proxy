#pragma once

#include "thread/thread_pool.h"

constexpr auto AI_HOST = "127.0.0.1";
constexpr auto AI_PORT = 5000;

class Proxy_Server {
    private:
        int         listen_fd;
        int         epoll_fd;
        Thread_Pool pool;

        void handle_client(int client_fd);
        
    public:
        Proxy_Server(int port, int threads);
        void run();
};
