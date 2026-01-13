#include <sys/epoll.h>
#include <sys/socket.h>
#include <chrono>

#include "net/proxy_server.h"
#include "ai/ai_client.h"
#include "ai/feature_extractor.h"
#include "ai/policy_engine.h"
#include "util/net_utils.h"
#include "util/csv_logger.h"

constexpr int MAX_EVENTS  = 1024;
constexpr int BUFFER_SIZE = 8192;

using namespace std::chrono;

Proxy_Server::Proxy_Server(
    int port, 
    int threads
) : pool(threads) {
    listen_fd = create_listen_socket(port);
    epoll_fd  = epoll_create1(0);

    epoll_event ev{};
    ev.events  = EPOLLIN;
    ev.data.fd = listen_fd;

    epoll_ctl(
        epoll_fd, 
        EPOLL_CTL_ADD, 
        listen_fd, 
        &ev
    );
}

void Proxy_Server::run() {
    epoll_event events[MAX_EVENTS];

    while (true) {
        int n = epoll_wait(
            epoll_fd, 
            events, 
            MAX_EVENTS, 
            -1
        );

        for (int i = 0; i < n; i++) {
            int fd = events[i].data.fd;

            if (fd == listen_fd) {
                int client_fd = accept4(
                    listen_fd,
                    nullptr,
                    nullptr,
                    SOCK_NONBLOCK
                );

                epoll_event ev {};
                ev.events  = EPOLLIN | EPOLLET;
                ev.data.fd = client_fd;

                epoll_ctl(
                    epoll_fd, 
                    EPOLL_CTL_ADD, 
                    client_fd, 
                    &ev
                );
            } else {
                epoll_ctl(
                    epoll_fd,
                    EPOLL_CTL_ADD,
                    fd,
                    nullptr 
                );

                pool.enqueue(
                    [this, fd] {
                        handle_client(fd);
                    }
                );
            }
        }
    }
}

void Proxy_Server::handle_client(int client_fd) {
    char buffer[BUFFER_SIZE];
    ssize_t n = recv(
        client_fd,
        buffer,
        BUFFER_SIZE,
        0
    );

    if (n <= 0) {
        close(client_fd);
        return;
    }

    auto t_start = high_resolution_clock::now();

    /* Feature Extraction */
    auto t0 = high_resolution_clock::now();
    Feature_Extractor extractor;
    auto features = extractor.extract(buffer, n);
    auto t1 = high_resolution_clock::now();
    
    /* AI Inference */
    auto t2 = high_resolution_clock::now();
    static thread_local AI_Client ai(AI_HOST, AI_PORT);
    auto result = ai.classify(features);
    auto t3 = high_resolution_clock::now();
    
    /* Policy Decision */
    auto t4 = high_resolution_clock::now();
    Policy_Engine policy;
    bool allowed = policy.allow(result);
    auto t5 = high_resolution_clock::now();
    
    auto t_end = high_resolution_clock::now();

    if (!allowed) {
        close(client_fd);
        return;
    }

    long feature_us = duration_cast<microseconds>(t1 - t1).count();
    long ai_us      = duration_cast<microseconds>(t3 - t2).count();
    long policy_us  = duration_cast<microseconds>(t5 - t4).count();
    long total_us   = duration_cast<microseconds>(t_end - t_start).count();
    long timestamp  = duration_cast<microseconds>(t_start.time_since_epoch()).count();

    
    CSV_Logger::instance().log(
        std::to_string(timestamp) + "," +
        std::to_string(client_fd) + "," +
        std::to_string(feature_us) + "," +
        std::to_string(ai_us) + "," +
        std::to_string(policy_us) + "," +
        std::to_string(total_us) + "," +
        (allowed ? "ALLOW" : "BLOCK") + "," +
        std::to_string(static_cast<int>(result.type)) + "," +
        std::to_string(result.confidence)
    );

    /* Upstream forwarding */
    int upstream = connect_upstream("127.0.0.1", 8080);
    send(upstream, buffer, n, 0);

    while ((n = recv(upstream, buffer, BUFFER_SIZE, 0)) > 0) {
        ssize_t sent = 0;
        while (sent < n) {
            ssize_t s = send(client_fd, buffer + sent, n - sent, 0);
            if (s <= 0) break;
            sent += s;
        }
    }

    close(upstream);
    close(client_fd);
}