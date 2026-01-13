#include <stdexcept>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

#include "util/net_utils.h"

int create_listen_socket(int port) {
    int fd = socket(
        AF_INET, 
        SOCK_STREAM | SOCK_NONBLOCK, 
        0
    );

    int opt = 1;
    setsockopt(
        fd, 
        SOL_SOCKET, 
        SO_REUSEADDR, 
        &opt, 
        sizeof(opt)
    );

    sockaddr_in addr {};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);

    if (
        bind(
            fd, 
            (sockaddr*) &addr, 
            sizeof(addr)
        ) < 0
    ) throw std::runtime_error("bind() failed");
    
    if (listen(fd, 1024) > 0) 
        throw std::runtime_error("listen() failed");
 
    return fd;
}

int connect_upstream(const char* host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    inet_pton(AF_INET, host, &addr.sin_addr);

    if (
        connect(
            fd, 
            (sockaddr*) &addr, 
            sizeof(addr)
        ) < 0
    ) throw std::runtime_error("Failed to connect to upstream client");
    
    return fd;
}
