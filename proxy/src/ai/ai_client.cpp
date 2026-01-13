#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdexcept>
#include <sys/socket.h>
#include <cstring>
#include <unistd.h>

#include "ai/ai_client.h"

AI_Client::AI_Client(const char* host, int port) {
    sock_fd = socket(
        AF_INET,
        SOCK_STREAM,
        0
    );
    if (sock_fd < 0) 
        throw std::runtime_error("Failed to create socket");
    
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    
    if (inet_pton(AF_INET, host, &addr.sin_addr) <= 0) {
        close(sock_fd);
        throw std::runtime_error("Invalid AI server address");
    }

    if (connect(sock_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock_fd);
        throw std::runtime_error("Cannot connect to AI server");
    }
}

AI_Result AI_Client::classify(const Feature_Vector& f_vec) {
    ssize_t sent = send(sock_fd, &f_vec, sizeof(f_vec), MSG_NOSIGNAL);
    if (sent != sizeof(f_vec)) {
        throw std::runtime_error("Failed to send feature vector");
    }

    AI_Result result{};
    ssize_t received = recv(sock_fd, &result, sizeof(result), MSG_WAITALL);
    if (received != sizeof(result)) {
        throw std::runtime_error("Failed to receive AI result");
    }

    return result;
}
