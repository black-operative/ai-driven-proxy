#include <iostream>

#include "net/proxy_server.h"

constexpr auto PROXY_PORT = 4005;

int main() {
    Proxy_Server server(
        PROXY_PORT, 
        8
    );
    std::cout << "Proxy Server running on port " << PROXY_PORT << std::endl;
    
    server.run();
    
    return 0;
}