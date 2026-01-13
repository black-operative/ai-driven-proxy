#pragma once

int create_listen_socket(int port);

int connect_upstream(const char* host, int port);
