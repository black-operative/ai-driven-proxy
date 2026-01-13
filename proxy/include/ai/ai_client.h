#pragma once

#include "common.h"

class AI_Client {
    private:
        int sock_fd = -1;

    public:
        AI_Client(const char* host, int port);

        AI_Result classify(const Feature_Vector& f_vec);
};