#pragma once

#include <fstream>
#include <mutex>
#include <string>

using std::string;
using std::ofstream;
using std::mutex;

class CSV_Logger {
    private:
        CSV_Logger();
        ofstream file;
        mutex mtx;
    
    public: 
        static CSV_Logger& instance();
        void log(const string& line);
};
