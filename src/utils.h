#include <iostream>
#include <chrono>
#include <iomanip>


struct TicTac
{
    /**
     * Tutorial here: https://akrzemi1.wordpress.com/2022/04/11/using-stdchrono/
     * 
     * "steady_clock is intented for handling time interval measurements"
    */

    using clock = std::chrono::steady_clock;
    // Duration represented in nanoseconds as 64 bit unsigned int 
    //   -> would take ~600 years before integer overflow
    using nanoseconds = std::chrono::duration<uint64_t, std::nano>;

    // member variables
    std::chrono::time_point<clock, nanoseconds> tstart;
    std::chrono::time_point<clock, nanoseconds> tend;
    double duration_ms;

    TicTac(): 
        tstart(clock::now()),
        tend(clock::now()),
        duration_ms(-1.0) 
    {}

    void tic()
    {
        tstart = clock::now();
    }

    double tac()
    {
        tend = clock::now();
        duration_ms = (tend - tstart).count()/1e6;
        return duration_ms;
    }

    void print_tac(const std::string& msg)
    {
        std::cout << std::setprecision(9) << msg << tac() << std::endl;
    }

};
