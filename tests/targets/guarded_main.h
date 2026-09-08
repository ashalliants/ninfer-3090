#pragma once

// A real-model test that throws should say what happened.
//
// Four of these tests had a bare `int main()` with nothing catching, so any exception -- an
// artifact rejection, a CUDA failure during Engine construction, a std::filesystem error --
// unwound out of main and Windows terminated the process with 0xC0000409, printing nothing at all.
// CTest then reported `Exit code 0xc0000409` with no line to go on, which reads like memory
// corruption and is not: e06d7363 in a debugger is a C++ throw. Recovering the message meant
// attaching cdb.
//
// Use NINFER_GUARDED_TEST_MAIN(run) in place of `int main()`, with the body in a function
// returning int. Exit codes are unchanged, including 77 for a skip.

#include <exception>
#include <iostream>

#define NINFER_GUARDED_TEST_MAIN(run_function)                                                     \
    int main() {                                                                                   \
        try {                                                                                      \
            return (run_function)();                                                               \
        } catch (const std::exception& error) {                                                    \
            std::cerr << "FATAL: " << error.what() << std::endl;                                   \
            return 1;                                                                              \
        } catch (...) {                                                                            \
            std::cerr << "FATAL: unknown exception" << std::endl;                                  \
            return 1;                                                                              \
        }                                                                                          \
    }
