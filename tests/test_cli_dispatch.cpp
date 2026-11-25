#include "cli/cli_main.h"

#include <iostream>

// `ninfer-serve` runs the CLI when it is started as `ninfer`, which is how a release archive carries
// one copy of the kernels (apps/serve/main.cpp). Only the executable's file name may decide that: a
// path that merely contains "ninfer", or the server's own name, must keep the server.

int ninfer_cli_main(int, char**) { return 0; }  // the header declares it; the dispatch rule needs no CLI

namespace {

int failures = 0;

void expect(const char* argv0, bool want) {
    const bool got = ninfer_started_as_cli(argv0);
    if (got == want) { return; }
    std::cerr << "argv[0] = " << (argv0 ? argv0 : "(null)") << ": expected " << (want ? "CLI" : "server")
              << " mode, got " << (got ? "CLI" : "server") << '\n';
    ++failures;
}

}  // namespace

int main() {
    // The CLI: the name `ninfer`, with or without a directory, an `.exe` suffix or a different case.
    expect("ninfer", true);
    expect("./ninfer", true);
    expect("/opt/ninfer-v100/ninfer", true);
    expect("ninfer.exe", true);
    expect("NINFER.EXE", true);
    expect("Ninfer", true);

    // The server, and every near miss, stay the server.
    expect("ninfer-serve", false);
    expect("./ninfer-serve", false);
    expect("ninfer-serve.exe", false);
    expect("/opt/ninfer/ninfer-serve", false);  // a directory named ninfer does not make it the CLI
    expect("ninfer_bench", false);
    expect("ninfer-cli", false);
    expect("my-ninfer", false);
    expect("ninfer2", false);
    expect("ninfer.sh", false);
    expect("", false);
    expect(nullptr, false);

    if (failures != 0) { return 1; }
    std::cout << "ninfer_cli_dispatch_test passed\n";
    return 0;
}
