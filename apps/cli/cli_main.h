#pragma once

// The one-shot CLI's entry point, as a function. The `ninfer` executable calls it from a one-line
// main(); `ninfer-serve` calls it when it is started under the name `ninfer`, so a release archive
// can ship the CLI as a link to the server instead of a second copy of every kernel.
int ninfer_cli_main(int argc, char** argv);
