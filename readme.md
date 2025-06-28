# Neural Network C++ Library (Work in Progress)

This project is a simple feedforward neural network implemented in C++. It is designed to be used as a shared library, with basic Python integration via `ctypes`.

## Project Structure

- [`Layer.hpp`](Layer.hpp) / [`Layer.cpp`](Layer.cpp): Defines and implements the `Layer` class, representing a single neural network layer.
- [`Network.hpp`](Network.hpp) / [`Network.cpp`](Network.cpp): Defines and implements the `Network` class, representing a multi-layer neural network.
- [`Main.py`](Main.py): Python script that builds the C++ code and loads the shared library.
- [`Makefile`](Makefile): Build instructions for compiling the C++ code into a shared library.
- [`ai_data.json`](ai_data.json), [`data.json`](data.json): Placeholder data files.
- [`.vscode/`](.vscode/): VS Code configuration files.

## Building

To build the shared library, run:

```sh
make
```

This will produce `libmain.so`.

## Python Integration

The Python script [`Main.py`](Main.py) will:
1. Run `make` to build the shared library.
2. Load `libmain.so` using `ctypes`.

## Status

This project is a work in progress. Features such as training, serialization, and a Python API are not yet implemented.

## License

MIT License