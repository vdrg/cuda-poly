#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fft.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(_core, m, py::mod_gil_not_used(), py::multiple_interpreters::per_interpreter_gil()) {
    m.doc() = R"pbdoc(
        Cuda poly
        -----------------------

        .. currentmodule:: cuda_poly

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    m.def("fft", &cuda_poly::fft::forward, R"pbdoc(
        Run a forward complex-to-complex FFT on the GPU.
    )pbdoc");

    m.def("ifft", &cuda_poly::fft::inverse, R"pbdoc(
        Run an inverse complex-to-complex FFT on the GPU.
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
