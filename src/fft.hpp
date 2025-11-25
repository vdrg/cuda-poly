#pragma once

#include <complex>
#include <vector>

namespace cuda_poly::fft {

using Complex = std::complex<float>;

// Forward 1D FFT executed on the GPU.
std::vector<Complex> forward(const std::vector<Complex>& host_input);

// Inverse 1D FFT executed on the GPU.
std::vector<Complex> inverse(const std::vector<Complex>& host_input);

void forward_inplace(std::vector<Complex>& data);
void inverse_inplace(std::vector<Complex>& data);

}  // namespace cuda_poly::fft
