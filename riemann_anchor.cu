#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdlib>

// Предрасчёт log(n) и 1/sqrt(n) один раз — убираем из горячего цикла
__global__ void precompute_constants(int N, double* d_log_n, double* d_inv_sqrt_n) {
    int n = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (n <= N) {
        d_log_n[n - 1] = log((double)n);
        d_inv_sqrt_n[n - 1] = 1.0 / sqrt((double)n);
    }
}

// ПРАВИЛЬНОЕ ядро:
// phi = theta_mod[k] + Cn[n] - delta_t * log(n)
// theta_mod[k] = theta(t_k) mod 2pi  — посчитан Python+mpmath (точно!)
// Cn[n]        = -t_anchor*log(n) mod 2pi — посчитан Python+mpmath (точно!)
// delta_t      <= 50 — маленькое число, double справляется!
// => cos() всегда получает маленький аргумент — нет потери точности!
__global__ void scan_zeros_correct(
    double delta_t_base, double step, int num_steps, int N,
    double* d_theta_mod, double* d_Cn,
    double* d_log_n, double* d_inv_sqrt_n,
    double* results)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < num_steps) {
        double th_k      = d_theta_mod[k];
        double delta_t_k = delta_t_base + k * step;  // <= 50
        double z = 0.0;
        for (int n = 0; n < N; ++n) {
            double phi = th_k + d_Cn[n] - delta_t_k * d_log_n[n];
            z += cos(phi) * d_inv_sqrt_n[n];
        }
        results[k] = 2.0 * z;
    }
}

void load_binary(const char* filename, double* array, int count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "ERROR_FILE: " << filename << std::endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(array), count * sizeof(double));
    file.close();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: riemann_anchor.exe <N> <t_anchor> <t_current>" << std::endl;
        return 1;
    }
    int    N         = std::atoi(argv[1]);
    double t_anchor  = std::atof(argv[2]);
    double t_current = std::atof(argv[3]);
    double step      = 0.001;
    int    num_steps = 50000;

    double delta_t_base = t_current - t_anchor;

    double* h_Cn        = new double[N];
    double* h_theta_mod = new double[num_steps];
    double* h_results   = new double[num_steps];

    load_binary("Cn.bin",    h_Cn,        N);
    load_binary("theta.bin", h_theta_mod, num_steps);

    double *d_Cn, *d_theta_mod, *d_log_n, *d_inv_sqrt_n, *d_results;
    cudaMalloc(&d_Cn,         N         * sizeof(double));
    cudaMalloc(&d_theta_mod,  num_steps * sizeof(double));
    cudaMalloc(&d_log_n,      N         * sizeof(double));
    cudaMalloc(&d_inv_sqrt_n, N         * sizeof(double));
    cudaMalloc(&d_results,    num_steps * sizeof(double));

    cudaMemcpy(d_Cn,        h_Cn,        N         * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta_mod, h_theta_mod, num_steps * sizeof(double), cudaMemcpyHostToDevice);

    // Предрасчёт констант один раз в VRAM
    int pre_t = 256, pre_b = (N + pre_t - 1) / pre_t;
    precompute_constants<<<pre_b, pre_t>>>(N, d_log_n, d_inv_sqrt_n);
    cudaDeviceSynchronize();

    // Основной сканирующий запуск
    int threads = 256, blocks = (num_steps + threads - 1) / threads;
    scan_zeros_correct<<<blocks, threads>>>(
        delta_t_base, step, num_steps, N,
        d_theta_mod, d_Cn, d_log_n, d_inv_sqrt_n, d_results);
    cudaDeviceSynchronize();

    cudaMemcpy(h_results, d_results, num_steps * sizeof(double), cudaMemcpyDeviceToHost);

    int zeros_found = 0;
    for (int i = 0; i < num_steps - 1; ++i) {
        if (h_results[i] * h_results[i + 1] < 0.0)
            zeros_found++;
    }
    std::cout << "[RESULT] ZEROS=" << zeros_found << std::endl;

    cudaFree(d_Cn); cudaFree(d_theta_mod);
    cudaFree(d_log_n); cudaFree(d_inv_sqrt_n); cudaFree(d_results);
    delete[] h_Cn; delete[] h_theta_mod; delete[] h_results;
    return 0;
}
