/*
 * riemann_os.cu — FP32 ускорение для внутреннего цикла
 *
 * Ключевая идея: RTX 3050 имеет 32-64x больше FP32 ядер чем FP64.
 *
 * ИЗМЕНЕНИЯ v2:
 * - Theta вычисляется прямо в GPU ядре (больше не передаётся theta_f32.bin)
 * - Python передаёт только 2 числа: theta_base и step_theta
 * - Сохраняется anchor-reset каждые 80 блоков → delta_t ВСЕГДА < 65536
 * - cosf() точен для |x| < 65536 (CUDA гарантия)  
 *
 * Компиляция: nvcc -O3 riemann_os.cu -o riemann_os.exe
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdlib>

// --- Предрасчёт констант в FP32 ---
__global__ void precompute_fp32(int N, float* d_log_n_f32, float* d_inv_sqrt_n_f32) {
    int n = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (n <= N) {
        d_log_n_f32[n - 1]     = logf((float)n);
        d_inv_sqrt_n_f32[n - 1] = rsqrtf((float)n);
    }
}

// --- Конвертация Cn.bin из FP64 в FP32 ---
__global__ void convert_Cn_to_fp32(int N, double* d_Cn_f64, float* d_Cn_f32) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        d_Cn_f32[n] = (float)d_Cn_f64[n];
    }
}

// Главное ядро v2: theta вычисляется прямо на GPU
// phi = theta_k + Cn[n] - delta_t_k * log(n)
// theta_k = fmodf(theta_base + k*step_theta, 2pi)  -- вычисляется здесь!
// delta_t_k = k * step (т.к. Cn уже сдвинут к t_current при anchor reset)
// CUDA cosf() точен для |phi| < 65536 — anchor reset каждые 80 блоков гарантирует это
#define TWO_PI_F32 6.28318530718f

__global__ void scan_zeros_fp32(
    float theta_base, float step_theta,
    float delta_t_base, float step,
    int num_steps, int N,
    float* d_Cn_f32,
    float* d_log_n_f32,
    float* d_inv_sqrt_n_f32,
    float* results)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < num_steps) {
        // Theta вычисляется прямо здесь — не нужен theta_f32.bin!
        float th_k = theta_base + (float)k * step_theta;
        // fmodf чтобы держать в [0, 2pi)
        th_k = fmodf(th_k, TWO_PI_F32);
        if (th_k < 0.0f) th_k += TWO_PI_F32;

        float delta_t_k = delta_t_base + (float)k * step;
        float z = 0.0f;

        // Главный FP32 цикл с точным cosf (содержит аппаратный range reduction)
        for (int n = 0; n < N; ++n) {
            float phi = th_k + d_Cn_f32[n] - delta_t_k * d_log_n_f32[n];
            z = fmaf(cosf(phi), d_inv_sqrt_n_f32[n], z);
        }
        results[k] = 2.0f * z;
    }
}

void load_binary_f32(const char* filename, float* array, int count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) { std::cerr << "ERROR_FILE: " << filename << std::endl; exit(1); }
    file.read(reinterpret_cast<char*>(array), count * sizeof(float));
    file.close();
}

int main(int argc, char** argv) {
    // Args: N  t_anchor  t_current  theta_base  step_theta
    if (argc < 6) {
        std::cerr << "Usage: riemann_os.exe <N> <t_anchor> <t_current> <theta_base> <step_theta>" << std::endl;
        return 1;
    }
    int    N           = std::atoi(argv[1]);
    double t_anchor    = std::atof(argv[2]);
    double t_current   = std::atof(argv[3]);
    float  theta_base  = (float)std::atof(argv[4]);  // theta(t_current) mod 2pi
    float  step_theta  = (float)std::atof(argv[5]);  // delta * theta'(t_current)
    float  step        = 0.001f;
    int    num_steps   = 50000;
    float  delta_t_base = (float)(t_current - t_anchor); // всегда < 80*50=4000 (anchor reset)

    float*  h_Cn_f32  = new float[N];
    float*  h_results = new float[num_steps];

    load_binary_f32("Cn.bin", h_Cn_f32, N);

    float *d_Cn_f32, *d_log_n_f32, *d_inv_sqrt_n_f32, *d_results;
    cudaMalloc(&d_Cn_f32,         N         * sizeof(float));
    cudaMalloc(&d_log_n_f32,      N         * sizeof(float));
    cudaMalloc(&d_inv_sqrt_n_f32, N         * sizeof(float));
    cudaMalloc(&d_results,        num_steps * sizeof(float));

    cudaMemcpy(d_Cn_f32, h_Cn_f32, N * sizeof(float), cudaMemcpyHostToDevice);

    // Предрасчёт log(n) и rsqrt(n)
    int pre_t = 256, pre_b = (N + pre_t - 1) / pre_t;
    precompute_fp32<<<pre_b, pre_t>>>(N, d_log_n_f32, d_inv_sqrt_n_f32);
    cudaDeviceSynchronize();

    // Главный FP32 сканирующий запуск
    int threads = 256, blocks = (num_steps + threads - 1) / threads;
    scan_zeros_fp32<<<blocks, threads>>>(
        theta_base, step_theta,
        delta_t_base, step, num_steps, N,
        d_Cn_f32, d_log_n_f32, d_inv_sqrt_n_f32, d_results);
    cudaDeviceSynchronize();

    cudaMemcpy(h_results, d_results, num_steps * sizeof(float), cudaMemcpyDeviceToHost);

    int zeros_found = 0;
    for (int i = 0; i < num_steps - 1; ++i) {
        if (h_results[i] * h_results[i + 1] < 0.0f)
            zeros_found++;
    }
    std::cout << "[RESULT] ZEROS=" << zeros_found << std::endl;

    cudaFree(d_Cn_f32); cudaFree(d_log_n_f32);
    cudaFree(d_inv_sqrt_n_f32); cudaFree(d_results);
    delete[] h_Cn_f32; delete[] h_results;
    return 0;
}
