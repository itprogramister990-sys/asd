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
#include <string>

// --- Предрасчёт констант в FP32 ---
__global__ void precompute_fp32(int N, float* d_log_n_f32, float* d_inv_sqrt_n_f32) {
    int n = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (n <= N) {
        d_log_n_f32[n - 1]     = logf((float)n);
        d_inv_sqrt_n_f32[n - 1] = rsqrtf((float)n);
    }
}

// --- Сдвиг Cn.bin на начало блока в FP64 ---
__global__ void shift_Cn_fp64(int N, double delta_t_base, double* d_Cn_f64, double* d_log_n_f64, float* d_Cn_f32) {
    int n = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (n <= N) {
        double TWO_PI = 6.28318530717958647692;
        double shift = delta_t_base * d_log_n_f64[n - 1];
        double val = d_Cn_f64[n - 1] - shift;
        val = fmod(val, TWO_PI);
        if (val < 0.0) val += TWO_PI;
        d_Cn_f32[n - 1] = (float)val;
    }
}

// Главное ядро v2: theta вычисляется прямо на GPU
// phi = theta_k + Cn[n] - delta_t_k * log(n)
// theta_k = fmodf(theta_base + k*step_theta, 2pi)  -- вычисляется здесь!
// delta_t_k = k * step (т.к. Cn уже сдвинут к t_current при anchor reset)
// CUDA cosf() точен для |phi| < 65536 — anchor reset каждые 80 блоков гарантирует это
#define TWO_PI_F32 6.28318530718f

__global__ void scan_zeros_fp32(
    double theta_base_f64, double step_theta_f64,
    float step,
    int num_steps, int N,
    float* d_Cn_f32,
    float* d_log_n_f32,
    float* d_inv_sqrt_n_f32,
    float* results)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < num_steps) {
        // Theta вычисляется в FP64 для точности, затем кастуется в FP32
        double TWO_PI_F64 = 6.28318530717958647692;
        double th_k_f64 = theta_base_f64 + (double)k * step_theta_f64;
        th_k_f64 = fmod(th_k_f64, TWO_PI_F64);
        if (th_k_f64 < 0.0) th_k_f64 += TWO_PI_F64;
        float th_k = (float)th_k_f64;

        // Внимание: delta_t_base УЖЕ вычтена из d_Cn_f32 в ядре shift_Cn_fp64!
        // Поэтому здесь delta_t_k идет только от 0 до 50.0!
        float delta_t_k = (float)k * step;
        float z = 0.0f;

        // Главный FP32 цикл (очень короткий delta_t_k -> максимальная точность)
        for (int n = 0; n < N; ++n) {
            float phi = th_k + d_Cn_f32[n] - delta_t_k * d_log_n_f32[n];
            z = fmaf(cosf(phi), d_inv_sqrt_n_f32[n], z);
        }
        results[k] = 2.0f * z;
    }
}

void load_binary_f64(const char* filename, double* array, int count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) { std::cerr << "ERROR_FILE: " << filename << std::endl; exit(1); }
    file.read(reinterpret_cast<char*>(array), count * sizeof(double));
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
    double theta_base  = std::stod(argv[4]);
    double step_theta  = std::stod(argv[5]);
    float  step        = 0.001f;
    int    num_steps   = 50000;
    double delta_t_base = t_current - t_anchor; // FP64 precision for shift

    double* h_Cn_f64  = new double[N];
    float*  h_results = new float[num_steps];

    load_binary_f64("Cn.bin", h_Cn_f64, N);

    double *d_Cn_f64, *d_log_n_f64;
    float  *d_Cn_f32, *d_log_n_f32, *d_inv_sqrt_n_f32, *d_results;
    
    cudaMalloc(&d_Cn_f64,         N         * sizeof(double));
    cudaMalloc(&d_log_n_f64,      N         * sizeof(double));
    cudaMalloc(&d_Cn_f32,         N         * sizeof(float));
    cudaMalloc(&d_log_n_f32,      N         * sizeof(float));
    cudaMalloc(&d_inv_sqrt_n_f32, N         * sizeof(float));
    cudaMalloc(&d_results,        num_steps * sizeof(float));

    cudaMemcpy(d_Cn_f64, h_Cn_f64, N * sizeof(double), cudaMemcpyHostToDevice);

    // Предрасчёт log(n) для FP64 (чтобы сделать точный сдвиг)
    double* h_log_n_f64 = new double[N];
    for (int n = 1; n <= N; ++n) {
        h_log_n_f64[n - 1] = log((double)n);
    }
    cudaMemcpy(d_log_n_f64, h_log_n_f64, N * sizeof(double), cudaMemcpyHostToDevice);
    delete[] h_log_n_f64;

    // Предрасчёт log(n) и rsqrt(n) для FP32 (для быстрого цикла)
    int pre_t = 256, pre_b = (N + pre_t - 1) / pre_t;
    precompute_fp32<<<pre_b, pre_t>>>(N, d_log_n_f32, d_inv_sqrt_n_f32);
    cudaDeviceSynchronize();

    // Сдвиг Cn на дельту текущего блока в FP64 с сохранением в FP32
    shift_Cn_fp64<<<pre_b, pre_t>>>(N, delta_t_base, d_Cn_f64, d_log_n_f64, d_Cn_f32);
    cudaDeviceSynchronize();

    // Главный FP32 сканирующий запуск (delta_t_k теперь идёт только от 0 до 50)
    int threads = 256, blocks = (num_steps + threads - 1) / threads;
    scan_zeros_fp32<<<blocks, threads>>>(
        theta_base, step_theta,
        step, num_steps, N,
        d_Cn_f32, d_log_n_f32, d_inv_sqrt_n_f32, d_results);
    cudaDeviceSynchronize();

    cudaMemcpy(h_results, d_results, num_steps * sizeof(float), cudaMemcpyDeviceToHost);

    int zeros_found = 0;
    for (int i = 0; i < num_steps - 1; ++i) {
        if (h_results[i] * h_results[i + 1] < 0.0f)
            zeros_found++;
    }
    std::cout << "[RESULT] ZEROS=" << zeros_found << std::endl;

    cudaFree(d_Cn_f64); cudaFree(d_log_n_f64);
    cudaFree(d_Cn_f32); cudaFree(d_log_n_f32);
    cudaFree(d_inv_sqrt_n_f32); cudaFree(d_results);
    delete[] h_Cn_f64; delete[] h_results;
    return 0;
}
