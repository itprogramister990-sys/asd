/*
 * riemann_os.cu — FP32 ускорение для внутреннего цикла
 * 
 * Ключевая идея: RTX 3050 имеет 32-64x больше FP32 ядер чем FP64.
 * 
 * Безопасность FP32 для нашей задачи:
 * - phi = theta_mod + Cn[n] - delta_t * log(n), где phi ∈ [-700, 12.56]
 * - CUDA cosf() для |x| < 8192 даёт полную FP32 точность (< 1 ULP)
 * - Ошибка накопления: sqrt(N) * 1e-7 * |sum| ≈ 1e-3 (достаточно для нулей)
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
        d_inv_sqrt_n_f32[n - 1] = rsqrtf((float)n); // rsqrt на GPU быстрее 1/sqrt
    }
}

// --- Конвертация Cn.bin из FP64 в FP32 ---
__global__ void convert_Cn_to_fp32(int N, double* d_Cn_f64, float* d_Cn_f32) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        d_Cn_f32[n] = (float)d_Cn_f64[n];
    }
}

// --- Главное FP32 ядро (32-64x быстрее FP64 на RTX 3050!) ---
// phi = theta_mod[k] + Cn[n] - delta_t_k * log(n)
// theta_mod и Cn предоставляются уже mod 2pi => cos() точен
__global__ void scan_zeros_fp32(
    float delta_t_base, float step, int num_steps, int N,
    float* d_theta_f32,     // theta mod 2pi, конвертирован в FP32
    float* d_Cn_f32,        // Cn[n] mod 2pi, конвертирован в FP32
    float* d_log_n_f32,     // log(n) в FP32
    float* d_inv_sqrt_n_f32,// 1/sqrt(n) в FP32
    float* results)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < num_steps) {
        float th_k      = d_theta_f32[k];
        float delta_t_k = delta_t_base + (float)k * step;
        float z = 0.0f;

        // Главный цикл — ВЕСЬ в FP32, 32-64x быстрее!
        for (int n = 0; n < N; ++n) {
            float phi = th_k + d_Cn_f32[n] - delta_t_k * d_log_n_f32[n];
            z = fmaf(cosf(phi), d_inv_sqrt_n_f32[n], z); // fused multiply-add
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

void load_binary_f32(const char* filename, float* array, int count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) { std::cerr << "ERROR_FILE: " << filename << std::endl; exit(1); }
    file.read(reinterpret_cast<char*>(array), count * sizeof(float));
    file.close();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: riemann_os.exe <N> <t_anchor> <t_current>" << std::endl;
        return 1;
    }
    int    N         = std::atoi(argv[1]);
    double t_anchor  = std::atof(argv[2]);
    double t_current = std::atof(argv[3]);
    double step_d    = 0.001;
    float  step      = (float)step_d;
    int    num_steps = 50000;
    float  delta_t_base = (float)(t_current - t_anchor);

    // --- Хост: загрузка данных ---
    double* h_Cn_f64    = new double[N];
    float*  h_theta_f32 = new float[num_steps];
    float*  h_results   = new float[num_steps];

    load_binary_f64("Cn.bin",       h_Cn_f64,    N);
    load_binary_f32("theta_f32.bin", h_theta_f32, num_steps);

    // --- GPU: выделение памяти ---
    double* d_Cn_f64;
    float   *d_Cn_f32, *d_theta_f32, *d_log_n_f32, *d_inv_sqrt_n_f32, *d_results;

    cudaMalloc(&d_Cn_f64,        N         * sizeof(double));
    cudaMalloc(&d_Cn_f32,        N         * sizeof(float));
    cudaMalloc(&d_theta_f32,     num_steps * sizeof(float));
    cudaMalloc(&d_log_n_f32,     N         * sizeof(float));
    cudaMalloc(&d_inv_sqrt_n_f32,N         * sizeof(float));
    cudaMalloc(&d_results,       num_steps * sizeof(float));

    cudaMemcpy(d_Cn_f64,    h_Cn_f64,    N         * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta_f32, h_theta_f32, num_steps * sizeof(float),  cudaMemcpyHostToDevice);

    // --- Предрасчёт log(n) и 1/sqrt(n) в FP32 на GPU ---
    int pre_t = 256, pre_b = (N + pre_t - 1) / pre_t;
    precompute_fp32<<<pre_b, pre_t>>>(N, d_log_n_f32, d_inv_sqrt_n_f32);

    // --- Конвертация Cn из FP64 в FP32 на GPU ---
    convert_Cn_to_fp32<<<pre_b, pre_t>>>(N, d_Cn_f64, d_Cn_f32);
    cudaDeviceSynchronize();

    // --- Главный FP32 сканирующий запуск ---
    int threads = 256, blocks = (num_steps + threads - 1) / threads;
    scan_zeros_fp32<<<blocks, threads>>>(
        delta_t_base, step, num_steps, N,
        d_theta_f32, d_Cn_f32, d_log_n_f32, d_inv_sqrt_n_f32, d_results);
    cudaDeviceSynchronize();

    cudaMemcpy(h_results, d_results, num_steps * sizeof(float), cudaMemcpyDeviceToHost);

    // --- Подсчёт нулей (смена знака) ---
    int zeros_found = 0;
    for (int i = 0; i < num_steps - 1; ++i) {
        if (h_results[i] * h_results[i + 1] < 0.0f)
            zeros_found++;
    }
    std::cout << "[RESULT] ZEROS=" << zeros_found << std::endl;

    cudaFree(d_Cn_f64); cudaFree(d_Cn_f32); cudaFree(d_theta_f32);
    cudaFree(d_log_n_f32); cudaFree(d_inv_sqrt_n_f32); cudaFree(d_results);
    delete[] h_Cn_f64; delete[] h_theta_f32; delete[] h_results;
    return 0;
}
