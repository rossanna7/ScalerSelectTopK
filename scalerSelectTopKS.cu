// Includes and defines
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

#define N 10000000   // 10 Million
#define THREADS 1024 // block size
#define MAX_ITERATIONS 5
#define BINS 32
#define K 10000 // The TOP-K 10K values

#define CUDA_CHECK(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__global__ void initIndex(int *index)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        index[idx] = idx;
}

// --------------------------------------
// Min/Max Reduction Kernel
__global__ void findMinMax(float *input, float *min_val, float *max_val, int n)
{
    __shared__ float s_min[THREADS];
    __shared__ float s_max[THREADS];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    s_min[threadIdx.x] = (idx < n) ? input[idx] : FLT_MAX;
    s_max[threadIdx.x] = (idx < n) ? input[idx] : -FLT_MAX;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_min[threadIdx.x] = fminf(s_min[threadIdx.x], s_min[threadIdx.x + stride]);
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) // global memory //not gonna work properly for negative numbers
    {
        atomicMin((int *)min_val, __float_as_int(s_min[0])); // address, value
        atomicMax((int *)max_val, __float_as_int(s_max[0])); // chatgpt said: This is dangerous when s_min[0] or s_max[0] are negative, because bitwise comparison of __float_as_int() doesn't preserve float ordering across negatives. So your min/max might be wrong, and sometimes completely broken. Fix this using atomic operations only on positive values, or use a proper two-level reduction: per-block reduction into shared memory → host-side final reduction. Or: use Thrust's reduce if you want to go fast and easy.
    }
}

__global__ void ReverseScaling(float *d_min_val, float *d_max_val, int *threshold_bin, int iteration)
{
    float min_val = *d_min_val;
    float max_val = *d_max_val;
    float range = max_val - min_val;
    // float bin_width = range / (BINS - 1);

    if (iteration == 1)
    {
        max_val = min_val - 1.0f + powf(range, (float)(*threshold_bin + 1) / (BINS - 1));
        if (*threshold_bin != 0)
        {
            min_val = min_val - 1.0f + powf(range, (float)(*threshold_bin) / (BINS - 1));
        }
        // printf("reverse log min and max \n");
    }
    else
    {
        max_val = min_val + ((float)(*threshold_bin + 1) / (BINS - 1)) * range;
        if (*threshold_bin != 0)
        {
            min_val = min_val + ((float)*threshold_bin / (BINS - 1)) * range;
        }
        // printf("reverse normal scaling min and max \n");
    }

    // Write results back
    *d_min_val = min_val;
    *d_max_val = max_val;
}
// --------------------------------------

// --------------------------------------
// Logarithmic Scaling Kernel and Histogram Kernel using atomic operations
// --------------------------------------
__global__ void scaleToLogRange(float *input, uint8_t *scaled, float *d_min_val, float *d_max_val, int *histogram, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float range = *d_max_val - *d_min_val; // avoid div by zero
    range = (range == 0) ? range + 1e-5 : range;

    if (idx < n)
    {
        float scale = (logf(input[idx] - *d_min_val + 1) / logf(range)) * (BINS - 1); // logarithmic scaling

        // Clamp to [0, BINS-1] and convert to uint8_t
        uint8_t bin = (uint8_t)(fminf(fmaxf(scale, 0.0f), BINS - 1));
        scaled[idx] = bin;

        atomicAdd(&histogram[bin], 1); // fused histogram update
    }
}

__global__ void scale(float *input, uint8_t *scaled, int *index, float *d_min_val, float *d_max_val, int *histogram, int n)
{
    float min_val = *d_min_val;
    float max_val = *d_max_val;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n)
    {
        // float range = max_val - min_val + 1e-5;                      // avoid div by zero
        float range = max_val - min_val; // avoid div by zero
        range = (range == 0) ? range + 1e-5 : range;
        float scale = ((input[index[idx]] - min_val) / range) * (BINS - 1); // logarithmic scaling

        // Clamp to [0, BINS-1] and convert to uint8_t
        uint8_t bin = (uint8_t)(fminf(fmaxf(scale, 0.0f), BINS - 1));
        scaled[idx] = bin;

        atomicAdd(&histogram[bin], 1);
        // local_rank[idx] = atomicAdd(&histogram[bin], 1); // local ranking to be discovered
    }
}

__global__ void findThresholdBin(int *histogram, int missing_k, int *som, int *threshold_bin, int *threshold_bin_count)
{
    *threshold_bin = BINS; // fallback: not found
    *threshold_bin_count = 0;
    *som = 0;
    __shared__ int prefix_sum[BINS]; // Adjust to max BINS
    int tid = threadIdx.x;

    // Load histogram to shared memory
    if (tid < BINS)
        prefix_sum[tid] = histogram[tid];
    __syncthreads();

    // Inclusive scan
    for (int offset = 1; offset < blockDim.x; offset *= 2)
    {
        int val = 0;
        if (tid >= offset)
            val = prefix_sum[tid - offset];
        __syncthreads();
        prefix_sum[tid] += val;
        __syncthreads();
    }

    // One thread finds the threshold bin
    if (tid < BINS && prefix_sum[tid] >= missing_k)
    {
        // Use atomicMin to ensure only the first matching thread writes
        atomicMin(threshold_bin, tid);
        *threshold_bin_count = histogram[tid];
        *som = prefix_sum[tid];
    }
}

__global__ void filterByThresholdBin(uint8_t *scaled, int *result, int *index, int *size, int *threshold_bin, int offset, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n && scaled[idx] < *threshold_bin)
    {
        int pos = atomicAdd(size, 1);
        result[offset + pos] = index[idx]; // store the index of the scaled value
    }
}

// filterByK<<<blocks, THREADS>>>(d_data, d_scaled, d_result, d_index, d_new_size, missing_k, d_threshold_bin, sub_k, n);
// filterByThresholdBin<<<blocks, THREADS>>>(d_data, d_scaled, d_result, d_index, d_new_size, d_threshold_bin, sub_k, n); // old kernel calls
// filterByScaledBin<<<blocks, THREADS>>>(d_data, d_scaled, d_index, d_new_size, d_threshold_bin, n);

__global__ void filterByScaledBin(uint8_t *scaled, int *index, int *size, int *threshold_bin, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n && scaled[idx] == *threshold_bin)
    {
        int pos = atomicAdd(size, 1);
        index[pos] = index[idx]; // store the index of the scaled value
        // sub_data[pos] = input[index[idx]]; // store the value
    }
}

__global__ void filterByK(uint8_t *scaled, int *result, int *index, int *size, int missing_k, int *threshold_bin, int offset, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n && scaled[idx] <= *threshold_bin && *size <= missing_k)
    {
        int pos = atomicAdd(size, 1);
        result[offset + pos] = index[idx]; // store the index of the scaled value
    }
}

void printHistogram(int *histogram)
{
    int max_count = 0;
    for (int i = 0; i < BINS; ++i)
        if (histogram[i] > max_count)
            max_count = histogram[i];

    for (int i = 0; i < BINS; ++i)
    {
        int bar_length = (histogram[i] * 50) / max_count; // normalize to max 50 chars ---> You should guard against max_count == 0.
        printf("[%2d] %5d | ", i, histogram[i]);
        for (int j = 0; j < bar_length; ++j)
            printf("#");
        printf("\n");
    }
}

int main()
{
    int n = N;

    // First pass to count how many lines (floats) are in the file
    FILE *fp = fopen("input_data.txt", "r");
    if (!fp)
    {
        fprintf(stderr, "Error: Cannot open input_data.txt\n");
        return 1;
    }

    int count = 0;
    float temp;
    while (fscanf(fp, "%f", &temp) == 1)
    {
        count++;
    }
    fclose(fp);

    n = count; // Update n to the actual number of floats in the file
    // Now allocate memory based on count
    float *h_data = (float *)malloc(n * sizeof(float));

    // Second pass to actually load the values
    fp = fopen("input_data.txt", "r");
    if (!fp)
    {
        fprintf(stderr, "Error: Cannot reopen input_data.txt\n");
        return 1;
    }

    for (int i = 0; i < n; ++i)
    {
        fscanf(fp, "%f", &h_data[i]); // host data
    }
    fclose(fp);

    printf("=== Simple Scaler for size N = %d k = %d ===\n", n, K);

    // Allocate host memory()
    // float *h_data = (float *)malloc(n * sizeof(float));
    int *h_result = (int *)malloc(K * sizeof(int)); // host result
    // float *h_filtered = (float *)malloc(n * sizeof(float));     // filtered data
    // uint8_t *h_scaled = (uint8_t *)malloc(n * sizeof(uint8_t)); // data scalled representation
    // float *h_sub_data = (float *)malloc(n * sizeof(float));     // host result
    float min_val, max_val; // min and max values

    int *h_histogram = (int *)malloc(BINS * sizeof(int));                         // the scale histogram
    int h_new_size, h_som = 0, h_threshold_bin = BINS, h_threshold_bin_count = 0; // new size of the filtered data

    // Allocate GPU memory()
    float *d_data, *d_min_val, *d_max_val;
    uint8_t *d_scaled;
    int *d_histogram, *d_index, *d_result, *d_som, *d_threshold_bin, *d_threshold_bin_count;
    // float *d_filtered, *d_sub_data; // filtered data and sub data
    int *d_new_size; // new size of the filtered data

    // // Generate input()
    // srand(time(NULL));
    // for (int i = 0; i < N; ++i)
    //     h_data[i] = (float)(rand() % 10000) / 10.0f;

    CUDA_CHECK(cudaMalloc(&d_min_val, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max_val, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_histogram, BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_size, sizeof(int))); //--------------------------------------------------
    CUDA_CHECK(cudaMalloc(&d_threshold_bin, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_threshold_bin_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_som, sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));
    // CUDA_CHECK(cudaMalloc(&d_sub_data, n * sizeof(float))); // max size for simplicity
    CUDA_CHECK(cudaMalloc(&d_index, n * sizeof(int))); // max size for simplicity
    CUDA_CHECK(cudaMalloc(&d_result, K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scaled, n * sizeof(uint8_t)));
    // CUDA_CHECK(cudaMalloc(&d_filtered, n * sizeof(float))); // max size for simplicity

    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
    min_val = FLT_MAX;
    max_val = -FLT_MAX;
    CUDA_CHECK(cudaMemcpy(d_min_val, &min_val, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max_val, &max_val, sizeof(float), cudaMemcpyHostToDevice));

    int blocks;
    blocks = (n + THREADS - 1) / THREADS;
    initIndex<<<blocks, THREADS>>>(d_index);

    int iteration = 0;
    bool done = false;
    int previous_threshold_bin_count, sub_k = 0, missing_k = K; // threshold bin
    // float range;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaError_t err;

    // Before you launch the sequence of kernels:
    cudaEventRecord(start);

    while (iteration < MAX_ITERATIONS && !done)
    {
        CUDA_CHECK(cudaMemset(d_histogram, 0, BINS * sizeof(int)));

        blocks = (n + THREADS - 1) / THREADS; // Calculate number of blocks needed

        // Step 1: Find min and max values
        if (iteration == 0)
        {
            // // Reset min/max and histogram and blocks

            findMinMax<<<blocks, THREADS>>>(d_data, d_min_val, d_max_val, n);
            cudaDeviceSynchronize();
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("CUDA Kernel Error (findMinMax): %s\n", cudaGetErrorString(err));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        else
        {
            // Reverse formula to find the min and max values
            ReverseScaling<<<1, 1>>>(d_min_val, d_max_val, d_threshold_bin, iteration);
            cudaDeviceSynchronize();
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("CUDA Kernel Error (ReverseScaling): %s\n", cudaGetErrorString(err));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Step 2: Scale values and compute histogram in one fused kernel
        if (iteration == 0)
        {
            scaleToLogRange<<<blocks, THREADS>>>(d_data, d_scaled, d_min_val, d_max_val, d_histogram, n);
            // printf("scalled using log scaling \n");
        }
        else
        {
            scale<<<blocks, THREADS>>>(d_data, d_scaled, d_index, d_min_val, d_max_val, d_histogram, n);
            // printf("scalled using normal scalling \n");
        }

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA Kernel Error (scale): %s\n", cudaGetErrorString(err));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 3: Histogram already computed in the previous step
        previous_threshold_bin_count = h_threshold_bin_count;
        missing_k = K - sub_k;
        // printf("missing k = %d\n", missing_k);
        findThresholdBin<<<1, BINS>>>(d_histogram, missing_k, d_som, d_threshold_bin, d_threshold_bin_count);

        // CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, BINS * sizeof(int), cudaMemcpyDeviceToHost));

        // printHistogram(h_histogram);

        CUDA_CHECK(cudaMemcpy(&h_som, d_som, sizeof(int), cudaMemcpyDeviceToHost));                                 // copy back to cpu to verify end of iterations and update sub_k
        CUDA_CHECK(cudaMemcpy(&h_threshold_bin, d_threshold_bin, sizeof(int), cudaMemcpyDeviceToHost));             // to verify if != 0 if == 32
        CUDA_CHECK(cudaMemcpy(&h_threshold_bin_count, d_threshold_bin_count, sizeof(int), cudaMemcpyDeviceToHost)); // copy back to cpu to verify end of iterations and update sub_k and update n

        // printf("som = %d, threshold_bin = %d, count = %d\n", h_som, h_threshold_bin, h_threshold_bin_count);

        if (h_threshold_bin == 32)
        {
            printf("WARNING: Early exit. Unable to find K values in available bins.\n");
            printf("[Iter %d] Threshold K=%d not reached when accumulating from bin[0] up to bin[32] \n",
                   iteration + 1, K);
            done = true;
        }
        else
        {
            // printf("[Iter %d] Threshold K=%d reached starting from bin[0] up to bin[%d] with count = %d, with freq som = %d\n", iteration + 1, K, h_threshold_bin, h_threshold_bin_count, h_som);
            // printf("Filtering with threshold_bin = %d, current som = %d\n", h_threshold_bin, h_som);
            // printf("previous_threshold_bin_count = %d \n", previous_threshold_bin_count);

            // Step 4: Filter values based on the scaled bin
            if ((h_som == missing_k) || (previous_threshold_bin_count == h_threshold_bin_count)) // Condition to stop iteration
            {
                // printf("[Iter %d] No need to rescale \n", iteration + 1);
                done = true;
                h_new_size = 0;
                cudaMemcpy(d_new_size, &h_new_size, sizeof(int), cudaMemcpyHostToDevice);
                filterByK<<<blocks, THREADS>>>(d_scaled, d_result, d_index, d_new_size, missing_k, d_threshold_bin, sub_k, n);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("CUDA Kernel Error (filterByK): %s\n", cudaGetErrorString(err));
                }
                CUDA_CHECK(cudaDeviceSynchronize());

                sub_k += missing_k;
            }
            else // if (h_som >>> K)
            {
                if (h_threshold_bin != 0)
                {
                    // Filter input based on scaled values
                    h_new_size = 0;
                    cudaMemcpy(d_new_size, &h_new_size, sizeof(int), cudaMemcpyHostToDevice);
                    filterByThresholdBin<<<blocks, THREADS>>>(d_scaled, d_result, d_index, d_new_size, d_threshold_bin, sub_k, n);

                    err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        printf("CUDA Kernel Error (filterByThresholdBin): %s\n", cudaGetErrorString(err));
                    }
                    CUDA_CHECK(cudaDeviceSynchronize());

                    sub_k += (h_som - h_threshold_bin_count);

                    // printf("sub_k = %d, h_new_size = %d \n", sub_k, h_new_size);
                }

                h_new_size = 0;
                cudaMemcpy(d_new_size, &h_new_size, sizeof(int), cudaMemcpyHostToDevice);
                filterByScaledBin<<<blocks, THREADS>>>(d_scaled, d_index, d_new_size, d_threshold_bin, n);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("CUDA Kernel Error (filterByScaledBin): %s\n", cudaGetErrorString(err));
                }
                CUDA_CHECK(cudaDeviceSynchronize());

                // CUDA_CHECK(cudaMemcpy(&h_new_size, d_new_size, sizeof(int), cudaMemcpyDeviceToHost));
                // printf("h_new_size = %d \n", h_new_size);
                // printf("threshold_bin_count = %d \n", h_threshold_bin_count);

                // CUDA_CHECK(cudaMemcpy(d_data, d_filtered, h_threshold_bin_count * sizeof(float), cudaMemcpyDeviceToDevice));

                n = h_threshold_bin_count;
                // printf("[Iter %d] New size: %d\n", iteration + 1, n);
                // printf("[Iter %d] Rescaling with subset: scale <= bin[%d]\n", iteration + 1, h_threshold_bin);
            }
        }

        // printf("=== End of Iteration %d ===\n", iteration + 1);
        iteration++;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    CUDA_CHECK(cudaMemcpy(h_result, d_result, K * sizeof(int), cudaMemcpyDeviceToHost));
    // Print top K values (you can print more if needed)
    printf("\nFiltered Top-%d values %d (from smallest bins):\n", K, sub_k);

    int print_count = (sub_k < K) ? sub_k : K;
    int begin = print_count - 100;
    int idx = 244;
    for (int i = begin; i < print_count; ++i)
    {
        idx = h_result[i];
        printf("%d -> [%d] %f\n", i, idx, h_data[idx]);
    }

    // Print elapsed time
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Iteration %d: GPU total time = %f ms\n", iteration, ms);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_scaled);
    cudaFree(d_min_val);
    cudaFree(d_max_val);
    cudaFree(d_histogram);
    cudaFree(d_result);
    // cudaFree(d_filtered);
    cudaFree(d_new_size);
    cudaFree(d_threshold_bin);
    cudaFree(d_threshold_bin_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_result);
    free(h_data);
    // free(h_scaled);
    free(h_histogram);
    // free(h_filtered);

    return 0;
}
// Compile with: nvcc -o simpleScaler simpleScaler.cu
// Run with: ./simpleScaler

// Note: The code is designed to be run on a CUDA-capable GPU. Ensure that you have the necessary CUDA toolkit installed.