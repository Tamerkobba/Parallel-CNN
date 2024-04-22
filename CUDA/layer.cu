#include "layer.h"

// Constructor
Layer::Layer(int M, int N, int O)
{
	this->M = M;
	this->N = N;
	this->O = O;

	float h_bias[N];
	float h_weight[N][M];

	output = NULL;
	preact = NULL;
	bias   = NULL;
	weight = NULL;

	for (int i = 0; i < N; ++i) {
		h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
		/*h_bias[i] = 0.0f;*/

		for (int j = 0; j < M; ++j) {
			h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
			/*h_weight[i][j] = 0.05f;*/
		}
	}

	cudaMalloc(&output, sizeof(float) * O);
	cudaMalloc(&preact, sizeof(float) * O);

	cudaMalloc(&bias, sizeof(float) * N);

	cudaMalloc(&weight, sizeof(float) * M * N);

	cudaMalloc(&d_output, sizeof(float) * O);
	cudaMalloc(&d_preact, sizeof(float) * O);
	cudaMalloc(&d_weight, sizeof(float) * M * N);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// Destructor
Layer::~Layer()
{
	cudaFree(output);
	cudaFree(preact);

	cudaFree(bias);

	cudaFree(weight);

	cudaFree(d_output);
	cudaFree(d_preact);
	cudaFree(d_weight);
}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data)
{
	cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer::clear()
{
	cudaMemset(output, 0x00, sizeof(float) * O);
	cudaMemset(preact, 0x00, sizeof(float) * O);
}

void Layer::bp_clear()
{
	cudaMemset(d_output, 0x00, sizeof(float) * O);
	cudaMemset(d_preact, 0x00, sizeof(float) * O);
	cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}


__device__ float step_function(float v)
{
	return 1 / (1 + exp(-v));
}

__global__ void apply_step_function(float *input, float *output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = step_function(input[i]);
    }
}


__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
	}
}

__global__ void apply_grad(float *output, float *grad, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] += dt * grad[idx];
	}
}


__global__ void fp_preact_c1( float input[28][28], float preact[6][24][24], float weight[6][5][5]) {
    __shared__ float input_tile[TILE_WIDTH + FILTER_SIZE - 1][TILE_WIDTH + FILTER_SIZE - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = blockIdx.y * TILE_WIDTH + ty; // Output row index
    int col_o = blockIdx.x * TILE_WIDTH + tx; // Output col index

    int row_i = row_o - FILTER_RADIUS; // Input row index
    int col_i = col_o - FILTER_RADIUS; // Input col index

    // Load tile into shared memory
    for (int m = blockIdx.z; m < 6; m += gridDim.z) {  // Handle more than one filter per block, if necessary
        if (row_i >= 0 && row_i < 28 && col_i >= 0 && col_i < 28) {
            input_tile[ty][tx] = input[row_i][col_i];
        } else {
            input_tile[ty][tx] = 0.0f;
        }

        __syncthreads();  // Wait for all threads to load the tile elements

        // Convolution computation for the tile
        if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < 24 && col_o < 24) {
            float sum = 0.0f;
            for (int i = 0; i < FILTER_SIZE; ++i) {
                for (int j = 0; j < FILTER_SIZE; ++j) {
                    sum += input_tile[ty + i][tx + j] * weight[m][i][j];
                }
            }
            preact[m][row_o][col_o] = sum;
        }
        __syncthreads();  // Wait for all threads to compute the convolution before starting next iteration
    }
}

__global__ void fp_bias_c1(float *preact,  float *bias, int width, int height) {
    int i = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = (i * width * height) + (x * height) + y;
        preact[index] += bias[i];
    }
}

__global__ void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 4*4*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 4);
		const int i2 = ((idx /= 4	) % 4);
		const int i3 = ((idx /= 4	) % 6);
		const int i4 = ((idx /= 6	) % 6);
		const int i5 = ((idx /= 6	) % 6);

		atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
	}
}

__global__ void fp_bias_s1(float preact[6][6][6], float bias[1])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		preact[i1][i2][i3] += bias[0];
	}
}

__global__ void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
	}
}

__global__ void fp_bias_f(float preact[10], float bias[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		preact[idx] += bias[idx];
	}
}

__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
	}
}

__global__ void bp_bias_f(float bias[10], float d_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		bias[idx] += dt * d_preact[idx];
	}
}

__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
	}
}

__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*4*4*6*6*6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 4);
		const int i3 = ((idx /= 4	) % 4);
		const int i4 = ((idx /= 4	) % 6);
		const int i5 = ((idx /= 6	) % 6);
		const int i6 = ((idx /= 6	) % 6);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3]);
	}
}

__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		atomicAdd(&bias[0], dt * d_preact[i1][i2][i3] / d);
	}
}

__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6]) {
    int i4 = blockIdx.z;  // Index for the output depth dimension
    int i5 = blockIdx.y;  // Index for the output height dimension
    int i6 = blockIdx.x;  // Index for the output width dimension

    int i2 = threadIdx.y; // Index within the weight height dimension
    int i3 = threadIdx.x; // Index within the weight width dimension

    // Each thread computes one element of the d_output tenso
    int x = i5 * 4 + i2;
    int y = i6 * 4 + i3;

    if (i2 < 4 && i3 < 4 && i4 < 6 && x < 24 && y < 24) {
        atomicAdd(&d_output[i4][x][y], n_weight[0][i2][i3] * nd_preact[i4][i5][i6]);
    }
}

__global__ void bp_preact_c1(
    float d_preact[6][24][24],
  float d_output[6][24][24],
   float preact[6][24][24]
) {
    int i = blockIdx.z; // Index for the depth dimension
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Index for the height dimension
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Index for the width dimension

    if (i < 6 && j < 24 && k < 24) {
        d_preact[i][j][k] = d_output[i][j][k] * activation_derivative(preact[i][j][k]);
    }
}
__global__ void bp_weight_c1(
    float d_weight[6][5][5],
    const float d_preact[6][24][24],
    const float p_output[28][28]
) {
    int i1 = blockIdx.z;  // Filter index
    int i2 = blockIdx.y;  // Kernel row index
    int i3 = blockIdx.x;  // Kernel column index

    float update = 0.0f;
    float d = 24.0f * 24.0f;  

    if (i1 < 6 && i2 < 5 && i3 < 5) {
        // Accumulate updates in a local variable to reduce the number of atomicAdds
        for (int i4 = 0; i4 < 24; ++i4) {
            for (int i5 = 0; i5 < 24; ++i5) {
                update += d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d;
            }
        }
        // Atomic add to update the weight
        atomicAdd(&d_weight[i1][i2][i3], update);
    }
}

__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24]) {
    __shared__ float s_accumulators[6][blockDim.x]; // Shared memory to store partial sums

    int filter_index = blockIdx.x;
    int tid = threadIdx.x;

    // Compute partial sums within each block
    float accumulator = 0.0f;
    for (int i = 0; i < 24; ++i) {
        for (int j = 0; j < 24; ++j) {
            accumulator += d_preact[filter_index][i][j];
        }
    }
    s_accumulators[filter_index][tid] = accumulator;

    // Perform block-level reduction to compute total sum for each filter
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_accumulators[filter_index][tid] += s_accumulators[filter_index][tid + stride];
        }
        __syncthreads();
    }

    // Write the block-level sum to global memory
    if (tid == 0) {
        atomicAdd(&bias[filter_index], LEARNING_RATE * s_accumulators[filter_index][0] / (24.0 * 24.0));
    }
}