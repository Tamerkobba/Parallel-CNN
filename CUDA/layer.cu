#include "layer_c.h"

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
    cudaMemcpyToSymbol(c_dt, &dt, sizeof(float), 0, cudaMemcpyHostToDevice);

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

__global__ void apply_step_function(float *input, float *output, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = step_function(input[idx]);
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
__global__ void fp_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5], float bias[6]) {
    int m = blockIdx.x; // One block per output feature map
    int x = threadIdx.x; // Thread along x dimension of output feature map
    int y = threadIdx.y; // Thread along y dimension of output feature map

    if (m < 6 && x < 24 && y < 24) {
        float sum = 0.0f;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                sum += input[x + i][y + j] * weight[m][i][j];
            }
        }
        preact[m][x][y] = sum + bias[m];
    }
}


__global__ void fp_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4], float bias[1]) {
    int m = blockIdx.z;  // Use z-dimension in grid to handle different feature maps
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate global x index
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // Calculate global y index

    if (m < 6 && x < 6 && y < 6) {
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {  // kernel width
            for (int j = 0; j < 4; ++j) {  // kernel height
               
                sum += weight[0][i][j] * input[m][x * 4 + i][y * 4 + j];
            }
        }
        // Add bias and store the result in the corresponding location in preact
        preact[m][x][y] = sum + bias[0];
    }
}

__global__ void fp_f(float input[6][6][6], float preact[10], float weight[10][6][6][6], float bias[10]) {
    int o = blockIdx.x * blockDim.x + threadIdx.x; // Index for the output dimension
    if (o < 10) {
        float sum = 0.0f;
        for (int j = 0; j < 6; ++j) { // First dimension of input
            for (int k = 0; k < 6; ++k) { // Second dimension of input
                for (int l = 0; l < 6; ++l) { // Third dimension of input
                    sum += weight[o][j][k][l] * input[j][k][l];
                }
            }
        }
        atomicAdd(&preact[o], sum); // Atomically add the sum to the output to avoid write conflicts
        preact[o] += bias[o]; // Add bias to each element
    }
}

__global__ void bp_f(float d_weight[10][6][6][6], float bias[10], float d_preact[10], float p_output[6][6][6]) {
    __shared__ float shared_d_weight[10][6][6][6];
    __shared__ float shared_p_output[6][6][6];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 10) {
        // Load d_weight and p_output into shared memory
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                for (int l = 0; l < 6; ++l) {
                    shared_d_weight[i][j][k][l] = d_weight[i][j][k][l];
                    shared_p_output[j][k][l] = p_output[j][k][l];
                }
            }
        }
        __syncthreads(); // Ensure all threads have loaded the data before proceeding

        // Update weights using shared memory
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                for (int l = 0; l < 6; ++l) {
                    d_weight[i][j][k][l] = d_preact[i] * shared_p_output[j][k][l];
                }
            }
        }
        // Update bias
        bias[i] += c_dt* d_preact[i];
    }
}


// Kernel launch parameters should be set according to the device's capabilities and the problem's needs
__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10]) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global index

    // Calculate indices for d_output
    int i2 = idx / 36; // First dimension index of d_output
    int i3 = (idx % 36) / 6; // Second dimension index of d_output
    int i4 = idx % 6; // Third dimension index of d_output

    if (i2 < 6 && i3 < 6 && i4 < 6) {
        float accumulation = 0.0f;

        // Accumulate contributions from each output neuron's weight and pre-activation gradient
        for (int i1 = 0; i1 < 10; ++i1) {
            accumulation += n_weight[i1][i2][i3][i4] * nd_preact[i1];
        }

        d_output[i2][i3][i4] = accumulation; // Store accumulated value
    }
}

__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6]) {
    // Calculate the global index, assuming a 3D block and grid configuration
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread is within bounds of the 6x6x6 array
    if (i < 6 && j < 6 && k < 6) {
        float o = step_function(preact[i][j][k]); // Assuming sigmoid is defined
        d_preact[i][j][k] = d_output[i][j][k] * o * (1 - o);
    }
}
__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6],float p_output[6][24][24]) {
    // Indices for the kernel weight to update
    int j = blockIdx.x * blockDim.x + threadIdx.x; // kernel width index
    int k = blockIdx.y * blockDim.y + threadIdx.y; // kernel height index

    if (j < 4 && k < 4) { // Ensure within bounds of the weight dimensions
        float accum = 0.0f;

        // Accumulate gradient contributions across all preact and corresponding output values
        for (int i4 = 0; i4 < 6; ++i4) { // over each output feature map dimension
            for (int i5 = 0; i5 < 6; ++i5) { // first dimension of output
                for (int i6 = 0; i6 < 6; ++i6) { // second dimension of output
                    accum += d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + j][i6 * 4 + k];
                }
            }
        }

        d_weight[0][j][k] = accum; // Single weight map
    }
}


__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6]) {
    // Define shared memory for reduction
    __shared__ float sharedSum[6 * 6 * 6];  // Allocate enough space for each thread in the block to hold one element

    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Global index

    if (idx < 6 * 6 * 6) {  // Check if within bounds
        int i = idx / 36;          // First dimension index
        int j = (idx % 36) / 6;    // Second dimension index
        int k = idx % 6;           // Third dimension index

        // Copy data to shared memory
        sharedSum[threadIdx.x] = d_preact[i][j][k];
    } else {
        sharedSum[threadIdx.x] = 0.0f;
    }

    __syncthreads();  // Synchronize threads to ensure all loads are complete

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedSum[threadIdx.x] += sharedSum[threadIdx.x + s];
        }
        __syncthreads();  // Ensure all additions are done before next step
    }

    // Update the bias in global memory from the thread 0 of each block
    if (threadIdx.x == 0) {
        atomicAdd(&bias[0], c_dt * sharedSum[0] / (6 * 6 * 6));
    }
}


__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6]) {
    int c = blockIdx.z;  // Channel
    int x = blockIdx.y * blockDim.y + threadIdx.y;  // Spatial y-coordinate
    int y = blockIdx.x * blockDim.x + threadIdx.x;  // Spatial x-coordinate

    if (x < 24 && y < 24 && c < 6) {
        float sum = 0.0f;

        // Determine which section of the nd_preact and n_weight affects this particular output
        // Reverse calculate the indices in nd_preact that affect this output
        for (int i2 = 0; i2 < 4; ++i2) {  // Kernel width
            for (int i3 = 0; i3 < 4; ++i3) {  // Kernel height
                int preact_x = (x - i2) / 4;
                int preact_y = (y - i3) / 4;

                // Check bounds and make sure we're considering valid mappings
                if (preact_x >= 0 && preact_x < 6 && preact_y >= 0 && preact_y < 6) {
                    sum += n_weight[0][i2][i3] * nd_preact[c][preact_x][preact_y];
                }
            }
        }

        // Only update the output if it's within bounds (this check might be redundant due to the initial condition)
        d_output[c][x][y] = sum;
    }
}

__global__ void bp_preact_c1(float d_preact[6][24][24],  float d_output[6][24][24], float preact[6][24][24]) {
    int i = blockIdx.z;  // Feature map index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Row index
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // Column index

    if (i < 6 && j < 24 && k < 24) {
        float s = 1.0f / (1.0f + expf(-preact[i][j][k]));  // Sigmoid function
        float ds = s * (1.0f - s);  // Derivative of the sigmoid function
        d_preact[i][j][k] = d_output[i][j][k] * ds;
    }
}


__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24],  float p_output[28][28]) {
    int filter = blockIdx.z; // Each block handles one filter
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Index for rows in weight tensor
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Index for columns in weight tensor

    if (i < 5 && j < 5) {
        float sum = 0.0;
        for (int x = 0; x < 24; ++x) {
            for (int y = 0; y < 24; ++y) {
                sum += d_preact[filter][x][y] * p_output[x + i][y + j];
            }
        }
        float d = 24.0f * 24.0f; // Normalization factor
        d_weight[filter][i][j] = sum / d;
    }
}

__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24]) {
    int feature = blockIdx.x; // Each block handles one feature map
    int idx = threadIdx.y * blockDim.x + threadIdx.x; // Flattened index for threads in a block

    __shared__ float partialSum[256]; // Shared memory for thread partial sums, assuming a block size of 256

    // Initialize shared memory
    if (idx < 256) {
        partialSum[idx] = 0;
    }
    __syncthreads();

    // Each thread computes a partial sum
    int stride = blockDim.x * blockDim.y;
    int start = idx;
    for (int i = start; i < 24 * 24; i += stride) {
        int row = i / 24;
        int col = i % 24;
        atomicAdd(&partialSum[idx], d_preact[feature][row][col]);
    }
    __syncthreads();

    // Reduce partial sums to a single sum per block
    if (idx == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x * blockDim.y; i++) {
            sum += partialSum[i];
        }
        float d = 24.0f * 24.0f;  // Normalization factor
        atomicAdd(&bias[feature], c_dt* sum / d);
    }
}
