#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer_c.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

 float total_convolution_time = 0.0f;
 float total_pooling_time = 0.0f;
 float total_fully_connected_time = 0.0f;
 float total_gradient_time=0.0f;
static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer l_input = Layer(0, 0, 28*28);
static Layer l_c1 = Layer(5*5, 6, 24*24*6);
static Layer l_s1 = Layer(4*4, 1, 6*6*6);
static Layer l_f = Layer(6*6*6, 10, 10);

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static void forward_pass(double data[28][28]);
static void back_pass();

struct KernelConfig {
    dim3 blocks;
    dim3 threads;
};

static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main(int argc, const  char **argv)
{
	srand(time(NULL));

	CUresult err = cuInit(0);
	if (err != CUDA_SUCCESS) {
		fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
		return 1;
	}
    loaddata();
    learn();
    test();
  float millisecond =  total_convolution_time+total_pooling_time+total_fully_connected_time+total_gradient_time;

printf("Total Convolution Time: %f ms\n", total_convolution_time);
    printf("Total Pooling Time: %f ms\n", total_pooling_time);
    printf("Total Fully Connected Time: %f ms\n", total_fully_connected_time);
      printf("Total Time on applying gradients: %f ms\n", total_gradient_time);
    printf("Total Time on Computation GPU :%f ms \n",millisecond);
    return 0;
}
// Forward propagation of a single row in dataset
static void forward_pass(double data[28][28])
{
	float input[28][28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	l_input.clear();
	l_c1.clear();
	l_s1.clear();
	l_f.clear();
float milliseconds=0;
	clock_t start, end;



	l_input.setOutput((float *)input);
	KernelConfig configLayer1 = {dim3(6), dim3(24, 24)};
  start = clock();
  fp_c1<<<configLayer1.blocks, configLayer1.threads>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight,l_c1.bias);
	  apply_step_function<<<configLayer1.blocks, configLayer1.threads>>>(l_c1.preact, l_c1.output, l_c1.O);
   end = clock();
    milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    total_convolution_time += milliseconds;



		  // Pooling layer

		// Configuration for the subsampling layer
KernelConfig configSubsample1 = {
    dim3((6 + 2 - 1) / 2, (6 + 2 - 1) / 2, 6), // Grid size, rounding up if not a perfect multiple
    dim3(2, 2, 1)  // Block size
};
	KernelConfig configBiasS1 = {
    dim3(2, 2, 2), // Blocks
    dim3(3, 3, 3)  // Threads per block
};
start = clock();
	fp_s1<<<configSubsample1.blocks, configSubsample1.threads>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight,l_s1.bias);

	apply_step_function<<<configSubsample1.blocks, configSubsample1.threads>>>(l_s1.preact, l_s1.output, l_s1.O);
    end = clock();
    milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    total_pooling_time += milliseconds;

  

		 // Fully connected layer

	  KernelConfig configFullyConnected = {dim3(10), dim3(256)};
// Kernel launch
	start = clock();
fp_f<<<configFullyConnected.blocks, configFullyConnected.threads>>>((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight,l_f.bias);
apply_step_function<<<1, 10>>>(l_f.preact, l_f.output, l_f.O);
   end = clock();
    milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    total_fully_connected_time += milliseconds;
	


}

// Back propagation to update weights
static void back_pass()
{
	clock_t start, end;
float milliseconds=0;

int blockSize = 256;  // Optimal block size
int numOutputs = 10;
int gridSize = (numOutputs + blockSize - 1) / blockSize;
	start = clock();
bp_f<<<gridSize, blockSize>>>((float (*)[6][6][6])l_f.d_weight,l_f.bias, l_f.d_preact, (float (*)[6][6])l_s1.output);
   end = clock();
    milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    total_fully_connected_time += milliseconds;
start = clock();
  bp_output_s1<<<5,(216 + 5 - 1) / 5>>>((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
	dim3 threadsPerBlock_s1(6, 6, 6); // One thread for each element in the 6x6x6 block
dim3 numBlocks_s1(1, 1, 1);
	bp_preact_s1<<<numBlocks_s1, threadsPerBlock_s1>>>((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
	dim3 threadsPerBlock_w_s1(4, 4); // Perfect fit for 4x4 kernel weight dimensions
dim3 numBlocks_w_s1(1, 1);
	bp_weight_s1<<<numBlocks_w_s1, threadsPerBlock_w_s1>>>((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
	int totalThreads=6*6*6;
	int numBlocks = (totalThreads + 256 - 1);
	bp_bias_s1<<<numBlocks, 256>>>(l_s1.bias, (float (*)[6][6])l_s1.d_preact);
     end = clock();
    milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    total_pooling_time += milliseconds;
dim3 threadsPerBlock_output_c1(8,8 );  // 4x4 threads to handle the 4x4 weight matrix
dim3 numBlocks_output_c1((24 + threadsPerBlock_output_c1.x - 1) / threadsPerBlock_output_c1.x,
               (24 + threadsPerBlock_output_c1.y - 1) / threadsPerBlock_output_c1.y,
               6);
						start = clock();	 
	bp_output_c1<<<numBlocks_output_c1, threadsPerBlock_output_c1>>>((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
	
	dim3 threadsPerBlock_bp_preact_c1(8, 8); // This can be tuned based on the device capabilities
dim3 numBlocks_bp_preact_c1(
    (24 + threadsPerBlock_bp_preact_c1.x - 1) / threadsPerBlock_bp_preact_c1.x,
    (24 + threadsPerBlock_bp_preact_c1.y - 1) / threadsPerBlock_bp_preact_c1.y,
    6
);
  bp_preact_c1<<<numBlocks_bp_preact_c1, threadsPerBlock_bp_preact_c1>>>((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
	dim3 threadsPerBlock_weight_c1(5, 5); // Assuming the kernel size is small enough to fit a block
dim3 numBlocks_weight_c1(1, 1, 6); 
	bp_weight_c1<<<numBlocks_weight_c1, threadsPerBlock_weight_c1>>>((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
	dim3 blocks_bias_c1(6); // One block per feature map
dim3 threads_bias_c1(16, 16);
	bp_bias_c1<<<blocks_bias_c1, threads_bias_c1>>>(l_c1.bias, (float (*)[24][24])l_c1.d_preact);
end = clock();
    milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    total_convolution_time += milliseconds;
start=clock();
	apply_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
	apply_grad<<<64, 64>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
	apply_grad<<<64, 64>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);
	  end = clock();
    milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    total_gradient_time += milliseconds;

}

static void learn()
{
	static cublasHandle_t blas;
	cublasCreate(&blas);

	float err;
	int iter = 1;


	fprintf(stdout ,"Learning\n");

	while (iter < 0 || iter-- > 0) {
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i) {
			float tmp_err;

			 forward_pass(train_set[i].data);

			l_f.bp_clear();
			l_s1.bp_clear();
			l_c1.bp_clear();

			// Euclid distance of train_set[i]
			makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
			cublasSnrm2(blas, 10, l_f.d_preact, 1, &tmp_err);
			err += tmp_err;

	back_pass();
		}

		err /= train_cnt;
		fprintf(stdout, "error: %e\n", err);

		if (err < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}

	}
	
}


// Returns label of given data (0-9)
static unsigned int classify(double data[28][28])
{
	float res[10];

	forward_pass(data);

	unsigned int max = 0;

	cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0);
}
