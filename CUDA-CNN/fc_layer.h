#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <curand.h>

#define LEARNING_RATE 0.01
#define MOMENTUM 0.6
#define WEIGHT_DECAY 0.001

typedef enum{
  fc
}layer_type;

typedef struct{
  float *data;
  int x,y,z;
}tensor_t;

typedef struct {
  int x, y, z;
} point_t;


typedef struct {
    float gradientValue;
    float grad;      
    float oldgrad;   
} gradient_t;


tensor_t* initialise_tensor(int x, int y, int z){
  tensor_t* tensor = (tensor_t*)malloc(sizeof(tensor_t));
  if(!tensor){
    return NULL;
  }
  tensor -> data = (float*)malloc(x * y * z * sizeof(float));
  if(!tensor -> data){
    free(tensor);
    return NULL;
  }

  tensor -> x = x;
  tensor -> y = y;
  tensor ->z = z;

  return tensor;
}

void free_tensor ( tensor_t *tensor){
  free(tensor->data);
  free(tensor);
}

gradient_t* create_gradients (int size){
  gradient_t* gradients = (gradient_t*) malloc(size * sizeof(gradient_t));
  if(!gradients){
    return NULL;
  }
  return gradients;
}

void free_gradients(gradient_t* gradients){
  free(gradients);
}


typedef struct {
  layer_type type;
  tensor_t grads_in;
  tensor_t in;
  tensor_t out;
  tensor_t weights;

  float *input;
  gradient_t *gradients;
  
} fc_layer_t;


__device__ float activator_function(float x){
  return 1.0f / (1.0f + exp(-x));
}

__device__ float activator_derivative(float x) {
  float sig = activator_function(x);
  return sig * (1 - sig);
}

__global__ void init_weights_kernel(float *weights, int in_size, int out_size, float maxVal, curandState_t* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState_t state = states[idx];
    for (int i = idx; i < in_size * out_size; i += stride) {
        float random_value = curand_uniform(&state);
        weights[i] = 2.19722f / maxVal * random_value;
    }
}


__global__ void activate_kernel(fc_layer_t *layer, float *input, float *weights, float *output, int in_size_x, int in_size_y, int in_size_z, int out_size) {
    __shared__ float shared_weights[256]; 
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < out_size) {
        float inputValue = 0.0f;
        if (threadIdx.x < in_size_x * in_size_y * in_size_z) {
            shared_weights[threadIdx.x] = weights[threadIdx.x];
        }
        __syncthreads(); 
        for (int i = 0; i < in_size_x; i++) {
            for (int j = 0; j < in_size_y; j++) {
                for (int z = 0; z < in_size_z; z++) {
                    int m = z * (in_size_x * in_size_y) + j * in_size_x + i;
                    inputValue += input[m] * shared_weights[m * out_size + n];
                }
            }
        }
        output[n] = activator_function(inputValue);
    }
}
__global__ void fix_weights_kernel(float *weights, float *input, float *gradients, int in_size_x, int in_size_y, int in_size_z, int out_size) {
    __shared__ float shared_weights[256];
    __shared__ float shared_inputs[256];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (threadIdx.x < in_size_x * in_size_y * in_size_z) {
        shared_weights[threadIdx.x] = weights[threadIdx.x];
        shared_inputs[threadIdx.x] = input[threadIdx.x];
    }

    __syncthreads();

    for (int n = index; n < out_size; n += stride) {
        gradient_t grad;
        grad.gradientValue = gradients[n];
        float gradValue = grad.gradientValue + grad.gradientValue * MOMENTUM; 

        for (int i = 0; i < in_size_x; i++) {
            for (int j = 0; j < in_size_y; j++) {
                for (int z = 0; z < in_size_z; z++) {
                    int m = z * (in_size_x * in_size_y) + j * in_size_x + i;
                    float *w = &weights[m * out_size + n];
                    float inputValue = shared_inputs[m];
                    float weightValue = shared_weights[m * out_size + n];
                    float newWeight = weightValue - LEARNING_RATE * gradValue * inputValue + LEARNING_RATE * WEIGHT_DECAY * weightValue;
                    *w = newWeight;
                }
            }
        }
    }
}

__global__ void calc_grads_kernel(float *grads_in, float *grads_next_layer, float *input, float *weights, int in_size_x, int in_size_y, int in_size_z, int out_size) {
    __shared__ float shared_weights[256];
    __shared__ float shared_grads_next_layer[256];

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < out_size) {
        float grad = grads_next_layer[n] * activator_derivative(input[n]);

        if (threadIdx.x < in_size_x * in_size_y * in_size_z) {
            shared_weights[threadIdx.x] = weights[threadIdx.x];
            shared_grads_next_layer[threadIdx.x] = grads_next_layer[threadIdx.x];
        }

        __syncthreads();
        float update = 0.0f;
        for (int i = 0; i < in_size_x; i++) {
            for (int j = 0; j < in_size_y; j++) {
                for (int z = 0; z < in_size_z; z++) {
                    int m = z * (in_size_x * in_size_y) + j * in_size_x + i;
                    update += grad * shared_weights[m * out_size + n];
                }
            }
        }
        atomicAdd(&grads_in[n], update);
    }
}


fc_layer_t* create_fc(int in_size_x, int in_size_y, int in_size_z, int out_size) {
    fc_layer_t* layer = (fc_layer_t*)malloc(sizeof(fc_layer_t));
    if (!layer) {
        return NULL;
    }
    layer->type = fc;
    layer->grads_in = *initialise_tensor(in_size_x, in_size_y, in_size_z);
    layer->in = *initialise_tensor(in_size_x, in_size_y, in_size_z);
    layer->out = *initialise_tensor(out_size, 1, 1);
    layer->input = (float*)malloc(out_size * sizeof(float));

    if (!layer->input) {
        free_tensor(&layer->grads_in);
        free_tensor(&layer->in);
        free_tensor(&layer->out);
        free(layer);
        return NULL;
    }

    int in_size = in_size_x * in_size_y * in_size_z;
    layer->weights = *initialise_tensor(in_size, 1, out_size);
    float *d_weights;
    cudaError_t err;

    err = cudaMalloc(&d_weights, in_size * out_size * sizeof(float));
    if (err != cudaSuccess) {
        free_tensor(&layer->grads_in);
        free_tensor(&layer->in);
        free_tensor(&layer->out);
        free(layer->input);
        free(layer);
        return NULL;
    }

    curandState_t* d_states;
    err = cudaMalloc(&d_states, in_size * out_size * sizeof(curandState_t));
    if (err != cudaSuccess) {
        cudaFree(d_weights);
        free_tensor(&layer->grads_in);
        free_tensor(&layer->in);
        free_tensor(&layer->out);
        free(layer->input);
        free(layer);
        return NULL;
    }

    curand_init(clock(), d_states, in_size * out_size, NULL);

    int threadsPerBlock = 256;
    int blocksPerGrid = (in_size * out_size + threadsPerBlock - 1) / threadsPerBlock;
    float maxVal = in_size_x + in_size_y + in_size_z;
    init_weights_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_weights, in_size, out_size, maxVal, d_states);

    err = cudaMemcpy(layer->weights.data, d_weights, in_size * out_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_weights);
        cudaFree(d_states);
        free_tensor(&layer->grads_in);
        free_tensor(&layer->in);
        free_tensor(&layer->out);
        free(layer->input);
        free(layer);
        return NULL;
    }

    cudaFree(d_weights);
    cudaFree(d_states);

    layer->gradients = create_gradients(out_size);
    if (!layer->gradients) {
        free_tensor(&layer->grads_in);
        free_tensor(&layer->in);
        free_tensor(&layer->out);
        free(layer->input);
        free(layer);
        return NULL;
    }

    return layer;
}





int map(tensor_t* tensor, point_t d){
  return d.z * (tensor->x * tensor->y) + d.y * (tensor->x) + d.x; 
}


void activate(fc_layer_t *layer, tensor_t *in, tensor_t *weights, tensor_t *out) {
    float *d_input, *d_weights, *d_output;
    cudaMalloc(&d_input, in->x * in->y * in->z * sizeof(float));
    cudaMalloc(&d_weights, weights->x * weights->y * weights->z * sizeof(float));
    cudaMalloc(&d_output, out->x * sizeof(float));

    cudaMemcpy(d_input, in->data, in->x * in->y * in->z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights->data, weights->x * weights->y * weights->z * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (out->x + threadsPerBlock - 1) / threadsPerBlock;
    
    activate_kernel<<<blocksPerGrid, threadsPerBlock>>>(layer, d_input, d_weights, d_output, in->x, in->y, in->z, out->x);

    cudaMemcpy(out->data, d_output, out->x * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
}


void fix_weights(fc_layer_t *layer) {
    float *d_weights, *d_input, *d_gradients;
    cudaMalloc(&d_weights, layer->weights.x * layer->weights.y * layer->weights.z * sizeof(float));
    cudaMalloc(&d_input, layer->in.x * layer->in.y * layer->in.z * sizeof(float));
    cudaMalloc(&d_gradients, layer->out.x * sizeof(float));

    cudaMemcpy(d_weights, layer->weights.data, layer->weights.x * layer->weights.y * layer->weights.z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, layer->in.data, layer->in.x * layer->in.y * layer->in.z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradients, layer->gradients, layer->out.x * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (layer->out.x + threadsPerBlock - 1) / threadsPerBlock;
    fix_weights_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_weights, d_input, d_gradients, layer->in.x, layer->in.y, layer->in.z, layer->out.x);

    cudaMemcpy(layer->weights.data, d_weights, layer->weights.x * layer->weights.y * layer->weights.z * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_gradients);
}

void calc_grads(fc_layer_t *layer, tensor_t *grad_next_layer) {
    float *d_grads_in, *d_grads_next_layer, *d_input, *d_weights;
    cudaMalloc(&d_grads_in, layer->grads_in.x * layer->grads_in.y * layer->grads_in.z * sizeof(float));
    cudaMalloc(&d_grads_next_layer, grad_next_layer->x * sizeof(float));
    cudaMalloc(&d_input, layer->in.x * layer->in.y * layer->in.z * sizeof(float));
    cudaMalloc(&d_weights, layer->weights.x * layer->weights.y * layer->weights.z * sizeof(float));

    cudaMemcpy(d_grads_in, layer->grads_in.data, layer->grads_in.x * layer->grads_in.y * layer->grads_in.z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grads_next_layer, grad_next_layer->data, grad_next_layer->x * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, layer->in.data, layer->in.x * layer->in.y * layer->in.z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, layer->weights.data, layer->weights.x * layer->weights.y * layer->weights.z * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (layer->out.x + threadsPerBlock - 1) / threadsPerBlock;
    calc_grads_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_grads_in, d_grads_next_layer, d_input, d_weights, layer->in.x, layer->in.y, layer->in.z, layer->out.x);

    cudaMemcpy(layer->grads_in.data, d_grads_in, layer->grads_in.x * layer->grads_in.y * layer->grads_in.z * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_grads_in);
    cudaFree(d_grads_next_layer);
    cudaFree(d_input);
    cudaFree(d_weights);
}

static float update_weight( float w, gradient_t grad, float multp)
{
	float m = (grad.grad + grad.oldgrad * MOMENTUM);
    w -= LEARNING_RATE * m * multp + LEARNING_RATE * WEIGHT_DECAY * w;
    return w;
}

static void update_gradient( gradient_t *grad )
{
	grad->oldgrad = (grad->grad + grad->oldgrad * MOMENTUM);
}
