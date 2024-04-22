#include <cstdlib>
#include <vector>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <cstring>

#ifndef LAYER_H
#define LAYER_H
#endif

#define LEARNING_RATE 0.01
const static float threshold = 1.0E-02f;

class Layer {
	public:
	int M, N, O;

	float *output;
	float *preact;

	float *bias;
	float *weight;

	float *d_output;
	float *d_preact;
	float *d_weight;

	Layer(int M, int N, int O);

	~Layer();

	void setOutput(float *data);
	void clear();
	void bp_clear();
};

// Constructor: initializes layer dimensions and allocates memory for neuron and weight data
Layer::Layer(int M, int N, int O) : M(M), N(N), O(O) {
    output = new float[O]();
    preact = new float[O]();
    bias = new float[N]();
    weight = new float[M * N]();
    d_output = new float[O]();
    d_preact = new float[O]();
    d_weight = new float[M * N]();

    // Initialize biases and weights with random values between -0.5 and 0.5
    for (int i = 0; i < N; ++i) {
        bias[i] = 0.5f - static_cast<float>(rand()) / RAND_MAX;

        for (int j = 0; j < M; ++j) {
            weight[i * M + j] = 0.5f - static_cast<float>(rand()) / RAND_MAX;
        }
    }
}


// Destructor: frees dynamically allocated memory
Layer::~Layer() {
    delete[] output;
    delete[] preact;
    delete[] bias;
    delete[] weight;
    delete[] d_output;
    delete[] d_preact;
    delete[] d_weight;
}

// Copies input data to the layer's output buffer
void Layer::setOutput(float *data) {
    memcpy(output, data, sizeof(float) * O);
}

// Clears the output and preactivation buffers (set to zero)
void Layer::clear() {
    memset(output, 0, sizeof(float) * O);
    memset(preact, 0, sizeof(float) * O);
}

// Clears the derivatives of the weights (set to zero), used in backpropagation
void Layer::bp_clear() {
    memset(d_weight, 0, sizeof(float) * M * N);
}

// Activation function, applies sigmoid function to a value
float activation_function(float v) {
    return 1 / (1 + exp(-v));
}

// Applies the activation function to each element in the input array
void apply_activation_function(float *input, float *output, int N) {
    for (int i = 0; i < N; ++i) {
        output[i] = activation_function(input[i]);
    }
}

// Calculates error for each output neuron based on expected output 'Y'
void makeError(float *err, float *output, unsigned int Y, int N) {
    for (int i = 0; i < N; ++i) {
        err[i] = (i == Y) ? 1.0f - output[i] : -output[i];
    }
}

void apply_grad(float *output, float *grad, int N) {
    for (int i = 0; i < N; ++i) {
        output[i] += LEARNING_RATE* grad[i];
    }
}
// Computes the preactivation values for a convolutional layer C1
void fp_preact_c1(const float input[28][28], float preact[6][24][24], const float weight[6][5][5]) {
    for (int m = 0; m < 6; ++m) {
        for (int x = 0; x < 24; ++x) {
            for (int y = 0; y < 24; ++y) {
                float sum = 0.0f;
                for (int i = 0; i < 5; ++i) {
                    for (int j = 0; j < 5; ++j) {
                        sum += input[x + i][y + j] * weight[m][i][j];
                    }
                }
                preact[m][x][y] += sum;
            }
        }
    }
}



// Adds biases to the preactivation values of the convolutional layer C1
void fp_bias_c1(float preact[6][24][24], const float bias[6]) {
    // Add bias to each element in the feature map
    for (int i = 0; i < 6; ++i) {
        for (int x = 0; x < 24; ++x) {
            for (int y = 0; y < 24; ++y) {
                preact[i][x][y] += bias[i];
            }
        }
    }
}
// Forward pass for the subsampling layer S1, applying weighted sum pooling
void fp_preact_s1(const float input[6][24][24], float preact[6][6][6], const float weight[1][4][4]) {
    for (int m = 0; m < 6; ++m) {          
        for (int x = 0; x < 6; ++x) {      
            for (int y = 0; y < 6; ++y) {
                float sum = 0.0f;
                for (int i = 0; i < 4; ++i) {  
                    for (int j = 0; j < 4; ++j) {  
                        
                        sum += weight[0][i][j] * input[m][x * 4 + i][y * 4 + j];
                    }
                }
              
                preact[m][x][y] += sum; 
            }
        }
    }
}

// Adds bias to the preactivation values of subsampling layer S1
void fp_bias_s1(float preact[6][6][6], const float bias[1]) {
    // Add bias to each output in the feature map
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                preact[i][j][k] += bias[0];
            }
        }
    }
}


// Computes preactivation for a fully connected layer F
void fp_preact_f(const float input[6][6][6], float preact[10], const float weight[10][6][6][6]) {
    // Initialize preactivation values
    for (int i = 0; i < 10; ++i) {
        preact[i] = 0;
        // Perform dot product operation
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                for (int l = 0; l < 6; ++l) {
                    preact[i] += weight[i][j][k][l] * input[j][k][l];
                }
            }
        }
    }
}


// Adds biases to the preactivation values of a fully connected layer F
void fp_bias_f(float preact[10], const float bias[10]) {
    // Add bias to each output neuron
    for (int i = 0; i < 10; ++i) {
        preact[i] += bias[i];
    }
}

// Backpropagation for weights of the fully connected layer F
void bp_weight_f(float d_weight[10][6][6][6], const float d_preact[10], const float p_output[6][6][6]) {
    // Calculate gradients for weights based on output errors and input activations
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                for (int l = 0; l < 6; ++l) {
                    d_weight[i][j][k][l] = d_preact[i] * p_output[j][k][l];
                }
            }
        }
    }
}

// Backpropagation for biases of the fully connected layer F
void bp_bias_f(float bias[10], const float d_preact[10]) {
    // Update biases based on the gradient of preactivation
    for (int i = 0; i < 10; ++i) {
        bias[i] += LEARNING_RATE * d_preact[i];
    }
}

// Backpropagation output gradient calculation for subsampling layer S1
void bp_output_s1(float d_output[6][6][6], const float n_weight[10][6][6][6], const float nd_preact[10]) {
    
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                d_output[i][j][k] = 0;
            }
        }
    }
    // Calculate output gradient contributions from subsequent layers
    for (int i1 = 0; i1 < 10; ++i1) { 
        for (int i2 = 0; i2 < 6; ++i2) { 
            for (int i3 = 0; i3 < 6; ++i3) { 
                for (int i4 = 0; i4 < 6; ++i4) { 
                    d_output[i2][i3][i4] += n_weight[i1][i2][i3][i4] * nd_preact[i1];
                }
            }
        }
    }
}


// Backpropagation preactivation gradient calculation for subsampling layer S1
void bp_preact_s1(float d_preact[6][6][6], const float d_output[6][6][6], const float preact[6][6][6]) {
    // Calculate gradient of preactivation using the derivative of the activation function
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                float o = activation_function(preact[i][j][k]);
                d_preact[i][j][k] = d_output[i][j][k] * o * (1 - o);
            }
        }
    }
}
// Backpropagation for weights of subsampling layer S1
void bp_weight_s1(float d_weight[1][4][4], const float d_preact[6][6][6], const float p_output[6][24][24]) {
    for (int i1 = 0; i1 < 1; ++i1) { 
        for (int i2 = 0; i2 < 4; ++i2) { 
            for (int i3 = 0; i3 < 4; ++i3) { 
                for (int i4 = 0; i4 < 6; ++i4) { 
                    for (int i5 = 0; i5 < 6; ++i5) { 
                        for (int i6 = 0; i6 < 6; ++i6) { 
                            
                            #pragma omp atomic
                            d_weight[i1][i2][i3] += d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3];
                        }
                    }
                }
            }
        }
    }
}

// Backpropagation for bias of subsampling layer S1
void bp_bias_s1(float bias[1], const float d_preact[6][6][6]) {
    // Calculate gradient contribution for bias and update
    float sum = 0.0f;
    int total_elements = 6 * 6 * 6;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                sum += d_preact[i][j][k];
            }
        }
    }
    bias[0] += LEARNING_RATE * sum / total_elements;
}

// Backpropagation output gradient calculation for convolutional layer C1
void bp_output_c1(float d_output[6][24][24], const float n_weight[1][4][4], const float nd_preact[6][6][6]) {
    for (int i1 = 0; i1 < 1; ++i1) { 
        for (int i2 = 0; i2 < 4; ++i2) { 
            for (int i3 = 0; i3 < 4; ++i3) { 
                for (int i4 = 0; i4 < 6; ++i4) { 
                    for (int i5 = 0; i5 < 6; ++i5) { 
                        for (int i6 = 0; i6 < 6; ++i6) { 
                            
                            int x = i5 * 4 + i2;
                            int y = i6 * 4 + i3;
                            d_output[i4][x][y] += n_weight[i1][i2][i3] * nd_preact[i4][i5][i6];
                        }
                    }
                }
            }
        }
    }
}

// Backpropagation preactivation gradient calculation for convolutional layer C1
void bp_preact_c1(float d_preact[6][24][24], const float d_output[6][24][24], const float preact[6][24][24]) {
    // Calculate gradient of preactivation using the derivative of the activation function
    auto sigmoid = [](float x) {
        return 1.0f / (1.0f + exp(-x));
    };
    auto sigmoid_derivative = [](float x) {
        float s = 1.0f / (1.0f + exp(-x));
        return s * (1 - s);
    };
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 24; ++j) {
            for (int k = 0; k < 24; ++k) {
                float o = sigmoid(preact[i][j][k]);
                d_preact[i][j][k] = d_output[i][j][k] * sigmoid_derivative(preact[i][j][k]);
            }
        }
    }
}
// Backpropagation for weights of convolutional layer C1
void bp_weight_c1(float d_weight[6][5][5], const float d_preact[6][24][24], const float p_output[28][28]) {
    float d = 24.0f * 24.0f;  
    for (int i1 = 0; i1 < 6; ++i1) {
        for (int i2 = 0; i2 < 5; ++i2) {
            for (int i3 = 0; i3 < 5; ++i3) {
                for (int i4 = 0; i4 < 24; ++i4) {
                    for (int i5 = 0; i5 < 24; ++i5) {
                        d_weight[i1][i2][i3] += d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d;
                    }
                }
            }
        }
    }
}

// Backpropagation for bias of convolutional layer C1
void bp_bias_c1(float bias[6], const float d_preact[6][24][24]) {
    // Calculate gradient contribution for bias and update
    float accumulators[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 24; ++j) {
            for (int k = 0; k < 24; ++k) {
                accumulators[i] += d_preact[i][j][k];
            }
        }
        bias[i] += LEARNING_RATE * accumulators[i] / (24.0 * 24.0);
    }
}