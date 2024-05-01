#include <cstdlib>
#include <vector>
#include <memory>

#ifndef LAYER_H
#define LAYER_H
#endif

const static float dt = 1.0E-01f;
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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

// Constructor
Layer::Layer(int M, int N, int O) : M(M), N(N), O(O) {
    output = new float[O]();
    preact = new float[O]();
    bias = new float[N]();
    weight = new float[M * N]();
    d_output = new float[O]();
    d_preact = new float[O]();
    d_weight = new float[M * N]();

    for (int i = 0; i < N; ++i) {
        bias[i] = 0.5f - static_cast<float>(rand()) / RAND_MAX;

        for (int j = 0; j < M; ++j) {
            weight[i * M + j] = 0.5f - static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

// Destructor
Layer::~Layer() {
    delete[] output;
    delete[] preact;
    delete[] bias;
    delete[] weight;
    delete[] d_output;
    delete[] d_preact;
    delete[] d_weight;
}

void Layer::setOutput(float *data) {
    memcpy(output, data, sizeof(float) * O);
}

void Layer::clear() {
    memset(output, 0, sizeof(float) * O);
    memset(preact, 0, sizeof(float) * O);
}

void Layer::bp_clear() {
    memset(d_weight, 0, sizeof(float) * M * N);
}

float step_function(float v) {
    return 1 / (1 + exp(-v));
}

void apply_step_function(float *input, float *output, int N) {
    for (int i = 0; i < N; ++i) {
        output[i] = step_function(input[i]);
    }
}

void makeError(float *err, float *output, unsigned int Y, int N) {
    for (int i = 0; i < N; ++i) {
        err[i] = (i == Y) ? 1.0f - output[i] : -output[i];
    }
}

void apply_grad(float *output, float *grad, int N) {
    for (int i = 0; i < N; ++i) {
        output[i] += dt * grad[i];
    }
}

void flattenTensor(float* flattenedTensor, float*** tensor, int numOutputs, int outputDim1, int outputDim2) {
    int idx = 0;
    for (int i = 0; i < numOutputs * outputDim1 * outputDim2; ++i) {
        flattenedTensor[i] = tensor[idx / (outputDim1 * outputDim2)][idx / outputDim2 % outputDim1][idx % outputDim2];
        ++idx;
    }
}

void buildTensor62424(float builtTensor[6][24][24], float flattenedTensor[3456]) {
    int numOutputs = 6;
    int outputDim1 = 24;
    int outputDim2 = 24;
    int idx = 0;
    for (int i = 0; i < numOutputs * outputDim1 * outputDim2; ++i) {
        builtTensor[idx / (outputDim1 * outputDim2)][idx / outputDim2 % outputDim1][idx % outputDim2] = flattenedTensor[i];
        ++idx;
    }
}

void buildTensor666(float builtTensor[6][6][6], float flattenedTensor[216]){ 
    int numOutputs = 6;
    int outputDim1 = 6;
    int outputDim2 = 6; 
    int idx = 0;
    for (int i = 0; i < numOutputs * outputDim1 * outputDim2; ++i) {
        builtTensor[idx / (outputDim1 * outputDim2)][idx / outputDim2 % outputDim1][idx % outputDim2] = flattenedTensor[i];
        ++idx;
    }
}

void buildTensor655(float builtTensor[6][5][5], float flattenedTensor[150]){ 
    int numOutputs = 6;
    int outputDim1 = 5;
    int outputDim2 = 5; 
    int idx = 0;
    for (int i = 0; i < numOutputs * outputDim1 * outputDim2; ++i) {
        builtTensor[idx / (outputDim1 * outputDim2)][idx / outputDim2 % outputDim1][idx % outputDim2] = flattenedTensor[i];
        ++idx;
    }
}

void buildTensor144(float builtTensor[1][4][4], float flattenedTensor[16]){ 
    int numOutputs = 1;
    int outputDim1 = 4;
    int outputDim2 = 4; 
    int idx = 0;
    for (int i = 0; i < numOutputs * outputDim1 * outputDim2; ++i) {
        builtTensor[idx / (outputDim1 * outputDim2)][idx / outputDim2 % outputDim1][idx % outputDim2] = flattenedTensor[i];
        ++idx;
    }
}

int Min(int a, int b){
    if (a>b)
        return b;
    return a;
}

void fp_preact_c1(const float input[28][28], float preact[6][24][24], const float weight[6][5][5]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    float placeHolder[6 * 24 * 24];
    float result[6 * 24 * 24];

    int S = 6 * 24 * 24;
    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), S);

    for (int i = start; i < end; ++i) {
        placeHolder[i] = 0;
        result[i] = 0;
    }
    
    for (int l = start; l < end; ++l) {
        int m = l / (24 * 24);
        int x = (l / 24) % 24;
        int y = l % 24;
        float sum = 0.0f;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                sum += input[x + i][y + j] * weight[m][i][j];
            }
        }
        placeHolder[l] += sum;
    }
    int size = end - start;
    MPI_Reduce(placeHolder, result, size, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (r==0){
        buildTensor62424(preact, result);
    }
}



void fp_bias_c1(float preact[6][24][24], const float bias[6]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 6 * 24 * 24;
    float placeHolder[S];
    float result[S];

    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), S);

    for (int l = 0; l < S; ++l) {
        placeHolder[l] = 0;
        result[l] = 0;
    }

    for (int l = start; l < end; ++l) {
        int m = l / (24 * 24);
        int x = (l / 24) % 24;
        int y = l % 24;
        placeHolder[l] = preact[m][x][y] + bias[m];
    }

    MPI_Reduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (r==0){
        buildTensor62424(preact, result);
    }
}

void fp_preact_s1(const float input[6][24][24], float preact[6][6][6], const float weight[1][4][4]) {
    
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 6 * 6 * 6;
    float placeHolder[S];
    float result[S];

    int it = ceil(6*6*6/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), S);

    for (int i = 0; i < S; ++i) {
        placeHolder[i] = 0;
        result[i] = 0;
    }
    
    for (int l = start; l < end; ++l) {
        int m = l / (6 * 6);
        int x = (l / 6) % 6;
        int y = l % 6;
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {  
            for (int j = 0; j < 4; ++j) {  
                sum += weight[0][i][j] * input[m][x * 4 + i][y * 4 + j];
            }
        }
        placeHolder[l] += sum;
    }

    MPI_Reduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (r==0){
        buildTensor666(preact, result);
    }
}

void fp_bias_s1(float preact[6][6][6], const float bias[1]) {

    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    
    int S = 6 * 6 * 6;
    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), S);

    float placeHolder[S];
    float result[S];
    for (int l = 0; l < S; ++l) {
        placeHolder[l] = 0;
        result[l]=0;
    }

    for (int l = start; l < end; ++l) {
        int m = l / (6 * 6);
        int j = (l / 6) % 6;
        int k = l % 6;
        placeHolder[l] = preact[m][j][k] + bias[m];
    }

    MPI_Reduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (r==0){
        buildTensor666(preact, result);
    }
}


void fp_preact_f(const float input[6][6][6], float preact[10], const float weight[10][6][6][6]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 10;
    float placeHolder[S];

    
    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), S);

    for (int i = 0; i < S; ++i) {
        placeHolder[i] = 0;
    }

    it = ceil(10 * 6 * 6 * 6/(float)p);
    start = r * it;
    end = Min(r * (it + 1), 10 * 6 * 6 * 6);
    
    for (int i = start; i < end; ++i) {
        int m = i / (6 * 6 * 6);
        int j = (i / (6 * 6)) % 6;
        int k = (i / 6) % 6;
        int l = i % 6;
        placeHolder[m] += weight[m][j][k][l] * input[j][k][l];
    }

    MPI_Reduce(placeHolder, preact, S, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
}


void fp_bias_f(float preact[10], const float bias[10]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);


    int S = 10;
    float placeHolder[S];

    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), S);

    for (int i = start; i < end; ++i) {
        placeHolder[i] = preact[i] + bias[i];
    }

    MPI_Reduce(placeHolder, preact, S, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
}


void bp_weight_f(float d_weight[10][6][6][6], const float d_preact[10], const float p_output[6][6][6]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 10 * 6 * 6 * 6;
    float placeHolder[S];
    float result[S];

    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), S);

    for (int i = 0; i < S; ++i) {
        placeHolder[i] = 0;
        result[i] = 0;
    }

    for (int i = start; i < end; ++i) {
        int m = i / (6 * 6 * 6);
        int j = (i / (6 * 6)) % 6;
        int k = (i / 6) % 6;
        int l = i % 6;
        placeHolder[i] = d_preact[m] * p_output[j][k][l];
    }

    MPI_Reduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (r==0){
        for (int i = 0; i < 9; i++)
            buildTensor666(d_weight[i], result);
    }
}

void bp_bias_f(float bias[10], const float d_preact[10]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 10;
    float placeHolder[S];

    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), S);

    for (int i = 0; i < S; ++i) {
        placeHolder[i] = 0;
    }

    for (int i = start; i < end; ++i) {
        placeHolder[i] = bias[i] + dt * d_preact[i];
    }

    MPI_Reduce(placeHolder, bias, S, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
}

void bp_output_s1(float d_output[6][6][6], const float n_weight[10][6][6][6], const float nd_preact[10]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    
    int S = 6 * 6 * 6;
    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), S);

    float placeHolder[S];
    float result[S];

    for (int l = start; l < end; ++l) {
        placeHolder[l] = 0;
        result[l] = 0;
    }

    it = ceil(10 * S/(float)p);
    start = r * it;
    end = Min(r * (it + 1), 10 * S);

    for (int i = start; i < end; ++i) {
        int i1 = i / (6 * 6 * 6);
        int i2 = (i / (6 * 6)) % 6;
        int i3 = (i / 6) % 6;
        int i4 = i % 6; 
        placeHolder[i % (6 * 6 * 6)] += n_weight[i1][i2][i3][i4] * nd_preact[i1];
    }

    MPI_Reduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (r==0){
        buildTensor666(d_output, result);
    }
}


void bp_preact_s1(float d_preact[6][6][6], const float d_output[6][6][6], const float preact[6][6][6]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    
    int S = 6 * 6 * 6;
    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), S);

    float placeHolder[S];
    float result[S];

    for (int l = 0; l < S; ++l) {
        placeHolder[l] = 0;
        result[l] = 0;
    }

    for (int l = start; l < end; ++l) {
        int m = l / (6 * 6);
        int j = (l / 6) % 6;
        int k = l % 6;
        float o = step_function(preact[m][j][k]);
        placeHolder[l] = d_output[m][j][k] * o * (1 - o);;
    }

     MPI_Reduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (r==0){
        buildTensor666(d_preact, result);
    }
}

void bp_weight_s1(float d_weight[1][4][4], const float d_preact[6][6][6], const float p_output[6][24][24]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    
    int S = 4 * 4;
    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), S);

    float placeHolder[S];
    float result[S];

    for (int i = 0; i < S; ++i) {
        placeHolder[i] = 0;
        result[i] = 0;
    }
    

    it = ceil(4 * 4 * 6 * 6 * 6/(float)p);
    start = r * it;
    end = Min(r * (it + 1), 4 * 4 * 6 * 6 * 6);

    for (int i = 0; i < 4 * 4 * 6 * 6 * 6; ++i) {
        int i2 = i / (4 * 6 * 6 * 6);
        int i3 = (i / (6 * 6 * 6)) % 4;
        int i4 = (i / (6 * 6)) % 6;
        int i5 = (i / 6) % 6;
        int i6 = i % 6;
        placeHolder[i2 * 4 + i3] += d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3];
    }

    MPI_Reduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (r==0){
        buildTensor144(d_weight, result);
    }
}

float bp_bias_s1(float bias[1], const float d_preact[6][6][6]) {
     int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    
    int S = 6 * 6 * 6;
    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), S);

    float placeHolder = 0.0;

    float sum = 0.0f;
    for (int l = start; l < end; ++l) {
        int m = l / (6 * 6);
        int j = (l / 6) % 6;
        int k = l % 6;
        sum += d_preact[m][j][k];
    }

    placeHolder += dt * sum / S;
    MPI_Reduce(&placeHolder, &bias[0], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    return bias[0];
}

void bp_output_c1(float d_output[6][24][24], const float n_weight[1][4][4], const float nd_preact[6][6][6]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 6 * 24 * 24;
    float placeHolder[S];
    float result[S];

    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), S);

    for (int i = start; i < end; ++i) {
        placeHolder[i] = 0;
        result[i] = 0;
    }

    it = ceil(4 * 4 * 6 * 6 * 6/(float)p);
    start = r * it;
    end = Min(r * (it + 1), 4 * 4 * 6 * 6 * 6);

    for (int i = start; i < end; ++i) {
        int i1 = i / (4 * 4 * 6 * 6 * 6);
        int i2 = (i / (4 * 6 * 6 * 6)) % 4;
        int i3 = (i / (6 * 6 * 6)) % 4;
        int i4 = (i / (6 * 6)) % 6;
        int i5 = (i / 6) % 6;
        int i6 = i % 6;

        int x = i5 * 4 + i2;
        int y = i6 * 4 + i3;
        placeHolder[i4 * 6 * 24 + x * 24 + y] += n_weight[i1][i2][i3] * nd_preact[i4][i5][i6];
    }

    MPI_Reduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (r==0){
        buildTensor62424(d_output, result);
    }
}

void bp_preact_c1(float d_preact[6][24][24], const float d_output[6][24][24], const float preact[6][24][24]) {
    auto sigmoid = [](float x) {
        return 1.0f / (1.0f + exp(-x));
    };

    auto sigmoid_derivative = [](float x) {
        float s = 1.0f / (1.0f + exp(-x));
        return s * (1 - s);
    };

     int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 6 * 24 * 24;
    float placeHolder[S];
    float result[S];

    int it = ceil(6 * 24 * 24/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), 6 * 24 * 24);

    for (int i =0 ; i < S; i++){
        placeHolder[i] = 0;
        result[i] = 0;
    }

    for (int i = start; i < end; ++i) {
        int m = i / (24 * 24);
        int j = (i / 24) % 24;
        int k = i % 24;
        float o = sigmoid(preact[i][j][k]);
        placeHolder[i] = d_output[m][j][k] * sigmoid_derivative(preact[m][j][k]);
    }

    MPI_Reduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (r==0){
        buildTensor62424(d_preact, result);
    }
}

void bp_weight_c1(float d_weight[6][5][5], const float d_preact[6][24][24], const float p_output[28][28]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 6 * 5 * 5;
    float placeHolder[S];
    float result[S];

    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), S);

    for (int i =0 ; i < S; i++){
        placeHolder[i] = 0;
        result[i] = 0;
    }

    float d = 24.0f * 24.0f;

    for (int i = start; i < end; ++i) {
        int i1 = i / (5 * 5 * 24 * 24);
        int remainder = i % (5 * 5 * 24 * 24);
        int i2 = remainder / (5 * 24 * 24);
        remainder %= (5 * 24 * 24);
        int i3 = remainder / (24 * 24);
        remainder %= (24 * 24);
        int i4 = remainder / 24;
        int i5 = remainder % 24;
        placeHolder[i/(24*24)] += d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d;
    }

    MPI_Reduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (r==0){
        buildTensor655(d_weight, result);
    }
}


void bp_bias_c1(float bias[6], const float d_preact[6][24][24]) {
    float accumulators[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    float d = 24.0f * 24.0f; 
    
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 6;
    float placeHolder[S];

    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(r * (it + 1), S);

    for (int i =0 ; i < S; i++){
        placeHolder[i] = 0;
    }

    for (int i = start; i < end; ++i) {
        int m = i / (24 * 24);
        int j = (i / 24) % 24;
        int k = i % 24;
        accumulators[i] += d_preact[m][j][k];
    }
    for (int i =0 ; i < S; i++){
        placeHolder[i] += dt * accumulators[i] / d;
    }
    
    MPI_Reduce(placeHolder, bias, S, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
}
