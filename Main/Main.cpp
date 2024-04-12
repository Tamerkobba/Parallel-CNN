#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <utility>
#include <vector>
#include <chrono>
#include "byteswap.h"
#include "CNN/cnn.h"

using namespace std;
using namespace std::chrono;

float cross_entropy_loss(const tensor_t<float>& predicted, const tensor_t<float>& expected) {
    float loss = 0.0f;
    int total_elements = predicted.size.x * predicted.size.y * predicted.size.z;

    for (int i = 0; i < total_elements; i++) {
        
        float p = max(predicted.data[i], 1e-6f);
        float q = expected.data[i];
        loss -= q * log(p) + (1 - q) * log(1 - p);
    }

    return loss / total_elements;  
}
float train(vector<layer_t*>& layers, tensor_t<float>& data, tensor_t<float>& expected) {
	for ( int i = 0; i < layers.size(); i++ )
	{
		if ( i == 0 )
			activate( layers[i], data );
		else
			activate( layers[i], layers[i - 1]->out );
	}

tensor_t<float> grads = layers.back()->out - expected;

	for ( int i = layers.size() - 1; i >= 0; i-- )
	{
		if ( i == layers.size() - 1 )
			calc_grads( layers[i], grads );
		else
			calc_grads( layers[i], layers[i + 1]->grads_in );
	}

	for ( int i = 0; i < layers.size(); i++ )
	{
		fix_weights( layers[i] );
	}

float loss = cross_entropy_loss(layers.back()->out, expected);  
    return loss;
}


void forward( vector<layer_t*>& layers, tensor_t<float>& data )
{
	for ( int i = 0; i < layers.size(); i++ )
	{
		if ( i == 0 )
			activate( layers[i], data );
		else
			activate( layers[i], layers[i - 1]->out );
	}
}

struct case_t
{
	tensor_t<float> data;
	tensor_t<float> out;
};

uint8_t* read_file( const char* szFile )
{
	ifstream file( szFile, ios::binary | ios::ate );
	streamsize size = file.tellg();
	file.seekg( 0, ios::beg );

	if ( size == -1 )
		return nullptr;

	uint8_t* buffer = new uint8_t[size];
	file.read( (char*)buffer, size );
	return buffer;
}

vector<pair<vector<case_t>, vector<case_t>>> read_test_cases(float validation_split = 0.2) {
    vector<case_t> cases;
    vector<case_t> training_cases;
    vector<case_t> validation_cases;

    uint8_t* train_image = read_file("train-images.idx3-ubyte");
    uint8_t* train_labels = read_file("train-labels.idx1-ubyte");

    if (!train_image || !train_labels) {
        cerr << "Error reading files!" << endl;
        return make_pair(vector<case_t>{}, vector<case_t>{});
    }

    uint32_t case_count = byteswap_uint32(*(uint32_t*)(train_image + 4));
    cases.reserve(case_count);

    for (int i = 0; i < case_count; i++) {
        case_t c { tensor_t<float>(28, 28, 1), tensor_t<float>(10, 1, 1) };

        uint8_t* img = train_image + 16 + i * (28 * 28);
        uint8_t* label = train_labels + 8 + i;

        for (int x = 0; x < 28; x++)
            for (int y = 0; y < 28; y++)
                c.data(x, y, 0) = img[x + y * 28] / 255.f;

        for (int b = 0; b < 10; b++)
            c.out(b, 0, 0) = (*label == b) ? 1.0f : 0.0f;

        cases.push_back(c);
    }

    delete[] train_image;
    delete[] train_labels;

    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(cases.begin(), cases.end(), g);

    
    size_t split_index = static_cast<size_t>(cases.size() * (1 - validation_split));
    training_cases.assign(cases.begin(), cases.begin() + split_index);
    validation_cases.assign(cases.begin() + split_index, cases.end());

    return {training_cases, validation_cases};
}
float validate(vector<layer_t*>& layers, const vector<case_t>& validation_cases) {
    float validation_loss = 0.0f;
    int count = 0;

    for (const case_t& v : validation_cases) {
        
        forward(layers, v.data);
        tensor_t<float>& predicted = layers.back()->out;  

        
        validation_loss += cross_entropy_loss(predicted, v.out);
        count++;
    }

    
    if (count > 0) {
        validation_loss /= count;
    }

    return validation_loss;
}
int main() {
    
    auto [training_cases, validation_cases] = read_test_cases(0.2);  

    
    vector<layer_t*> layers;
    conv_layer_t* layer1 = new conv_layer_t(1, 5, 8, training_cases[0].data.size);
    relu_layer_t* layer2 = new relu_layer_t(layer1->out.size);
    pool_layer_t* layer3 = new pool_layer_t(2, 2, layer2->out.size);
    fc_layer_t* layer4 = new fc_layer_t(layer3->out.size, 10);

    layers.push_back((layer_t*)layer1);
    layers.push_back((layer_t*)layer2);
    layers.push_back((layer_t*)layer3);
    layers.push_back((layer_t*)layer4);

    
    auto training_start = high_resolution_clock::now();
    for (case_t& t : training_cases) {
        float loss = train(layers, t.data, t.out);
        cout << "Training Loss: " << loss << endl;
    }
    auto training_end = high_resolution_clock::now();
    auto training_duration = duration_cast<milliseconds>(training_end - training_start);

    
    cout << "Training completed in " << training_duration.count() << " milliseconds." << endl;

    
    auto validation_start = high_resolution_clock::now();
    float validation_error = validate(layers, validation_cases);
    auto validation_end = high_resolution_clock::now();
    auto validation_duration = duration_cast<milliseconds>(validation_end - validation_start);

    
    cout << "Validation Loss: " << validation_error << endl;
    cout << "Validation completed in " << validation_duration.count() << " milliseconds." << endl;

    
    for (auto layer : layers) {
        delete layer;
    }

    return 0;
}