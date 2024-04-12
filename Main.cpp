#include <cassert>
#include <cstdint>
#include <cstdio>
#include <random>
#include <cmath> 
#include <iostream>
#include <fstream>
#include <algorithm>
#include<chrono>
#include "byteswap.h"
#include "CNN/cnn.h"

using namespace std;
struct case_t
{
	tensor_t<float> data;
	tensor_t<float> out;
};
void split_data(vector<case_t>& cases, vector<case_t>& train_cases, vector<case_t>& val_cases, vector<case_t>& test_cases, float train_split = 0.7, float val_split = 0.15) {
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(cases.begin(), cases.end(), default_random_engine(seed));

    int total_cases = cases.size();
    int train_size = int(total_cases * train_split);
    int val_size = int(total_cases * val_split);

    train_cases.assign(cases.begin(), cases.begin() + train_size);
    val_cases.assign(cases.begin() + train_size, cases.begin() + train_size + val_size);
    test_cases.assign(cases.begin() + train_size + val_size, cases.end());
}

float train(vector<layer_t*>& layers, tensor_t<float>& data, tensor_t<float>& expected)
{
    for (int i = 0; i < layers.size(); i++)
    {
        if (i == 0)
            activate(layers[i], data);
        else
            activate(layers[i], layers[i - 1]->out);
    }

    tensor_t<float>& preds = layers.back()->out;
float max_val = *max_element(preds.data, preds.data + preds.size.x * preds.size.y * preds.size.z);
float sum = 0;
for (int i = 0; i < preds.size.x * preds.size.y * preds.size.z; i++)
    sum += exp(preds.data[i] - max_val);
for (int i = 0; i < preds.size.x * preds.size.y * preds.size.z; i++)
    preds.data[i] = exp(preds.data[i] - max_val) / sum;

// Compute Cross-Entropy Loss
float loss = 0;
for (int i = 0; i < preds.size.x * preds.size.y * preds.size.z; i++) {
    float p = preds.data[i];
    float y = expected.data[i];
    loss += -y * log(p + 1e-9); // Ensure p is never zero
}

    tensor_t<float> grads = preds - expected;

    for (int i = layers.size() - 1; i >= 0; i--)
    {
        if (i == layers.size() - 1)
            calc_grads(layers[i], grads);
        else
            calc_grads(layers[i], layers[i + 1]->grads_in);
    }

    for (int i = 0; i < layers.size(); i++)
    {
        fix_weights(layers[i]);
    }

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

vector<case_t> read_test_cases()
{
	vector<case_t> cases;

	uint8_t* train_image = read_file( "train-images.idx3-ubyte" );
	uint8_t* train_labels = read_file( "train-labels.idx1-ubyte" );

	uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );

	for ( int i = 0; i < case_count; i++ )
	{
		case_t c {tensor_t<float>( 28, 28, 1 ), tensor_t<float>( 10, 1, 1 )};

		uint8_t* img = train_image + 16 + i * (28 * 28);
		uint8_t* label = train_labels + 8 + i;

		for ( int x = 0; x < 28; x++ )
			for ( int y = 0; y < 28; y++ )
				c.data( x, y, 0 ) = img[x + y * 28] / 255.f;

		for ( int b = 0; b < 10; b++ )
			c.out( b, 0, 0 ) = *label == b ? 1.0f : 0.0f;

		cases.push_back( c );
	}
	delete[] train_image;
	delete[] train_labels;

	return cases;
}

int main()
{
	vector<case_t> cases = read_test_cases();
    
    vector<case_t> train_cases, val_cases, test_cases;
    split_data(cases, train_cases, val_cases, test_cases);
	    int num_epochs = 50;
  vector<layer_t*> layers;

	conv_layer_t * layer1 = new conv_layer_t( 1, 5, 8, cases[0].data.size );		// 28 * 28 * 1 -> 24 * 24 * 8
	relu_layer_t * layer2 = new relu_layer_t( layer1->out.size );
	pool_layer_t * layer3 = new pool_layer_t( 2, 2, layer2->out.size );				// 24 * 24 * 8 -> 12 * 12 * 8
	fc_layer_t * layer4 = new fc_layer_t(layer3->out.size, 10);					// 4 * 4 * 16 -> 10

	layers.push_back( (layer_t*)layer1 );
	layers.push_back( (layer_t*)layer2 );
	layers.push_back( (layer_t*)layer3 );
	layers.push_back( (layer_t*)layer4 );



 for (int epoch = 0; epoch < num_epochs; epoch++) {
        float train_loss = 0;
        for (auto& t : train_cases) {
            float xerr = train(layers, t.data, t.out);
            train_loss += xerr;
        }
        train_loss /= train_cases.size();
        cout << "Epoch " << epoch << " Training Loss: " << train_loss << endl;

        // Validation
        float val_loss = 0;
        for (auto& v : val_cases) {
            forward(layers, v.data);
            tensor_t<float>& out = layers.back()->out;
            float case_loss = 0;
            for (int i = 0; i < out.size.x * out.size.y * out.size.z; i++) {
                float p = out.data[i];
                float y = v.out.data[i];
                case_loss += -y * log(p + 1e-9);
            }
            val_loss += case_loss;
        }
        val_loss /= val_cases.size();
        cout << "Epoch " << epoch << " Validation Loss: " << val_loss << endl;
    }

    // Testing
    float test_loss = 0;
    for (auto& t : test_cases) {
        forward(layers, t.data);
        tensor_t<float>& out = layers.back()->out;
        float case_loss = 0;
        for (int i = 0; i < out.size.x * out.size.y * out.size.z; i++) {
            float p = out.data[i];
            float y = t.out.data[i];
            case_loss += -y * log(p + 1e-9);
        }
        test_loss += case_loss;
    }
    test_loss /= test_cases.size();
    cout << "Test Loss: " << test_loss << endl;

    return 0;
}
