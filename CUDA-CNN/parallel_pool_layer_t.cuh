#pragma once
#include "layer_t.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_HEIGHT 32
#define TILE_WIDTH 32
__global__ void abstract_parallelized_activate(float* in, float* out, int width, int height, int depth, int filterSize, int stride);
#pragma pack(push, 1)
struct parallel_pool_layer_t
{
	layer_type type = layer_type::pool;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	uint16_t stride;
	uint16_t extend_filter;

	parallel_pool_layer_t( uint16_t stride, uint16_t extend_filter, tdsize in_size )
		:
		grads_in( in_size.x, in_size.y, in_size.z ),
		in( in_size.x, in_size.y, in_size.z ),
		out(
		(in_size.x - extend_filter) / stride + 1,
			(in_size.y - extend_filter) / stride + 1,
			in_size.z
		)

	{
		this->stride = stride;
		this->extend_filter = extend_filter;
		assert( (float( in_size.x - extend_filter ) / stride + 1)
				==
				((in_size.x - extend_filter) / stride + 1) );

		assert( (float( in_size.y - extend_filter ) / stride + 1)
				==
				((in_size.y - extend_filter) / stride + 1) );
	}

	point_t map_to_input( point_t out, int z )
	{
		out.x *= stride;
		out.y *= stride;
		out.z = z;
		return out;
	}

	struct range_t
	{
		int min_x, min_y, min_z;
		int max_x, max_y, max_z;
	};

	int normalize_range( float f, int max, bool lim_min )
	{
		if ( f <= 0 )
			return 0;
		max -= 1;
		if ( f >= max )
			return max;

		if ( lim_min ) // left side of inequality
			return ceil( f );
		else
			return floor( f );
	}

	range_t map_to_output( int x, int y )
	{
		float a = x;
		float b = y;
		return
		{
			normalize_range( (a - extend_filter + 1) / stride, out.size.x, true ),
			normalize_range( (b - extend_filter + 1) / stride, out.size.y, true ),
			0,
			normalize_range( a / stride, out.size.x, false ),
			normalize_range( b / stride, out.size.y, false ),
			(int)out.size.z - 1,
		};
	}

	void activate( tensor_t<float>& in )
	{
		this->in = in;
        float *deviceIn, *deviceOut;
        size_t numBytesIn = in.size.x * in.size.y * in.size.z * sizeof(float);
        size_t numBytesOut = out.size.x * out.size.y * out.size.z * sizeof(float);

        cudaMalloc(&deviceIn, numBytesIn);
        cudaMalloc(&deviceOut, numBytesOut);
        cudaMemcpy(deviceIn, in.data, numBytesIn, cudaMemcpyHostToDevice);

        // Define grid and block dimensions
        dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
        dim3 gridSize((in.size.x + TILE_WIDTH - 1) / TILE_WIDTH, (in.size.y + TILE_HEIGHT - 1) / TILE_HEIGHT, in.size.z);

        // Launch the kernel
        abstract_parallelized_activate<<<gridSize, blockSize>>>(deviceIn, deviceOut, in.size.x, in.size.y, in.size.z, extend_filter, stride);
        cudaDeviceSynchronize();

        // Copy results back to host
        cudaMemcpy(out.data, deviceOut, numBytesOut, cudaMemcpyDeviceToHost);

        // Free memory
        cudaFree(deviceIn);
        cudaFree(deviceOut);
    }

	void activate()
	{
		for ( int x = 0; x < out.size.x; x++ )
		{
			for ( int y = 0; y < out.size.y; y++ )
			{
				point_t mapped = map_to_input( { (uint16_t)x, (uint16_t)y, 0 }, 0 );
				for ( int z = 0; z < out.size.z; z++ )
				{
					float mval = -__FLT_MAX__;
					for ( int i = 0; i < extend_filter; i++ )
						for ( int j = 0; j < extend_filter; j++ )
						{
							float v = in( mapped.x + i, mapped.y + j, z );
							if ( v > mval )
								mval = v;
						}
					out( x, y, z ) = mval;
				}
			}
		}
	}

	void fix_weights()
	{

	}

		grads_in( x, y, z ) = sum_error;
	}
	void calc_grads( tensor_t<float>& grad_next_layer )
	{
		for ( int x = 0; x < in.size.x; x++ )
		{
			for ( int y = 0; y < in.size.y; y++ )
			{
				range_t rn = map_to_output( x, y );
				for ( int z = 0; z < in.size.z; z++ )
				{
					float sum_error = 0;
					for ( int i = rn.min_x; i <= rn.max_x; i++ )
					{
						int minx = i * stride;
						for ( int j = rn.min_y; j <= rn.max_y; j++ )
						{
							int miny = j * stride;

							int is_max = in( x, y, z ) == out( i, j, z ) ? 1 : 0;
							sum_error += is_max * grad_next_layer( i, j, z );
						}
					}
					grads_in( x, y, z ) = sum_error;
				}
			}
		}
	}
};
#pragma pack(pop)

	__global__
	void abstract_parallelized_activate()
	{
		int blocksPerDepth = ceil( float(out.size.x * out.size.y) / 1024 );

		__shared__ float ds_in[TILE_HEIGHT][TILE_WIDTH];
		// __shared__ float ds_out[ceil( out.size.x / blocksPerDepth )][ceil( out.size.y / blocksPerDepth)] = {0};
		
		int x = (blockIdx.x % blocksPerDepth) * blockDim.x + threadIdx.x;
		int y = (blockIdx.y % blocksPerDepth) * blockDim.y + threadIdx.y;
		int z = blockIdx.x / blocksPerDepth;
		
		for ( int i = 0; x < out.size.x; x++ )
		{
			for ( int j = 0; y < out.size.y; y++ )
			{
				ds_in(x + i, y + j) = in(x + i, y + j, z)
			}
		}
		__syncthreads();

		point_t mapped = map_to_input( { (uint16_t)x, (uint16_t)y, 0 }, 0 );

		float mval = -__FLT_MAX__;
		for ( int i = 0; i < extend_filter; i++ )
			for ( int j = 0; j < extend_filter; j++ )
			{
				float v = ds_in( mapped.x + i, mapped.y + j, z );
				if ( v > mval )
					mval = v;
			}

		out( x, y, z ) = mval;
	}