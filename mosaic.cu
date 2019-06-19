/* @file    mosaic.c
* @Author	Ziling Li
* @brief	COM6521 - Assignment 1
* @deadline	18 March 2019
*/

// Include standard library
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include "device_functions.h"

// Definition 
#define FAILURE 0
#define SUCCESS !FAILURE
#define USER_NAME "acv18zl"	

// Define a struct named pixel to contains rgb values as characters
typedef struct single_pixel {
	unsigned char red;
	unsigned char green;
	unsigned char blue;
} pixel;

typedef struct mosaic_pixel {
	unsigned int red;
	unsigned int green;
	unsigned int blue;
} sum_pixel;
typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;

// cuda functions
__global__ void merge_gpu(int *dev_c, int *dev_width, int *dev_height, pixel *dev_arr, sum_pixel *dev_merge);
__global__ void add_mosaic_gpu(int *dev_c, int *dev_width, int *dev_height, pixel *dev_arr, sum_pixel *dev_merge);

// cpu functions
void print_help();
int process_command_line(int argc, char *argv[]);
void file_exist(FILE *file);
int read_header_numbers(FILE *input);
void read_header(FILE *input, int *width, int *height, int *maxColor, int *version);
void read_binary(FILE *input, pixel *buffer, int *width, int *height, int *maxColor, int *version, char *argv[]);
void read_text(FILE *input, pixel *buffer, int *width, int *height, int *maxColor, int *version, char *argv[]);
void initial_rgb_zero(unsigned long long *r, unsigned long long *g, unsigned long long *b);
void add_value(pixel *array, int p, unsigned long long *r, unsigned long long *g, unsigned long long *b);
void module_average(pixel *array, int p, int r, int g, int b, int w, int h);
void module_pixel_same_color(pixel *small_part, pixel *module, int p, int m);
void merge_openmp(pixel *buffer, pixel *new_image, int *width, int *height);
void add_mosaic_openmp(pixel *new_image, pixel *out, int *width, int *height);
void merge(pixel *buffer, pixel *new_image, int *width, int *height);
void add_mosaic(pixel *new_image, pixel *out, int *width, int *height);
void write_PPM3(FILE *output, pixel *out_image, int *width, int *height, int *maxColor, int *version, char *argv[]);
void write_PPM6(FILE *output, pixel *out_image, int *width, int *height, int *maxColor, int *version, char *argv[]);
void average(pixel *arr, int *width, int *height, int *avg_r, int *avg_g, int *avg_b);
void cpu_mode(FILE *inFile, FILE *outFile, int *width, int *height, int *maxColor, int *version, char *argv[]);
void openmp_mode(FILE *inFile, FILE *outFile, int *width, int *height, int *maxColor, int *version, char *argv[]);
void cuda_mode(FILE *inFile, FILE *outFile, int *width, int *height, int *maxColor, int *version, char *argv[]);

// Global variable
unsigned int c = 0;
MODE execution_mode = CPU;
pixel *buffer;
int width, height, maxColorValue, version, choice;
int i, j, w, h, a, b;
int right_w, low_h, actual_w, actual_h;
int value_r, value_g, value_b;
int point, cell, merge_length;

int main(int argc, char *argv[]) {
	if (process_command_line(argc, argv) == FAILURE)
		return 1;
	// file operation
	FILE *inFile = fopen(argv[4], "rb");
	file_exist(inFile);
	FILE *outFile = fopen(argv[6], "wb");
	// read the header of ppm
	read_header(inFile, &width, &height, &maxColorValue, &version);
	// judge if c is suitable for this input file
	if (c> width || c > height) {
		printf("c is greater than scope, please try another number!");
		exit(-1);
	}
	// execute the mosaic filter based on the mode
	switch (execution_mode) {
	case (CPU): {
		cpu_mode(inFile, outFile, &width, &height, &maxColorValue, &version, argv);
		fclose(inFile);
		break;
	}
	case (OPENMP): {
		openmp_mode(inFile, outFile, &width, &height, &maxColorValue, &version, argv);
		fclose(inFile);
		break;
	}
	case (CUDA): {
		cuda_mode(inFile, outFile, &width, &height, &maxColorValue, &version, argv);
		fclose(inFile);
		break;
	}
	case (ALL): {
		cpu_mode(inFile, outFile, &width, &height, &maxColorValue, &version, argv);
		fclose(inFile);

		inFile = fopen(argv[4], "rb");
		openmp_mode(inFile, outFile, &width, &height, &maxColorValue, &version, argv);
		fclose(inFile);

		inFile = fopen(argv[4], "rb");
		cuda_mode(inFile, outFile, &width, &height, &maxColorValue, &version, argv);
		fclose(inFile);
		break;
	}
	}
	// close file
	fclose(outFile);
	return 0;
}

void print_help() {
	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size which should be any positive\n"
		"\t               power of 2 number \n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
		"\t               ALL. The mode specifies which version of the simulation\n"
		"\t               code should execute. ALL should execute each mode in\n"
		"\t               turn.\n");
	printf("\t-i input_file  Specifies an input image file\n");
	printf("\t-o output_file Specifies an output image file which will be used\n"
		"\t               to write the mosaic image\n");
	printf("[options]:\n");
	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
		"\t               PPM_PLAIN_TEXT\n ");
}

int process_command_line(int argc, char *argv[]) {
	if (argc < 7) {
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}

	//read in the non optional command line arguments
	c = (unsigned int)atoi(argv[1]);
	// judge if c in the power of 2
	if (((c % 2) != 0 && c != 1) || (c <= 0)) {
		printf("c should be 2 ^ n, please try another number");
		exit(-1);
	}
	int divide = c / 2;
	while (divide > 1) {
		if ((divide % 2) != 0) {
			printf("c should be 2 ^ n, please try another number");
			exit(-1);
		}
		divide /= 2;
	}

	// read in the mode
	char str_1[] = "CPU";
	char str_2[] = "OPENMP";
	char str_3[] = "ALL";
	char str_4[] = "CUDA";
	char ** temp = argv;
	while (*++temp != NULL) {
		// pass the space
		if (**++temp == ' ') {
			break;
		}
		// choose the related mode by comparing
		if (strcmp(*temp, str_1) == 0) {
			execution_mode = CPU;
		}
		else if (strcmp(*temp, str_2) == 0) {
			execution_mode = OPENMP;
		}
		else if (strcmp(*temp, str_3) == 0) {
			execution_mode = ALL;
		}
		else if (strcmp(*temp, str_4) == 0) {
			execution_mode = CUDA;
		}
		break;
	}
	return SUCCESS;
}

/* Process CUDA module
*
* @param inFile		ppm file
* @param outFile	ppm file
* @param width		a pointer to the width of picture
* @param height		a pointer to the height of picture
* @param maxColor	a pointer to the maximum RGB color
* @param version	a pointer to the version of picture
* @param argv[]		parameter of command line
* @return			none
*/
void cuda_mode(FILE *inFile, FILE *outFile, int *width, int *height, int *maxColor, int *version, char *argv[]) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int *dev_c, *dev_width, *dev_height;
	int avg_r = 0;
	int avg_g = 0;
	int avg_b = 0;

	// compute the amount of mosaic modules
	w = (*width) / c;
	h = (*height) / c;
	int actual_w = (((*width) % c) != 0) ? w + 1 : w;
	int actual_h = (((*height) % c) != 0) ? h + 1 : h;

	//copy important number from host to device
	cudaMalloc((void**)&dev_c, sizeof(int));
	cudaMalloc((void**)&dev_width, sizeof(int));
	cudaMalloc((void**)&dev_height, sizeof(int));
	cudaMemcpy(dev_c, &c, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_width, width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_height, height, sizeof(int), cudaMemcpyHostToDevice);

	// allocate memory for host and device
	pixel *image_cuda, *dev_arr;
	sum_pixel *dev_merge;
	cudaMallocHost((void **)&image_cuda, sizeof(pixel) * (*width) * (*height));
	cudaMalloc((void**)&dev_arr, (*width) * (*height) * sizeof(pixel));
	cudaMalloc((void**)&dev_merge, actual_w * actual_h * sizeof(sum_pixel));

	// choose dimension for blockDim and gridDim
	int dim = (c > 32) ? 32 : c;
	int shift = (c > 32) ? c / 32 : 1;
	dim3 blockDim(dim, dim, 1);
	dim3 gridDim(actual_w, actual_h, shift*shift);

	// Judge whether the infile is P3 OR P6 and read file
	if (*version == 6) {
		read_binary(inFile, image_cuda, width, height, maxColor, version, argv);
		average(image_cuda, width, height, &avg_r, &avg_g, &avg_b);
	}

	if (*version == 3) {
		read_text(inFile, image_cuda, width, height, maxColor, version, argv);
		average(image_cuda, width, height, &avg_r, &avg_g, &avg_b);
	}

	// cuda timing
	cudaEventRecord(start);
	// copy rgb data to GPU
	cudaMemcpy(dev_arr, image_cuda, (*width) * (*height) * sizeof(pixel), cudaMemcpyHostToDevice);
	// kernel
	merge_gpu << <gridDim, blockDim >> >(dev_c, dev_width, dev_height, dev_arr, dev_merge);
	add_mosaic_gpu << <gridDim, blockDim >> >(dev_c, dev_width, dev_height, dev_arr, dev_merge);
	cudaDeviceSynchronize();
	// copy rgb data back to CPU
	cudaMemcpy(image_cuda, dev_arr, (*width) * (*height) * sizeof(pixel), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	// free memory
	cudaFree(dev_arr);
	cudaFree(dev_merge);

	//write output
	if (*version == 6) {
		write_PPM6(outFile, image_cuda, width, height, maxColor, version, argv);
	}
	if (*version == 3) {
		write_PPM3(outFile, image_cuda, width, height, maxColor, version, argv);
	}
	// Output the average colour value for the image
	printf("CUDA Average image colour red = %d, green = %d, blue = %d \n", avg_r, avg_g, avg_b);
	// free memory
	cudaFree(image_cuda);
	// timing
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	int second = milliseconds / CLOCKS_PER_SEC;
	int msecond = milliseconds - second * CLOCKS_PER_SEC;
	printf("CUDA mode execution time took %d s and %d ms\n", second, msecond);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}


/* Merge all pixel into different small module and stores the pixel data into array.
*
* @param dev_c			is equal to c
* @param dev_width		is equal to width, which is the width of picture
* @param dev_height		is equal to height, which is the height of picture
* @param dev_arr		the array contains all pixel data
* @param dev_merge		the array contains all mosaic module data
* @return				None
*/
__global__ void merge_gpu(int *dev_c, int *dev_width, int *dev_height, pixel *dev_arr, sum_pixel *dev_merge) {
	int mosaic, point, block_offset, thread_offset, z_offset;
	// the width and height of mosaic modules without partial mosaic
	int w = (*dev_width) / (*dev_c);
	int h = (*dev_height) / (*dev_c);
	// the width of partial mosaic
	int large_right_w = (*dev_width) % (*dev_c);
	int large_low_h = (*dev_height) % (*dev_c);
	// the width of rightest partial mosaic if c > 32
	int right_w = (*dev_c > 32) ? (*dev_width) % 32 : (*dev_width) % (*dev_c);
	int low_h = (*dev_c > 32) ? (*dev_height) % 32 : (*dev_height) % (*dev_c);
	// actual width of mosaic
	int actual_w = (large_right_w != 0) ? w + 1 : w;
	int actual_h = (large_low_h != 0) ? h + 1 : h;
	// the width of 32x32 in mosaic if c > 32
	int shift = (*dev_c > 32) ? *dev_c / 32 : 1;
	// the shift distance of 32x32 in partial mosaic module
	int shift_r = (*dev_c > 32) ? (large_right_w / 32) + 1 : 1;
	int shift_l = (*dev_c > 32) ? (large_low_h / 32) + 1 : 1;
	// the limitation of partial mosaic
	int limit2 = (actual_h == h) ? actual_h : actual_h - 1;
	int limit3 = (actual_w == w) ? actual_w : actual_w - 1;
	// situations
	block_offset = blockIdx.x * (*dev_c) + (*dev_c) * (*dev_width) * blockIdx.y;
	thread_offset = threadIdx.y * (*dev_width) + threadIdx.x;
	z_offset = (blockIdx.z / shift) * 32 * (*dev_width) + (blockIdx.z % shift) * 32;

	// if the picture is square and can be divided by c
	if ((blockIdx.y < h) && (blockIdx.x < w)) {
		// position of mosaic
		mosaic = blockIdx.y * (actual_w)+blockIdx.x;
		// position of pixel point
		point = block_offset + thread_offset + z_offset;
		atomicAdd(&(dev_merge[mosaic].red), dev_arr[point].red);
		atomicAdd(&(dev_merge[mosaic].green), dev_arr[point].green);
		atomicAdd(&(dev_merge[mosaic].blue), dev_arr[point].blue);
	}

	// rightest part of mosaic
	if ((blockIdx.y < limit2) && (blockIdx.x == w)) {
		mosaic = blockIdx.y * actual_w + blockIdx.x;
		// part 1, the rect part of partial mosaic
		if ((blockIdx.z % shift) == (shift_r - 1)) {
			if (threadIdx.x < right_w) {
				point = block_offset + thread_offset + z_offset;
				atomicAdd(&(dev_merge[mosaic].red), dev_arr[point].red);
				atomicAdd(&(dev_merge[mosaic].green), dev_arr[point].green);
				atomicAdd(&(dev_merge[mosaic].blue), dev_arr[point].blue);
			}
		}
		// part 2, the square part of partial mosaic
		if ((blockIdx.z % shift) < (shift_r - 1)) {
			point = block_offset + thread_offset + z_offset;
			atomicAdd(&(dev_merge[mosaic].red), dev_arr[point].red);
			atomicAdd(&(dev_merge[mosaic].green), dev_arr[point].green);
			atomicAdd(&(dev_merge[mosaic].blue), dev_arr[point].blue);
		}
	}

	// lowest part of mosaic
	if ((blockIdx.y == h) && (blockIdx.x < limit3)) {
		mosaic = blockIdx.y * actual_w + blockIdx.x;
		// part 1, the rect part of partial mosaic
		if ((blockIdx.z / shift) == (shift_l - 1)) {
			if (threadIdx.y < low_h) {
				point = block_offset + thread_offset + z_offset;
				atomicAdd(&(dev_merge[mosaic].red), dev_arr[point].red);
				atomicAdd(&(dev_merge[mosaic].green), dev_arr[point].green);
				atomicAdd(&(dev_merge[mosaic].blue), dev_arr[point].blue);
			}
		}
		// part 2, the square part of partial mosaic
		if ((blockIdx.z / shift) < (shift_l - 1)) {
			point = block_offset + thread_offset + z_offset;
			atomicAdd(&(dev_merge[mosaic].red), dev_arr[point].red);
			atomicAdd(&(dev_merge[mosaic].green), dev_arr[point].green);
			atomicAdd(&(dev_merge[mosaic].blue), dev_arr[point].blue);
		}

	}

	// corner part of mosaic
	// it is divided to 4 parts if they exists, and handle different part by different limitations
	if ((blockIdx.y == h) && (blockIdx.x == w)) {
		mosaic = blockIdx.y * (actual_w)+blockIdx.x;
		// part 1, square
		if (((blockIdx.z % shift) == (shift_r - 1)) && ((blockIdx.z / shift) == (shift_l - 1))) {
			if (threadIdx.x < right_w) {
				if (threadIdx.y < low_h) {
					point = block_offset + thread_offset + z_offset;
					atomicAdd(&(dev_merge[mosaic].red), dev_arr[point].red);
					atomicAdd(&(dev_merge[mosaic].green), dev_arr[point].green);
					atomicAdd(&(dev_merge[mosaic].blue), dev_arr[point].blue);
				}
			}
		}
		// part 2, right partial mosaic
		if (((blockIdx.z % shift) == (shift_r - 1)) && ((blockIdx.z / shift) < (shift_l - 1))) {
			if (threadIdx.x < right_w) {
				point = block_offset + thread_offset + z_offset;
				atomicAdd(&(dev_merge[mosaic].red), dev_arr[point].red);
				atomicAdd(&(dev_merge[mosaic].green), dev_arr[point].green);
				atomicAdd(&(dev_merge[mosaic].blue), dev_arr[point].blue);
			}
		}
		// part 3, low partial mosaic
		if (((blockIdx.z % shift) < (shift_r - 1)) && ((blockIdx.z / shift) == (shift_l - 1))) {
			if (threadIdx.y < low_h) {
				point = block_offset + thread_offset + z_offset;
				atomicAdd(&(dev_merge[mosaic].red), dev_arr[point].red);
				atomicAdd(&(dev_merge[mosaic].green), dev_arr[point].green);
				atomicAdd(&(dev_merge[mosaic].blue), dev_arr[point].blue);
			}
		}
		// part 4, rect in the corner
		if (((blockIdx.z % shift) < (shift_r - 1)) && ((blockIdx.z / shift) < (shift_l - 1))) {
			point = block_offset + thread_offset + z_offset;
			atomicAdd(&(dev_merge[mosaic].red), dev_arr[point].red);
			atomicAdd(&(dev_merge[mosaic].green), dev_arr[point].green);
			atomicAdd(&(dev_merge[mosaic].blue), dev_arr[point].blue);
		}
	}
}

/*
* Allocate all original pixels with the value from merged pixel array to a new array--out_array.
*
* @param dev_c			is equal to c
* @param dev_width		is equal to width, which is the width of picture
* @param dev_height		is equal to height, which is the height of picture
* @param dev_arr		the array contains all pixel data
* @param dev_merge		the array contains all mosaic module data
* @return				None
*/
__global__ void add_mosaic_gpu(int *dev_c, int *dev_width, int *dev_height, pixel *dev_arr, sum_pixel *dev_merge) {
	int mosaic, point, block_offset, thread_offset, z_offset;
	// the width and height of mosaic modules without partial mosaic
	int w = (*dev_width) / (*dev_c);
	int h = (*dev_height) / (*dev_c);
	// the width of partial mosaic
	int large_right_w = (*dev_width) % (*dev_c);
	int large_low_h = (*dev_height) % (*dev_c);
	// the width of rightest partial mosaic if c > 32
	int right_w = (*dev_c > 32) ? (*dev_width) % 32 : (*dev_width) % (*dev_c);
	int low_h = (*dev_c > 32) ? (*dev_height) % 32 : (*dev_height) % (*dev_c);
	// actual width of mosaic
	int actual_w = (large_right_w != 0) ? w + 1 : w;
	int actual_h = (large_low_h != 0) ? h + 1 : h;
	// the width of 32x32 in mosaic if c > 32
	int shift = (*dev_c > 32) ? *dev_c / 32 : 1;
	// the shift distance of 32x32 in partial mosaic module
	int shift_r = (*dev_c > 32) ? (large_right_w / 32) + 1 : 1;
	int shift_l = (*dev_c > 32) ? (large_low_h / 32) + 1 : 1;
	// the limitation of partial mosaic
	int limit2 = (actual_h == h) ? actual_h : actual_h - 1;
	int limit3 = (actual_w == w) ? actual_w : actual_w - 1;
	// situations
	block_offset = blockIdx.x * (*dev_c) + (*dev_c) * (*dev_width) * blockIdx.y;
	thread_offset = threadIdx.y * (*dev_width) + threadIdx.x;
	z_offset = (blockIdx.z / shift) * 32 * (*dev_width) + (blockIdx.z % shift) * 32;

	// if the picture is square and can be divided by c
	if ((blockIdx.y < h) && (blockIdx.x < w)) {
		mosaic = blockIdx.y * (actual_w)+blockIdx.x;
		point = block_offset + thread_offset + z_offset;
		// get the average rgb value
		dev_arr[point].red = dev_merge[mosaic].red / ((*dev_c)*(*dev_c));
		dev_arr[point].green = dev_merge[mosaic].green / ((*dev_c)*(*dev_c));
		dev_arr[point].blue = dev_merge[mosaic].blue / ((*dev_c)*(*dev_c));
	}

	// rightest part of mosaic
	if ((blockIdx.x == w) && (blockIdx.y < limit2)) {
		mosaic = blockIdx.y * actual_w + blockIdx.x;
		// part 2, the rect part of partial mosaic
		if ((blockIdx.z % shift) == (shift_r - 1)) {
			if (threadIdx.x < right_w) {
				// average by dividing the actual pixels points in the partial mosaic
				point = block_offset + thread_offset + z_offset;
				dev_arr[point].red = dev_merge[mosaic].red / ((*dev_c)*large_right_w);
				dev_arr[point].green = dev_merge[mosaic].green / ((*dev_c)*large_right_w);
				dev_arr[point].blue = dev_merge[mosaic].blue / ((*dev_c)*large_right_w);
			}
		}
		// part 2, the square part of partial mosaic
		if ((blockIdx.z % shift) < (shift_r - 1)) {
			point = block_offset + thread_offset + z_offset;
			dev_arr[point].red = dev_merge[mosaic].red / ((*dev_c)*large_right_w);
			dev_arr[point].green = dev_merge[mosaic].green / ((*dev_c)*large_right_w);
			dev_arr[point].blue = dev_merge[mosaic].blue / ((*dev_c)*large_right_w);
		}
	}

	// lowest part of mosaic
	if ((blockIdx.y == h) && (blockIdx.x < limit3)) {
		mosaic = blockIdx.y * actual_w + blockIdx.x;
		// part 2, the rect part of partial mosaic
		if ((blockIdx.z / shift) == (shift_l - 1)) {
			if (threadIdx.y < low_h) {
				point = block_offset + thread_offset + z_offset;
				dev_arr[point].red = dev_merge[mosaic].red / ((*dev_c)*large_low_h);
				dev_arr[point].green = dev_merge[mosaic].green / ((*dev_c)*large_low_h);
				dev_arr[point].blue = dev_merge[mosaic].blue / ((*dev_c)*large_low_h);
			}
		}
		// part 2, the square part of partial mosaic
		if ((blockIdx.z / shift) < (shift_l - 1)) {
			point = block_offset + thread_offset + z_offset;
			dev_arr[point].red = dev_merge[mosaic].red / ((*dev_c)*large_low_h);
			dev_arr[point].green = dev_merge[mosaic].green / ((*dev_c)*large_low_h);
			dev_arr[point].blue = dev_merge[mosaic].blue / ((*dev_c)*large_low_h);
		}
	}

	// corner part of mosaic
	// it is divided to 4 parts if they exists, and handle different part by different limitations
	if ((blockIdx.y == h) && (blockIdx.x == w)) {
		mosaic = blockIdx.y * (actual_w)+blockIdx.x;
		// part 1, square
		if (((blockIdx.z % shift) == (shift_r - 1)) && ((blockIdx.z / shift) == (shift_l - 1))) {
			if (threadIdx.x < right_w) {
				if (threadIdx.y < low_h) {
					point = block_offset + thread_offset + z_offset;
					dev_arr[point].red = dev_merge[mosaic].red / (large_right_w*large_low_h);
					dev_arr[point].green = dev_merge[mosaic].green / (large_right_w*large_low_h);
					dev_arr[point].blue = dev_merge[mosaic].blue / (large_right_w*large_low_h);
				}
			}
		}
		// part 2, right partial mosaic
		if (((blockIdx.z % shift) == (shift_r - 1)) && ((blockIdx.z / shift) < (shift_l - 1))) {
			if (threadIdx.x < right_w) {
				point = block_offset + thread_offset + z_offset;
				dev_arr[point].red = dev_merge[mosaic].red / (large_right_w*large_low_h);
				dev_arr[point].green = dev_merge[mosaic].green / (large_right_w*large_low_h);
				dev_arr[point].blue = dev_merge[mosaic].blue / (large_right_w*large_low_h);
			}
		}
		// part 3, low partial mosaic
		if (((blockIdx.z % shift) < (shift_r - 1)) && ((blockIdx.z / shift) == (shift_l - 1))) {
			if (threadIdx.y < low_h) {
				point = block_offset + thread_offset + z_offset;
				dev_arr[point].red = dev_merge[mosaic].red / (large_right_w*large_low_h);
				dev_arr[point].green = dev_merge[mosaic].green / (large_right_w*large_low_h);
				dev_arr[point].blue = dev_merge[mosaic].blue / (large_right_w*large_low_h);
			}
		}
		// part 4, rect in the corner
		if (((blockIdx.z % shift) < (shift_r - 1)) && ((blockIdx.z / shift) < (shift_l - 1))) {
			point = block_offset + thread_offset + z_offset;
			dev_arr[point].red = dev_merge[mosaic].red / (large_right_w*large_low_h);
			dev_arr[point].green = dev_merge[mosaic].green / (large_right_w*large_low_h);
			dev_arr[point].blue = dev_merge[mosaic].blue / (large_right_w*large_low_h);
		}
	}
}

/* Process CPU module
*
* @param inFile		ppm file
* @param outFile	ppm file
* @param width		a pointer to the width of picture
* @param height		a pointer to the height of picture
* @param maxColor	a pointer to the maximum RGB color
* @param version	a pointer to the version of picture
* @param argv[]		parameter of command line
* @return			none
*/
void cpu_mode(FILE *inFile, FILE *outFile, int *width, int *height, int *maxColor, int *version, char *argv[]) {
	clock_t begin, end;
	int avg_r = 0;
	int avg_g = 0;
	int avg_b = 0;
	// allocate memory for array contains all value of RGB
	w = (*width) / c;
	h = (*height) / c;
	pixel *image_cpu = (pixel *)malloc(sizeof(pixel) * (*width) * (*height));
	pixel *new_image_cpu = (pixel *)malloc(sizeof(pixel) * (w + 1) * (h + 1));
	// Judge whether the infile is P3 OR P6
	if (*version == 6) {
		read_binary(inFile, image_cpu, width, height, maxColor, version, argv);
		begin = clock();
		merge(image_cpu, new_image_cpu, width, height);
		add_mosaic(new_image_cpu, image_cpu, width, height);
		end = clock();
		write_PPM6(outFile, image_cpu, width, height, maxColor, version, argv);
		free(new_image_cpu);
	}
	if (*version == 3) {
		read_text(inFile, image_cpu, width, height, maxColor, version, argv);
		begin = clock();
		merge(image_cpu, new_image_cpu, width, height);
		add_mosaic(new_image_cpu, image_cpu, width, height);
		end = clock();
		write_PPM3(outFile, image_cpu, width, height, maxColor, version, argv);
		free(new_image_cpu);
	}
	// Output the average colour value for the image
	average(image_cpu, width, height, &avg_r, &avg_g, &avg_b);
	free(image_cpu);
	printf("CPU Average image colour red = %d, green = %d, blue = %d \n", avg_r, avg_g, avg_b);
	// timing
	int second = (end - begin) / CLOCKS_PER_SEC;
	int msecond = (end - begin) - second * CLOCKS_PER_SEC;
	printf("CPU mode execution time took %d s and %d ms\n", second, msecond);
}

/* Process OPENMP module
*
* @param inFile		ppm file
* @param outFile	ppm file
* @param width		a pointer to the width of picture
* @param height		a pointer to the height of picture
* @param maxColor	a pointer to the maximum RGB color
* @param version	a pointer to the version of picture
* @param argv[]		parameter of command line
* @return			none
*/
void openmp_mode(FILE *inFile, FILE *outFile, int *width, int *height, int *maxColor, int *version, char *argv[]) {
	clock_t begin, end;
	int avg_r = 0;
	int avg_g = 0;
	int avg_b = 0;
	// allocate memory for array contains all value of RGB
	w = (*width) / c;
	h = (*height) / c;
	pixel *image_omp = (pixel *)malloc(sizeof(pixel) * (*width) * (*height));
	pixel *new_image_omp = (pixel *)malloc(sizeof(pixel) * (w + 1) * (h + 1));
	// Judge whether the infile is P3 OR P6
	if (*version == 6) {
		read_binary(inFile, image_omp, width, height, maxColor, version, argv);
		begin = clock();
		merge_openmp(image_omp, new_image_omp, width, height);
		add_mosaic_openmp(new_image_omp, image_omp, width, height);
		end = clock();
		write_PPM6(outFile, image_omp, width, height, maxColor, version, argv);
		free(new_image_omp);
	}
	if (*version == 3) {
		read_text(inFile, image_omp, width, height, maxColor, version, argv);
		begin = clock();
		merge_openmp(image_omp, new_image_omp, width, height);
		add_mosaic_openmp(new_image_omp, image_omp, width, height);
		end = clock();
		write_PPM3(outFile, image_omp, width, height, maxColor, version, argv);
		free(new_image_omp);
	}
	// Output the average colour value for the image
	average(image_omp, width, height, &avg_r, &avg_g, &avg_b);
	free(image_omp);
	printf("OPENMP Average image colour red = %d, green = %d, blue = %d \n", avg_r, avg_g, avg_b);
	// timing
	int second = (end - begin) / CLOCKS_PER_SEC;
	int msecond = (end - begin) - second * CLOCKS_PER_SEC;
	printf("OPENMP mode execution time took %d s and %d ms\n", second, msecond);
}

/* Judge whether the input file exist
*
* @param file		input file
* @return			none
*/
void file_exist(FILE *file) {
	// if the input file does not exist, exit and report error
	if (file == NULL) {
		printf("Error! Could not open this file, try another one!\n");
		exit(-1);
	}
}

/* Read weight, height and maxColor
*
* @param file		input file
* @return			number
*/
int read_header_numbers(FILE *input) {
	int number = 0;
	char c;
	c = getc(input);
	while (c == ' ' || c == '\n') {
		c = getc(input);
	}
	do {
		number = number * 10 + c - '0';
		c = getc(input);
	} while ((c != ' ') && (c != '\n'));
	return number;
}

/* Skip the comment line
*
* @param file		input file
* @return			none
*/
void skip_comment(FILE *input) {
	char c;
	//Skip Comments in the Header
	c = getc(input);
	while (c == '#') {
		while (c != '\n') {
			//head is '\n' when jump out loop
			c = getc(input);
		}
	}
	ungetc(c, input);
}

/* Reads the header of a ppm file
*
* @param input		ppm file
* @param width		a pointer to the width of picture
* @param height	a pointer to the height of picture
* @param maxColor	a pointer to the maximum RGB color
* @param version	a pointer to the version of picture
* @return			none
*/
void read_header(FILE *input, int *width, int *height, int *maxColor, int *version) {
	char magic_num, head, f;
	// get the magic number of P%d
	f = getc(input);
	if (f != 'P') {
		printf("Invalid file, try another one!\n");
		exit(-1);
	}
	magic_num = getc(input);
	*version = atoi(&magic_num);
	head = getc(input);
	//Skip Comments in the Header
	skip_comment(input);
	// skim space and get the number
	*width = read_header_numbers(input);
	skip_comment(input);
	*height = read_header_numbers(input);
	skip_comment(input);
	*maxColor = read_header_numbers(input);
}

/* Reads the contents of a P6 PPM file and stores the pixel data into a array--image.
*
* @param input		the ppm file for reading
* @param buffer		the pixel structure to contain rgb data
* @param width		a pointer to the width
* @param height		a pointer to the height
* @param maxColor   a pointer to the maximum RGB color
* @param version	a pointer to the version of picture
* @param argv[]		command parametre
* @return           none
*/
void read_binary(FILE *input, pixel *buffer, int *width, int *height, int *maxColor, int *version, char *argv[]) {
	unsigned char *inBuf = (unsigned char *)malloc(sizeof(unsigned char));
	int value;
	int length = (*width) * (*height);
	i = 0;
	// read original file one by one
	while (i < length) {
		// get each pixel into the char buffer
		// get the red value and allocate it to buffer.red
		fread(inBuf, 1, 1, input);
		value = *inBuf;
		buffer[i].red = value;
		// get the green value and allocate it to buffer.green
		fread(inBuf, 1, 1, input);
		value = *inBuf;
		buffer[i].green = value;
		// get the blue value and allocate it to buffer.blue
		fread(inBuf, 1, 1, input);
		value = *inBuf;
		buffer[i].blue = value;
		i++;
	}
}

/* Reads the contents of a P3 PPM file and stores the pixel data into a array--image.
*
* @param input         the ppm file for reading
* @param buffer        the pixel structure to contain rgb data
* @param width         a pointer to the width
* @param height        a pointer to the height
* @param maxColor      a pointer to the maximum RGB color
* @param version		a pointer to the version of picture
* @param argv[]		command parametre
* @return              none
*/
void read_text(FILE *input, pixel *buffer, int *width, int *height, int *maxColor, int *version, char *argv[]) {
	int re, gr, bl;
	char str;
	// read rgb value into array
	int length = (*width) * (*height);
	// read bytes one by one and transform bytes into int
	for (i = 0; i<length; i++) {
		// judge different value by passing space
		re = 0;
		while ((str = getc(input)) != ' ') {
			re = re * 10 + str - '0';
		}
		gr = 0;
		while ((str = getc(input)) != ' ') {
			gr = gr * 10 + str - '0';
		}
		bl = 0;
		str = getc(input);
		// when face '\n', pass it
		while (str != ' ' && str != '\t' && str != '\n'&& str != EOF) {
			bl = bl * 10 + str - '0';
			str = getc(input);
		}
		// allocate triple value into pixel
		buffer[i].red = re;
		buffer[i].green = gr;
		buffer[i].blue = bl;
	}
}

/* add all rgb value in this module together.
*
* @param array	the pixel structure contains rgb data in a mosaic module
* @param p		the present pixel we are handling
* @param r		a pointer to the present red color
* @param g		a pointer to the present green color
* @param b		a pointer to the present blue color
* @return		none
*/
void add_value(pixel *array, int p, unsigned long long *r, unsigned long long *g, unsigned long long *b) {
	*r += array[p].red;
	*g += array[p].green;
	*b += array[p].blue;
}

/* Average all rgb value in a mosaic module.
*
* @param array	the pixel structure contains rgb data in a mosaic module
* @param r		present sum of red color in this module
* @param g		present sum of green color in this module
* @param b		present sum of blue color in this module
* @param w		present weight of this module
* @param h		present hwight of this module
* @return		none
*/
void module_average(pixel *array, int p, int r, int g, int b, int w, int h) {
	array[p].red = r / (w*h);
	array[p].green = g / (w*h);
	array[p].blue = b / (w*h);
}

/* initial all local rgb value for sum up the rgb value in a module.
*
* @param r		a pointer to the present red color
* @param g		a pointer to the present green color
* @param b		a pointer to the present blue color
* @return		none
*/
void initial_rgb_zero(unsigned long long *r, unsigned long long *g, unsigned long long *b) {
	*r = 0;
	*g = 0;
	*b = 0;
}

/* allocate rgb value in a module to all pixels in it.
*
* @param small_part	the pixel structure contains rgb data in a single pixel
* @param module		the pixel structure contains rgb data in a mosaic module
* @param p			a pointer to the present pixel we are handling
* @param m			a pointer to the present module
* @return			none
*/
void module_pixel_same_color(pixel *small_part, pixel *module, int p, int m) {
	small_part[p].red = module[m].red;
	small_part[p].green = module[m].green;
	small_part[p].blue = module[m].blue;
}

/* Merge all pixel into different small module and stores the pixel data into a array--new_image.
*
* @param buffer        the pixel structure contains all rgb data in the file
* @param new_image		the pixel structure contains all merged rgb data in the file,for example:
* 						one element in the array represents the average rgb of a box of pixels
* @param width         a pointer to the width
* @param height        a pointer to the height
* @return              none
*/
void merge(pixel *buffer, pixel *new_image, int *width, int *height) {
	unsigned long long r_local = 0;
	unsigned long long g_local = 0;
	unsigned long long b_local = 0;
	// get the amount of square mosaic modules in a raw and column
	w = (*width) / c;
	h = (*height) / c;
	// pre set the amount of square mosaic modules in a raw and column
	int actual_w = w;
	int actual_h = h;
	// compute the weight of rightest rectangle mosaic module if exist
	right_w = (*width) % c;
	low_h = (*height) % c;
	if (right_w != 0) {
		actual_w = w + 1;
	}
	if (low_h != 0) {
		actual_h = h + 1;
	}
	// initial the amount of mosaic modules
	for (i = 0; i<actual_h; i++) { // loop for raws
		if ((low_h == 0) || i < (actual_h - 1)) {
			for (j = 0; j<actual_w; j++) { // loop for columns
				if ((right_w == 0) || j < (actual_w - 1)) {
					cell = j + i * actual_w;
					for (a = 0; a<c; a++) { // loop for raws in a mosaic module
						for (b = 0; b<c; b++) { // loop for columns in a mosaic module
												// based on the index of module, get the indexs of pixels in this module
							point = j * c + c * (*width)*i + b + a * (*width);
							add_value(buffer, point, &r_local, &g_local, &b_local);
						}
					}
					module_average(new_image, cell, r_local, g_local, b_local, c, c);
					initial_rgb_zero(&r_local, &g_local, &b_local);
				}
				if (j == (actual_w - 1) && right_w != 0) { // judge if there would be a rectangle mosaic in the rightest side of ppm for every raw
					cell = j + i * actual_w;
					for (a = 0; a<c; a++) {
						for (b = 0; b<right_w; b++) { // loop for columns in a mosaic module and it is less than c in here
							point = j * c + c * (*width)*i + b + a * (*width);
							add_value(buffer, point, &r_local, &g_local, &b_local);
						}
					}
					module_average(new_image, cell, r_local, g_local, b_local, c, c);
					initial_rgb_zero(&r_local, &g_local, &b_local);
				}
			}
		}
		if (i == (actual_h - 1) && low_h != 0) { // judge if there would be a rectangle mosaic in the lowest side of ppm
			for (j = 0; j<actual_w; j++) { // loop the lowest and left rectangles if exist
				cell = j + i * actual_w;
				if ((right_w == 0) || j < (actual_w - 1)) {
					for (a = 0; a<low_h; a++) { // loop for raws in a mosaic module and it is less than c in here
						for (b = 0; b<c; b++) {
							point = j * c + c * (*width)*i + b + a * (*width);
							add_value(buffer, point, &r_local, &g_local, &b_local);
						}
					}
					module_average(new_image, cell, r_local, g_local, b_local, c, c);
					initial_rgb_zero(&r_local, &g_local, &b_local);
				}
				if (j == (actual_w - 1) && right_w != 0) { // loop the lowest and rightest corner rectangles if exist
					cell = j + i * actual_w;
					for (a = 0; a<low_h; a++) {
						for (b = 0; b<right_w; b++) {
							point = j * c + c * (*width)*i + b + a * (*width);
							add_value(buffer, point, &r_local, &g_local, &b_local);
						}
					}
					module_average(new_image, cell, r_local, g_local, b_local, c, c);
					initial_rgb_zero(&r_local, &g_local, &b_local);
				}
			}
		}
	}
}

/*
* Allocate all original pixels with the value from merged pixel array to a new array--out_array.
*
* @param new_image		the pixel structure contains all merged and averaged rgb data in the file
* @param out			the pixel structure contains all rgb data for writing mosaic
* @param width         a pointer to the width
* @param height        a pointer to the height
* @return              none
*/
void add_mosaic(pixel *new_image, pixel *out, int *width, int *height) {
	pixel *out_image = (pixel *)malloc(sizeof(pixel) * (*width) * (*height));
	merge_length = 0;
	int cell;
	right_w = (*width) % c;
	low_h = (*height) % c;
	int actual_w = w;
	int actual_h = h;
	if (right_w != 0) {
		actual_w = w + 1;
	}
	if (low_h != 0) {
		actual_h = h + 1;
	}
	for (i = 0; i<actual_h; i++) { // loop for raws
		if ((low_h == 0) || i < (actual_h - 1)) {
			for (j = 0; j<actual_w; j++) { // loop for columns
				cell = j + i * actual_w;
				if ((right_w == 0) || j < (actual_w - 1)) {
					for (a = 0; a<c; a++) { // loop for raws in a mosaic module
						for (b = 0; b<c; b++) { // loop for columns in a mosaic module
												// based on the index of module, get the indexs of pixels in this module
							point = j * c + c * (*width)*i + b + a * (*width);
							// every point in this module would get the same rgb value
							module_pixel_same_color(out, new_image, point, cell);
						}
					}
				}
				if (j == (actual_w - 1) && right_w != 0) { // judge if there would be a rectangle mosaic in the rightest side of ppm for every raw
					cell = j + i * actual_w;
					for (a = 0; a<c; a++) {
						for (b = 0; b<right_w; b++) { // loop for columns in a mosaic module and it is less than c in here
							point = j * c + c * (*width)*i + b + a * (*width);
							module_pixel_same_color(out, new_image, point, cell);
						}
					}
				}
			}
		}

		if (i == (actual_h - 1) && low_h != 0) { // judge if there would be a rectangle mosaic in the lowest side of ppm
			for (j = 0; j<actual_w; j++) { // loop the lowest and left rectangles if exist
				cell = j + i * actual_w;
				if (j != (actual_w - 1)) {
					for (a = 0; a<low_h; a++) { // loop for raws in a mosaic module and it is less than c in here
						for (b = 0; b<c; b++) {
							point = j * c + c * (*width)*i + b + a * (*width);
							module_pixel_same_color(out, new_image, point, cell);
						}
					}
				}
				else { // loop the lowest and rightest corner rectangles if exist
					cell = j + i * actual_w;
					for (a = 0; a<low_h; a++) {
						for (b = 0; b<right_w; b++) {
							point = j * c + c * (*width)*i + b + a * (*width);
							module_pixel_same_color(out, new_image, point, cell);
						}
					}
				}
			}
		}
	}
}

/* Merge all pixel into different small module and stores the pixel data into a array--new_image.
*
* @param buffer			the pixel structure contains all rgb data in the file
* @param new_image		the pixel structure contains all merged rgb data in the file,for example:
* 						one element in the array represents the average rgb of a box of pixels
* @param width		    a pointer to the width
* @param height		    a pointer to the height
* @return              none
*/
void merge_openmp(pixel *buffer, pixel *new_image, int *width, int *height) {
	unsigned long long r_local = 0;
	unsigned long long g_local = 0;
	unsigned long long b_local = 0;
	// get the amount of square mosaic modules in a raw and column
	w = (*width) / c;
	h = (*height) / c;
	// pre set the amount of square mosaic modules in a raw and column
	int actual_w = w;
	int actual_h = h;
	// compute the weight of rightest rectangle mosaic module if exist
	right_w = (*width) % c;
	low_h = (*height) % c;
	if (right_w != 0) {
		actual_w = w + 1;
	}
	if (low_h != 0) {
		actual_h = h + 1;
	}
	// initial the amount of mosaic modules
#pragma omp parallel for private(i,j,a,b,cell,point) \
reduction(+:r_local)\
reduction(+:g_local)\
reduction(+:b_local)
	for (i = 0; i<actual_h; i++) { // loop for raws
		if ((low_h == 0) || i < (actual_h - 1)) {
			for (j = 0; j<actual_w; j++) { // loop for columns
				if ((right_w == 0) || j < (actual_w - 1)) {
					cell = j + i * actual_w;
					for (a = 0; a<c; a++) { // loop for raws in a mosaic module
						for (b = 0; b<c; b++) { // loop for columns in a mosaic module
												// based on the index of module, get the indexs of pixels in this module
							point = j * c + c * (*width)*i + b + a * (*width);
							add_value(buffer, point, &r_local, &g_local, &b_local);
						}
					}
#pragma omp critical
					{
						module_average(new_image, cell, r_local, g_local, b_local, c, c);
						initial_rgb_zero(&r_local, &g_local, &b_local);
					}
				}
				if (j == (actual_w - 1) && right_w != 0) { // judge if there would be a rectangle mosaic in the rightest side of ppm for every raw
					cell = j + i * actual_w;
					for (a = 0; a<c; a++) {
						for (b = 0; b<right_w; b++) { // loop for columns in a mosaic module and it is less than c in here
							point = j * c + c * (*width)*i + b + a * (*width);
							add_value(buffer, point, &r_local, &g_local, &b_local);
						}
					}
#pragma omp critical
					{
						module_average(new_image, cell, r_local, g_local, b_local, right_w, c);
						initial_rgb_zero(&r_local, &g_local, &b_local);
					}
				}
			}
		}
		if (i == (actual_h - 1) && low_h != 0) { // judge if there would be a rectangle mosaic in the lowest side of ppm
			for (j = 0; j<actual_w; j++) { // loop the lowest and left rectangles if exist
				cell = j + i * actual_w;
				if ((right_w == 0) || j < (actual_w - 1)) {
					for (a = 0; a<low_h; a++) { // loop for raws in a mosaic module and it is less than c in here
						for (b = 0; b<c; b++) {
							point = j * c + c * (*width)*i + b + a * (*width);
							add_value(buffer, point, &r_local, &g_local, &b_local);
						}
					}
#pragma omp critical
					{
						module_average(new_image, cell, r_local, g_local, b_local, c, low_h);
						initial_rgb_zero(&r_local, &g_local, &b_local);
					}
				}
				if (j == (actual_w - 1) && right_w != 0) { // loop the lowest and rightest corner rectangles if exist
					cell = j + i * actual_w;
					for (a = 0; a<low_h; a++) {
						for (b = 0; b<right_w; b++) {
							point = j * c + c * (*width)*i + b + a * (*width);
							add_value(buffer, point, &r_local, &g_local, &b_local);
						}
					}
#pragma omp critical
					{
						module_average(new_image, cell, r_local, g_local, b_local, right_w, low_h);
						initial_rgb_zero(&r_local, &g_local, &b_local);
					}
				}
			}
		}
	}
}


/*
* Allocate all original pixels with the value from merged pixel array to a new array--out_array.
*
* @param new_image		the pixel structure contains all merged and averaged rgb data in the file
* @param out			the pixel structure contains all rgb data for writing mosaic
* @param width			a pointer to the width
* @param height			a pointer to the height
* @return				none
*/
void add_mosaic_openmp(pixel *new_image, pixel *out, int *width, int *height) {
	pixel *out_image = (pixel *)malloc(sizeof(pixel) * (*width) * (*height));
	merge_length = 0;
	int cell;
	right_w = (*width) % c;
	low_h = (*height) % c;
	int actual_w = w;
	int actual_h = h;
	if (right_w != 0) {
		actual_w = w + 1;
	}
	if (low_h != 0) {
		actual_h = h + 1;
	}
#pragma omp parallel for private(i,j,a,b,point)
	for (i = 0; i<actual_h; i++) { // loop for raws
		if ((low_h == 0) || i < (actual_h - 1)) {
			for (j = 0; j<actual_w; j++) { // loop for columns
				cell = j + i * actual_w;
				if ((right_w == 0) || j < (actual_w - 1)) {
					for (a = 0; a<c; a++) { // loop for raws in a mosaic module
						for (b = 0; b<c; b++) { // loop for columns in a mosaic module
												// based on the index of module, get the indexs of pixels in this module
							point = j * c + c * (*width)*i + b + a * (*width);
							// every point in this module would get the same rgb value
							module_pixel_same_color(out, new_image, point, cell);
						}
					}
				}
				if (j == (actual_w - 1) && right_w != 0) { // judge if there would be a rectangle mosaic in the rightest side of ppm for every raw
					cell = j + i * actual_w;
					for (a = 0; a<c; a++) {
						for (b = 0; b<right_w; b++) { // loop for columns in a mosaic module and it is less than c in here
							point = j * c + c * (*width)*i + b + a * (*width);
							module_pixel_same_color(out, new_image, point, cell);
						}
					}
				}
			}
		}

		if (i == (actual_h - 1) && low_h != 0) { // judge if there would be a rectangle mosaic in the lowest side of ppm
			for (j = 0; j<actual_w; j++) { // loop the lowest and left rectangles if exist
				cell = j + i * actual_w;
				if (j != (actual_w - 1)) {
					for (a = 0; a<low_h; a++) { // loop for raws in a mosaic module and it is less than c in here
						for (b = 0; b<c; b++) {
							point = j * c + c * (*width)*i + b + a * (*width);
							module_pixel_same_color(out, new_image, point, cell);
						}
					}
				}
				else { // loop the lowest and rightest corner rectangles if exist
					cell = j + i * actual_w;
					for (a = 0; a<low_h; a++) {
						for (b = 0; b<right_w; b++) {
							point = j * c + c * (*width)*i + b + a * (*width);
							module_pixel_same_color(out, new_image, point, cell);
						}
					}
				}
			}
		}
	}
}

/*
* Write mosaic for P3 files by the array--out_image.
*
* @param output		the ppm file for writing
* @param out		the pixel structure contains all rgb data for writing mosaic
* @param width      a pointer to the width
* @param height     a pointer to the height
* @param maxColor   a pointer to the maximum RGB color
* @param version	a pointer to the version of picture
* @param argv[]		command parametre
* @return           none
*/
void write_PPM3(FILE *output, pixel *out_image, int *width, int *height, int *maxColor, int *version, char *argv[]) {
	// write header
	fprintf(output, "P%d\n#Converted by ziling\n%d %d %d\n", *version, *width, *height, *maxColor);
	int lineLen = 1;
	int len = 0;
	while (len < (*width) * (*height)) {
		fprintf(output, "%d ", out_image[len].red);
		fprintf(output, "%d ", out_image[len].green);
		fprintf(output, "%d", out_image[len].blue);
		lineLen++;
		if (lineLen == (*width)) { //if achieves the maximum value,then change to a new line
			fprintf(output, "\n");
			lineLen = 1;
		}
		else if (lineLen != (*width)) { // add table aftering writing each pixel
			fprintf(output, "\t");
		}
		len++;
	}
}

/*
* Write mosaic for P6 files by the array--out_image.
*
* @param output		the ppm file for writing
* @param out		the pixel structure contains all rgb data for writing mosaic
* @param width      a pointer to the width
* @param height     a pointer to the height
* @param maxColor   a pointer to the maximum RGB color
* @param version	a pointer to the version of picture
* @param argv[]		command parametre
* @return           none
*/
void write_PPM6(FILE *output, pixel *out_image, int *width, int *height, int *maxColor, int *version, char *argv[]) {
	// write header
	int wr;
	fprintf(output, "P%d \n#Converted by ziling\n %d %d %d\n", *version, *width, *height, *maxColor);
	for (wr = 0; wr < (*width) * (*height); wr++) {
		fwrite(&(out_image[wr].red), sizeof(unsigned char), 1, output);
		fwrite(&(out_image[wr].green), sizeof(unsigned char), 1, output);
		fwrite(&(out_image[wr].blue), sizeof(unsigned char), 1, output);
	}
}

/*
* Compute the average rgb value for the outfile.
*
* @param out		the pixel structure contains all rgb data for writing mosaic
* @param width      a pointer to the width
* @param height     a pointer to the height
* @return           none
*/
void average(pixel *arr, int *width, int *height, int *avg_r, int *avg_g, int *avg_b) {
	unsigned long long sum_r = 0;
	unsigned long long sum_g = 0;
	unsigned long long sum_b = 0;
	// sum and average the triple value separately
	for (i = 0; i< (*width) * (*height); i++) {
		add_value(arr, i, &sum_r, &sum_g, &sum_b);
	}
	*avg_r = sum_r / ((*width) * (*height));
	*avg_g = sum_g / ((*width) * (*height));
	*avg_b = sum_b / ((*width) * (*height));
}

