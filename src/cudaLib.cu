
#include "cudaLib.cuh"
#include "cpuLib.h"
#include "curand_kernel.h"


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		y[index] += scale * x[index];
	}

}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here

	std::cout << "The current vector size is: " << vectorSize << "\n";
	std::cout << "Do you want to change it to: " << "\n";
	scanf("%d", &vectorSize);

	int size = vectorSize * sizeof(float);

	float* x_h, * y_h, * z_h;
	float scale = 2.0f;
	
	x_h = new float[vectorSize];
	y_h = new float[vectorSize];
	z_h = new float[vectorSize];

	// Initialize A and B with some values
	for (int i = 0; i < vectorSize; i++) {
		x_h[i] = (float)(rand() % 100);   
		y_h[i] = (float)(rand() % 100);
		z_h[i] = y_h[i];
	}

	//Beginning of GPU code
	float* x_d, * y_d;

	//GPU memory allocations
	cudaMalloc((void**)&x_d, size);
	cudaMalloc((void**)&y_d, size);

	//Print Block 1
	printf("\n Adding vectors : \n");
	printf(" scale = %f\n", scale);
	printf(" x_h = { ");
	for (int i = 0; i < 5; ++i) {
		printf("%3.4f, ", x_h[i]);
	}
	printf(" ... }\n");
	printf(" y_h = { ");
	for (int i = 0; i < 5; ++i) {
		printf("%3.4f, ", y_h[i]);
	}
	printf(" ... }\n");

	cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y_h, size, cudaMemcpyHostToDevice);
	
	int threadsPerBlock = 256;
	int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;

	saxpy_gpu << <blocksPerGrid, threadsPerBlock>> > (x_d, y_d, scale, size);

	cudaDeviceSynchronize();

	cudaMemcpy(z_h, y_d, size, cudaMemcpyDeviceToHost);

	printf(" z_h = { ");
	for (int i = 0; i < 5; ++i) {
		printf("%3.4f, ", z_h[i]);
	}
	printf(" ... }\n");

	int errorCount = verifyVector(x_h, y_h, z_h, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	delete[] x_h;
	delete[] y_h;
	cudaFree(x_d);
	cudaFree(y_d);

	// End of inserted code
	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints(uint64_t* pSums, uint64_t pSumSize, uint64_t sampleSize) {
	// Each thread must generate sampleSize points.

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	//RNG Thread-State-Independence
	curandState_t rng;
	curand_init(clock64(), index, 0, &rng);

	float x, y;
	uint64_t hitCount = 0;
	//uint64_t totalHitCount = 0;

	if (index < pSumSize) {
		for (int idx = 0; idx < sampleSize; ++idx) {
			x = curand_uniform(&rng);
			y = curand_uniform(&rng);

			if (int(x * x + y * y) == 0) {
				++hitCount;
			}

		}
		pSums[index] += hitCount;
		//atomicAdd((unsigned long long int*) & pSums[index], (unsigned long long int)hitCount);
		//printf("Index: %d | pSums[Index]: %lu \n", index, pSums[index]);
	}
	//printf("Index: %d | hitCount: %lu \n", index, hitCount);
}

__global__
void reduceCounts(uint64_t* pSums, uint64_t* totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	//	Inputs: pSums, pSumSize, reduceSize
	//	Outputs: totals
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("Index: %d \n",index);
	//First have to check if the thread index (which is basically going to be [0...31] is < reduceSize (also 32)
	if (index < reduceSize) {
		//Next the for-loop has to run through psums according to each thread
		//Thread 0 would cover [0..31], thread 1 would cover [32..63] -- this is [index*reduceSize + idx] thread 2 idx=4 is [2*32+4]
		/*
		
		index is going to be 0 to 31 which means that pSumsize needs to be /reducesize so that idx iterates through 

		512/32 = 16 indexs/thread 

		*/
		
		for (int idx = 0; idx < (pSumSize/reduceSize); ++idx) {
			int k = (index * (pSumSize / reduceSize) + idx);
			//printf("Position: %d \n", k);
			printf("Index: %d | idx: %d |  Position: %d \n", index, idx, k);
			totals[index] += pSums[index * (pSumSize / reduceSize) + idx];
		}
	}

	//	End of inserted code
}

int runGpuMCPi(uint64_t generateThreadCount, uint64_t sampleSize,
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	std::cout << "Here's your generate Thread Count: " << generateThreadCount << " \n"; // 1024
	std::cout << "Do you want to change it to: " << "\n";
	scanf("%" SCNu64, &generateThreadCount);

	std::cout << "Here's your Sample Size: " << sampleSize << " \n"; // 100000
	std::cout << "Do you want to change it to: " << "\n";
	scanf("%" SCNu64, &sampleSize);

	std::cout << "Here's your Reduce Thread Count: " << reduceThreadCount << " \n"; // 32
	std::cout << "Do you want to change it to: " << "\n";
	scanf("%" SCNu64, &reduceThreadCount);

	std::cout << "Here's your reduce Size: " << reduceSize << " \n"; // 32
	std::cout << "Do you want to change it to: " << "\n";
	scanf("%" SCNu64, &reduceSize);
	
	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();

	float approxPi = estimatePi(generateThreadCount, sampleSize,
		reduceThreadCount, reduceSize);

	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd - tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize,
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	double approxPi = 0;

	//      Insert code here
	uint64_t totalHitCount = 0;
	int totalThreads = generateThreadCount;
	std::cout << "Total Threads: " << totalThreads << "\n";
	uint64_t* pSums_h, * pSums_h2, * totals_h;
	pSums_h = new uint64_t[totalThreads];
	pSums_h2 = new uint64_t[totalThreads];
	totals_h = new uint64_t[totalThreads];

	// Initialize pSums with some zeros
	for (int i = 0; i < totalThreads; i++) {
		pSums_h[i] = 0;
		pSums_h2[i] = pSums_h[i];
	}

	// Initialize totals with zeros
	for (int i = 0; i < reduceSize; i++) {
		totals_h[i] = 0;
	}

	uint64_t* pSums_d, * totals_d;
	cudaMalloc((void**)&pSums_d, generateThreadCount * sizeof(uint64_t));
	cudaMalloc((void**)&totals_d, reduceThreadCount * sizeof(uint64_t));

	cudaMemcpy(pSums_d, pSums_h, generateThreadCount * sizeof(uint64_t), cudaMemcpyHostToDevice);


	//int threadsPerBlock = generateThreadCount; // 4 blocks * 256 = 1024 -- This will probably break if they send GTC > 1024
	//int blocksPerGrid = (threadsPerBlock + sampleSize - 1) / threadsPerBlock;

	int threadsPerBlock = 256; // 4 blocks * 256 = 1024 -- This will probably break if they send GTC > 1024
	int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;



	generatePoints << <blocksPerGrid, threadsPerBlock >> > (pSums_d, totalThreads, sampleSize);

	cudaDeviceSynchronize();

	//cudaMemcpy(pSums_h2, pSums_d, generateThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);


	//cudaMemcpy(pSums_d, pSums_h2, generateThreadCount * sizeof(uint64_t), cudaMemcpyHostToDevice);
	
	cudaMemcpy(totals_d, totals_h, reduceThreadCount * sizeof(uint64_t), cudaMemcpyHostToDevice);

	std::cout << "Here's your pSumSize: " << totalThreads << " \n"; // 32
	if (reduceSize > reduceThreadCount) { reduceSize = reduceThreadCount; }
	reduceCounts << <1, reduceThreadCount >> > (pSums_d, totals_d, totalThreads, reduceSize);

	cudaDeviceSynchronize();

	cudaMemcpy(totals_h, totals_d, reduceThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	printf(" totals_h = { ");
	for (int i = 0; i < reduceSize; ++i) {
		if (totals_h[i] != 0) {
			totalHitCount += totals_h[i];
			if (i < 5) {
				std::cout << totals_h[i] << ", ";
			}
		}
	}
	printf(" ... }\n");
	std::cout << "totalHitCount: " << totalHitCount << "\n ";
	
	approxPi = ((double)totalHitCount / sampleSize) / generateThreadCount;
	approxPi = approxPi * 4.0f;

	//		Insert code stop

	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}


/*
* 
* Reduce Thread count is 32 thus the reduced sum array should be 32?
* 
* 1024 threads, the reduce threads divides it such that 32 threads divde the 1024 pSum so each thread of reduce contains 32 threads of gen

Device Name: NVIDIA GeForce RTX 4070 Laptop GPU
Max Threads Per Block: 1024

printf(" pSums_h = { ");
	for (int i = 0; i < totalThreads; ++i) {
		if (pSums_h2[i] != 0) {
			std::cout << "pSums: " << pSums_h2[i] << "\n ";
		}
		//printf("%lu, ", pSums_h[i]);
	}
	printf(" ... }\n");

	25,128,139


*/