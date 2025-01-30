
//Max Memory efficient saxpy_gpu based on lecture 6/7

__global__ void saxpy_gpu(float* x, float* y, float scale, int size) {
	__shared__ float local_x[256];
	__shared__ float local_y[256];

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		local_x[threadIdx.x] = x[index];
		local_y[threadIdx.x] = y[index];
		__syncthreads();  // Ensure shared memory is populated before using

		local_y[threadIdx.x] += scale * local_x[threadIdx.x];

		__syncthreads();
		y[index] = local_y[threadIdx.x];
	}
}


__global__
void saxpy_gpu(float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		y[index] += scale * x[index];
	}

}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here

	int size = vectorSize * sizeof(float);

	std::cout << "Here's your size: " << (vectorSize * sizeof(float)) << " \n";
	float* x_h, * y_h, * z_h;
	float scale = 2.0f;

	x_h = new float[vectorSize];
	y_h = new float[vectorSize];
	z_h = new float[vectorSize];

	std::cout << "CPU Vectors initialized\n";

	// Initialize A and B with some values
	for (int i = 0; i < vectorSize; i++) {
		x_h[i] = (float)(rand() % 100);
		y_h[i] = (float)(rand() % 100);
		z_h[i] = y_h[i];
	}

	//printVector(x_h, size);
	//printVector(y_h, size);
	//printVector(z_h, size);

	std::cout << "Beginning GPU initialization, CPU initialization completed!\n";

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

	saxpy_gpu << <ceil(vectorSize / 256.0), 256 >> > (x_d, y_d, scale, size);

	cudaDeviceSynchronize();

	cudaMemcpy(z_h, y_d, size, cudaMemcpyDeviceToHost);

	printf(" z_h = { ");
	for (int i = 0; i < 5; ++i) {
		printf("%3.4f, ", z_h[i]);
	}
	printf(" ... }\n");

	//printVector(x_h, size);
	//printVector(y_h, size);
	//printVector(z_h, size);

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

__global__
void saxpy_gpu(float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		y[index] += scale * x[index];
	}

}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy! You're vector size is: " << vectorSize << "\n";

	//Custom vectorSize (seems to die at 2^29)

	std::cout << "Custom Vector Size: \n";
	scanf("%d", &vectorSize);

	//	Insert code here

	int size = vectorSize * sizeof(float);

	std::cout << "Here's your size: " << (vectorSize * sizeof(float)) << " \n";
	float* x_h, * y_h, * z_h;
	float scale = 2.0f;

	x_h = new float[vectorSize];
	y_h = new float[vectorSize];
	z_h = new float[vectorSize];

	std::cout << "CPU Vectors initialized\n";

	// Initialize A and B with some values
	for (int i = 0; i < vectorSize; i++) {
		x_h[i] = (float)(rand() % 100);
		y_h[i] = (float)(rand() % 100);
		z_h[i] = y_h[i];
	}

	//printVector(x_h, size);
	//printVector(y_h, size);
	//printVector(z_h, size);

	std::cout << "Beginning GPU initialization, CPU initialization completed!\n";

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

	saxpy_gpu << <ceil(vectorSize / 256.0), 256 >> > (x_d, y_d, scale, size);

	cudaDeviceSynchronize();

	cudaMemcpy(z_h, y_d, size, cudaMemcpyDeviceToHost);

	printf(" z_h = { ");
	for (int i = 0; i < 5; ++i) {
		printf("%3.4f, ", z_h[i]);
	}
	printf(" ... }\n");

	//printVector(x_h, size);
	//printVector(y_h, size);
	//printVector(z_h, size);

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

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
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
		//printf("Index: %d | pSums[Index]: %lu \n", index, pSums[index]);
	}
	//printf("Index: %d | hitCount: %lu \n", index, hitCount);
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	//	Inputs: pSums, pSumSize, reduceSize
	//	Outputs: totals
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	//First have to check if the thread index (which is basically going to be [0...31] is < reduceSize (also 32)
	if (index < reduceSize) {
		//Next the for-loop has to run through psums according to each thread
		//Thread 0 would cover [0..31], thread 1 would cover [32..63] -- this is [index*reduceSize + idx] thread 2 idx=4 is [2*32+4]
		for (int idx = 0; idx < reduceSize; ++idx) {
			totals[index] += pSums[index * reduceSize + idx];
		}
	}

	//	End of inserted code
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	std::cout << "Here's your generate Thread Count: " << generateThreadCount	<< " \n"; // 1024
	std::cout << "Here's your Sample Size: "		 << sampleSize				<< " \n"; // 100000
	std::cout << "Here's your Reduce Thread Count: " << reduceThreadCount		<< " \n"; // 32
	std::cout << "Here's your reduce Size: "		 << reduceSize				<< " \n"; // 32

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

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
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
	uint64_t* pSums_h, *pSums_h2, *totals_h;
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

	reduceCounts << <1, 32 >> > (pSums_d, totals_d, totalThreads, reduceSize);
	
	cudaDeviceSynchronize();

	cudaMemcpy(totals_h, totals_d, reduceThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	printf(" totals_h = { ");
	for (int i = 0; i < reduceSize; ++i) {
		if (totals_h[i] != 0) {
			std::cout << "pSums: " << totals_h[i] << "\n ";
			totalHitCount += totals_h[i];
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