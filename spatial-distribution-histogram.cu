#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */
#define BLOCK_WIDTH 32.0 /* amount of threads in a block */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

/* declaration of variables for gpu implememtation*/
bucket* histogram_gpu;
atom * atom_list_gpu;		

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;

struct timezone Idunno_gpu;	
struct timeval startTime_gpu, endTime_gpu;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
			// histogram[0].d_cnt++;

		} 
	}
	return 0;
}

void checkCudaErrors() {
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaerr));
        exit(1);
    }
}

/* distance GPU implementation */
__device__ double p2p_distance_gpu(atom* atom_list, int idx1, int idx2) {
    double x1 = atom_list[idx1].x_pos;
    double x2 = atom_list[idx2].x_pos;
    double y1 = atom_list[idx1].y_pos;
    double y2 = atom_list[idx2].y_pos;
    double z1 = atom_list[idx1].z_pos;
    double z2 = atom_list[idx2].z_pos;

    return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

/* SDH solution with GPU */
__global__ void SDH_gpu(atom* atom_list, int PDH_amnt, double PDH_res, bucket* histogram_gpu) {
    // __shared__ atom blockTile[BLOCK_WIDTH];
	// add to registers
	int hPos;	
	double bucketWidth = PDH_res;

    int tx = threadIdx.x;
    int bx = blockIdx.x; 
	int blockTileX = blockDim.x;

    // assign threads to blocks
    int focusPointIdx = bx * blockTileX + tx;
    double p2pDistance;

	for (int subsequentPointIdx = focusPointIdx+1; subsequentPointIdx < PDH_amnt; subsequentPointIdx++) {
		p2pDistance = p2p_distance_gpu(atom_list, focusPointIdx, subsequentPointIdx);	
		hPos = (int) (p2pDistance / bucketWidth);
		// __syncthreads();
		// histogram_gpu[hPos].d_cnt++;
		atomicAdd(&histogram_gpu[hPos].d_cnt, 1);
		// __syncthreads();
	}
}

// histogram output implementation 
void output_histogram_gpu(bucket* histogram_gpu, int num_buckets) {
	int i; 
	long long total_cnt = 0;
	for(i=0; i < num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram_gpu[i].d_cnt);
		total_cnt += histogram_gpu[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

double report_running_time(timeval startTime, timeval endTime, struct timezone Idunno) {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for GPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


// print the counts in all buckets of the histogram 
// void output_histogram(){
// 	int i; 
// 	long long total_cnt = 0;
// 	for(i=0; i< num_buckets; i++) {
// 		if(i%5 == 0) /* we print 5 buckets in a row */
// 			printf("\n%02d: ", i);
// 		printf("%15lld ", histogram[i].d_cnt);
// 		total_cnt += histogram[i].d_cnt;
// 	  	/* we also want to make sure the total distance count is correct */
// 		if(i == num_buckets - 1)	
// 			printf("\n T:%lld \n", total_cnt);
// 		else printf("| ");
// 	}
// }

void compare_histograms(bucket* histogram_cpu, bucket* histogram_gpu) {
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram_cpu[i].d_cnt - histogram_gpu[i].d_cnt);
		total_cnt += histogram_cpu[i].d_cnt - histogram_gpu[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}

}

/* arg1 = total amount of points
 * arg2 = bucket width
 */
int main(int argc, char **argv)
{
	printf("PROGRAM STARTS HERE ===============\n");
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	printf("args are %d and %f\n", PDH_acnt, PDH_res);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}

	// allocate histogram and atom list for gpu implementation
	cudaMalloc((void**)&histogram_gpu, sizeof(bucket)*num_buckets);
	cudaMalloc((void**)&atom_list_gpu, sizeof(atom)*PDH_acnt);
	cudaMemcpy(histogram_gpu, histogram, sizeof(bucket)*num_buckets, cudaMemcpyHostToDevice);
	cudaMemcpy(atom_list_gpu, atom_list, sizeof(atom)*PDH_acnt, cudaMemcpyHostToDevice);
	checkCudaErrors();


	/* start gpu counting time  ============*/
	gettimeofday(&startTime_gpu, &Idunno_gpu);
    SDH_gpu<<<ceil(PDH_acnt/BLOCK_WIDTH), BLOCK_WIDTH>>>(atom_list_gpu, PDH_acnt, PDH_res, histogram_gpu);
	cudaError_t cudaerr = cudaDeviceSynchronize();
    checkCudaErrors();
	report_running_time(startTime_gpu, endTime_gpu, Idunno_gpu);
	
	/* start cpu counting time ============== */
	gettimeofday(&startTime, &Idunno);
	PDH_baseline();
	report_running_time(startTime, endTime, Idunno);
	
	/* print out the histogram */
	printf("\nCPU IMPLEMENTATION");
	output_histogram_gpu(histogram, num_buckets);

	// send histogram_gpu back to host to output
	bucket result_histogram_gpu[num_buckets];
	cudaMemcpy(result_histogram_gpu, histogram_gpu, sizeof(bucket)*num_buckets, cudaMemcpyDeviceToHost);
	printf("\nGPU IMPLEMENTATION");
	output_histogram_gpu(result_histogram_gpu, num_buckets);

	// compare histograms
	printf("\nCOMPARISON");
	compare_histograms(histogram, result_histogram_gpu);
	
	
	cudaFree(histogram_gpu);
	free(atom_list);
	free(histogram);
	
	return 0;
}

