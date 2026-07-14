#include "helper.cu.h"
#include "pbb/hostSkel.cu.h"
#include "radix-sort-kers.cu.h"

template<typename Meta>
void oneIteration ( const uint64_t N
                  , const int beg_bit 
                  , uint64_t* d_tmp
                  , uint16_t* d_histo
                  , uint16_t* d_histoT
                  , uint64_t* d_histoTS
                  , uint64_t* d_histoTST
                  , typename Meta::uint* keys_in
                  , typename Meta::uint* keys_out
) {

    using uint = typename Meta::uint;

    const size_t num_blocks = (N + Meta::Q*Meta::B - 1) / (Meta::Q * Meta::B);
    const size_t histo_len  = num_blocks * (1 << Meta::lgH);
    dim3 block(Meta::B, 1, 1);
    dim3 grid (num_blocks,  1, 1);

    ker1<Meta><<<num_blocks, Meta::B>>>(N, beg_bit, keys_in, d_histo);

    transposeTiled<uint16_t, 32>( d_histo, d_histoT, num_blocks, 1<<Meta::lgH ); // (inp, res, height, width)

    // need to change the operator to allow a map converting from uint16_t to uint32_t
    scanInc< AddCast<uint16_t, uint64_t> > ( 256, num_blocks * (1<<Meta::lgH), d_histoTS, d_histoT, d_tmp );

    transposeTiled<uint64_t, 32>( d_histoTS, d_histoTST, 1<<Meta::lgH, num_blocks ); // (inp, res, height, width)

    //kerHisto<Meta><<<num_blocks, 1<<Meta::lgH>>>( d_histo, d_histoTST );

    ker2<Meta><<<num_blocks, Meta::B, Meta::Q*Meta::B*sizeof(uint)>>>(N, beg_bit, d_histo, d_histoTST, keys_in, keys_out);
}

template<typename Meta>
double radixSortByKey ( typename Meta::uint* data_keys_in
                      , typename Meta::uint* data_keys_out
                      , const uint64_t N
) {
    using uint = typename Meta::uint;

    const size_t num_blocks = (N + Meta::Q*Meta::B - 1) / (Meta::Q * Meta::B);
    const size_t histo_len  = num_blocks * (1 << Meta::lgH);

    cudaFuncSetAttribute(ker1<Meta>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304); // 65536
    cudaFuncSetAttribute(ker2<Meta>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    uint16_t* d_histo;
    uint16_t* d_histoT;
    uint64_t* d_histoTS;
    uint64_t* d_histoTST;
    cudaSucceeded(cudaMalloc((void**) &d_histo,   histo_len * sizeof(uint16_t)));
    cudaSucceeded(cudaMalloc((void**) &d_histoT,  histo_len * sizeof(uint16_t)));
    cudaSucceeded(cudaMalloc((void**) &d_histoTS, histo_len * sizeof(uint64_t)));
    cudaSucceeded(cudaMalloc((void**) &d_histoTST,histo_len * sizeof(uint64_t)));

    uint64_t* d_tmp;
    cudaSucceeded(cudaMalloc((void**) &d_tmp,   MAX_BLOCK * sizeof(uint64_t)));

    uint* d_tmp_data;
    cudaSucceeded(cudaMalloc((void**) &d_tmp_data, 2 * N * sizeof(uint)));
    uint* tmp1 = d_tmp_data;
    uint* tmp2 = d_tmp_data + N;

    // setup execution parameters
    dim3 block(Meta::B, 1, 1);
    dim3 grid (num_blocks,  1, 1);

    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int r=0; r<RUNS_GPU; r++) 
    {

        uint *data_inp = data_keys_in, *data_out = tmp2;
        int upper = (Meta::bits + Meta::lgH - 1) / Meta::lgH;

        for(int ii = 0; ii < upper; ii++) {
            if(ii == upper-1) { data_out = data_keys_out; }
            int beg_bit = ii * Meta::lgH;
#if 0
            // to do: use temporary buffers instead of `data_keys_in` and `data_keys_out`
            ker1<Meta><<<num_blocks, Meta::B>>>(N, beg_bit, data_inp, d_histo);

            transposeTiled<uint16_t, 32>( d_histo, d_histoT, num_blocks, 1<<Meta::lgH ); // (inp, res, height, width)

            // need to change the operator to allow a map converting from uint16_t to uint32_t
            scanInc< AddCast<uint16_t, uint32_t> > ( 256, num_blocks * (1<<Meta::lgH), d_histoTS, d_histoT, d_tmp );

            transposeTiled<uint32_t, 32>( d_histoTS, d_histoTST, 1<<Meta::lgH, num_blocks ); // (inp, res, height, width)

            ker2<Meta><<<num_blocks, Meta::B, Meta::Q*Meta::B*sizeof(uint)>>>(N, beg_bit, d_histo, d_histoTST, data_inp, data_out);
#else
            oneIteration<Meta>( N, beg_bit, d_tmp, d_histo, d_histoT, d_histoTS, d_histoTST, data_inp, data_out );
#endif
            if(ii == 0) { data_inp = tmp2; data_out = tmp1; }
            else { uint* tmp = data_inp; data_inp = data_out; data_out = tmp; }
        }
    }
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)RUNS_GPU);

    cudaCheckError();
    gpuAssert( cudaPeekAtLastError() );

    printf("Average runtime: %f, shmsize:%lu\n", elapsed, Meta::Q*Meta::B*sizeof(uint));

    cudaFree(d_histo); cudaFree(d_histoT); cudaFree(d_histoTS); cudaFree(d_histoTST); cudaFree(d_tmp_data);

    return elapsed;
}

template<class Meta>
int debug() {
    const uint32_t block_size = Meta::B;
    const uint32_t num_elems_in_block = Meta::B * Meta::Q;
    const uint32_t num_blocks = 4;
    const uint64_t N = num_elems_in_block * num_blocks;

    // host initialization
    uint32_t* h_keys      = (uint32_t*) malloc(N*sizeof(uint32_t));
    uint32_t* h_keys_res  = (uint32_t*) malloc(N*sizeof(uint32_t));
    for(int b = 0; b < num_blocks; b++) {
        int offset = b * num_elems_in_block;
        for(int i=0; i < num_elems_in_block; i++) {
            //h_keys[offset+i] = (num_elems_in_block - i - 1) % (1<<Meta::lgH);
            h_keys[offset+i] = (rand() % (1<<Meta::lgH)) << Meta::lgH;
        }
    }

    const size_t histo_len  = (1 << Meta::lgH);
    uint16_t* h_histo16  = (uint16_t*) malloc(num_blocks*histo_len*sizeof(uint16_t));
    uint32_t* h_histo32  = (uint32_t*) malloc(num_blocks*histo_len*sizeof(uint32_t));

    uint32_t* d_keys_in;
    uint32_t* d_keys_out;
    cudaSucceeded(cudaMalloc((void**) &d_keys_in,  N * sizeof(uint32_t)));
    cudaSucceeded(cudaMemcpy(d_keys_in, h_keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    cudaSucceeded(cudaMalloc((void**) &d_keys_out, N * sizeof(uint32_t)));

    {
        uint16_t* d_histo;
        uint16_t* d_histoT;
        uint32_t* d_histoTS;
        uint32_t* d_histoTST;
        cudaSucceeded(cudaMalloc((void**) &d_histo,   (num_blocks * histo_len + 1) * sizeof(uint16_t)));
        cudaSucceeded(cudaMalloc((void**) &d_histoT,  num_blocks * histo_len * sizeof(uint16_t)));
        cudaSucceeded(cudaMalloc((void**) &d_histoTS, num_blocks * histo_len * sizeof(uint32_t)));
        cudaSucceeded(cudaMalloc((void**) &d_histoTST,num_blocks * histo_len * sizeof(uint32_t)));
        uint32_t* d_tmp;
        cudaSucceeded(cudaMalloc((void**) &d_tmp,   MAX_BLOCK * sizeof(uint32_t)));

        int beg_bit = Meta::lgH; // 0

        ker1<Meta><<<num_blocks, block_size>>>(N, beg_bit, d_keys_in, d_histo);

        transposeTiled<uint16_t, 32>( d_histo, d_histoT, num_blocks, 1<<Meta::lgH ); // (inp, res, height, width)

        scanInc< AddCast<uint16_t, uint32_t> > ( 256, num_blocks * (1<<Meta::lgH), d_histoTS, d_histoT, d_tmp );

        transposeTiled<uint32_t, 32>( d_histoTS, d_histoTST, 1<<Meta::lgH, num_blocks ); // (inp, res, height, width)
        cudaCheckError();

        size_t shmem_size = max(num_elems_in_block*sizeof(uint), 3*histo_len*sizeof(uint32_t));

        ker2<Meta><<<num_blocks, block_size, shmem_size>>>(N, beg_bit, d_histo, d_histoTST, d_keys_in, d_keys_out);
        cudaDeviceSynchronize();
        cudaCheckError();

        cudaMemcpy(h_histo16,    d_histo,    num_blocks*histo_len*sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_histo32,    d_histoTST,  num_blocks*histo_len*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_keys_res, d_keys_out, N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaCheckError();
    }

    printf("Printing key-result for (num_elems_in_block=%d): \n", num_elems_in_block);
    printMatrix<uint32_t>(h_keys_res, num_blocks, num_elems_in_block);

    printf("Printing histo-global for (histo_len=%d): \n", histo_len);
    printMatrix<uint32_t>(h_histo32, num_blocks, histo_len);

    printf("Printing histo-local for (histo_len=%d): \n", histo_len);
    printMatrix<uint16_t>(h_histo16, num_blocks, histo_len);

    return 0;
}

int main (int argc, char * argv[]) {
    initHwd();
    cudaSetDevice(1);

    //ebug<Test32x8x4>();
    //debug<RadixMeta32x8x23>();
    //return 1;

    //srand(time(nullptr));

    if (argc != 2) {
        printf("Usage: %s <size-of-array>\n", argv[0]);
        exit(1);
    }
    const uint64_t N = atoi(argv[1]);

    //Allocate and Initialize Host data with random values
    uint32_t* h_keys  = (uint32_t*) malloc(N*sizeof(uint32_t));
    uint32_t* h_keys_res  = (uint32_t*) malloc(N*sizeof(uint32_t));

    randomInitNat(h_keys, N, N*10);

    //Allocate and Initialize Device data
    uint32_t* d_keys_in;
    uint32_t* d_keys_out;
    cudaSucceeded(cudaMalloc((void**) &d_keys_in,  N * sizeof(uint32_t)));
    cudaSucceeded(cudaMemcpy(d_keys_in, h_keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    cudaSucceeded(cudaMalloc((void**) &d_keys_out, N * sizeof(uint32_t)));
    cudaSucceeded(cudaMemcpy(d_keys_out, h_keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    double elapsed = radixSortByKey<RadixMeta32x8x23>( d_keys_in, d_keys_out, N );

    cudaMemcpy(h_keys_res, d_keys_out, N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckError();

//    printf("Printing key-results: \n");
//    printMatrix<uint32_t>(h_keys_res+(N-512*23-1), 512, 23);

    bool success = validateZ(h_keys_res, N);

    printf("Our Sorting for N=%lu runs in: %.2f us, VALID: %d\n", N, elapsed, success);

    // Cleanup and closing
    cudaFree(d_keys_in); cudaFree(d_keys_out);
    free(h_keys); free(h_keys_res);

    return success ? 0 : 1;
}
