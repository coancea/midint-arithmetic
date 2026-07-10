# GPU implementations for midsize integer operations

### This is the public code repository for the papers:

* (1) Cosmin E. Oancea and Stephen M. Watt, "GPU Implementations for Midsize Integer Addition and Multiplication", 2025, [bibtex](bibtext-for-cuda-arithm-paper.txt), also [available on arXiv](https://arxiv.org/abs/2405.14642)

* (2) Cosmin E. Oancea and Stephen M. Watt, "High-Level Big Integer Arithmetic in Futhark for GPUs", submitted to SYNASC, 2025, preprint available here(Futhark_Arithmetic_SYNASC.pdf)

Errata for (1): The paper incorrectly states that the prime fields for FFT multiplication use `15` out of the `16` bits and `31` out of `32` bits of the corresponding `uint` for *PrimeField32* and *PrimeField64*, respectively. This is incorrect because it does not considers the precision. We have amended the implementation, which validates now, to use `10` out of `16` bits and `25` out of `32` bits of the corresponding `uint` for *PrimeField32* and *PrimeField64*, respectively. This does not changes the overall story: FFT multiplication offers significant performance gains in comparison to classical/quadratic multiplication. The reported runtimes are still correct, except that the precision of the corresponding integers should be accordingly amended by `62.5%` and `78.125%` for *PrimeField32* and *PrimeField64*, respectively. For example, *PrimeField64* allows maximal precision of some `410000` bits instead of half a million. In principle, one can use more bits as the precision decreases, but we have not fine tuned it.

62.5% 78.125%

PrimeField32

### Code structure

* Folder [cuda](cuda) contains the CUDA implementation of addition and multiplication (quadratic and FFT) corresponding to paper (1). If you have a working CUDA installation then, in principle, you should be able to run the whole thing by using `make` inside that folder. If you want to run the corresponding CGBN tests you first need to install CGBN in folder [cuda/cgbn-tests](cuda/cgbn-tests) as summarized [here](cuda/cgbn-tests/README.md).  We validate our results for addition and multiplication by comparing them with the GMP results of the corresponding operations, but this can take a (long) time since validation is performed sequentially on CPU. You can turn validation off by setting `#define WITH_VALIDATION 0` at the beginning of the [cuda/main.cu](cuda/main.cu) file. As well, you can decrease the maximal amount of shared memory available per SM by adjusting the line [`const size_t MAX_SHM_SIZE = 163840; // 98304`](cuda/main.cu). This might prevent running the highest precision, which corresponds to roughly a half a million bits.

* Folder [futhark-ninja](futhark-ninja) contains an early Futhark implementation, before adding compiler support for register-mapping of intra-group kernels (and a manifest construct). This implementation is outdated! As the name suggests, the implementation attempts to bypass said compiler limitations by using expert knowledge of what the compiler could do at that time, e.g., it (1) fixes the sequentialization factor to `4`, (2) uses `concatenate` to short-circuit buffers in shared memory, and (3) implements addition's `scan o map` sequentialization by redundantly computing the `map` twice, i.e., per-thread `map o reduce`, then `scan` across threads, then per-thread `map o scan`.

* Folder [futhark-reg](futhark-reg) implements addition, classical/quadratic multiplication and division in Futhark and corresponds to paper (2). Since the compiler improvements are not yet available in the main Futhark branch, we have archived the Futhark compiler executable and are making it available [here](futhark-reg/futhark.tar.gz). If you are in a Linux environment and have a compatible `libgc` library, you may actually be able to use it to reproduce the experiments, provided that you have a working CUDA installation. More information about how to run futhark is provided in the corresponding folder [here](futhark-reg/README.md).

