# Addition, Classical Multiplication and Division in Futhark

* The code implementing said operations is avalidable in folder [lib](lib).

* Validation tests are available in folder [validation](validation): 

  ** Since datasets are big we only provide direct validation of division, which has by far the most complex implementation and, in particular, performs many instances of addition and multiplication of variant precision. Hence, if division validates than everything else should.  The futhark file that performs validation of division is [validate-div.fut](validation/validate-div.fut); we also perform validation of the whole shifted inverse in [validate-shinv.fut](validation/validate-shinv.fut). If you run the scripts as above, e.g., `futhark bench --backend=cuda validate-div.fut -r 1`, and **the runtime is reported then it means it also validates**, i.e., if it does not validate it will explicitly say so and the runtime is not reported.
  
  ** Dataset [data-div-1-2048-u64.in](validation/data-div/data-div-1-2048-u64.in) is thought for debugging (text based) and contains the inputs for one instance of division; its reference output is [data-div-1-2048-u64.out](validation/data-div/data-div-1-2048-u64.out). Similar datasets are provided for the whole shifted inverse. 
  
  ** Dataset [data-bin-div-1024-2048-u64.in](validation/data-div/data-bin-div-1024-2048-u64.in) is in binary form and contains the input for `1024` division instances in precision `[2048]u64`; its reference result is [data-bin-div-1024-2048-u64.out](validation/data-div/data-bin-div-1024-2048-u64.out). Similar datasets are provided for the whole shifted inverse.

  ** In addition to this we perform sanity tests that do not require storing datasets. For example, `sanity-div-xxx.fut` performs the division `(q,r) = u / v` and checks that `u == q * v + r`, for a large batch of integers of precision `[xxx]u64`. Similarly, [validation/validate-sub-add-inv.fut](validation/validate-sub-add-inv.fut) performs `a + b - b` and checks that the result equals `a`.

* Folder [performance](performance) contains the scripts for profiling performance. The runtimes for various experiments executed on an A100---including the CUDA implementation, Futhaark with and without support for mapping intermediate arrays in register memory, and the Ninja Futhark version---are backed up in files [A100-RegFuthark-Cuda.txt](performance/A100-RegFuthark-Cuda.txt), [A100-ShmFuthark](performance/A100-ShmFuthark.txt), [A100-NinjaFuthark](performance/A100-NinjaFuthark.txt). For most parts, you can run the tests as below and the runtime in microseconds will be reported.

```
futhark bench --backend=cuda name-of-file.fut
```

However, we have encountered some serious problems when using `futhark bench` on files that have multiple entry points: in some cases the performance is bottlenecked and in others it fails to validate (erroneously). Due to this, in some cases we have split the tests so that there exists only one entry point per file, see for example the `performance/perf-poly-xxx.fut` and the `validation/sanity-div-xxx.fut` series. 

As well, in some cases `futhark bench` requires a compilation time that is longer than one is willing to wait for. This typically happens for highest precision. In those cases it is probably necessary to compile the futhark program directly and then run it by piping the corresponding dataset, as demonstrated below:

```
futhark cuda perf-poly-8192Q8.fut
futhark dataset -b -g [8192][1024][8]u64 -g [8192][1024][8]u64 | ./perf-poly-8192Q8 -n -t /dev/stderr -r 5 --entry=poly8192Q8
```

