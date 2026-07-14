COMPILER?=nvcc
OPT_FLAGS?=-O3
LDFLAGS?=-lOpenCL

TEST_PBB=test-pbb

.PHONY: clean all run

default: compile run

compile: $(TEST_PBB)

$(TEST_PBB): testPBB.cu hostSkel.cu.h pbbKernels.cu.h constants.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(TEST_PBB) testPBB.cu

all: $(TEST_PBB)


run: $(TEST_PBB) $(TEST_SPMV)
	./$(TEST_PBB) 100003565 256

run-pbb: $(TEST_PBB)
	./$(TEST_PBB) 50003565 128

clean:
	rm -f $(TEST_PBB) $(TEST_SPMV)




