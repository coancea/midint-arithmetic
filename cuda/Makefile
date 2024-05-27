CXX = nvcc -O3 -arch=sm_80
#-arch=sm_70
#-arch=sm_80
#-maxrregcount=96
# --ptxas-options=-v
#-arch=sm_60

HELPERS = helper.h
SRC_OUR = main.cu
KER_OUR = ker-helpers.cu.h ker-addition.cu.h ker-classic-mul.cu.h ker-fft-mul.cu.h
EXE_OUR = big-num-our

default: compile run

#.cu.o: $(SRC) $(HELPERS)
#	$(CXX) -c $@ $<


compile: $(EXE_OUR)

$(EXE_OUR): $(SRC_OUR) $(KER_OUR) $(HELPERS)
	$(CXX) $(SRC_OUR) -o $(EXE_OUR) -lgmp

run: $(EXE_OUR)
	./$(EXE_OUR) 134217728
#833333

clean:
	rm -f $(EXE_OUR)

