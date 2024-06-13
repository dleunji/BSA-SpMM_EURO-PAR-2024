# G++
CXX		=g++
CXXFLAGS  =  -fpermissive -Wl,--no-as-needed -m64 -std=c++11 -fopenmp
INCLUDE = -I include

OBJ_DIR = ./bin
APP_DIR = ./programs
SRC_DIR = ./src

# CUDA
CUDA_CXXFLAGS = 
CUDA_CXXFLAGS += -std=c++11
CUDA_CXXFLAGS += -m64 
CUDA_CXXFLAGS += -O3 -arch=sm_${CUDA_ARCH}
CUDA_CXXFLAGS += -Xcompiler -fopenmp
CUDA_CXXFLAGS += -rdc=true

NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)
CUDA_INCLUDE  = -I include
# CUDA_INCLUDE += $(INCLUDE) 

CUDA_LDFLAGS = -lcusparse -lcublas

# Compile C++ source files to object files
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(SRC_DIR)/%.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -x c++ -c $< -o $@

# Compile C++ source files to object files
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_CXXFLAGS) $(CUDA_INCLUDE) $(CUDA_CXXFLAGS) -x cu -c $< -o $@

# Compile CUDA source files to object files (bsa_spmm, cusparse)
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu include/%.cuh
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_CXXFLAGS) $(CUDA_INCLUDE) $(CUDA_CXXFLAGS) -x cu -c $< -o $@

###############################################################################

bsa_spmm_benchmark: $(OBJ_DIR)/reordering_benchmark.o $(OBJ_DIR)/matrices.o $(OBJ_DIR)/reorder.o $(OBJ_DIR)/similarity.o $(OBJ_DIR)/logger.o $(OBJ_DIR)/reorder_gpu.o $(OBJ_DIR)/spmm.o
	$(NVCC) $(CUDA_CXXFLAGS) $(CUDA_INCLUDE) $(CUDA_LDFLAGS) $(CUDA_CXXFLAGS) $^ -o $@

reordering_benchmark: $(OBJ_DIR)/reordering_benchmark.o $(OBJ_DIR)/matrices.o $(OBJ_DIR)/reorder.o $(OBJ_DIR)/similarity.o $(OBJ_DIR)/logger.o $(OBJ_DIR)/reorder_gpu.o $(OBJ_DIR)/spmm.o
	$(NVCC) $(CUDA_CXXFLAGS) $(CUDA_INCLUDE) $(CUDA_LDFLAGS) $(CUDA_CXXFLAGS) $^ -o $@

clean:
	rm -rf bin/*
	rm -rf reordering_benchmark bsa_spmm_benchmark
all: bsa_spmm_benchmark reordering_benchmark

###############################################################################