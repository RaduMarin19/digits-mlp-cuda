NVCC   := nvcc
ARCH   ?= -arch=sm_70
FLAGS  := $(ARCH) -O3 -Iinclude
TESTBINS := build/test_matmul build/test_layers

SRCS   := $(wildcard src/*.cu)
OBJS   := $(patsubst src/%.cu,build/%.o,$(SRCS))
DEPS   := $(OBJS:.o=.d)

.PHONY: all test clean
all: mlp

mlp: $(OBJS)
	$(NVCC) $(FLAGS) $^ -o $@

build/%.o: src/%.cu | build
	$(NVCC) $(FLAGS) -MMD -MP -c $< -o $@

build:
	mkdir -p build

test: $(TESTBINS)
	for t in $(TESTBINS); do ./$$t || exit 1; done


build/test_matmul: build/matmul.o tests/test_matmul.cu | build
	$(NVCC) $(FLAGS) $^ -o $@

build/test_layers: build/layers.o tests/test_layers.cu | build
	$(NVCC) $(FLAGS) $^ -o $@

clean:
	rm -rf build mlp

-include $(DEPS)