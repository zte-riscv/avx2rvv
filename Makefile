# Compiler selection
CC      ?= gcc
CXX     ?= g++

# Architecture flags and cross-compilation
ifeq ($(origin CROSS_COMPILE), undefined)
    processor := $(shell uname -m)
    ifeq ($(processor), x86_64)
        ARCH_CFLAGS = -maes -mpclmul -mssse3 -msse4.2
    else ifeq ($(processor), i386)
        ARCH_CFLAGS = -maes -mpclmul -mssse3 -msse4.2
    else
        ARCH_CFLAGS =
    endif
else
    CC      = $(CROSS_COMPILE)gcc
    CXX     = $(CROSS_COMPILE)g++
    CXXFLAGS += -static
    LDFLAGS  += -static
    check_riscv := $(shell echo | $(CROSS_COMPILE)cpp -dM - | grep " __riscv_xlen " | cut -c22-)
    uname_result := $(shell uname -m)

    ifeq ($(check_riscv),64)
        processor = rv64
    else ifeq ($(uname_result),rv64imafdc)
        processor = rv64
    else ifeq ($(check_riscv),32)
        processor = rv32
    else ifeq ($(uname_result),rv32i)
        processor = rv32
    else
        $(error Unsupported cross-compiler)
    endif

    ifeq ($(processor),$(filter $(processor),i386 x86_64))
        ARCH_CFLAGS = -maes -mpclmul -mssse3 -msse4.2
    else
        ARCH_CFLAGS = -march=$(processor)gcv_zba
    endif

    ifeq ($(SIMULATOR_TYPE), qemu)
        SIMULATOR      = qemu-riscv64
        SIMULATOR_FLAGS= -cpu $(processor),v=true,zba=true,vlen=128
    else
        SIMULATOR      = spike
        SIMULATOR_FLAGS= --isa=$(processor)gcv_zba
    endif
endif

# Common flags
CXXFLAGS += -Wall -Wcast-qual -I. $(ARCH_CFLAGS)
LDFLAGS  += -lm

# Source and object files
SRCS     := tests/binding.cpp tests/common.cpp tests/debug_tools.cpp tests/sse_impl.cpp tests/main.cpp
OBJS     := $(SRCS:.cpp=.o)
deps     := $(OBJS:.o=.o.d)

EXEC     := tests/main

# Default target
all: $(EXEC)

# Build executable
$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

# Compile rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEFINED_FLAGS) -MMD -MF $@.d -c $< -o $@

# Test rule
test: $(EXEC)
ifeq ($(processor),$(filter $(processor),rv32 rv64))
	$(CC) $(ARCH_CFLAGS) -c sse2rvv.h
endif
	$(SIMULATOR) $(SIMULATOR_FLAGS) $^

# Build-test rule
build-test: $(EXEC)
ifeq ($(processor),$(filter $(processor),rv32 rv64))
	$(CC) $(ARCH_CFLAGS) -c sse2rvv.h
endif

# Formatting
format:
	@echo "Formatting files with clang-format.."
	@if ! hash clang-format 2>/dev/null; then \
        echo "clang-format is required to indent"; exit 1; \
    fi
	clang-format -i sse2rvv.h $(SRCS) tests/*.h

# Clean rules
clean:
	$(RM) $(OBJS) $(EXEC) $(deps) sse2rvv.h.gch

clean-all: clean
	$(RM) *.log

-include $(deps)

.PHONY: all clean clean-all test build-test format