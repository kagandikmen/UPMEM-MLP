DPU_UPMEM_CLANG = dpu-upmem-dpurte-clang
DPU_UPMEM_CFLAGS += 
CFLAGS += -std=c99 -Iinclude -D_GNU_SOURCE -DVERBOSE -DDEBUG -DBATCH_SIZE=2 -DMAX_EPOCH=2 -DNUM_TRAIN_SAMPLES=8
FILES_TO_DELETE = build/

UPMEM ?= 1
ifeq ($(UPMEM), 1)
	CFLAGS += -DUPMEM
endif

SAN ?= 0
ifeq ($(SAN), 1)
	CFLAGS += -fsanitize=address,undefined,leak -fno-omit-frame-pointer -g
endif

all:
	mkdir build; \
	$(DPU_UPMEM_CLANG) $(DPU_UPMEM_CFLAGS) -Iinclude -o build/dpu_program src/dpu/dpu_program.c; \
	gcc src/host/*.c $(CFLAGS) -o build/mlp -lm `dpu-pkg-config --cflags --libs dpu`

clean:
	rm -rf $(FILES_TO_DELETE)
