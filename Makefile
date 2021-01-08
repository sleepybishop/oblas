CFLAGS  = -D_DEFAULT_SOURCE -O3 -g -std=c99 -Wall -march=native 
CFLAGS += -funroll-loops -ftree-vectorize -fno-inline 
#CFLAGS += -fopt-info-vec
CFLAGS += -Wno-unused

OBJ=oblas.o oblas_octet.o util.o octmat.o

#all: liboblas.a
all: testgf4 testgf16 liboblas.a tablegen

testgf16: $(OBJ)

testgf4: $(OBJ)

tablegen: tablegen.c

gf2_8_tables.h: tablegen
	./$< 8 > $@

gf2_4_tables.h: tablegen
	./$< 4 > $@

gf2_2_tables.h: tablegen
	./$< 2 > $@

oblas.o: oblas.c oblas.h gf2_2_tables.h gf2_4_tables.h gf2_8_tables.h

liboblas.a: $(OBJ)
	$(AR) rcs $@ $^

.PHONY: clean indent scan
clean:
	$(RM) *.o *.a tablegen 

indent:
	clang-format -style=LLVM -i *.c *.h

scan:
	scan-build $(MAKE) clean all

