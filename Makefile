CPPFLAGS= -D_DEFAULT_SOURCE
CFLAGS  = -O3 -std=c99 -Wall -ftree-vectorize -mmmx -msse -msse2 -msse3 -march=native -funroll-loops
LDFLAGS +=  

OBJ=oblas.o octmat.o

all : liboblas.a

tablegen: tablegen.o

octtables.h: tablegen
	./$< > $@
	clang-format -style=LLVM -i $@

oblas.o: oblas.c oblas.h octtables.h

liboblas.a: $(OBJ)
	$(AR) rcs $@ $^

.PHONY: clean indent scan
clean:
	$(RM) *.o *.a

indent:
	clang-format -style=LLVM -i *.c *.h

scan:
	scan-build $(MAKE) clean all

