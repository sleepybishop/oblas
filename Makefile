CPPFLAGS= -D_DEFAULT_SOURCE
CFLAGS  = -O3 -std=c99 -Wall -ftree-vectorize -mmmx -msse -msse2 -msse3 -march=native -funroll-loops
LDFLAGS +=  

all: oblas.o octmat.o

tablegen: tablegen.c
	$(CC) $(CFLAGS) -o$@ $< $(LDFLAGS)

octtables.h: tablegen
	./$< > $@
	clang-format -style=LLVM -i $@

oblas.o: oblas.c oblas.h octtables.h

.PHONY: clean
clean:
	$(RM) tablegen *.o

.PHONY: indent
indent:
	clang-format -style=LLVM -i *.c *.h
