#include <stdio.h>

// This is the function we want to call from LLVM
int test_poly_fn(int x);

int main(int argc, char *argv[]) {
  int i = 1;
  int result = test_poly_fn(i);
  printf("Result: %d\n", result);

  return 0;
}
