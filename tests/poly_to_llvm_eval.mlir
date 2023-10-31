// RUN: tutorial-opt --poly-to-llvm %s | mlir-translate --mlir-to-llvmir | llc --relocation-model=pic -filetype=obj > %t
// RUN: clang -c %project_source_dir/tests/poly_to_llvm_main.c
// RUN: clang poly_to_llvm_main.o %t -o eval_test.out
// RUN: ./eval_test.out | FileCheck %s

// CHECK: 9
func.func @test_poly_fn(%arg : i32) -> i32 {
  // 2 + 3x + 4x^2 evaluated at x=1, should be 2+3+4
  %input = poly.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly.poly<3>
  %0 = poly.eval %input, %arg: (!poly.poly<3>, i32) -> i32
  return %0 : i32
}
