// RUN: mlir-opt %s \
// RUN:   -pass-pipeline="builtin.module( \
// RUN:      convert-math-to-funcs{convert-ctlz}, \
// RUN:      func.func(convert-scf-to-cf,convert-arith-to-llvm), \
// RUN:      convert-func-to-llvm, \
// RUN:      convert-cf-to-llvm, \
// RUN:      reconcile-unrealized-casts)" \
// RUN: | mlir-cpu-runner -e test_7i32_to_29 -entry-point-result=i32 > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_7i32_TO_29 < %t

func.func @test_7i32_to_29() -> i32 {
  %arg = arith.constant 7 : i32
  %0 = math.ctlz %arg : i32
  func.return %0 : i32
}
// CHECK_TEST_7i32_TO_29: 29


// RUN: mlir-opt %s \
// RUN:   -pass-pipeline="builtin.module( \
// RUN:      convert-math-to-funcs{convert-ctlz}, \
// RUN:      func.func(convert-scf-to-cf,convert-arith-to-llvm), \
// RUN:      convert-func-to-llvm, \
// RUN:      convert-cf-to-llvm, \
// RUN:      reconcile-unrealized-casts)" \
// RUN: | mlir-cpu-runner -e test_7i64_to_61 -entry-point-result=i64 > %t
// RUN:  FileCheck %s --check-prefix=CHECK_TEST_7i64_TO_61 < %t
func.func @test_7i64_to_61() -> i64 {
  %arg = arith.constant 7 : i64
  %0 = math.ctlz %arg : i64
  func.return %0 : i64
}
// CHECK_TEST_7i64_TO_61: 61
