// RUN: tutorial-opt %s > %t
// RUN FileCheck %s < %t

module {
  // CHECK-LABEL: test_type_syntax
  func.func @test_type_syntax(%arg0: !poly.poly<10>) -> !poly.poly<10> {
    // CHECK: poly.poly
    return %arg0 : !poly.poly<10>
  }

  // CHECK-LABEL: test_binop_syntax
  func.func @test_binop_syntax(%arg0: !poly.poly<10>, %arg1: !poly.poly<10>) -> !poly.poly<10> {
    // CHECK: poly.add
    %0 = poly.add %arg0, %arg1 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
    // CHECK: poly.sub
    %1 = poly.sub %arg0, %arg1 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
    // CHECK: poly.mul
    %2 = poly.mul %arg0, %arg1 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
    return %2 : !poly.poly<10>
  }
}
