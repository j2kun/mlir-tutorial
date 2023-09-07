// RUN: tutorial-opt %s > %t
// RUN FileCheck %s < %t

module {
  // CHECK-LABEL: test_type_syntax
  func.func @test_type_syntax(%arg0: !poly.poly<10>) -> !poly.poly<10> {
    // CHECK: poly.poly
    return %arg0 : !poly.poly<10>
  }

  // CHECK-LABEL: test_op_syntax
  func.func @test_op_syntax(%arg0: !poly.poly<10>, %arg1: !poly.poly<10>) -> !poly.poly<10> {
    // CHECK: poly.add
    %0 = poly.add %arg0, %arg1 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
    // CHECK: poly.sub
    %1 = poly.sub %arg0, %arg1 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
    // CHECK: poly.mul
    %2 = poly.mul %arg0, %arg1 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>

    %3 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
    // CHECK: poly.from_tensor
    %4 = poly.from_tensor %3 : tensor<3xi32> -> !poly.poly<10>

    %5 = arith.constant 7 : i32
    // CHECK: poly.eval
    %6 = poly.eval %4, %5 : (!poly.poly<10>, i32) -> i32

    %7 = tensor.from_elements %arg0, %arg1 : tensor<2x!poly.poly<10>>
    // CHECK: poly.add
    %8 = poly.add %7, %7 : (tensor<2x!poly.poly<10>>, tensor<2x!poly.poly<10>>) -> tensor<2x!poly.poly<10>>
    // CHECK: poly.add
    %9 = poly.add %7, %4 : (tensor<2x!poly.poly<10>>, !poly.poly<10>) -> tensor<2x!poly.poly<10>>

    return %4 : !poly.poly<10>
  }
}
