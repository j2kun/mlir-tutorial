// RUN: tutorial-opt %s > %t
// RUN FileCheck %s < %t

module {
  // CHECK-LABEL: test_type_syntax
  func.func @test_type_syntax(%arg0: !poly.poly<10>) -> !poly.poly<10> {
    // CHECK: poly.poly
    return %arg0 : !poly.poly<10>
  }

  // CHECK-LABEL: test_add_syntax
  func.func @test_add_syntax(%arg0: !poly.poly<10>, %arg1: !poly.poly<10>) -> !poly.poly<10> {
    // CHECK: poly.add
    %0 = poly.add %arg0, %arg1 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
    return %0 : !poly.poly<10>
  }
}
