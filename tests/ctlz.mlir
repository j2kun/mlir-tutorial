// RUN: mlir-opt %s --convert-math-to-funcs=convert-ctlz | FileCheck %s

func.func @main(%arg0: i32) {
  %0 = math.ctlz %arg0 : i32
  func.return
}
// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                       %[[VAL_0:.*]]: i32
// CHECK-SAME:                       ) {
// CHECK:           %[[VAL_1:.*]] = call @__mlir_math_ctlz_i32(%[[VAL_0]]) : (i32) -> i32
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @__mlir_math_ctlz_i32(
// CHECK-SAME:            %[[ARG:.*]]: i32
// CHECK-SAME:            ) -> i32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[C_32:.*]] = arith.constant 32 : i32
// CHECK:           %[[C_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[ARGCMP:.*]] = arith.cmpi eq, %[[ARG]], %[[C_0]] : i32
// CHECK:           %[[OUT:.*]] = scf.if %[[ARGCMP]] -> (i32) {
// CHECK:             scf.yield %[[C_32]] : i32
// CHECK:           } else {
// CHECK:             %[[C_1INDEX:.*]] = arith.constant 1 : index
// CHECK:             %[[C_1I32:.*]] = arith.constant 1 : i32
// CHECK:             %[[C_32INDEX:.*]] = arith.constant 32 : index
// CHECK:             %[[N:.*]] = arith.constant 0 : i32
// CHECK:             %[[FOR_RET:.*]]:2 = scf.for %[[I:.*]] = %[[C_1INDEX]] to %[[C_32INDEX]] step %[[C_1INDEX]]
// CHECK:                 iter_args(%[[ARG_ITER:.*]] = %[[ARG]], %[[N_ITER:.*]] = %[[N]]) -> (i32, i32) {
// CHECK:               %[[COND:.*]] = arith.cmpi slt, %[[ARG_ITER]], %[[C_0]] : i32
// CHECK:               %[[IF_RET:.*]]:2 = scf.if %[[COND]] -> (i32, i32) {
// CHECK:                 scf.yield %[[ARG_ITER]], %[[N_ITER]] : i32, i32
// CHECK:               } else {
// CHECK:                 %[[N_NEXT:.*]] = arith.addi %[[N_ITER]], %[[C_1I32]] : i32
// CHECK:                 %[[ARG_NEXT:.*]] = arith.shli %[[ARG_ITER]], %[[C_1I32]] : i32
// CHECK:                 scf.yield %[[ARG_NEXT]], %[[N_NEXT]] : i32, i32
// CHECK:               }
// CHECK:               scf.yield %[[IF_RET]]#0, %[[IF_RET]]#1 : i32, i32
// CHECK:             }
// CHECK:             scf.yield %[[FOR_RET]]#1 : i32
// CHECK:           }
// CHECK:           return %[[OUT]] : i32
// CHECK:         }
// NOCVT-NOT: __mlir_math_ctlz_i32
