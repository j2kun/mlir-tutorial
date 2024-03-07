

// CHECK-LABEL: func @test_affine_analysis_debug
func.func @test_affine_analysis_debug(%arg0: memref<64xi32>) -> memref<64xi32> {
  %alloca = memref.alloca() : memref<64xi32>
  affine.for %arg1 = 1 to 63 {
    %0 = affine.load %arg0[%arg1] : memref<64xi32>
    %1 = affine.load %arg0[%arg1 + 1] : memref<64xi32>
    %2 = affine.load %arg0[%arg1 - 1] : memref<64xi32>
    %3 = arith.addi %0, %1 : i32
    %4 = arith.addi %3, %2 : i32
    affine.store %4, %alloca[%arg1] : memref<64xi32>
  }
  return %alloca : memref<64xi32>
}
