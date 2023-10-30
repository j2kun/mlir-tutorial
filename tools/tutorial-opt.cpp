#include "lib/Conversion/PolyToStandard/PolyToStandard.h"
#include "lib/Dialect/Poly/PolyDialect.h"
#include "lib/Transform/Affine/Passes.h"
#include "lib/Transform/Arith/Passes.h"
#include "mlir/include/mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/include/mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/include/mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/include/mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/include/mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/include/mlir/Dialect/Linalg/Passes.h"
#include "mlir/include/mlir/InitAllDialects.h"
#include "mlir/include/mlir/InitAllPasses.h"
#include "mlir/include/mlir/Pass/PassManager.h"
#include "mlir/include/mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/include/mlir/Transforms/Passes.h"
//

void polyToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // Poly
  manager.addPass(mlir::tutorial::poly::createPolyToStandard());
  manager.addPass(mlir::createCanonicalizerPass());

  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // Does nothing yet!
  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  manager.addPass(mlir::createConvertSCFToCFPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createArithToLLVMConversionPass());
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::tutorial::poly::PolyDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::tutorial::registerAffinePasses();
  mlir::tutorial::registerArithPasses();

  // Dialect conversion passes
  mlir::tutorial::poly::registerPolyToStandardPasses();

  mlir::PassPipelineRegistration<>("poly-to-llvm",
                             "Run passes to lower the poly dialect to LLVM",
                             polyToLLVMPipelineBuilder);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
