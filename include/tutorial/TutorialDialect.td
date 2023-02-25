#ifndef TUTORIAL_DIALECT
#define TUTORIAL_DIALECT

include "mlir/IR/OpBase.td"

def Tutorial_Dialect : Dialect {
    let name = "tutorial";
    let summary = "A tutorial MLIR dialect.";
    let description = [{
      j2kun's tutorial dialect.
    }];
    let cppNamespace = "::mlir::tutorial";
}

class Tutorial_Op<string mnemonic, list<Trait> traits = []> :
        Op<Tutorial_Dialect, mnemonic, traits>;

#endif // TUTORIAL_DIALECT
