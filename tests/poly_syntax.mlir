// RUN: tutorial-opt %s

module {
  func.func @main(%arg0: !poly.poly) -> !poly.poly {
    return %arg0 : !poly.poly
  }
}
