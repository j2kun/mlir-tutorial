// RUN: tutorial-opt %s

module {
  func.func @main(%arg0: !poly.poly<10>) -> !poly.poly<10> {
    return %arg0 : !poly.poly<10>
  }
}
