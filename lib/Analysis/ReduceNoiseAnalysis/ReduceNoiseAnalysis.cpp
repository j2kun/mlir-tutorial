#include "lib/Analysis/ReduceNoiseAnalysis/ReduceNoiseAnalysis.h"

#include <string>

#include "lib/Dialect/Noisy/NoisyOps.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/Value.h"
#include "ortools/linear_solver/linear_solver.h"
#include "llvm/Support/Debug.h"
#include "llvm/include/llvm/ADT/DenseMap.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"

using namespace operations_research;

namespace mlir {
namespace tutorial {

#define DEBUG_TYPE "ReduceNoiseAnalysis"

// This needs only be larger than 32, since we're hard coding i32s in this
// tutorial.
constexpr int IF_THEN_AUX = 100;

std::string nameAndLoc(Operation *op) {
  std::string varName;
  llvm::raw_string_ostream ss(varName);
  ss << op->getName() << "_" << op->getLoc();
  return ss.str();
}

ReduceNoiseAnalysis::ReduceNoiseAnalysis(Operation *op) {
  std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
  MPObjective *const objective = solver->MutableObjective();
  objective->SetMinimization();

  llvm::DenseMap<Operation *, MPVariable *> decisionVariables;
  llvm::DenseMap<Value, MPVariable *> ssaNoiseVariables;
  std::vector<MPVariable *> allVariables;

  // First walk the IR to define variables for all values and ops,
  // and constraint initial conditions.
  op->walk([&](Operation *op) {
    // FIXME: assumes all reduce_noise ops have already been removed and their
    // values forwarded.
    if (!llvm::isa<noisy::AddOp, noisy::SubOp, noisy::MulOp>(op)) {
      return;
    }

    std::string varName = "InsertReduceNoise_" + nameAndLoc(op);
    auto decisionVar = solver->MakeIntVar(0, 1, varName);
    decisionVariables.insert(std::make_pair(op, decisionVar));
    allVariables.push_back(decisionVar);
    objective->SetCoefficient(decisionVar, 1);

    int index = 0;
    for (auto operand : op->getOperands()) {
      if (ssaNoiseVariables.contains(operand)) {
        continue;
      }
      std::string varName =
          "NoiseAt_" + nameAndLoc(op) + "_arg_" + std::to_string(index++);
      auto ssaNoiseVar = solver->MakeNumVar(0, MAX_NOISE, varName);
      allVariables.push_back(ssaNoiseVar);
      ssaNoiseVariables.insert(std::make_pair(operand, ssaNoiseVar));
    }

    if (!ssaNoiseVariables.contains(op->getResult(0))) {
      std::string varName = "NoiseAt_" + nameAndLoc(op) + "_result";
      auto ssaNoiseVar = solver->MakeNumVar(0, MAX_NOISE, varName);
      allVariables.push_back(ssaNoiseVar);
      ssaNoiseVariables.insert(std::make_pair(op->getResult(0), ssaNoiseVar));
    }
  });

  // Define constraints on the noise at each SSA value
  for (auto item : ssaNoiseVariables) {
    auto value = item.first;
    auto var = item.second;
    // An input node has noise equal to the initial noise, though we're being a
    // bit sloppy by saying that EVERY block argument counts as an input node.
    // In the tutorial, there is no control flow, so these are the function
    // arguments of the main function being analyzed. A real compiler would
    // need to handle this more generically.
    if (value.isa<BlockArgument>() ||
        llvm::isa<noisy::EncodeOp>(value.getDefiningOp())) {
      MPConstraint *const ct =
          solver->MakeRowConstraint(INITIAL_NOISE, INITIAL_NOISE, "");
      ct->SetCoefficient(var, 1);
    }
  }

  std::string cstName;
  // Define the decision variable constraints
  op->walk([&](Operation *op) {
    llvm::TypeSwitch<Operation &>(*op)
        .Case<noisy::MulOp>([&](auto op) {
          // result_noise = input_noise (1 - reduce_decision) + 12 *
          // reduce_decision but linearized due to the quadratic term
          // input_noise * reduce_decision

          auto inf = solver->infinity();
          auto lhsNoiseVar = ssaNoiseVariables.lookup(op.getLhs());
          auto rhsNoiseVar = ssaNoiseVariables.lookup(op.getRhs());
          auto resultNoiseVar = ssaNoiseVariables.lookup(op.getResult());
          auto reduceNoiseDecision = decisionVariables.lookup(op);

          // result_noise >= 12 * reduce_decision
          cstName = "DecisionDynamics_" + nameAndLoc(op) + "_1";
          MPConstraint *const ct1 =
              solver->MakeRowConstraint(0.0, inf, cstName);
          ct1->SetCoefficient(resultNoiseVar, 1);
          ct1->SetCoefficient(reduceNoiseDecision, -INITIAL_NOISE);

          // result_noise <= 12 + (1 - reduce_decision) * BIG_CONST
          cstName = "DecisionDynamics_" + nameAndLoc(op) + "_2";
          MPConstraint *const ct2 = solver->MakeRowConstraint(
              0.0, INITIAL_NOISE * IF_THEN_AUX, cstName);
          ct2->SetCoefficient(resultNoiseVar, 1);
          ct2->SetCoefficient(reduceNoiseDecision, IF_THEN_AUX);

          // result_noise >= input_noise - reduce_decision * BIG_CONST
          cstName = "DecisionDynamics_" + nameAndLoc(op) + "_3";
          MPConstraint *const ct3 =
              solver->MakeRowConstraint(0.0, inf, cstName);
          ct3->SetCoefficient(resultNoiseVar, 1);
          ct3->SetCoefficient(reduceNoiseDecision, IF_THEN_AUX);
          // The input noise is the sum of the two argument noises
          if (op.getLhs() == op.getRhs()) {
            ct3->SetCoefficient(lhsNoiseVar, -2);
          } else {
            ct3->SetCoefficient(lhsNoiseVar, -1);
            ct3->SetCoefficient(rhsNoiseVar, -1);
          }

          // result_noise <= input_noise + reduce_decision * BIG_CONST
          cstName = "DecisionDynamics_" + nameAndLoc(op) + "_4";
          MPConstraint *const ct4 =
              solver->MakeRowConstraint(-inf, 0.0, cstName);
          ct4->SetCoefficient(resultNoiseVar, 1);
          ct4->SetCoefficient(reduceNoiseDecision, -IF_THEN_AUX);
          if (op.getLhs() == op.getRhs()) {
            ct4->SetCoefficient(lhsNoiseVar, -2);
          } else {
            ct4->SetCoefficient(lhsNoiseVar, -1);
            ct4->SetCoefficient(rhsNoiseVar, -1);
          }

          // ensure the noise before the reduce_noise op (input_noise)
          // also is not too large
          cstName = "DecisionDynamics_" + nameAndLoc(op) + "_5";
          MPConstraint *const ct5 =
              solver->MakeRowConstraint(0.0, MAX_NOISE, cstName);
          if (op.getLhs() == op.getRhs()) {
            ct5->SetCoefficient(lhsNoiseVar, 2);
          } else {
            ct5->SetCoefficient(lhsNoiseVar, 1);
            ct5->SetCoefficient(rhsNoiseVar, 1);
          }
        })
        .Case<noisy::AddOp, noisy::SubOp>([&](auto op) {
          // Same as for MulOp, but the noise combination function is more
          // complicated because it involves a maximum.
          auto inf = solver->infinity();
          auto lhsNoiseVar = ssaNoiseVariables.lookup(op.getLhs());
          auto rhsNoiseVar = ssaNoiseVariables.lookup(op.getRhs());
          auto resultNoiseVar = ssaNoiseVariables.lookup(op.getResult());
          auto reduceNoiseDecision = decisionVariables.lookup(op);

          // result_noise >= 12 * reduce_decision
          cstName = "DecisionDynamics_" + nameAndLoc(op) + "_1";
          MPConstraint *const ct1 =
              solver->MakeRowConstraint(0.0, inf, cstName);
          ct1->SetCoefficient(resultNoiseVar, 1);
          ct1->SetCoefficient(reduceNoiseDecision, -INITIAL_NOISE);

          // result_noise <= 12 + (1 - reduce_decision) * BIG_CONST
          cstName = "DecisionDynamics_" + nameAndLoc(op) + "_2";
          MPConstraint *const ct2 = solver->MakeRowConstraint(
              0.0, INITIAL_NOISE * IF_THEN_AUX, cstName);
          ct2->SetCoefficient(resultNoiseVar, 1);
          ct2->SetCoefficient(reduceNoiseDecision, IF_THEN_AUX);

          // for AddOp, the input noise is the max of the two argument noises
          // plus one. Model this with an extra variable Z and two constraints:
          //
          // lhs_noise + 1 <= Z <= MAX_NOISE
          // rhs_noise + 1 <= Z <= MAX_NOISE
          // input_noise := Z
          //
          // Then add theze Z variables to the minimization objective, and
          // they will be clamped to the larger of the two lower bounds.
          cstName = "Z_" + nameAndLoc(op);
          auto zVar = solver->MakeNumVar(0, MAX_NOISE, cstName);
          allVariables.push_back(zVar);
          // The objective coefficient is not all that important: the solver
          // cannot cheat by making Z larger than necessary, since making Z
          // larger than it needs to be would further increase the need to
          // insert reduce_noise ops, which would be more expensive.
          objective->SetCoefficient(zVar, 0.1);

          cstName = "DecisionDynamics_" + nameAndLoc(op) + "_z1";
          MPConstraint *const zCt1 =
              solver->MakeRowConstraint(1.0, inf, cstName);
          zCt1->SetCoefficient(zVar, 1);
          zCt1->SetCoefficient(lhsNoiseVar, -1);

          if (op.getLhs() != op.getRhs()) {
            cstName = "DecisionDynamics_" + nameAndLoc(op) + "_z2";
            MPConstraint *const zCt2 =
                solver->MakeRowConstraint(1.0, inf, cstName);
            zCt2->SetCoefficient(zVar, 1);
            zCt2->SetCoefficient(rhsNoiseVar, -1);
          }

          // result_noise >= input_noise - reduce_decision * BIG_CONST
          cstName = "DecisionDynamics_" + nameAndLoc(op) + "_3";
          MPConstraint *const ct3 =
              solver->MakeRowConstraint(0.0, inf, cstName);
          ct3->SetCoefficient(resultNoiseVar, 1);
          ct3->SetCoefficient(reduceNoiseDecision, IF_THEN_AUX);
          ct3->SetCoefficient(zVar, -1);

          // result_noise <= input_noise + reduce_decision * BIG_CONST
          cstName = "DecisionDynamics_" + nameAndLoc(op) + "_4";
          MPConstraint *const ct4 =
              solver->MakeRowConstraint(-inf, 0.0, cstName);
          ct4->SetCoefficient(resultNoiseVar, 1);
          ct4->SetCoefficient(reduceNoiseDecision, -IF_THEN_AUX);
          ct4->SetCoefficient(zVar, -1);

          // ensure the noise before the reduce_noise op (input_noise)
          // also is not too large
          cstName = "DecisionDynamics_" + nameAndLoc(op) + "_5";
          MPConstraint *const ct5 =
              solver->MakeRowConstraint(0.0, MAX_NOISE, cstName);
          ct5->SetCoefficient(zVar, 1);
        });
  });

  // Uncomment if you want to read the model's textual description,
  // generally not for those unfamiliar with linear programming.
  // std::string modelAsString;
  // solver->ExportModelAsLpFormat(false, &modelAsString);
  // LLVM_DEBUG(llvm::dbgs() << "Model string = " << modelAsString << "\n");

  solver->Solve();
  LLVM_DEBUG(llvm::dbgs() << "Problem solved in " << solver->wall_time()
                          << " milliseconds"
                          << "\n");

  LLVM_DEBUG(llvm::dbgs() << "Solution:\n");
  LLVM_DEBUG(llvm::dbgs() << "Objective value = " << objective->Value()
                          << "\n");
  // LLVM_DEBUG(llvm::dbgs() << "Variables:\n");
  // for (auto var : allVariables) {
  //   LLVM_DEBUG(llvm::dbgs() << "  " << var->name() << " = "
  //                           << var->solution_value() << "\n");
  // }

  for (auto item : decisionVariables) {
    solution.insert(std::make_pair(item.first, item.second->solution_value()));
  }
}

} // namespace tutorial
} // namespace mlir
