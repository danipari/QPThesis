#include <chrono>
#include <Eigen/Dense>
#include "Torus.h"
#include "PeriodicOrbit.h"
#include "POCollocationSolver.h"
#include "QPCollocationSolver.h"
#include "tools3bp.h"

int main() {

    // Create solver object
    int oneDay = 86400;
    double distanceSunEarth = 149597870700; // in m
    std::string primaryBody = "Sun";
    std::string secondaryBody = "Earth";

    auto POSolver = POCollocationSolver(primaryBody, secondaryBody, distanceSunEarth);
    double H = 3.00077;
    PeriodicOrbit myOrbit = POSolver.getLinearApproximation(POSolver.L1, POSolver.planar, 11, 6, H);
    POSolver.Solve(myOrbit, myOrbit, H);
    myOrbit.writeOrbit("perOrbit.dat");
//    QPCollocationSolver QPSolver = QPCollocationSolver(primaryBody, secondaryBody, distanceSunEarth);
//
//    // Propagate a trajectory
//    std::pair<double, double> tSpan(0.0, 2*1.5273942481118157);
//    Eigen::VectorXd initialState(6);
//    initialState << 0.9889513400735400, 0, 0.0026459726921073, 0, 0.0098145137610377, 0;
//    auto periodicSol = QPSolver.runSimulation(initialState, tSpan, 0.001);
//
//    // Create torus object
//    Torus myTorus = Torus(periodicSol, 20, 20, 5);
//    // Find tangent torus
//    Torus tanTorus = myTorus - periodicSol.first;
//    // Convert to collocation form
//    myTorus.toCollocationForm();
//    tanTorus.toCollocationForm();
//
//    // Solve collocation problem
//    std::pair<double,double> config(3.00078,0);
//    QPSolver.Solve(myTorus, myTorus, tanTorus, config, 1E-10);
//
//    auto aState = myTorus.getManifoldState(0, 0.1, true);
//    auto manifoldSol = QPSolver.runSimulation(aState, std::pair<double, double>(0.0, QPSolver.timeDimensionalToNormalized(300 * oneDay)), 0.001);
//    tools3BP::writeTrajectory(manifoldSol.first,  "aTrajectory.dat");

    return 0;
}
