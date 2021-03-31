#include <chrono>
#include <Eigen/Dense>
#include "Torus.h"
#include "QPCollocationSolver.h"

int main() {
    // Create solver object
    double distanceSunEarth = 149597870700; // in m
    std::string primaryBody = "Sun";
    std::string secondaryBody = "Earth";
    QPCollocationSolver QPSolver = QPCollocationSolver(primaryBody, secondaryBody, distanceSunEarth);

    // Propagate a trajectory
    std::pair<double, double> tSpan(0.0, 2*1.5273942481118157);
    Eigen::VectorXd initialState(6);
    initialState << 0.9889513400735400, 0, 0.0026459726921073, 0, 0.0098145137610377, 0;
    auto periodicSol = QPSolver.runSimulation(initialState, tSpan, 0.001);

    // Create torus object
    Torus myTorus = Torus(periodicSol, 20, 20, 5);
    // Find tangent torus
    Torus tanTorus = myTorus - periodicSol.first;
    // Convert to collocation form
    myTorus.toCollocationForm();
    tanTorus.toCollocationForm();

    // Solve collocation problem
    std::pair<double,double> config(3.00078,0);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); // start chrono
    QPSolver.Solve(myTorus, myTorus, tanTorus, config, 1E-10);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();   // stop chrono

    std::cout << "Elapsed time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0 << " [ms]" << std::endl;

    return 0;
}
