#include <iostream>
#include <complex>
#include <string>
#include <map>
#include <Eigen/Dense>
#include <fstream>
#include "Solver3BP.h"
#include "Torus.h"
#include <algorithm>
#include <boost/math//special_functions/legendre.hpp>
#include "interp.h"
#include "QPCollocationSolver.h"

void writeState(std::string filename, std::map<double, Eigen::VectorXd> stateDict)
{
    std::ofstream ofile(filename, std::ios::out);
    for (auto const& x : stateDict)
    {
        ofile << x.first << " " << x.second[0] << " " << x.second[1] << " " << x.second[2]
              << " " << x.second[3] << " " << x.second[4] << " " << x.second[5] << "\n";
    }
}


//Eigen::VectorXd gaussLegendreCollocationArray(int N, int m)
//{
//    Eigen::VectorXd collocationArray(N * (m + 1) + 1); // allocate solution
//
//    // Generate Legendre zeros
//    std::vector<double> legendreZeros;
//    for (auto & element :boost::math::legendre_p_zeros<double>(m))
//    {
//        if (element == 0) {
//            legendreZeros.push_back(0);
//        }
//        else {
//            legendreZeros.push_back(+element);
//            legendreZeros.push_back(-element);
//        }
//    }
//    sort(legendreZeros.begin(), legendreZeros.end());
//
//    double interValue = 0;
//    for (int i = 0; i <= N; i++)
//    {
//        interValue = double(i) / N;
//        collocationArray[i * (m+1)] = interValue;
//        // Break the last iteration before filling
//        if (interValue == 1.0)
//            break;
//
//        // Fill Legendre roots
//        int j = 1;
//        for (auto & element :legendreZeros)
//        {
//            collocationArray[i * (m+1) + j] = interValue + (element / 2.0 + 0.5) / N;
//            j++;
//        }
//    }
//    return collocationArray;
//}

int main() {
    // Create solver object
    double distanceSunEarth = 149597870700; // in m
    std::string primaryBody = "Sun";
    std::string secondaryBody = "Earth";
    Solver3BP mySolver(primaryBody, secondaryBody, distanceSunEarth);

    // Propagate a trajectory
    double oneDay = 86400;
    std::pair<double, double> tSpan(0.0, 2*1.5273942481118157);
    Eigen::VectorXd initialState(6);
    initialState << 0.9889513400735400, 0, 0.0026459726921073, 0, 0.0098145137610377, 0;
    auto sol = mySolver.runSimulation(initialState, tSpan, 0.001);

    // Create torus object
    auto myTorus = Torus(sol);

    // Find tangent torus // TODO: Turn into a fucntion
    auto tanTorus = myTorus;
    for (auto p = std::make_pair(tanTorus.data.begin(), sol.first.begin()); p.first != tanTorus.data.end(); ++p.first, ++p.second)
    {
        for (int i = 0; i < tanTorus.N2; i++)
            tanTorus.data[p.first->first].row(i) -= p.second->second.transpose();
    }
    myTorus.toCollocationForm();
    tanTorus.toCollocationForm();

    // Solve collocation problem
    myTorus.rho += 1;
    std::pair<double,double> config(3.00078,0);
    QPCollocationSolver QPSolver = QPCollocationSolver(primaryBody, secondaryBody, distanceSunEarth);
    QPSolver.Solve(myTorus, myTorus, tanTorus, config);
    myTorus.writeTorus("aTorus.dat");

    return 0;
}
