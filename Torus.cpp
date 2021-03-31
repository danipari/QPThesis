//
// Created by Daniel on 24/03/2021.
//

#define _USE_MATH_DEFINES

#include "Torus.h"
#include <cmath>
#include <vector>
#include <fstream>
#include <boost/math//special_functions/legendre.hpp>
#include "interp.h"
#include "tools3bp.h"

Torus::Torus(std::pair<stateDict, stateTransDict> &pairSolution, int N1, int N2, int m, double K):
        N1(N1), N2(N2), m(m)
{
    std::cout << "Initializing torus from periodic orbit..." << std::endl;

    if (N2%2 != 0) // check that N2 is even
        throw std::runtime_error("N2 has to be even!");

    // Allocate input data
    stateDict stateEvol = pairSolution.first;
    stateTransDict stateTransEvol = pairSolution.second;

    // Check that pair solution is not empty
    if (stateEvol.rbegin() == stateEvol.rend() || stateTransEvol.rbegin() == stateTransEvol.rend())
        throw std::invalid_argument("Solution object is empty!");

    this->T = stateEvol.rbegin()->first;    // save period
    Matrix6d monodromyMatrix = stateTransEvol.rbegin()->second;     // find monodromy matrix

    // Find eigenvalues and eigenvector with QP term
    std::pair<std::complex<double>, Eigen::VectorXcd> eigenQP = findQPEigen(monodromyMatrix);
    std::complex<double> eigenValue = eigenQP.first;
    Eigen::VectorXcd eigenVector = eigenQP.second;

    this->rho = std::arg(eigenValue) / (2 * M_PI);

    // Create invariant circle
    auto invCircle = createInvariantCircle(eigenVector, K);
    // Propagate invariant circle
    this->data = propagateInvariantCircle(invCircle, pairSolution);

    std::cout << "Torus initialized. Transform to collocation form and use a QPSolver to converge." << std::endl;
}


void Torus::toCollocationForm()
{
    if (torusInCollocation) // avoid double collocation
        std::cout << "Torus already in collocation form!" << std::endl;

    std::map<double, MatrixCircle> torus = this->data; // retrieve data

    // Times where to save new torus are given by Lengendre roots within intervals
    Eigen::VectorXd legendreRoots = tools3BP::gaussLegendreCollocationArray(N1, m);
    std::map<double, MatrixCircle> torusCol; // allocate solution
    for (const auto &time :legendreRoots)       // initialize solution
        torusCol[time] = MatrixCircle(N2,6);

    // Iterate through each state to interpolate
    for (int stateNum = 0; stateNum < 6; stateNum++)
    {
        // Iterate along the angles
        for (int angleNum = 0; angleNum < N2; angleNum++)
        {
            // Create data set to interpolate
            int numSet = torus.size();
            std::vector<double> t(numSet), x(numSet);
            int timeNum = 0;
            for (const auto &dictCircle: torus)
            {
                t[timeNum] = dictCircle.first;
                x[timeNum] = dictCircle.second(angleNum,stateNum);
                timeNum++;
            }
            // Interpolate data
            auto stateInterp = interp_linear(1, numSet, t.data(), x.data(), legendreRoots.size(), legendreRoots.data());

            // Save data in new torus
            for (int i = 0; i < legendreRoots.size(); i++)
                torusCol[legendreRoots[i]](angleNum,stateNum) = stateInterp[i];
        }
    }
    std::cout << "Torus transformed to collocation form." << std::endl;
    this->torusInCollocation = true;
    this->data = torusCol;
}


void Torus::writeTorus(const std::string& fileName)
{
    if (!torusConverged)
        std::cout << "Warning, the torus is not converged!" << std::endl;

    std::ofstream ofile(fileName, std::ios::out);
    ofile.precision(16);
    for (auto const& dictTorus : data)
    {
        for (int i = 0; i < N2; i++)
        {
            double t = dictTorus.first;
            Vector6d x = dictTorus.second.row(i);
            ofile << t << " " << i << " " << x[0] << " " << x[1] << " " << x[2]
                  << " " << x[3] << " " << x[4] << " " << x[5] << "\n";
        }
    }
}


std::pair<std::complex<double>, Eigen::VectorXcd> Torus::findQPEigen(Matrix6d &monodromyMatrix)
{
    // Initialize eigenvalue/vector solver
    Eigen::EigenSolver<Matrix6d> eigenSolver;
    eigenSolver.compute(monodromyMatrix);
    Eigen::VectorXcd eigenValues = eigenSolver.eigenvalues();
    Eigen::MatrixXcd eigenVectors = eigenSolver.eigenvectors();

    // Iterate through eigenvalues to find the one lying on unit circle
    int elementNumber = 0;
    for (const auto &val :eigenValues)
    {
        if (std::abs(val) > 0.99 && std::abs(val) < 1.01 && std::arg(val) > 0)
        {
            std::cout << "Eigenvalue with QP form found: " << eigenValues[elementNumber] << std::endl;
            return std::pair<std::complex<double>, Eigen::VectorXcd>(eigenValues[elementNumber],eigenVectors.col(elementNumber));
        }
        elementNumber++;
    }
    throw std::runtime_error("Error! Seed orbit doesn't have QP component");
}


MatrixCircle Torus::createInvariantCircle(const Eigen::VectorXcd &eigenVector, double K) const
{
    MatrixCircle circleStates(N2,6);
    std::vector<double> state(6); // allocate state solution
    // Iterate along the circle
    for (int i = 0; i < N2; i++)
    {
        double angle = double(i)/N2;
        // Iterate on each state
        for (int j = 0; j < 6; j++)
            state[j] = K * (eigenVector[j].real() * cos(2 * M_PI * angle) - eigenVector[j].imag() * sin(2 * M_PI * angle));
        // Save circle state
        circleStates.row(i) << state[0], state[1], state[2], state[3], state[4], state[5];
    }
    return circleStates;
}


std::map<double, MatrixCircle> Torus::propagateInvariantCircle(const MatrixCircle &invCircle,
                                                               const std::pair<stateDict, stateTransDict> &pairSolution) const
{
    std::map<double, MatrixCircle> torus; // allocate solution
    const std::complex<double> imagNum(0, 1); // define imaginary number
    stateDict stateEvol = pairSolution.first;
    stateTransDict stateTransEvol = pairSolution.second;

    double i = 0;
    for (auto dictStateTrans = stateTransEvol.begin(); dictStateTrans != stateTransEvol.end(); dictStateTrans++)
    {
        double time = i / (stateTransEvol.size() - 1); // last element has time = 1
        MatrixCircle circleStates(N2,6);

        // Iterate along elements in each circle
        for (int j = 0; j < N2; j++)
        {
            Vector6d stateBase = stateEvol[dictStateTrans->first];  // state from the periodic orbit
            Matrix6d stateTrans = dictStateTrans->second;
            Vector6d stateProp = stateBase + (exp(-imagNum * (2 * M_PI * rho * time)) * stateTrans * invCircle.row(j).transpose()).real();
            // Save circle state
            circleStates.row(j) << stateProp[0], stateProp[1], stateProp[2], stateProp[3], stateProp[4], stateProp[5];
        }
        // Save circle in torus
        torus[time] = circleStates;
        i++;  // time counter
    }
    return torus;
}


Torus operator-(Torus &torus, stateDict& stateDict)
{
    if (!torus.torusInCollocation)
        std::cout << "Warning torus in collocation form, sizes may not match" << std::endl;

    Torus tanTorus = torus;
    for (auto p = std::make_pair(tanTorus.data.begin(), stateDict.begin()); p.first != tanTorus.data.end(); ++p.first, ++p.second)
    {
        for (int i = 0; i < tanTorus.N2; i++)
            tanTorus.data[p.first->first].row(i) -= p.second->second.transpose();
    }
    return tanTorus;
}


Torus operator-(Torus &torus1, Torus &torus2)
{
    if (torus1.data.size() != torus2.data.size() || torus1.N2 != torus2.N2)
        throw std::runtime_error("Torus sizes do not match!");

    Torus tanTorus = torus1;
    for (auto tor1dict = torus1.data.begin(), tor2dict = torus2.data.begin(); tor1dict != torus1.data.end(); tor1dict++, tor2dict++)
        tanTorus.data[tor1dict->first] = tor1dict->second - tor2dict->second;

    return tanTorus;
}

