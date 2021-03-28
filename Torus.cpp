//
// Created by Daniel on 24/03/2021.
//

#define _USE_MATH_DEFINES

#include "Torus.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <vector>
#include <fstream>
#include <boost/math//special_functions/legendre.hpp>
#include "interp.h"


Torus::Torus(std::pair<stateDict, stateTransDict> &pairSolution){
    std::cout << "Initializing torus from periodic orbit..." << std::endl;
    // Allocate input data
    stateDict stateEvol = pairSolution.first;
    stateTransDict stateTransEvol = pairSolution.second;

    // Check that pair solution is not empty
    if (stateEvol.rbegin() == stateEvol.rend() || stateTransEvol.rbegin() == stateTransEvol.rend())
        throw std::invalid_argument("Solution object is empty!");

    this->T = stateEvol.rbegin()->first;    // save period
    Matrix6d monodromyMatrix = stateTransEvol.rbegin()->second;     // find monodromy matrix

    // TODO: Add try catch
    std::pair<std::complex<double>, Eigen::VectorXcd> eigenQP = findQPEigen(monodromyMatrix);
    std::complex<double> eigenValue = eigenQP.first;
    Eigen::VectorXcd eigenVector = eigenQP.second;

    this->rho = std::arg(eigenValue) / (2 * M_PI);

    // Create invariant circle
    auto invCircle = createInvariantCircle(eigenVector, 0.001);
    // Propagate invariant circle
    this->data = propagateInvarianCircle(invCircle, pairSolution, rho);

    std::cout << "Torus initialized. Transform to collocation form and use a QPSolver to converge." << std::endl;
}


void Torus::toCollocationForm()
{
    std::map<double, MatrixCircle> torus = this->data;
    // Times where to save new torus are given by Lengendre roots within intervals
    Eigen::VectorXd legendreRoots = gaussLegendreCollocationArray(N1, m);
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
            for (auto dictCircle = torus.begin(); dictCircle != torus.end(); dictCircle++)
            {
                t[timeNum] = dictCircle->first;
                x[timeNum] = dictCircle->second(angleNum,stateNum);
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
    this->data = torusCol;
}


std::pair<std::complex<double>, Eigen::VectorXcd> Torus::findQPEigen(Matrix6d monodromyMatrix)
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


MatrixCircle Torus::createInvariantCircle(const Eigen::VectorXcd &eigenVector, const double K)
{
    MatrixCircle circleStates(N2,6);
    // Iterate along the circle
    for (int i = 0; i < N2; i++)
    {
        std::vector<double> state; // allocate state solution
        double angle = double(i)/N2;
        // Iterate on each state
        for (int j = 0; j < 6; j++)
            state.push_back(K * (eigenVector[j].real() * cos(2 * M_PI * angle) - eigenVector[j].imag() * sin(2 * M_PI * angle)));
        // Save circle state
        circleStates.row(i) << state[0], state[1], state[2], state[3], state[4], state[5];
    }
    return circleStates;
}


std::map<double, MatrixCircle> Torus::propagateInvarianCircle(const MatrixCircle &invCircle,
                                                                 const std::pair<stateDict, stateTransDict> &pairSolution,
                                                                 const double rho)
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




Eigen::VectorXd Torus::gaussLegendreCollocationArray(int N, int m)
{
    Eigen::VectorXd collocationArray(N * (m + 1) + 1); // allocate solution

    // Generate Legendre zeros
    std::vector<double> legendreZeros;
    for (auto & element :boost::math::legendre_p_zeros<double>(m))
    {
        if (element == 0) {
            legendreZeros.push_back(0);
        }
        else {
            legendreZeros.push_back(+element);
            legendreZeros.push_back(-element);
        }
    }
    sort(legendreZeros.begin(), legendreZeros.end());

    double interValue = 0;
    for (int i = 0; i <= N; i++)
    {
        interValue = double(i) / N;
        collocationArray[i * (m+1)] = interValue;
        // Break the last iteration before filling
        if (interValue == 1.0)
            break;

        // Fill Legendre roots
        int j = 1;
        for (auto & element :legendreZeros)
        {
            collocationArray[i * (m+1) + j] = interValue + (element / 2.0 + 0.5) / N;
            j++;
        }
    }
    return collocationArray;
}

void Torus::writeTorus(std::string fileName)
{
    std::ofstream ofile(fileName, std::ios::out);
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

