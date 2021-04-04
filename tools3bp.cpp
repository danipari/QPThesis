//
// Created by Daniel on 24/03/2021.
//
#ifndef THESIS_TOOLS3BP_CPP
#define THESIS_TOOLS3BP_CPP

#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/FFT>
#include <boost/math//special_functions/legendre.hpp>
#include "tools3bp.h"


Eigen::VectorXd tools3BP::gaussLegendreCollocationArray(int N, int m)
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


Eigen::MatrixXcd tools3BP::DFT(int N)
{
    Eigen::MatrixXd input(N,N);
    input.setIdentity();
    Eigen::MatrixXcd output(N, N);
    output.setZero();

    Eigen::FFT<double> fft;
    for (int k=0; k<input.cols(); ++k)
        output.col(k) = fft.fwd( input.col(k) );

    return output;
}


Eigen::MatrixXcd tools3BP::IDFT(int N)
{
    Eigen::MatrixXcd input(N,N);
    input.setIdentity();
    Eigen::MatrixXcd output(N, N);
    output.setZero();

    Eigen::FFT<double> fft;
    for (int k=0; k<input.cols(); ++k)
        output.col(k) = fft.inv( input.col(k) );

    return output;
}


void tools3BP::writeTrajectory(stateDict &stateDict, const std::string& fileName)
{
    std::ofstream ofile(fileName, std::ios::out);
    for (const auto &stateRow : stateDict)
    {
        double time = stateRow.first;
        Vector6d state = stateRow.second;
        ofile << time << " " << state[0] << " " << state[1] << " " << state[2] << " " << state[3] << " " << state[4] << " " << state[5] << "\n";
    }
}


#endif //THESIS_TOOLS3BP_CPP

//    orbitDict readOrbit(const std::string &fileName) {
//        // Allocate state and orbit dictionary
//        Eigen::VectorXd state(6);
//        orbitDict readOrbit;
//
//        // Loop through the file and save entries
//        std::ifstream inFile(fileName, std::ios::in);
//        float t, x, y, z, dx, dy, dz;
//        while (inFile >> t >> x >> y >> z >> dx >> dy >> dz) {
//            state << x, y, z, dx, dy, dz;
//            readOrbit[t] = state;
//        }
//        return readOrbit;
//    }
//
//    void writeState(std::string filename, std::map<double, Eigen::VectorXd> stateDict)
//    {
//        std::ofstream ofile(filename, std::ios::out);
//        for (auto const& x : stateDict)
//        {
//            ofile << x.first << " " << x.second[0] << " " << x.second[1] << " " << x.second[2]
//                << " " << x.second[3] << " " << x.second[4] << " " << x.second[5] << "\n";
//        }
//    }