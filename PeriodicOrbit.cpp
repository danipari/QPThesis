//
// Created by Daniel on 02/04/2021.
//

#include "PeriodicOrbit.h"
#include <fstream>

PeriodicOrbit::PeriodicOrbit(stateDict &orbitData, double T, int N1, int m)
    : data(orbitData), T(T), N1(N1), m(m){ }


void PeriodicOrbit::writeOrbit(std::string fileName)
{
    std::ofstream ofile(fileName, std::ios::out);
    ofile.precision(16);
    for (auto const& dictOrbit : data)
    {
        double t = dictOrbit.first;
        Vector6d x = dictOrbit.second;
        ofile << t << " " << x[0] << " " << x[1] << " " << x[2]
              << " " << x[3] << " " << x[4] << " " << x[5] << "\n";
    }
}