//
// Created by Daniel on 02/04/2021.
//

#ifndef THESIS_PERIODICORBIT_H
#define THESIS_PERIODICORBIT_H

#include <iostream>
#include <map>
#include <Eigen/Dense>

// Type definitions
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef std::map<double, Vector6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Vector6d>>> stateDict;

class PeriodicOrbit {
public:
    stateDict data;
    int N1, m;
    double T;
    bool orbitConverged = false;

    PeriodicOrbit(stateDict &orbitData, double T, int N1, int m);

    void writeOrbit(std::string fileName);
};


#endif //THESIS_PERIODICORBIT_H
