//
// Created by Daniel on 24/03/2021.
//

#ifndef THESIS_SOLVER3BP_H
#define THESIS_SOLVER3BP_H

#include <stdexcept>
#include <string>
#include <map>
#include <vector>
#include <Eigen/Dense>
#include <cmath>

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef std::map<double, Vector6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Vector6d>>> stateDict;
typedef Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Matrix6d;
typedef std::map<double, Matrix6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Matrix6d>>> stateTransDict;

class Solver3BP {
public:
    double gravParamPrimary, gravParamSecondary, massParameter, distanceBodies;

    std::map<std::string, double> gravitationalParameter = {
            {"Sun", 1.32712440018E20},
            {"Earth", 3.986004418E14},
    };

    Solver3BP(std::string &bodyPrimary, std::string &bodySecondary, double &distanceBodies);

    std::pair<stateDict, stateTransDict> runSimulation(const Eigen::VectorXd &initialState, std::pair<double, double> timeSpan, double deltaTime);

    Vector6d dynamicsCR3BP(Vector6d &state);

    double getJacobi(Vector6d &state);

    double timeDimensionalToNormalized(const double dimensionalTime){
        return dimensionalTime * sqrt((gravParamPrimary + gravParamSecondary) / pow(distanceBodies, 3));
    }

private:

    Matrix6d stateTransitionDerivative(const std::vector<double> &state, const Matrix6d &phi) const;

    void modelDynamicsSolver(std::vector< double> &state, std::vector< double> &stateDer, double t);
};


#endif //THESIS_SOLVER3BP_H
