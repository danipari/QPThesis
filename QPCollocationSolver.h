//
// Created by Daniel on 25/03/2021.
//

#ifndef THESIS_QPCOLLOCATIONSOLVER_H
#define THESIS_QPCOLLOCATIONSOLVER_H

#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Solver3BP.h"
#include "Torus.h"

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef std::map<double, Vector6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Vector6d>>> stateDict;
typedef Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Matrix6d;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixCircle;
typedef std::map<double, Matrix6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Matrix6d>>> stateTransDict;
typedef Eigen::SparseMatrix<double> SparseMatrix;

class QPCollocationSolver : public Solver3BP {
public:
    QPCollocationSolver(std::string &bodyPrimary, std::string &bodySecondary, double &distanceBodies):
        Solver3BP(bodyPrimary, bodySecondary, distanceBodies){ }

    void Solve(Torus &torus, Torus &prevTorus, Torus &tanTorus, std::pair<double,double>config);
private:
    Eigen::VectorXd torusToArray(Torus &torus);

    void arrayToTorus(Eigen::VectorXd &array, Torus &torus);

    Eigen::VectorXd getCollocationSyst(Eigen::VectorXd &torusV,
                                       Eigen::VectorXd &prevTorusV,
                                       Eigen::VectorXd &tanTorusV,
                                       Torus &torus,
                                       std::pair<double,double>config);

    SparseMatrix getCollocationJacobian(Eigen::VectorXd &torusV,
                                        Eigen::VectorXd &prevTorusV,
                                        Eigen::VectorXd &tanTorusV,
                                        Torus &torus,
                                        std::pair<double,double>config);

    Vector6d modelDynamicCR3BP(Vector6d &state, Vector6d &dxdth2Section, double l1, double l2);

    Vector6d QPCollocationSolver::jacobiGradient(Vector6d &state);

    Eigen::VectorXd gaussLegendreCollocationArray(int N, int m);

    Matrix6d getJacobianCR3BP(Vector6d &state);

    void printSparseMatrix(SparseMatrix &matrix);
};


#endif //THESIS_QPCOLLOCATIONSOLVER_H
