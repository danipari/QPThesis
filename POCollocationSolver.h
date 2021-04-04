//
// Created by Daniel on 02/04/2021.
//

#ifndef THESIS_POCOLLOCATIONSOLVER_H
#define THESIS_POCOLLOCATIONSOLVER_H

#include <iostream>
#include <map>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Solver3BP.h"
#include "PeriodicOrbit.h"

// Type definitions
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef std::map<double, Vector6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Vector6d>>> stateDict;
typedef Eigen::SparseMatrix<double> SparseMatrix;

class POCollocationSolver: public Solver3BP
{
public:
    enum lagrangePoint { L1, L2, L3, Last };
    enum periodicOrbit { planar, vertical, northHalo, southHalo };
    std::map<lagrangePoint, Eigen::Vector3d> lagrangePosition;

    POCollocationSolver(std::string &bodyPrimary, std::string &bodySecondary, double &distanceBodies);

    Eigen::Vector3d getLagrangePoint(lagrangePoint &point);

    void Solve(PeriodicOrbit &orbit, PeriodicOrbit &prevOrbit, double Href, double tol=1E-10);

    Eigen::VectorXd getCollocationSyst(Eigen::VectorXd &orbitV,
                                       Eigen::VectorXd &prevOrbitV,
                                       PeriodicOrbit &orbit,
                                       double Href);

    SparseMatrix getCollocationJacobian(Eigen::VectorXd &orbitV,
                                        Eigen::VectorXd &prevOrbitV,
                                        PeriodicOrbit &orbit,
                                        double Href);

    Vector6d modelDynamicCR3BP(Vector6d &state, double l1);

    static Eigen::VectorXd orbitToArray(PeriodicOrbit &orbit);

    template <class T>
    static void fillSpBlock(std::vector<Eigen::Triplet<float>> &tripletList,
                            const T &block, int row, int col, int height, int width);


    PeriodicOrbit getLinearApproximation(lagrangePoint point, periodicOrbit orbit, int N1, int m, double Href);

    PeriodicOrbit threeDlinearApproximation(int N1, int m, double Href, lagrangePoint point, periodicOrbit orbit);


private:

    PeriodicOrbit planarLinearApproximation(int N1, int m, Eigen::Vector3d lagrangePos, double Href);

    double energyCondPlanar(const double &x, double s, double v, double Href, Eigen::Vector3d &lagrangePos);

    Vector6d state3DOrbitApproximation(double theta, double Az, lagrangePoint point, periodicOrbit orbit);

    double energyCondition3D(const double &Az, double Href, lagrangePoint point, periodicOrbit orbit);
};


#endif //THESIS_POCOLLOCATIONSOLVER_H
