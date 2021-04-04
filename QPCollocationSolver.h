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

// Type definitions
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef std::map<double, Vector6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Vector6d>>> stateDict;
typedef Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Matrix6d;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixCircle;
typedef std::map<double, Matrix6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Matrix6d>>> stateTransDict;
typedef Eigen::SparseMatrix<double> SparseMatrix;


class QPCollocationSolver : public Solver3BP {
public:

    /**
     * Constructs a quasi-periodic solver that uses the collocation method to converge the torus.
     * The class constructor is inherited from the Solver3BP.
     *
     * @param bodyPrimary Name of the first -more massive- primary of the system.
     * @param bodySecondary Name of the second -less massive- primary of the system.
     * @param distanceBodies Distance -in meters- between the primaries.
     */
    QPCollocationSolver(std::string &bodyPrimary, std::string &bodySecondary, double &distanceBodies):
        Solver3BP(bodyPrimary, bodySecondary, distanceBodies){ }

    /**
     * Method that converges a torus by solving the flow, continuity, periodicity and scalar
     * equations. Required a previous known torus and a tangent torus. For initialization
     * purposes the torus and prevTorus can be the same, the tanTorus can be obtained via
     * the '-' operand implemented within the class Torus.
     *
     * @param torus Torus to be converged.
     * @param prevTorus Previously known converged torus.
     * @param tanTorus Tangent directions between two known solutions.
     * @param config Contains the energy level Href and dsRef for the continuation scheme.
     * @param tol Stopping tolerance for convergence.
     */
    void Solve(Torus &torus, Torus &prevTorus, Torus &tanTorus,
               std::pair<double,double>config, double tol=1E-8, bool getManifolds=true);

    /**
     * Converts a torus parametrization to an array.
     * @param torus Torus to convert.
     * @return Array with all the states on the torus.
     */
    static Eigen::VectorXd torusToArray(Torus &torus);

    /**
     * Converts an array to its torus object.
     * @param array Array to convert.
     * @param torus Torus to update.
     */
    static void arrayToTorus(Eigen::VectorXd &array, Torus &torus);



private:

    /**
     * Returns the collocation system G(x) = 0 value.
     * It contains the flow, continuity, periodicity and scalar equations.
     *
     * @param torusV Array of torus states.
     * @param prevTorusV Array of previous torus states.
     * @param tanTorusV Array of tangent torus.
     * @param torus Torus to be converged.
     * @param config Configuration pair (Href, dsRef)
     * @return Vector with the values of G(x)
     */
    Eigen::VectorXd getCollocationSyst(Eigen::VectorXd &torusV,
                                       Eigen::VectorXd &prevTorusV,
                                       Eigen::VectorXd &tanTorusV,
                                       Torus &torus,
                                       std::pair<double,double>config);

    /**
     * Returns the Jacobian of the collocation system.
     *
     * @param torusV Array of torus states.
     * @param prevTorusV Array of previous torus states.
     * @param tanTorusV Array of tangent torus.
     * @param torus Torus to be converged.
     * @param config Configuration pair (Href, dsRef)
     * @return Matrix with all the Jacobian values.
     */
    SparseMatrix getCollocationJacobian(Eigen::VectorXd &torusV,
                                        Eigen::VectorXd &prevTorusV,
                                        Eigen::VectorXd &tanTorusV,
                                        Torus &torus,
                                        std::pair<double,double>config);

    /**
     * It computes the manifold vectors at each of the points of the torus.
     * @param torus Torus to save the manifold data.
     * @param stateTransitionMatrices Matrix with all the state transition matrices
     *      in a column.
     */
    static void findManifolds(Torus &torus, SparseMatrix &stateTransitionMatrices);

    /**
     * Method that computes the extended -with unfolding parameters- flow vector.
     * @param state State at which to compute the flow.
     * @param dxdth2Section Derivative of the state wrt theta_2.
     * @param l1 First unfolding parameter.
     * @param l2 Second unfolding parameter.
     * @return Flow vector.
     */
    Vector6d modelDynamicCR3BP(Vector6d &state, Vector6d &dxdth2Section, double l1, double l2);

    /**
     * Derivative of I2 used to compute a local constant to have an extra
     * unfolding parameter.
     * @param dxdth2Section Derivative of the state wrt theta_2.
     * @return Value of the gradient of the local constant.
     */
    static Vector6d i2Gradient(Vector6d &dxdth2Section);

    /**
     * Method to compute a rotation matrix using DFT.
     * @param N Number of element in circle section.
     * @param rho Rotating angle [0,1].
     * @return Rotation matrix.
     */
    static Eigen::MatrixXcd rotationMatrix(int N, double rho);

    /**
     * Method to compute a rotation matrix derivative using DFT.
     * @param N Number of element in circle section.
     * @param rho Rotating angle [0,1].
     * @return Rotation matrix derivative.
     */
    static Eigen::MatrixXcd rotationMatrixDer(int N, double rho);

    /**
     * Method to fill a triplet used to create a sparse matrix.
     * Triplets are used for optimization since accessing the sparse matrix
     * many times is a slow process.
     *
     * @tparam T Type of element to introduce (either EigenMatrix or EigenVector)
     * @param tripletList Triplet list being used to construct the sparse matrix.
     * @param block Block (array or matrix) to fill into the sparse matrix.
     * @param row Starting row to fill.
     * @param col Starting column to fill.
     * @param height Number of rows to fill.
     * @param width Number of columns to fill.
     */
    template <class T>
    static void fillSpBlock(std::vector<Eigen::Triplet<float>> &tripletList,
                            const T &block, int row, int col, int height, int width);
};


#endif //THESIS_QPCOLLOCATIONSOLVER_H
