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
#include <Eigen/Sparse>
#include <cmath>

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef std::map<double, Vector6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Vector6d>>> stateDict;
typedef Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Matrix6d;
typedef std::map<double, Matrix6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Matrix6d>>> stateTransDict;
typedef Eigen::SparseMatrix<double> SparseMatrix;

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

    double getJacobi(const Vector6d &state);

    double timeDimensionalToNormalized(const double dimensionalTime){
        return dimensionalTime * sqrt((gravParamPrimary + gravParamSecondary) / pow(distanceBodies, 3));
    }

    /**
     * Computes a normalized time.
     * @param t Time to normalize.
     * @param ti Smallest time.
     * @param tii Largest time.
     * @return Normalized time.
     */
    static double tauHat(double t, double ti, double tii);

    /**
     * Returns the value of the Lagrange polynomial at a time.
     * @param time Time to compute the Lagrange polynomial.
     * @param timeSampleK Node of the time segment.
     * @param timeSegment Time segment to use, note that last element is not included!
     * @return Value of the Lagrange polynomial.
     */
    static double lagrangePoly(double time, double timeSampleK, Eigen::VectorXd &timeSegment);

    /**
     * Returns the value of the derivative of the Lagrange polynomial at a time.
     * @param time Time to compute the derivative of the Lagrange polynomial.
     * @param timeSampleK Node of the time segment.
     * @param timeSegment Time segment to use, note that last element is not included!
     * @return Value of the Lagrange polynomial derivative.
     */
    static double lagrangePolyDeriv(double time, double timeSampleK, Eigen::VectorXd &timeSegment);

    /**
     * Returns the energy gradient.
     * @param state State at which to compute the gradient.
     * @return Vector with the gradient.
     */
    Vector6d jacobiGradient(Vector6d &state);

    /**
     * Method to expand a matrix to match the Jacobian format required.
     * @param matrix Matrix to expand.
     * @param N Number of element in circle section.
     * @return Expanded (N x 6) x (N x 6) matrix.
     */
    static Eigen::MatrixXd expandMatrix(Eigen::MatrixXd &matrix, int N);

    /**
     * Method to compute the transform matrix to compute derivative wrt theta_2.
     * @param N Number of element in circle section.
     * @return Value of the transformation matrix.
     */
    static Eigen::MatrixXcd th2Derivative(int N);

    /**
     * Method that returns the Jacobian matrix of the energy equation.
     * @param state State at which to compute the Jacobian.
     * @return Matrix with the Jacobian.
     */
    Matrix6d getJacobianCR3BP(Vector6d &state);

    /**
     * Debug method used to print a sparse matrix in a file -"sparseMatrix.dat"-
     * where non-empty element are filled.
     * @param matrix Sparse matrix to print.
     */
    static void printSparseMatrix(SparseMatrix &matrix);

private:

    Matrix6d stateTransitionDerivative(const std::vector<double> &state, const Matrix6d &phi) const;

    void modelDynamicsSolver(std::vector< double> &state, std::vector< double> &stateDer, double t);
};


#endif //THESIS_SOLVER3BP_H
