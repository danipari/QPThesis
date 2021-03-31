//
// Created by Daniel on 24/03/2021.
//

#ifndef THESIS_TORUS_H
#define THESIS_TORUS_H

#include <iostream>
#include <complex>
#include <map>
#include <Eigen/Dense>

// Types definition
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef std::map<double, Vector6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Vector6d>>> stateDict;
typedef Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Matrix6d;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixCircle;
typedef std::map<double, Matrix6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Matrix6d>>> stateTransDict;

class Torus {
public:
    int N1, N2, m;
    double T, rho;
    std::map<double, MatrixCircle> data;
    bool torusInCollocation = false;
    bool torusConverged = false;

    /**
     * Torus object constructor. Initialization of torus using a periodic orbit as a seed.
     * For more information about the collocation parameters N1, N2, m check toCollocationForm()
     *
     * @param pairSolution Pair containing the trajectory dictionary and the state transition
     * dictionary.
     * @param N1 Number of equally-spaced divisions in the theta_1 dimension (including last element)
     * @param N2 Number of equally-spaced divisions in the theta_2 dimension (excluding last element)
     * @param m Number of divisions based on the shifted and transformed [0,1] Legendre roots.
     * @param K Constant used to build the first invariant circle.
     */
    explicit Torus(std::pair<stateDict, stateTransDict> &pairSolution, int N1, int N2, int m, double K = 1E-3);

    /**
     * Method that transforms the torus into its collocation form.
     * The collocation form is used in the collocation process to converge the algorithm.
     * The torus is discretized into N1 -including the last section- equally-spaced segments
     * in its theta_1 dimension and each section is subdivided into m parts using the Legendre
     * roots shifted and transformed. Then the theta_2 (each circle) dimension is divided into N2
     * parts -excluding the last point.
     * The method is separated form the constructor to allow operations with the bulk data, such as
     * finding the tangent solution for firs case.
     */
    void toCollocationForm();

    /**
     * @return A copy of the torus data
     */
    std::map<double, MatrixCircle> getData() const { return data; }

    /**
     * Writes the torus in a file containing the state of each point parameterizing the torus.
     * @param fileName Name of the file to save.
     */
    void writeTorus(const std::string &fileName);

private:
    /**
     * Method to find the eigen value and vector with quasi-periodic component of a periodic orbit.
     * @param monodromyMatrix Monodromy matrix of a periodic orbit.
     * @return Pair containing selected eigenvalue (first) and eigenvector (second).
     */
    static std::pair<std::complex<double>, Eigen::VectorXcd> findQPEigen(Matrix6d &monodromyMatrix);

    /**
     * Method to create an invariant circle using a eigenvector and scaling factor.
     * @param eigenVector Eigenvector with quasi-periodic form, from findQPEigen().
     * @param K Scaling factor applied to the invariant circle.
     * @return Matrix with each state of the circle as a row, dim = N2 x 6.
     */
    MatrixCircle createInvariantCircle(const Eigen::VectorXcd &eigenVector, double K) const;

    /**
     * Method that propagates a invariant circle using a state transition history to
     * create a seed torus. Used for creating the seed torus before convergence.
     *
     * @param invCircle Invariant circle states to be propagated.
     * @param pairSolution Solution -both states and transition matrix- evolution
     *          from where to perform the propagation.
     * @return Map containing for each N1 times [0,1] a matrix with the circle states
     *          arranged in rows, as in row 0 represents the state at angle [0,1] = 0.
     */
    std::map<double, MatrixCircle> propagateInvariantCircle(const MatrixCircle &invCircle,
                                                           const std::pair<stateDict, stateTransDict> &pairSolution) const;

};

/**
 * Subtraction operator between torus and periodic orbit soltion.
 * @param torus Torus object.
 * @param stateDict Periodic orbit solution dictionary.
 * @return Tangent torus solution.
 */
Torus operator-(Torus &torus, stateDict& stateDict);

/**
 * Subtraction operator between two tori.
 * @param torus1 First torus object.
 * @param torus2 Second torus object.
 * @return Tangent torus solution.
 */
Torus operator-(Torus &torus1, Torus &torus2);


#endif //THESIS_TORUS_H
