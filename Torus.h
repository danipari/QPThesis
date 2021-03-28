//
// Created by Daniel on 24/03/2021.
//

#ifndef THESIS_TORUS_H
#define THESIS_TORUS_H

#include <iostream>
#include <complex>
#include <map>
#include <Eigen/Dense>

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef std::map<double, Vector6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Vector6d>>> stateDict;
typedef Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Matrix6d;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixCircle;
typedef std::map<double, Matrix6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Matrix6d>>> stateTransDict;

class Torus {
public:
    int N1 = 4, N2 = 4, m = 2; // dummy values
    double T, rho;
    std::map<double, MatrixCircle> data;

    /**
     * Torus object constructor. Initialization of torus using a periodic orbit as a seed.
     *
     * @param pairSolution Pair containing the trajectory dictionary and the state transition
     * dictionary.
     */
    explicit Torus(std::pair<stateDict, stateTransDict> &pairSolution);

    /**
     * Method that transforms the torus into its collocation form.
     * The collocation form is used in the collocation process to converge the algorithm.
     * The torus is discretized into N1 -including the last section- equally-spaced segments
     * in its theta_1 dimension and each section is subdivided into m parts using the Legendre
     * roots shifted and transformed. Then the theta_2 (each circle) dimension is divided into N2
     * parts -excluding the last point.
     *
     * @param N1 Number of equally-spaced divisions in the theta_1 dimension (including last element)
     * @param N2 Number of equally-spaced divisions in the theta_2 dimension (excluding last element)
     * @param m Number of divisions based on the shifted and transformed [0,1] Legendre roots.
     */
    void toCollocationForm();

    std::map<double, MatrixCircle> getData(){ return data; }

    Eigen::VectorXd gaussLegendreCollocationArray(int N, int m);

    void writeTorus(std::string fileName);

private:
    bool torusCollocated = false;
    bool torusConverged = false;

    std::pair<std::complex<double>, Eigen::VectorXcd> findQPEigen(Matrix6d monodromyMatrix);

    MatrixCircle createInvariantCircle(const Eigen::VectorXcd &eigenVector, const double K);

    std::map<double, MatrixCircle> propagateInvarianCircle(const MatrixCircle &invCircle, const std::pair<stateDict, stateTransDict> &pairSolution, const double rho);

};


#endif //THESIS_TORUS_H
