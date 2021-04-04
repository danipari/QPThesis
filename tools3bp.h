//
// Created by Daniel on 30/03/2021.
//

#ifndef THESIS_TOOLS3BP_H
#define THESIS_TOOLS3BP_H

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef std::map<double, Vector6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Vector6d>>> stateDict;

namespace tools3BP {
    /**
     * Creates a list containing N equally spaced element and between these
     * elements, there are m elements located using the Legendre roots.
     *
     * @param N Number of equally spaced elements.
     * @param m Number of elements located using shifted transformed
     *      Legendre roots.
     * @return Eigen vector with the Gauss-Legendre collocation patter.
     */
    Eigen::VectorXd gaussLegendreCollocationArray(int N, int m);

    Eigen::MatrixXcd DFT(int N);

    Eigen::MatrixXcd IDFT(int N);

    void writeTrajectory(stateDict &state, const std::string& fileName);


}

#endif //THESIS_TOOLS3BP_H
