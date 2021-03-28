//
// Created by Daniel on 24/03/2021.
//
#ifndef THESIS_TOOLS_CPP
#define THESIS_TOOLS_CPP

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <fstream>

namespace tools {

//    orbitDict readOrbit(const std::string &fileName) {
//        // Allocate state and orbit dictionary
//        Eigen::VectorXd state(6);
//        orbitDict readOrbit;
//
//        // Loop through the file and save entries
//        std::ifstream inFile(fileName, std::ios::in);
//        float t, x, y, z, dx, dy, dz;
//        while (inFile >> t >> x >> y >> z >> dx >> dy >> dz) {
//            state << x, y, z, dx, dy, dz;
//            readOrbit[t] = state;
//        }
//        return readOrbit;
//    }

    void writeState(std::string filename, std::map<double, Eigen::VectorXd> stateDict)
    {
        std::ofstream ofile(filename, std::ios::out);
        for (auto const& x : stateDict)
        {
            ofile << x.first << " " << x.second[0] << " " << x.second[1] << " " << x.second[2]
                << " " << x.second[3] << " " << x.second[4] << " " << x.second[5] << "\n";
        }
    }

//    Eigen::VectorXd gaussLegendreCollocationArray(int N, int m)
//    {
//        Eigen::VectorXd collocationArray(N * (m + 1) + 1); // allocate solution
//        double interValue = 0;
//        for (int i = 0; i <= N+1; i++)
//        {
//            interValue += double(i) / (N + 1);
//            collocationArray[i * N] = interValue;
//            // Break the last iteration before filling
//            if (interValue == 1.0)
//                break;
//            // Fill Legendre roots
//            int j = 0;
//            for (auto & element :boost::math::legendre_p_zeros<double>(m))
//            {
//                collocationArray[i * N + j] = interValue + (element / 2.0 + 0.5) / N;
//                j++;
//            }
//        }
//        return collocationArray;
//    }

}

#endif //THESIS_TOOLS_CPP