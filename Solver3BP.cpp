//
// Created by Daniel on 24/03/2021.
//
#include "Solver3BP.h"
#include <boost/numeric/odeint.hpp>
#include <fstream>
#include "tools3bp.h"


Solver3BP::Solver3BP(std::string &bodyPrimary, std::string &bodySecondary, double &distanceBodies)
{
    // Check if the bodies exist in the dictionary
    if ( gravitationalParameter.find(bodyPrimary) == gravitationalParameter.end() ||
         gravitationalParameter.find(bodySecondary) == gravitationalParameter.end())
        throw std::invalid_argument("Some of the bodies has not a gravitational parameter value associated.");

    this->gravParamPrimary = gravitationalParameter[bodyPrimary];
    this->gravParamSecondary = gravitationalParameter[bodySecondary];
    this->distanceBodies = distanceBodies;
    this->massParameter = gravParamSecondary / (gravParamPrimary + gravParamSecondary);
}


std::pair<stateDict, stateTransDict> Solver3BP::runSimulation(const Eigen::VectorXd &initialState, std::pair<double, double> timeSpan, double deltaTime)
{
    // Check time boundaries
    double tmin = timeSpan.first;
    double tmax = timeSpan.second;
    if (tmax < tmin) throw std::invalid_argument("tmin larger than tmax!");
    int nSteps = int((tmax - tmin) / deltaTime);

    // Bind function to object
    std::function<void(std::vector<double>  &state, std::vector<double>  &stateDer, double t)> model =
            std::bind(&Solver3BP::modelDynamicsSolver, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

    boost::numeric::odeint::runge_kutta4< std::vector<double> > rk4; // define integrator state
    stateDict solDict; // allocate solutions
    stateTransDict matDict;

    std::vector< double> state(initialState.data(), initialState.data() + initialState.size());
    Matrix6d stateTransitionMat = Eigen::MatrixXd::Identity(6,6);
    for (int i = 0; i < nSteps; i++)
    {
        double t = tmin + i * deltaTime;
        // Save solutions
        solDict[t] = Eigen::Map<Vector6d, Eigen::Unaligned>(state.data(), state.size());
        matDict[t] = stateTransitionMat;

        // Propagate state and transition matrix
        rk4.do_step( model, state , t , deltaTime );
        stateTransitionMat += deltaTime * stateTransitionDerivative(state, stateTransitionMat);
    }
    return std::pair<stateDict, stateTransDict>(solDict, matDict);
}

Vector6d Solver3BP::dynamicsCR3BP(Vector6d &state)
{
    double u = massParameter;
    double x = state[0], y = state[1], z = state[2], dx = state[3], dy = state[4], dz = state[5];

    double r1 = sqrt(pow((u + x), 2) + y * y + z * z);
    double r2 = sqrt(pow((1 - u - x), 2) + y * y + z * z);

    Vector6d stateDer;
    stateDer[0] = dx;
    stateDer[1] = dy;
    stateDer[2] = dz;
    stateDer[3] = x - (u + x) * (1 - u) / pow(r1, 3) + (1 - u - x) * u / pow(r2, 3) + 2 * dy;
    stateDer[4] = y       - y * (1 - u) / pow(r1, 3)           - y * u / pow(r2, 3) - 2 * dx;
    stateDer[5] =         - z * (1 - u) / pow(r1, 3)           - z * u / pow(r2, 3);

    return stateDer;
}


void Solver3BP::modelDynamicsSolver(std::vector<double> &state, std::vector<double> &stateDer, double t)
{
    Vector6d stateConverted = Eigen::Map<Vector6d, Eigen::Unaligned>(state.data(), state.size());
    Vector6d derivative = dynamicsCR3BP(stateConverted);
    for (int i = 0; i < 6; i++)
        stateDer[i] = derivative[i];
}


Matrix6d Solver3BP::stateTransitionDerivative(const std::vector<double> &state, const Matrix6d &phi) const
{
    double u = massParameter;
    double x = state[0], y = state[1], z = state[2];
    double r1 = sqrt(pow((u + x), 2) + y * y + z * z);
    double r2 = sqrt(pow((1 - u - x), 2) + y * y + z * z);

    Matrix6d A(6,6);
    A.setZero();
    A(0,3) = 1;
    A(1,4) = 1;
    A(2,5) = 1;

    A(3,0) = 1 + 3 * (1 - u) * pow((x + u),2) / pow(r1,5) - (1 - u) / pow(r1,3) + 3 * u * pow((1 - u - x),2) / pow(r2,5) - u / pow(r2,3);
    A(3,1) = 3 * (1 - u) * (x + u) * y / pow(r1,5) - 3 * u * (1 - u - x) * y / pow(r2,5);
    A(3,2) = 3 * (1 - u) * (x + u) * z / pow(r1,5) - 3 * u * (1 - u - x) * z / pow(r2,5);

    A(4,0) = 3 * (1 - u) * (x + u) * y / pow(r1,5) - 3 * u * (1 - u - x) * y / pow(r2,5);
    A(4,1) = 1 + 3 * (1 - u) * y * y / pow(r1,5) - (1 - u) / pow(r1,3) + 3 * u * y * y / pow(r2,5) - u / pow(r2,3);
    A(4,2) = 3 * (1 - u) * y * z / pow(r1,5) + 3 * u * y * z / pow(r2,5);

    A(5,0) = 3 * (1 - u) * (x + u) * z / pow(r1,5) - 3 * u * (1 - u - x) * z / pow(r2,5);
    A(5,1) = 3 * (1 - u) * y * z / pow(r1,5) + 3 * u * y * z / pow(r2,5);
    A(5,2) = 3 * (1 - u) * z * z / pow(r1,5) - (1 - u) / pow(r1,3) + 3 * u * z * z / pow(r2,5) - u / pow(r2,3);

    A(3,4) = 2;
    A(4,3) = -2;

    return A * phi;
}

double Solver3BP::getJacobi(const Vector6d &state)
{
    double u = massParameter;
    double x = state[0], y = state[1], z = state[2];

    double r1 = sqrt(pow((u + x), 2) + y * y + z * z);
    double r2 = sqrt(pow((1 - u - x), 2) + y * y + z * z);
    double U = (x * x + y * y) + 2 * ((1 - u) / r1 + u / r2);
    double V = state.segment(3,3).squaredNorm();

    return U - V;
}


double Solver3BP::tauHat(double t, double ti, double tii){ return (t - ti) / (tii - ti); }


double Solver3BP::lagrangePoly(double time, double timeSampleK, Eigen::VectorXd &timeSegment)
{
    double ti = timeSegment[0];
    double tii = timeSegment[timeSegment.size()-1];

    double sol = 1;
    for (const auto &timeJ :timeSegment.head(timeSegment.size()-1))
    {
        if (timeJ != timeSampleK)
            sol *= (tauHat(time, ti, tii) - tauHat(timeJ, ti, tii)) / (tauHat(timeSampleK, ti, tii) - tauHat(timeJ, ti, tii));
    }
    return sol;
}


double Solver3BP::lagrangePolyDeriv(double time, double timeSampleK, Eigen::VectorXd &timeSegment)
{
    double ti = timeSegment[0];
    double tii = timeSegment[timeSegment.size()-1];

    double sol1 = 0;
    for (const auto &timeJ1 :timeSegment.head(timeSegment.size()-1))
    {
        if (timeJ1 != timeSampleK)
        {
            double sol2 = 1;
            for (const auto &timeJ2 :timeSegment.head(timeSegment.size()-1))
            {
                if (timeJ2 != timeSampleK && timeJ2 != timeJ1)
                {
                    sol2 *= (tauHat(time, ti, tii) - tauHat(timeJ2, ti, tii)) / (tauHat(timeSampleK,ti,tii) - tauHat(timeJ2,ti,tii));
                }
            }
            sol1 += sol2 / (tauHat(timeSampleK,ti,tii) - tauHat(timeJ1,ti,tii));
        }
    }
    return sol1;
}


Vector6d Solver3BP::jacobiGradient(Vector6d &state)
{
    Vector6d jacobiGradient;
    double u = massParameter;
    double x = state[0], y = state[1], z = state[2], dx = state[3], dy = state[4], dz = state[5];
    double r1 = sqrt(pow((u + x), 2) + y * y + z * z);
    double r2 = sqrt(pow((1 - u - x), 2) + y * y + z * z);

    double dHx = - 2 * (x + u) * (1 - u) / pow(r1,3) + 2 * (1 - u - x) * u / pow(r2,3) + 2 * x;
    double dHy = - 2 *       y * (1 - u) / pow(r1,3) - 2 *           y * u / pow(r2,3) + 2 * y;
    double dHz = - 2 *       z * (1 - u) / pow(r1,3) - 2 *           z * u / pow(r2,3);
    double dHdx = - 2 * dx;
    double dHdy = - 2 * dy;
    double dHdz = - 2 * dz;

    jacobiGradient << dHx, dHy, dHz, dHdx, dHdy, dHdz;
    return jacobiGradient;
}


Eigen::MatrixXd Solver3BP::expandMatrix(Eigen::MatrixXd &matrix, int N)
{
    Eigen::MatrixXd expandedMatrix(N*6, N*6);
    expandedMatrix.setZero();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            expandedMatrix.block(i * 6, j * 6, 6, 6) = Eigen::MatrixXd::Identity(6, 6) * matrix(i,j);
    }
    return expandedMatrix;
}


Eigen::MatrixXcd Solver3BP::th2Derivative(int N)
{
    Eigen::MatrixXcd shiftMat(N,N);
    shiftMat.setZero();
    Eigen::VectorXd kCoeff(N);
    kCoeff << Eigen::ArrayXd::LinSpaced(int(N/2), 0, int(N/2)-1), 0.0, Eigen::ArrayXd::LinSpaced(int(N/2)-1, int(-N/2)+1, -1);

    int count = 0;
    for (const auto &k: kCoeff)
    {
        shiftMat(count,count) = std::complex<double>(0, k);
        count++;
    }

    Eigen::MatrixXcd dftMat = tools3BP::DFT(N), idftMat = tools3BP::IDFT(N);
    return idftMat * (shiftMat * dftMat);
}


Matrix6d Solver3BP::getJacobianCR3BP(Vector6d &state)
{
    Matrix6d subBlock; subBlock.setZero();
    double u = massParameter;
    double x = state[0], y = state[1], z = state[2];
    double r1 = sqrt(pow((u + x), 2) + y * y + z * z);
    double r2 = sqrt(pow((1 - u - x), 2) + y * y + z * z);

    subBlock(0,3) = 1;
    subBlock(1,4) = 1;
    subBlock(2,5) = 1;

    subBlock(3,0) = 1 + 3 * (1 - u) * pow((x + u),2) / pow(r1,5) - (1 - u) / pow(r1,3) + 3 * u * pow((1 - u - x),2) / pow(r2,5) - u / pow(r2,3);
    subBlock(3,1) = 3 * (1 - u) * (x + u) * y / pow(r1,5) - 3 * u * (1 - u - x) * y / pow(r2,5);
    subBlock(3,2) = 3 * (1 - u) * (x + u) * z / pow(r1,5) - 3 * u * (1 - u - x) * z / pow(r2,5);

    subBlock(4,0) = 3 * (1 - u) * (x + u) * y / pow(r1,5) - 3 * u * (1 - u - x) * y / pow(r2,5);
    subBlock(4,1) = 1 + 3 * (1 - u) * y * y / pow(r1,5) - (1 - u) / pow(r1,3) + 3 * u * y * y / pow(r2,5) - u / pow(r2,3);
    subBlock(4,2) = 3 * (1 - u) * y * z / pow(r1,5) + 3 * u * y * z / pow(r2,5);

    subBlock(5,0) = 3 * (1 - u) * (x + u) * z / pow(r1,5) - 3 * u * (1 - u - x) * z / pow(r2,5);
    subBlock(5,1) = 3 * (1 - u) * y * z / pow(r1,5) + 3 * u * y * z / pow(r2,5);
    subBlock(5,2) = 3 * (1 - u) * z * z / pow(r1,5) - (1 - u) / pow(r1,3) + 3 * u * z * z / pow(r2,5) - u / pow(r2,3);

    subBlock(3,4) = +2;
    subBlock(4,3) = -2;

    return subBlock;
}


void Solver3BP::printSparseMatrix(SparseMatrix &matrix)
{
    int height = matrix.rows(); int width = matrix.cols();
    std::ofstream ofile("sparseMatrix.dat", std::ios::out);
    for (int i = 0; i < height; i ++)
    {
        for (int j = 0; j < width; j++)
        {
            if (matrix.coeffRef(i,j) != 0)
                ofile << 'x';
            else
                ofile << ' ';
        }
        ofile << '\n';
    }
}
