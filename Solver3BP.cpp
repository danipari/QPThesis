//
// Created by Daniel on 24/03/2021.
//
#include "Solver3BP.h"
#include <boost/numeric/odeint.hpp>


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

double Solver3BP::getJacobi(Vector6d &state)
{
    double u = massParameter;
    double x = state[0], y = state[1], z = state[2];

    double r1 = sqrt(pow((u + x), 2) + y * y + z * z);
    double r2 = sqrt(pow((1 - u - x), 2) + y * y + z * z);
    double U = (x * x + y * y) + 2 * ((1 - u) / r1 + u / r2);
    double V = state.segment(3,3).squaredNorm();

    return U - V;
}