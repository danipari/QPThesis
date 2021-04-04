//
// Created by Daniel on 02/04/2021.
//

#define _USE_MATH_DEFINES


#include <cmath>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/math/tools/roots.hpp>
#include <utility>
#include "POCollocationSolver.h"
#include "tools3bp.h"

using boost::math::tools::bisect;
using boost::math::tools::newton_raphson_iterate;

POCollocationSolver::POCollocationSolver(std::string &bodyPrimary, std::string &bodySecondary, double &distanceBodies)
    : Solver3BP(bodyPrimary, bodySecondary, distanceBodies)
{
    // Find values of collinear Lagrange points
    for (int lagrangeNumber = 0; lagrangeNumber < 3; lagrangeNumber++ )
    {
        auto equilibriumPoint = lagrangePoint(lagrangeNumber);
        lagrangePosition[equilibriumPoint] = getLagrangePoint(equilibriumPoint);
    }
}

void arrayToOrbit(Eigen::VectorXd &array, PeriodicOrbit &orbit)
{
    int N1 = orbit.N1, m = orbit.m, dimState = 6, numTime = 0;
    for (auto &orbitDict :orbit.data)
    {
        orbitDict.second = array.segment(numTime * dimState, dimState);
        numTime++;
    }
}


void POCollocationSolver::Solve(PeriodicOrbit &orbit, PeriodicOrbit &prevOrbit, double Href, double tol)
{
    // Transform to array
    Eigen::VectorXd orbitV = orbitToArray(orbit);
    Eigen::VectorXd prevOrbitV = orbitToArray(prevOrbit);

//    // Fill fArray
//    Eigen::VectorXd fArray = getCollocationSyst(orbitV, prevOrbitV, orbit, Href);
//    // Fill jMatrix
//    SparseMatrix jMatrix = getCollocationJacobian(orbitV, prevOrbitV, orbit, Href);
//    printSparseMatrix(jMatrix);
    Eigen::VectorXd fArray;
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver; // create solver
    double error = 1;
    int count = 0;
    while(error > tol)
    {
        // Fill fArray
        fArray = getCollocationSyst(orbitV, prevOrbitV, orbit, Href);
        // Fill jMatrix (Jacobian)
        SparseMatrix jMatrix = getCollocationJacobian(orbitV, prevOrbitV, orbit, Href);

        if (count == 0)
            solver.analyzePattern(jMatrix);   // since the pattern of jMatrix is always the same, analyze only once
        solver.factorize(jMatrix);

        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Jacobian decomposition failed.");

        Eigen::VectorXd correction = solver.solve(fArray);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Solving procedure failed.");

        error = correction.norm(); // update error
        orbitV -= correction;   // update orbit
        std::cout << "Iteration no: " << ++count << " | Error: " << error << std::endl;
    }
    // Update input torus
    arrayToOrbit(orbitV, orbit);
    orbit.orbitConverged = true;
    std::cout << "Orbit converged!" << std::endl;
}


Eigen::VectorXd POCollocationSolver::getCollocationSyst(Eigen::VectorXd &orbitV,
                                                        Eigen::VectorXd &prevOrbitV,
                                                        PeriodicOrbit &orbit,
                                                        double Href)
{
    // Retrieve data
    int N1 = orbit.N1, m = orbit.m;
    int scalarEqs = 2, dimState = 6;
    int dimVector = (N1 * (m + 1) + 1) * dimState;

    // Extra parameters
    double T = orbitV[dimVector], lambda1 = orbitV[dimVector + 1];
    double Tprev = prevOrbitV[dimVector], lambda1Prev = prevOrbitV[dimVector + 1];

    // Allocate solution
    Eigen::VectorXd fArray(dimVector + scalarEqs);
    fArray.setZero();

    Eigen::VectorXd timeArray = tools3BP::gaussLegendreCollocationArray(N1, m);
    Eigen::MatrixXd thDerMatrix = th2Derivative(N1 * (m + 1) + 1).real();
    Eigen::VectorXd dxDth = expandMatrix(thDerMatrix, N1 * (m + 1) + 1) * prevOrbitV.segment(0, dimVector);
    // Loop over equally-spaced segments
    for (int numSegment = 0; numSegment < N1; numSegment++)
    {
        // Time-set including extremes of the segment
        Eigen::VectorXd timeSegment = timeArray.segment(numSegment * (m + 1), m + 2);
        Eigen::VectorXd timeCollocation = timeSegment.segment(1, m);
        double ti = timeSegment[0], tii = timeSegment[timeSegment.size() - 1];

        // Generate flow conditions m * N1 * dimState
        Eigen::VectorXd flowCondition(m * dimState); // allocate sol
        for (int numCollocation = 0; numCollocation < m; numCollocation++)
        {
            double time = timeCollocation[numCollocation];
            int indexTime = numSegment * (m + 1) * dimState + (numCollocation + 1) * dimState;
            Vector6d state = orbitV.segment(indexTime, dimState);    // retrieve state

            // Compute the state derivative using the interpolation
            Vector6d stateDerivative; stateDerivative.setZero();
            int numDer = 0;
            for (const auto &timeK :timeSegment.head(m + 1))
            {
                int indexDerivative = numSegment * (m + 1) * dimState + numDer * dimState;
                stateDerivative += orbitV.segment(indexDerivative, dimState) * lagrangePolyDeriv(time, timeK, timeSegment);
                numDer++;
            }
            // Evaluate flow condition
            flowCondition.segment(numCollocation * dimState, dimState) =
                    stateDerivative - T * (tii - ti) * modelDynamicCR3BP(state, lambda1);
        }
        // Continuity conditions
        int numTime = 0;
        Eigen::VectorXd stateFinalInterp(dimState); stateFinalInterp.setZero();    // allocate sol
        for (const auto &timeK :timeSegment.head(timeSegment.size() - 1))
        {
            int indexSection = numSegment * (m + 1) * dimState + numTime * dimState;
            Eigen::VectorXd stateSection = orbitV.segment(indexSection, dimState);
            stateFinalInterp += stateSection * lagrangePoly(tii, timeK, timeSegment);
            numTime++;
        }
        Eigen::VectorXd continuityCond = stateFinalInterp - orbitV.segment((numSegment + 1) * (m + 1) * dimState, dimState);

        // Pack blocks into fArray
        int flowIndex = numSegment * (m+1) * dimState;
        int contIndex = (numSegment+1) * (m+1) * dimState - dimState;
        fArray.segment(flowIndex, m * dimState) = flowCondition;
        fArray.segment(contIndex, dimState) = continuityCond;
    }
    // Periodicity condition
    int periodIndex = N1 * (m + 1) * dimState;
    Eigen::VectorXd initialState = orbitV.segment(0, dimState);
    Eigen::VectorXd finalState = orbitV.segment(periodIndex, dimState);
    fArray.segment(periodIndex, dimState) = initialState - finalState;

    // Scalar equations
    double phaseCondition = 0;
    double energyCondition = 0;
    int numTime = 0;
    for (const auto &time :timeArray)
    {
        // Retrieve states
        Eigen::VectorXd state = orbitV.segment(numTime * dimState, dimState);
        Eigen::VectorXd prevState = prevOrbitV.segment(numTime * dimState, dimState);
        Vector6d dxDthSection = dxDth.segment(numTime * dimState, dimState);

        // Phase condition
        phaseCondition += (state - prevState).transpose() * dxDthSection;
        // Energy condition
        energyCondition += getJacobi(state);
        numTime++;
    }

    // Pack condition
    fArray[dimVector] = phaseCondition;
    fArray[dimVector+1] = energyCondition / (N1 * (m + 1) + 1) - Href;

    return fArray;
}


SparseMatrix POCollocationSolver::getCollocationJacobian(Eigen::VectorXd &orbitV,
                                                         Eigen::VectorXd &prevOrbitV,
                                                         PeriodicOrbit &orbit,
                                                         double Href)
{
    // Retrieve data
    int N1 = orbit.N1, m = orbit.m;
    int scalarEqs = 2, dimState = 6;
    int dimVector = (N1 * (m + 1) + 1) * dimState;

    // Extra parameters
    double T = orbitV[dimVector], lambda1 = orbitV[dimVector + 1];
    double Tprev = prevOrbitV[dimVector], lambda1Prev = prevOrbitV[dimVector + 1];

    // Allocate solution
    SparseMatrix jMatrix(dimVector + scalarEqs, dimVector + scalarEqs);
    std::vector<Eigen::Triplet<float>> tripletList;
    tripletList.reserve(int(N1 * pow((m + 1) * dimState, 2) + 3 * N1 * (m + 1) * dimState));

    Eigen::VectorXd timeArray = tools3BP::gaussLegendreCollocationArray(N1, m);
    Eigen::MatrixXd thDerMatrix = th2Derivative(N1 * (m + 1) + 1).real();
    Eigen::VectorXd dxDth = expandMatrix(thDerMatrix, N1 * (m + 1) + 1) * prevOrbitV.segment(0, dimVector);
    Eigen::MatrixXd identityBlock = Eigen::MatrixXd::Identity(dimState, dimState);
    // Loop over equally-spaced segments
    for (int numSegment = 0; numSegment < N1; numSegment++)
    {
        // Time-set including extremes of the segment
        Eigen::VectorXd timeSegment = timeArray.segment(numSegment * (m + 1), m + 2);
        Eigen::VectorXd timeCollocation = timeSegment.segment(1, m);
        double ti = timeSegment[0], tii = timeSegment[m + 1];

        // Allocate block and solution vectors
        int dimBlock = (m + 1) * dimState;
        Eigen::MatrixXd segmentBlock(dimBlock, dimBlock+dimState); segmentBlock.setZero();
        Eigen::VectorXd periodDerivative(dimBlock,1); periodDerivative.setZero();
        Eigen::VectorXd l1Derivative(dimBlock); l1Derivative.setZero();
        // Generate flow derivatives (m+1) * N1 * dimState
        for (int numSection = 0; numSection < m + 2; numSection++)
        {
            double time = timeSegment[numSection];
            int indexTime = numSegment * (m + 1) * dimState + numSection * dimState;
            Vector6d state = orbitV.segment(indexTime, dimState);    // retrieve state

            // Fill state derivative
            if (time != timeSegment[timeSegment.size() - 1])
            {
                int colFillState = numSection * dimState;
                for (int numFillState = 0; numFillState < m; numFillState++)
                    segmentBlock.block(numFillState * dimState, colFillState, dimState, dimState) =
                            identityBlock * lagrangePolyDeriv(timeSegment[numFillState + 1], time, timeSegment);
            }
            // Generate flow derivatives only in collocation points
            if ((time == timeCollocation.array()).any())
            {
                // Derivative wrt the states
                int rowFlowIndex = (numSection - 1) * dimState;
                int colFlowIndex = rowFlowIndex + dimState;
                segmentBlock.block(rowFlowIndex, colFlowIndex, dimState, dimState) -= (tii - ti) * T * getJacobianCR3BP(state);

                // Period derivative of the flow equations
                periodDerivative.segment(rowFlowIndex, dimState) = -(tii - ti) * modelDynamicCR3BP(state, lambda1);
                // Lambda 1 derivative of the flow equations
                l1Derivative.segment(rowFlowIndex, dimState) = -(tii - ti) * T * jacobiGradient(state);

            }
            // Continuity derivatives
            int rowContIndex = m * dimState;
            int colContIndex = numSection * dimState;
            Eigen::MatrixXd continuityVal(dimState, dimState);
            if (time == tii)
                continuityVal = -identityBlock;
            else
                continuityVal = identityBlock * lagrangePoly(tii, time, timeSegment);
            segmentBlock.block(rowContIndex, colContIndex , dimState, dimState) = continuityVal;
        }
        // Pack blocks and vectors
        int segmentIndex = numSegment * (m+1) * dimState;
        int heightSegment = (m+1) * dimState;
        fillSpBlock<Eigen::MatrixXd>(tripletList, segmentBlock, segmentIndex, segmentIndex, heightSegment, heightSegment + dimState);
        fillSpBlock<Eigen::VectorXd>(tripletList, periodDerivative, segmentIndex, dimVector, heightSegment, 1);         // period
        fillSpBlock<Eigen::VectorXd>(tripletList, l1Derivative, segmentIndex, dimVector+1, heightSegment, 1);       // lambda 1
    }
    // Periodicity derivatives
    int rowPeriodIndex = N1 * (m + 1) * dimState;
    fillSpBlock<Eigen::MatrixXd>(tripletList, identityBlock, rowPeriodIndex, 0, dimState, dimState);  // wrt w_00
    fillSpBlock<Eigen::MatrixXd>(tripletList, -identityBlock, rowPeriodIndex, rowPeriodIndex, dimState, dimState);  // wrt w_N0

    // Scalar equations derivatives
    int numTime = 0;
    for (const auto &time :timeArray)
    {
        // Retrieve state
        Vector6d state = orbitV.segment(numTime * dimState, dimState);
        Eigen::VectorXd prevState = prevOrbitV.segment(numTime * dimState, dimState);
        Vector6d dxDthSection = dxDth.segment(numTime * dimState, dimState);

        // Phase condition derivative
        fillSpBlock<Eigen::RowVectorXd>(tripletList, dxDthSection.transpose(),
                                        dimVector, numTime * dimState, 1, dimState);
        // Energy condition derivatives
        fillSpBlock<Eigen::RowVectorXd>(tripletList, jacobiGradient(state).transpose() / (N1 * (m + 1) + 1),
                                        dimVector+1, numTime * dimState, 1, dimState);
        numTime++;
    }

    jMatrix.setFromTriplets(tripletList.begin(), tripletList.end());
    return jMatrix;
}


template <class T>
void POCollocationSolver::fillSpBlock(std::vector<Eigen::Triplet<float>> &tripletList, const T &block, int row, int col, int height, int width)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            double val = block(i,j);
            if (val != 0)
                tripletList.emplace_back(Eigen::Triplet<float>(row + i, col + j, val));
        }
    }
}


Vector6d POCollocationSolver::modelDynamicCR3BP(Vector6d &state, double l1)
{
    return dynamicsCR3BP(state) + l1 * jacobiGradient(state);
}


Eigen::VectorXd POCollocationSolver::orbitToArray(PeriodicOrbit &orbit)
{
    int dimVector = (orbit.N1 * (orbit.m + 1) + 1) * 6;
    Eigen::VectorXd orbitArray( dimVector + 2);   // allocate solution
    orbitArray.setZero();

    int count = 0;
    for (const auto &stateDict :orbit.data)
    {
        orbitArray.segment(count * 6, 6) = stateDict.second;
        count++;
    }
    orbitArray[dimVector] = orbit.T;

    return orbitArray;
}


double POCollocationSolver::energyCondPlanar(const double &x, double s, double v, double Href, Eigen::Vector3d &lagrangePos)
{
    Vector6d state = Vector6d(x, 0, 0, 0, -s * v * x, 0);
    state.segment(0,3) += lagrangePos;
    return getJacobi(state) - Href;
}

struct TerminationCondition  {
    bool operator() (double min, double max)  {
        return abs(min - max) <= 1E-5;
    }
};

PeriodicOrbit POCollocationSolver::planarLinearApproximation(int N1, int m, Eigen::Vector3d lagrangePos, double Href)
{
    stateDict periodicOrbitApproximation;   // allocate solution
    Eigen::VectorXd thetaArray = tools3BP::gaussLegendreCollocationArray(N1, m);

    // Compute constants
    double s = 2.087; // TODO: Generalize (this is only for L1)
    double v = 3.229;

    auto energyFun = std::bind(&POCollocationSolver::energyCondPlanar, this, std::placeholders::_1, s, v, Href, lagrangePos);
    auto result = bisect(energyFun, 0.0 ,1.0, TerminationCondition());
    double x0 = (result.first + result.second) / 2;

    for (const auto &t : thetaArray)
    {
        double x = x0 * cos(2 * M_PI * t);
        double y = -v * x0 * sin(2 * M_PI * t);
        double dx = -x0 * s * sin(2 * M_PI * t);
        double dy = -v * s * x0 * cos(2 * M_PI * t);

        Vector6d state = Vector6d(x, y, 0, dx, dy, 0);
        state.segment(0, 3) += lagrangePos;
        periodicOrbitApproximation[t] = state;
    }
    double T = 2 * M_PI / s;
    return PeriodicOrbit(periodicOrbitApproximation, T, N1, m);
}

PeriodicOrbit POCollocationSolver::getLinearApproximation(lagrangePoint point, periodicOrbit orbit, int N1, int m, double H)
{
    Eigen::Vector3d lPosition = lagrangePosition[point];

    if (orbit == planar)
        return planarLinearApproximation(N1, m, lPosition, H);
    else
        return threeDlinearApproximation(N1, m, H, point, orbit);
}


Eigen::Vector3d POCollocationSolver::getLagrangePoint(lagrangePoint &point)
{
    Eigen::Vector3d posLagrange; // allocate solution
    double u = massParameter, guess;
    std::pair<double, double> bound = std::make_pair(0.0, 2.0);

    // Implement root finding function for each point
    std::function<std::tuple<double,double>(const double&)> rootFun;
    switch (point)
    {
        case L1:
            guess = 0.5;
            rootFun = [u](const double& x) {
                return std::make_tuple(x - (1 - u) / pow(u + x, 2) + u / pow(1 - u - x, 2),
                                       1 + 2 * (1 - u) / pow(u + x, 3) + 2 * u / pow(1 - u - x, 3));};
            break;
        case L2:
            guess = 2.0;
            rootFun = [u](const double& x) {
                return std::make_tuple(x - (1 - u) / pow(u + x, 2) - u / pow(1 - u - x, 2),
                                       1 + 2 * (1 - u) / pow(u + x, 3) - 2 * u / pow(1 - u - x, 3));};
            break;
        case L3:
            guess = -1.0;
            bound = std::make_pair(-2.0, 0.0);
            rootFun = [u](const double& x) {
                return std::make_tuple(x + (1 - u) / pow(u + x, 2) - u / pow(1 - u - x, 2),
                                       1 - 2 * (1 - u) / pow(u + x, 3) - 2 * u / pow(1 - u - x, 3));};
            break;
        default:
            throw std::runtime_error("No valid Lagrange point selected.");
    }
    boost::uintmax_t maxit = 50;
    int get_digits = static_cast<int>(std::numeric_limits<double>::digits * 0.8);
    double xVal = newton_raphson_iterate(rootFun, guess, bound.first, bound.second, get_digits, maxit);

    // Collinear libration points only have x coordinate
    posLagrange << xVal, 0.0, 0.0;
    return posLagrange;
}





PeriodicOrbit POCollocationSolver::threeDlinearApproximation(int N1, int m, double Href, lagrangePoint point, periodicOrbit orbit)
{
    stateDict periodicOrbitApproximation;   // allocate solution
    Eigen::VectorXd thetaArray = tools3BP::gaussLegendreCollocationArray(N1, m);

    // Find conditions that meet energy requirement
    auto energyFun = std::bind(&POCollocationSolver::energyCondition3D, this, std::placeholders::_1, Href, point, orbit);
    std::pair<double, double> result = bisect(energyFun, 0.0 ,3.0, TerminationCondition()); // TODO: Hardcoded
    double Az = (result.first + result.second) / 2;
    std::cout << Az << std::endl;

    // Create linear approximation of 3D orbit
    for (const auto &theta :thetaArray)
        periodicOrbitApproximation[theta] = state3DOrbitApproximation(2 * M_PI * theta, Az, point, orbit);

    // Define period of the orbit
    double T = 2 * M_PI;
    if (orbit == vertical)
        T *= 2;

    return PeriodicOrbit(periodicOrbitApproximation, T, N1, m);
}


double POCollocationSolver::energyCondition3D(const double &Az, double Href, lagrangePoint point, periodicOrbit orbit)
{
    Vector6d state = state3DOrbitApproximation(0, Az, point, orbit);
    return getJacobi(state) - Href;
}


Vector6d POCollocationSolver::state3DOrbitApproximation(double theta, double Az, lagrangePoint point, periodicOrbit orbit)
{
    double u = massParameter;
    Eigen::Vector3d lPosition = lagrangePosition[point];
    double distLagrange = lPosition[0]; // TODO: Only for L123
    double gamma = (1 - u) - distLagrange;
    double c_2 = (u + (1 - u) * pow(gamma, 3) / pow(1 - gamma, 3)) / pow(gamma, 3);
    double c_3 = (u - (1 - u) * pow(gamma, 4) / pow(1 - gamma, 4)) / pow(gamma, 3);
    double c_4 = (u + (1 - u) * pow(gamma, 5) / pow(1 - gamma, 5)) / pow(gamma, 3);

    double lmda = sqrt(((2 - c_2) + sqrt(pow(c_2 - 2, 2) + 4 * (c_2 - 1) * (1 + 2 * c_2))) / 2);
    double k, Delta;
    if (orbit == vertical) {
        Delta = pow(lmda, 2) / 4 - c_2;
        k = 1;
    }
    else {
        Delta = pow(lmda, 2) - c_2;
        k = 2 * lmda / (pow(lmda, 2) + 1 - c_2);
    }

    double d_1 = 3 * pow(lmda, 2) / k * (k * (6 * pow(lmda, 2) - 1) - 2 * lmda);
    double d_2 = 8 * pow(lmda, 2) / k * (k * (11 * pow(lmda, 2) - 1) - 2 * lmda);

    double a_21 = 3 * c_3 * (pow(k, 2) - 2) / (4 * (1 + 2 * c_2));
    double a_22 = 3 * c_3 / (4 * (1 + 2 * c_2));
    double a_23 = -3 * c_3 * lmda / (4 * k * d_1) * (3 * pow(k, 3) * lmda - 6 * k * (k - lmda) + 4);
    double a_24 = -3 * c_3 * lmda / (4 * k * d_1) * (2 + 3 * k * lmda);
    double b_21 = -3 * c_3 * lmda / (2 * d_1) * (3 * k * lmda - 4);
    double b_22 = 3 * c_3 * lmda / d_1;
    double d_21 = -c_3 / (2 * pow(lmda, 2));

    double d_31 = 3 / (64 * pow(lmda, 2)) * (4 * c_3 * a_24 + c_4);
    double a_31 = -9 * lmda / (4 * d_2) * (4 * c_3 * (k * a_23 - b_21) + k * c_4 * (4 + pow(k, 2))) +
                  (9 * pow(lmda, 2) + 1 - c_2) / (2 * d_2) * (3 * c_3 * (2 * a_23 - k * b_21) + c_4 * (2 + 3 * pow(k, 2)));
    double a_32 = -1/d_2 * (9 * lmda / 4 * (4 * c_3 * (k * a_24 - b_22) + k * c_4) +
                            3.0/2 * (9 * pow(lmda, 2) + 1 - c_2) * (c_3 * (k * b_22 + d_21 - 2 * a_24) - c_4));
    double b_31 = 3 / (8 * d_2) * (8 * lmda * (3 * c_3 * (k * b_21 - 2 * a_23) - c_4 * (2 + 3 * pow(k, 2))) + \
            (9 * pow(lmda, 2) + 1 + 2 * c_2)*(4 * c_3 * (k * a_23 - b_21) + k * c_4 * (4 + pow(k, 2))));
    double b_32 = 1 / d_2 * (9 * lmda * (3 * c_3 * (k * b_22 + d_21 - 2 * a_24) - c_4) + \
            3.0/8 * (9 * pow(lmda, 2) + 1 + 2 * c_2) * (4 * c_3 * (k * a_24 - b_22) + k * c_4));
    double d_32 = 3 / (64 * pow(lmda, 2)) * (4 * c_3 * (a_23 - d_21) + c_4 * (4 + pow(k, 2)));

    double s_1 = (3.0/2 * c_3 * (2 * a_21 * (pow(k, 2) - 2) - a_23 * (pow(k, 2) + 2) - 2 * k * b_21) - \
            3.0/8 * c_4 * (3 * pow(k, 4) - 8 * pow(k, 2) + 8)) / (2 * lmda * (lmda * (1 + pow(k, 2)) - 2 * k));
    double s_2 = (3.0/2 * c_3 * (2 * a_22 * (pow(k, 2) - 2) + a_24 * (pow(k, 2) + 2) + 2 * k * b_22 + 5 * d_21) + \
            3.0/8 * c_4 * (12 - pow(k, 2))) / (2 * lmda * (lmda * (1 + pow(k, 2)) - 2 * k));

    double a_1 = -3.0/2 * c_3 * (2 * a_21 + a_23 + 5 * d_21) - 3.0/8 * c_4 * (12 - pow(k, 2));
    double a_2 = 3.0/2 * c_3 * (a_24 - 2 * a_22) + 9.0/8 * c_4;
    double l_1 = a_1 + 2 * pow(lmda, 2) * s_1;
    double l_2 = a_2 + 2 * pow(lmda, 2) * s_2;

    int r;
    if (orbit == this->northHalo)
        r = 1;
    else
        r = 3;
    int d_r = 2 - r;

    double Ax = sqrt(-(l_2 * Az * Az + Delta) / l_1);
    double w = 1 + s_1 * Ax * Ax + s_2 * Az * Az;
    double tau_1, tau_2;
    double x, y, z, dx, dy, dz;
    if (orbit == this->vertical)
    {
        tau_1 = theta * 2;
        tau_2 = theta + M_PI_2;
        x = a_21 * Ax * Ax + a_22 * Az * Az - Ax * cos(tau_1);
        y = Ax * sin(tau_1);
        z = Az * cos(tau_2);
        dx = Ax * sin(tau_1);
        dy = Ax * cos(tau_1);
        dz = -Az * sin(tau_2);
    }
    else
    {
        tau_1 = theta; tau_2 = theta;
        x = a_21 * Ax * Ax + a_22 * Az * Az - Ax * cos(tau_1) + (a_23 * Ax * Ax - a_24 * Az * Az) * cos(2 * tau_1) +
            (a_31 * pow(Ax, 3) - a_32 * Ax * Az * Az) * cos(3 * tau_1);
        y = k * Ax * sin(tau_1) + (b_21 * Ax * Ax - b_22 * Az * Az) * sin(2 * tau_1) +
            (b_31 * pow(Ax, 3) - b_32 * Ax * Az * Az) * sin(3 * tau_1);
        z = d_r * Az * cos(tau_2) + d_r * d_21 * Ax * Az * (cos(2 * tau_2) - 3) +
            d_r * (d_32 * Az * Ax * Ax - d_31 * pow(Az, 3)) * cos(3 * tau_2);

        dx = Ax * sin(tau_1) - 2 * (a_23 * Ax * Ax - a_24 * Az * Az) * sin(2 * tau_1) -
             3 * (a_31 * pow(Ax, 3) - a_32 * Ax * Az * Az) * sin(3 * tau_1);
        dy = k * Ax * cos(tau_1) + (b_21 * Ax * Ax - b_22 * Az * Az) * 2 * cos(2 * tau_1) +
             (b_31 * pow(Ax, 3) - b_32 * Ax * Az * Az) * 3 * cos(3 * tau_1);
        dz = -d_r * Az * sin(tau_2) - 2 * d_r * d_21 * Ax * Az * sin(2 * tau_2) -
             3 * d_r * (d_32 * Az * Ax * Ax - d_31 * pow(Az, 3)) * sin(3 * tau_2);
    }
    Vector6d state = Vector6d(x, y, z, dx, dy, dz);
    state.segment(0,3) = state.segment(0,3) * gamma + lPosition;
    state.segment(3,3) *= gamma * w;

    return state;
}