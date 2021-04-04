//
// Created by Daniel on 25/03/2021.
//

#define _USE_MATH_DEFINES

#include <cmath>
#include "QPCollocationSolver.h"
#include <boost/math//special_functions/legendre.hpp>
#include <complex>
#include <fstream>
#include "tools3bp.h"


void QPCollocationSolver::Solve(Torus &torus, Torus &prevTorus, Torus &tanTorus,
                                std::pair<double,double>config, double tol, bool getManifolds)
{
    // Transform torus data to array
    Eigen::VectorXd torusV = torusToArray(torus);
    Eigen::VectorXd prevTorusV = torusToArray(prevTorus);
    Eigen::VectorXd tanTorusV = torusToArray(tanTorus);

    // Iteration procedure
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver; // create solver
    int dimVector = (torus.N1 * (torus.m + 1) + 1) * torus.N2 * 6;
    Eigen::VectorXd fArray(dimVector), correction(dimVector);
    SparseMatrix jMatrix;
    double error = 1;
    int count = 0;
    while(error > tol)
    {
        // Fill fArray
        fArray = getCollocationSyst(torusV, prevTorusV, tanTorusV, torus, config);
        // Fill jMatrix (Jacobian)
        jMatrix = getCollocationJacobian(torusV, prevTorusV, tanTorusV, torus, config);

        if (count == 0)
            solver.analyzePattern(jMatrix);   // since the pattern of jMatrix is always the same, analyze only once
        solver.factorize(jMatrix);

        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Jacobian decomposition failed.");

        correction = solver.solve(fArray);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Solving procedure failed.");

        error = correction.norm(); // update error
        torusV -= correction;   // update torus
        std::cout << "Iteration no: " << ++count << " | Error: " << error << std::endl;
    }
    // Update input torus
    arrayToTorus(torusV, torus);
    torus.torusConverged = true;
    std::cout << "Torus converged!" << std::endl;

    if (getManifolds)
    {
        // Compute the matrix containing all state transition matrices
        int dimState = 6, dimMatrix = torus.N1 * (torus.m + 1) * torus.N2 * dimState;
        SparseMatrix matrixA = SparseMatrix(jMatrix.block(0, 0, dimMatrix, torus.N2 * dimState));
        SparseMatrix matrixB = SparseMatrix(-jMatrix.block(0, torus.N2 * dimState, dimMatrix, dimMatrix));
        solver.compute(matrixB);
        SparseMatrix stateTransitionMatrices = solver.solve(matrixA);

        // Fill manifold information
        findManifolds(torus, stateTransitionMatrices);
        std::cout << "Manifolds computed!" << std::endl;
    }
}


Eigen::VectorXd QPCollocationSolver::torusToArray(Torus &torus)
{
    int N1 = torus.N1, N2 = torus.N2, m = torus.m, dimState = 6, scalarEq = 6;
    // Allocate vector solution
    int dimVector = (N1 * (m + 1) + 1) * N2 * dimState;
    Eigen::VectorXd torusArray(dimVector + scalarEq);
    int i = 0;
    for (auto const &torusDict :torus.getData())
    {
        for (int j = 0; j < torusDict.second.size(); j++)
            torusArray[i * N2 * dimState + j] = *(torusDict.second.data() + j);
        i++;
    }
    torusArray[dimVector] = torus.T;
    torusArray[dimVector+1] = torus.rho;
    torusArray[dimVector+2] = 1 / torus.T;
    torusArray[dimVector+3] = torus.rho / torus.T;
    torusArray[dimVector+4] = 0;
    torusArray[dimVector+5] = 0;

    return torusArray;
}


void QPCollocationSolver::arrayToTorus(Eigen::VectorXd &array, Torus &torus)
{
    int N1 = torus.N1, N2 = torus.N2, m = torus.m, dimState = 6, numTime = 0;
    for (auto &torusDict :torus.data)
    {
        for (int i = 0; i < N2; i++)
            torusDict.second.row(i) = array.segment(numTime * N2 * dimState + i * dimState, dimState);
        numTime++;
    }
}

Eigen::VectorXd QPCollocationSolver::getCollocationSyst(Eigen::VectorXd &torusV,
                                                        Eigen::VectorXd &prevTorusV,
                                                        Eigen::VectorXd &tanTorusV,
                                                        Torus &torus,
                                                        std::pair<double,double>config)
{
    // Retrieve data
    int N1 = torus.N1, N2 = torus.N2, m = torus.m;  // geometry of the torus
    int scalarEqs = 6, dimState = 6;
    int dimVector = (N1 * (m + 1) + 1) * N2 * dimState;
    double Href = config.first, dsref = config.second;  // configuration

    // Extra parameters
    double T = torusV[dimVector], rho = torusV[dimVector+1], w1 = torusV[dimVector+2],
        w2 = torusV[dimVector+3], lambda1 = torusV[dimVector+4], lambda2 = torusV[dimVector+5];
    double Tprev = prevTorusV[dimVector], rhoPrev = prevTorusV[dimVector+1],
        lambda1Prev = prevTorusV[dimVector+4], lambda2Prev = prevTorusV[dimVector+5];
    double Ttan = tanTorusV[dimVector], rhoTan = tanTorusV[dimVector+1];

    // Allocate solution
    Eigen::VectorXd fArray(dimVector + scalarEqs);
    fArray.setZero();

    Eigen::VectorXd collocationArray = tools3BP::gaussLegendreCollocationArray(N1, m);
    Eigen::MatrixXd th2Der = th2Derivative(N2).real();
    Eigen::MatrixXd rotMat = rotationMatrix(N2,-rho).real();
    // Loop over equally-spaced segments
    for (int numSegment = 0; numSegment < N1; numSegment++)
    {
        // Time-set including extremes of the segment
        Eigen::VectorXd timeSegment = collocationArray.segment(numSegment * (m + 1), m + 2);
        Eigen::VectorXd timeCollocation = timeSegment.segment(1, m);
        double ti = timeSegment[0], tii = timeSegment[timeSegment.size() - 1];

        // Generate flow conditions m * N1 * N2 * dimState
        Eigen::VectorXd flowCondition(m * N2 * dimState); // allocate sol
        for (int numCollocation = 0; numCollocation < m; numCollocation++)
        {
            double time = timeCollocation[numCollocation];
            int indexTime = numSegment * (m + 1) * N2 * dimState + (numCollocation + 1) * N2 * dimState;
            // Retrieve circle of states
            Eigen::VectorXd stateCircle = torusV.segment(indexTime,N2 * dimState);
            // Compute derivative along the circle
            Eigen::VectorXd dxDth2 = expandMatrix(th2Der, N2) * stateCircle;

            // Iterate through the states in the circle
            for (int numState = 0; numState < N2; numState++)
            {
                Vector6d state = stateCircle.segment(numState * dimState, dimState);
                Vector6d dxDth2Section = dxDth2.segment(numState * dimState, dimState);

                // Compute the state derivative using the interpolation
                Vector6d stateDerivative; stateDerivative.setZero();
                int numDer = 0;
                for (const auto &timeK :timeSegment.head(m + 1))
                {
                    int indexDerivative = numSegment * (m + 1) * N2 * dimState + numDer * N2 * dimState + numState * dimState;
                    stateDerivative += torusV.segment(indexDerivative, dimState) * lagrangePolyDeriv(time, timeK, timeSegment);
                    numDer++;
                }

                // Evaluate flow condition
                flowCondition.segment(numCollocation * N2 * dimState + numState * dimState, dimState) =
                        stateDerivative - T * (tii - ti) * modelDynamicCR3BP(state, dxDth2Section, lambda1, lambda2);
            }
        }
        // Continuity conditions
        int numTime = 0;
        Eigen::VectorXd stateFinalInterp(N2 * dimState); stateFinalInterp.setZero();    // allocate sol
        for (const auto &timeK :timeSegment.head(timeSegment.size() - 1))
        {
            int indexSection = numSegment * (m + 1) * N2 * dimState + numTime * N2 * dimState;
            Eigen::VectorXd stateSection = torusV.segment(indexSection,N2 * dimState);
            stateFinalInterp += stateSection * lagrangePoly(tii, timeK, timeSegment);
            numTime++;
        }
        Eigen::VectorXd continuityCond = stateFinalInterp - torusV.segment((numSegment + 1) * (m + 1) * N2 * dimState, N2 * dimState);

        // Pack blocks into fArray
        int flowIndex = numSegment * (m+1) * N2 * dimState;
        int contIndex = (numSegment+1) * (m+1) * N2 * dimState - N2 * dimState;
        fArray.segment(flowIndex, m * N2 * dimState) = flowCondition;
        fArray.segment(contIndex, N2 * dimState) = continuityCond;
    }
    // Periodicity condition
    int periodIndex = N1 * (m + 1) * N2 * dimState;
    Eigen::VectorXd initialCircle = torusV.segment(0, N2 * dimState);
    Eigen::VectorXd finalCircle = torusV.segment(periodIndex, N2 * dimState);

    fArray.segment(periodIndex, N2 * dimState) = initialCircle - expandMatrix(rotMat, N2) * finalCircle;

    // Scalar equations
    double phaseCondition1 = 0;
    double phaseCondition2 = 0;
    double energyCondition = 0;
    double contCondition = 0;
    int numTime = 0;
    for (const auto &time :collocationArray)
    {
        // Retrieve circle of states
        Eigen::VectorXd stateCircle = torusV.segment(numTime * N2 * dimState,N2 * dimState);
        Eigen::VectorXd prevStateCircle = prevTorusV.segment(numTime * N2 * dimState,N2 * dimState);
        // Compute derivative along the circle
        Eigen::VectorXd dxDth2 = expandMatrix(th2Der, N2) * prevStateCircle;
        for (int numState = 0; numState < N2; numState++)
        {
            Vector6d state = stateCircle.segment(numState * dimState, dimState);
            Vector6d prevState = prevStateCircle.segment(numState * dimState, dimState);
            Vector6d dxDth2Section = dxDth2.segment(numState * dimState, dimState);
            Vector6d dxDth1Section = Tprev * modelDynamicCR3BP(prevState, dxDth2Section, lambda1Prev, lambda2Prev) - rhoPrev * dxDth2Section;

            int indexState = numTime * N2 * dimState + numState * dimState;
            // Phase condition 1
            phaseCondition1 += (state - prevTorusV.segment(indexState, dimState)).transpose() * dxDth1Section;
            // Phase condition 2
            phaseCondition2 += (state - prevTorusV.segment(indexState, dimState)).transpose() * dxDth2Section;
            // Energy condition
            energyCondition += getJacobi(state);
            // Pseudo-arc length continuation
            contCondition += (state - prevTorusV.segment(indexState, dimState)).transpose() * tanTorusV.segment(indexState, dimState);
        }
        numTime++;
    }
    // Pack condition
    fArray[dimVector] = phaseCondition1 / ((N1 * (m + 1) + 1) * N2);
    fArray[dimVector+1] = phaseCondition2 / ((N1 * (m + 1) + 1) * N2);
    fArray[dimVector+2] = energyCondition / ((N1 * (m + 1) + 1) * N2) - Href;
    fArray[dimVector+3] = contCondition / ((N1 * (m + 1) + 1) * N2) - dsref;

    // Extra relationship 1
    fArray[dimVector+4] = T * w1 - 1;
    // Extra relationship 2
    fArray[dimVector+5] = T * w2 - rho;

    return fArray;
}


SparseMatrix QPCollocationSolver::getCollocationJacobian(Eigen::VectorXd &torusV,
                                                         Eigen::VectorXd &prevTorusV,
                                                         Eigen::VectorXd &tanTorusV,
                                                         Torus &torus,
                                                         std::pair<double,double>config)
{
    // Retrieve data
    int N1 = torus.N1, N2 = torus.N2, m = torus.m;  // geometry of the torus
    int scalarEqs = 6, dimState = 6;
    int dimVector = (N1 * (m + 1) + 1) * N2 * dimState;
    double Href = config.first, dsref = config.second;  // configuration

    // Extra parameters
    double T = torusV[dimVector], rho = torusV[dimVector+1], w1 = torusV[dimVector+2],
            w2 = torusV[dimVector+3], lambda1 = torusV[dimVector+4], lambda2 = torusV[dimVector+5];
    double Tprev = prevTorusV[dimVector], rhoPrev = prevTorusV[dimVector+1],
            lambda1Prev = prevTorusV[dimVector+4], lambda2Prev = prevTorusV[dimVector+5];
    double Ttan = tanTorusV[dimVector], rhoTan = tanTorusV[dimVector+1];

    // Allocate solution
    SparseMatrix jMatrix(dimVector + scalarEqs, dimVector + scalarEqs);
    std::vector<Eigen::Triplet<float>> tripletList;
    tripletList.reserve(int(N1 * pow((m + 1) * N2 * dimState, 2) + 7 * N1 * (m + 1) * N2 * dimState));

    Eigen::VectorXd collocationArray = tools3BP::gaussLegendreCollocationArray(N1, m);
    Eigen::MatrixXd th2Der = th2Derivative(N2).real();
    Eigen::MatrixXd rotMat = rotationMatrix(N2, -rho).real();
    Eigen::MatrixXd rotMatDer = rotationMatrixDer(N2, -rho).real();
    Eigen::MatrixXd identityBlock = Eigen::MatrixXd::Identity(N2 * dimState, N2 * dimState);
    // Loop over equally-spaced segments
    for (int numSegment = 0; numSegment < N1; numSegment++)
    {
        // Time-set including extremes of the segment
        Eigen::VectorXd timeSegment = collocationArray.segment(numSegment * (m + 1), m + 2);
        Eigen::VectorXd timeCollocation = timeSegment.segment(1, m);
        double ti = timeSegment[0], tii = timeSegment[m + 1];

        // Allocate block and solution vectors
        int dimBlock = (m + 1) * N2 * dimState;
        Eigen::MatrixXd segmentBlock(dimBlock, dimBlock+N2*dimState); segmentBlock.setZero();
        Eigen::VectorXd periodDerivative(dimBlock,1); periodDerivative.setZero();
        Eigen::VectorXd l1Derivative(dimBlock); l1Derivative.setZero();
        Eigen::VectorXd l2Derivative(dimBlock); l2Derivative.setZero();
        // Generate flow derivatives (m+1) * N1 * N2 * dimState
        for (int numSection = 0; numSection < m + 2; numSection++)
        {
            double time = timeSegment[numSection];
            int indexTime = numSegment * (m + 1) * N2 * dimState + numSection * N2 * dimState;
            // Retrieve circle of states
            Eigen::VectorXd stateCircle = torusV.segment(indexTime,N2 * dimState);
            // Compute derivative along the circle
            Eigen::VectorXd dxDth2 = expandMatrix(th2Der, N2) * stateCircle;

            // Fill state derivative
            if (time != timeSegment[timeSegment.size() - 1])
            {
                int colFillState = numSection * N2 * dimState;
                for (int numFillState = 0; numFillState < m; numFillState++)
                    segmentBlock.block(numFillState * N2 * dimState, colFillState, N2*dimState, N2*dimState) =
                            identityBlock * lagrangePolyDeriv(timeSegment[numFillState + 1], time, timeSegment);
            }
            // Generate flow derivatives only in collocation points
            if ((time == timeCollocation.array()).any())
            {
                for (int numState = 0; numState < N2; numState++)
                {
                    // Derivative wrt the states
                    int rowFlowIndex = (numSection - 1) * N2 * dimState + numState * dimState;
                    int colFlowIndex = rowFlowIndex + N2 * dimState;
                    Vector6d state = stateCircle.segment(numState * dimState, dimState);
                    segmentBlock.block(rowFlowIndex, colFlowIndex, dimState, dimState) -= (tii - ti) * T * getJacobianCR3BP(state);

                    Vector6d dxDth2Section = dxDth2.segment(numState * dimState, dimState);
                    // Period derivative of the flow equations
                    periodDerivative.segment(rowFlowIndex, dimState) = -(tii - ti) * modelDynamicCR3BP(state, dxDth2Section, lambda1, lambda2);
                    // Lambda 1 derivative of the flow equations
                    l1Derivative.segment(rowFlowIndex, dimState) = -(tii - ti) * T * jacobiGradient(state);
                    // Lambda 2 derivative of the flow equations
                    l2Derivative.segment(rowFlowIndex, dimState) = -(tii - ti) * T * i2Gradient(dxDth2Section);
                }
            }
            // Continuity derivatives
            int rowContIndex = m * N2 * dimState;
            int colContIndex = numSection * N2 * dimState;
            Eigen::MatrixXd continuityVal(N2 * dimState, N2 * dimState);
            if (time == tii)
                continuityVal = -identityBlock;
            else
                continuityVal = identityBlock * lagrangePoly(tii, time, timeSegment);
            segmentBlock.block(rowContIndex, colContIndex ,N2 * dimState,N2 * dimState) = continuityVal;
        }
        // Pack blocks and vectors
        int segmentIndex = numSegment * (m+1) * N2 * dimState;
        int heightSegment = (m+1) * N2 * dimState;

        fillSpBlock<Eigen::MatrixXd>(tripletList, segmentBlock, segmentIndex, segmentIndex, heightSegment, heightSegment + N2 * dimState);
        fillSpBlock<Eigen::VectorXd>(tripletList, periodDerivative, segmentIndex, dimVector, heightSegment, 1);         // period
        fillSpBlock<Eigen::VectorXd>(tripletList, l1Derivative, segmentIndex, dimVector+4, heightSegment, 1);       // lambda 1
        fillSpBlock<Eigen::VectorXd>(tripletList, l2Derivative, segmentIndex, dimVector+5, heightSegment, 1);       // lambda 2
    }
    // Periodicity derivatives
    int rowPeriodIndex = N1 * (m + 1) * N2 * dimState;
    fillSpBlock<Eigen::MatrixXd>(tripletList, identityBlock, rowPeriodIndex, 0, N2*dimState, N2*dimState);  // wrt w_00
    fillSpBlock<Eigen::MatrixXd>(tripletList, -expandMatrix(rotMat,N2), rowPeriodIndex, rowPeriodIndex, N2*dimState, N2*dimState);  // wrt w_N0
    Eigen::VectorXd finalCircle = torusV.segment(rowPeriodIndex,N2 * dimState);
    fillSpBlock<Eigen::VectorXd>(tripletList, -expandMatrix(rotMatDer,N2) * finalCircle, rowPeriodIndex, dimVector+1, N2*dimState, 1);

    // Scalar equations derivatives
    int numTime = 0;
    for (const auto &time :collocationArray)
    {
        // Retrieve circle of states
        Eigen::VectorXd stateCircle = torusV.segment(numTime * N2 * dimState, N2 * dimState);
        Eigen::VectorXd prevStateCircle = prevTorusV.segment(numTime * N2 * dimState, N2 * dimState);
        // Compute derivative along the circle
        Eigen::VectorXd dxDth2 = expandMatrix(th2Der, N2) * prevStateCircle;

        for (int numState = 0; numState < N2; numState++)
        {
            Vector6d state = stateCircle.segment(numState * dimState, dimState);
            Vector6d prevState = prevStateCircle.segment(numState * dimState, dimState);
            Vector6d dxDth2Section = dxDth2.segment(numState * dimState, dimState);
            Vector6d dxDth1Section = Tprev * modelDynamicCR3BP(prevState, dxDth2Section, lambda1Prev, lambda2Prev) - rhoPrev * dxDth2Section;

            int indexState = numTime * N2 * dimState + numState * dimState;
            // Phase condition 1 derivative
            fillSpBlock<Eigen::RowVectorXd>(tripletList, dxDth1Section.transpose() / ((N1 * (m + 1) + 1) * N2),
                                            dimVector, indexState, 1, dimState);
            // Phase condition 2 derivative
            fillSpBlock<Eigen::RowVectorXd>(tripletList, dxDth2Section.transpose() / ((N1 * (m + 1) + 1) * N2),
                                            dimVector+1, indexState, 1, dimState);
            // Energy condition derivatives
            fillSpBlock<Eigen::RowVectorXd>(tripletList, jacobiGradient(state).transpose() / ((N1 * (m + 1) + 1) * N2),
                                            dimVector+2, indexState, 1, dimState);
            // Pseudo-arc length continuation derivatives
            fillSpBlock<Eigen::RowVectorXd>(tripletList, tanTorusV.segment(indexState, dimState).transpose() / ((N1 * (m + 1) + 1) * N2),
                                            dimVector+3, indexState, 1, dimState);
        }
        numTime++;
    }
    // Extra relationship 1 derivatives
    Eigen::VectorXd extraVect1(4); extraVect1 << w1, 0, T, 0;
    fillSpBlock<Eigen::RowVectorXd>(tripletList, extraVect1.transpose(), dimVector+4, dimVector, 1, 4);
    // Extra relationships 2 derivatives
    Eigen::VectorXd extraVect2(4); extraVect2 << w2, -1, 0, T;
    fillSpBlock<Eigen::RowVectorXd>(tripletList, extraVect2.transpose(), dimVector+5, dimVector, 1, 4);

    jMatrix.setFromTriplets(tripletList.begin(), tripletList.end());
    return jMatrix;
}


void QPCollocationSolver::findManifolds(Torus &torus, SparseMatrix &stateTransitionMatrices)
{
    int dimState = 6, N1 = torus.N1, N2 = torus.N2, m = torus.m;
    int dimMatrix = N1 * (m + 1) * N2 * dimState, dimBlock = N2 * dimState;
    Eigen::VectorXd collocationArray = tools3BP::gaussLegendreCollocationArray(N1, m);

    // Compute rotated monodromy matrix
    Eigen::MatrixXd monodromyMatrix = stateTransitionMatrices.block(dimMatrix - dimBlock, 0, dimBlock, dimBlock);
    Eigen::MatrixXd rotMatrix = rotationMatrix(N2, -torus.rho).real();
    Eigen::MatrixXd linearFixedCircle = expandMatrix(rotMatrix, N2) * monodromyMatrix;

    // Find eigenvalues
    Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver;
    eigenSolver.compute(linearFixedCircle);
    Eigen::VectorXcd eigenValues = eigenSolver.eigenvalues();
    Eigen::MatrixXcd eigenVectors = eigenSolver.eigenvectors();

    // Find the manifold vector
    double firstManifoldValue;
    Eigen::VectorXd firstManifoldVector(dimBlock);
    int indexElement = 0;
    for (const auto &eigenValue :eigenValues)
    {
        if (eigenValue.imag() == 0 && eigenValue.real() > 1.5) // TODO: Do not used hard-coded value
        {
            firstManifoldValue = eigenValues[indexElement].real();
            firstManifoldVector = eigenVectors.col(indexElement).real();
            break;
        }
        indexElement++;
    }

    // Propagate the manifold eigenvector
    Eigen::VectorXd manifoldVector(dimBlock);
    MatrixCircle manifoldMatrix(N2, dimState);
    for (int i = 0; i < N1 * (m + 1) + 1; i++)
    {
        double time = collocationArray[i];

        if (i == 0 || i == N1 * (m + 1))    // do not propagate first and last eigenvectors
            manifoldVector = firstManifoldVector;
        else                 // remove scaling produced by state transition matrix
            manifoldVector = pow(firstManifoldValue,-i / double(N1 * (m + 1))) *
                             stateTransitionMatrices.block(i * dimBlock, 0, dimBlock, dimBlock) * firstManifoldVector;

        // Transform vector into matrix form
        for (int j = 0; j < N2; j++)
            manifoldMatrix.row(j) = manifoldVector.segment(j*dimState, dimState);

        torus.manifoldData[time] = 1E-5 * manifoldMatrix;  // fill torus information    // TODO: Do not used hard-coded value
    }
}


Vector6d QPCollocationSolver::modelDynamicCR3BP(Vector6d &state, Vector6d &dxdth2Section, double l1, double l2)
{
    return dynamicsCR3BP(state) + l1 * jacobiGradient(state) + l2 * i2Gradient(dxdth2Section);
}


Vector6d QPCollocationSolver::i2Gradient(Vector6d &dxdth2Section)
{
    Matrix6d matrixJ;
    matrixJ.setZero();
    matrixJ(0,1) = +2;
    matrixJ(1,0) = -2;
    matrixJ.block(0,3,3,3) = -Eigen::MatrixXd::Identity(3,3);
    matrixJ.block(3,0,3,3) = Eigen::MatrixXd::Identity(3,3);
    return matrixJ * dxdth2Section;
}


Eigen::MatrixXcd QPCollocationSolver::rotationMatrix(int N, double rho)
{
    Eigen::MatrixXcd shiftMat(N,N);
    shiftMat.setZero();
    for (int k = 0; k < N; k++)
    {
        if (k <= N/2 - 1)
            shiftMat(k,k) = exp(std::complex<double>(0,2 * M_PI * rho * k));
        else
            shiftMat(k,k) = exp(std::complex<double>(0,2 * M_PI * rho * (k - N)));
    }

    Eigen::MatrixXcd dftMat = tools3BP::DFT(N), idftMat = tools3BP::IDFT(N);
    return idftMat * (shiftMat * dftMat);
}


Eigen::MatrixXcd QPCollocationSolver::rotationMatrixDer(int N, double rho)
{
    Eigen::MatrixXcd shiftMat(N,N); shiftMat.setZero();
    Eigen::VectorXd kCoeff(N);
    kCoeff << Eigen::ArrayXd::LinSpaced(int(N/2), 0, int(N/2)-1), 0.0, Eigen::ArrayXd::LinSpaced(int(N/2)-1, int(-N/2)+1, -1);

    int count = 0;
    for (const auto &k: kCoeff)
    {
        shiftMat(count,count) = std::complex<double>(0, -1) * (2 * M_PI * k) * exp(std::complex<double>(0,1) * (2 * M_PI * rho * k));
        count++;
    }

    Eigen::MatrixXcd dftMat = tools3BP::DFT(N), idftMat = tools3BP::IDFT(N);
    return idftMat * (shiftMat * dftMat);
}

template <class T>
void QPCollocationSolver::fillSpBlock(std::vector<Eigen::Triplet<float>> &tripletList, const T &block, int row, int col, int height, int width)
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
