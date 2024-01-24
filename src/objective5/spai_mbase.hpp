/*
 * spai_mbase.hpp
 *
 *  Created on: Jan 1, 2024
 *      Author: Ying Zhang
 */

#ifndef HH_SPAI_MBASE___HH
#define HH_SPAI_MBASE___HH
//*****************************************************************
// SParse Approximate Inverse algorithm - SPAI supporting Matrix.hpp and OpenMP
// 
// Every thread in OpenMP executes one to several columns of matrix M in parallel.
// Matrix.hpp supports a parallel version of Matrix Vector multiplication
// which also decreases the whole processing time.
// 
// The SPAI algorithm computes a sparse approximate inverse M of a general 
// sparse matrix A. It is inherently parallel, since the columns of M are 
// calculated independently of one another.
//
// Grote M J, Huckle T. Parallel preconditioning with sparse 
// approximate inverses[J]. SIAM Journal on Scientific Computing, 
// 1997, 18(3): 838-853.
//
// Abstract. 
// A parallel preconditioner is presented for the solution of 
// general sparse linear systems of equations. A sparse approximate 
// inverse is computed explicitly and then applied as a preconditioner 
// to an iterative method. The computation of the preconditioner is 
// inherently parallel, and its application only requires a 
// matrix-vector product. The sparsity pattern of the approximate 
// inverse is not imposed a priori but captured automatically. This keeps 
// the amount of work and the number of nonzero entries in the 
// preconditioner to a minimum. Rigorous bounds on the clustering of 
// the eigenvalues and the singular values are derived for the 
// preconditioned system, and the proximity of the approximate to the 
// true inverse is estimated.
// 
// Upon successful return, output arguments have the following values:
//
//        A  --  matrix A in equation Ax=b
//        M  --  the sparse approximate inverse of matrix A
// max_iter  --  the number of iterations performed to limit 
//               the maximal fill-in per column in M
//  epsilon  --  the threshold of stopping criterion on residual ||r||
//               for every column of M
//
//*****************************************************************

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <cstddef>
#include "Matrix/Matrix.hpp"
#include "MatrixWithVecSupport.hpp"
#include <omp.h>


/*! 
 * Extract a block from matrix `A` specified by the indices set provided in 
 * vectors `I` and `J` and stores the result in the Eigen::MatrixXd `AIJ`.
 */
template <class Matrix>
void getBlockByTwoIndicesSet(const Matrix&A, Eigen::MatrixXd &AIJ, 
    const std::vector<int> &I, const std::vector<int> &J) {

    int n1 = I.size();
    int n2 = J.size();
    for (int i=0; i<n1; i++) {
        for (int j=0; j<n2; j++) {
            AIJ(i,j) = A(I[i], J[j]);
        }
    }
    // print to verify correctness
    // std::cout << "\nRun getBlockByTwoIndicesSet function......\nI: ";
    for (const auto &element : I) { 
        // std::cout << element << " ";
    }
    // std::cout << "\nJ: ";
    for (const auto &element : J) { 
        // std::cout << element << " ";
    }
    // std::cout << "\nAIJ matrix:\n" << AIJ << std::endl;

}

/*!
 * Performs the QR decomposition of a given matrix `A` and returns 
 * the results in matrices `Q` and `R`. The QR decomposition is achieved 
 * using the ColPivHouseholderQR decomposition method provided 
 * by the Eigen library.
 */
void qrDecomposition(const Eigen::MatrixXd &A, Eigen::MatrixXd &Q, Eigen::MatrixXd &R) {

    // // std::cout << "\nBegin the QR decomposition process......";
    // // std::cout << "\nOriginal matrix:\n" << A << std::endl;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
    Q = qr.householderQ();
    R = qr.matrixQR().topLeftCorner(A.cols(), A.cols());
    // // std::cout << "Q matrix:\n" << Q << std::endl;
    // // std::cout << "R matrix:\n" << R << "\n\n";

}

/*!
 * According to the process required in the paper, compute k-th column `m_k` of M
 * and the corresponding residual `r`. This function will be used many times in the
 * entire algorithm, but the first computing for `r` is different from others for
 * every column of M so we use the variable `flag` to distinguish
 */
template <class Matrix, typename Scalar>
void computeRAndMk(const Matrix&A, const int Size, const Eigen::MatrixXd &Q, 
    const Eigen::MatrixXd &R, Eigen::VectorXd &r, Eigen::VectorXd &m_k,
    const Eigen::VectorXd &e_k, const std::vector<int> &I, const std::vector<int> &J, int flag) {
    
    int n1 = I.size();
    int n2 = J.size();
    // e_k_triangular = e_k(I)
    Eigen::VectorXd e_k_triangular(n1);
    for (int i=0; i<n1; i++) {
        e_k_triangular[i] = e_k[I[i]];
    }
    Eigen::VectorXd c_triangular = Q.transpose() * e_k_triangular;
    Eigen::VectorXd m_k_triangular = R.inverse() * c_triangular.segment(0,n2-1);
    // m_k
    int index = 0;
    for (int i=0; i<Size; i++) {
        if (index<n2 && J[index]==i) {
            m_k[i] = m_k_triangular[index];
            index++;
        } else {
            m_k[i] = 0;
        }
    }
    // AJ be the A(.,J)
    Eigen::MatrixXd AJ(Size, n2);
    for (int i=0; i<Size; i++) {
        for (int j=0; j<n2; j++) {
            AJ(i,j) = A(i, J[j]);
        }
    }
    // r
    if (flag==0) { r = AJ * m_k_triangular - e_k;} 
    if (flag==1) {
        // A * m_k - e_k
        std::vector<Scalar> m_k_vector(Size, 0.0);
        for (int i=0; i<Size; i++) {
            m_k_vector[i] = m_k[i];
        }
        std::vector<Scalar> intermediate = A * m_k_vector;
        Eigen::Map<Eigen::VectorXd> intermediate_eigenVector(intermediate.data(), intermediate.size());
        r = intermediate_eigenVector - e_k;
    } 

}


/*!
 * @note Scalar is usually double
 * @param max_iter is usually Size or you can set your own
 * @param epsilon can be selected from set {0.2, 0.3, 0.4, 0.5, 0.6} 
 */
namespace LinearAlgebra {
template <class Matrix, typename Scalar>
int SPAI_MBASE(const Matrix &A, Matrix &M, const int max_iter, Scalar epsilon) {

        ASSERT((A.rows() == A.cols()), "The matrix must be squared!\n");
        const int Size = A.rows();

        // // strategy 1: the initial sparsity of M is chosen to be diagonal
        std::cout << "The initial sparsity of M which is chosen to be diagonal..." << std::endl;
        M.resize(Size, Size);
        const Scalar diagonal_value = 1;
        const Scalar zero = 0;
        for (int i=0; i<Size; i++) {
            for (int j=0; j<Size; j++) {
                if (i==j) { M(i, j) = diagonal_value; }
                else { M(i,j) = zero; } 
            }
        }
        // // std::cout << "matrix M:\n" << M << std::endl;

        // // strategy 2: M is chosen to be not diagonal
        // std::cout << "The initial sparsity of M which is chosen to be lkie..." << std::endl;
        // M.resize(Size, Size);
        // const Scalar diagonal_value = 1;
        // const Scalar zero = 0;
        // for (int i=0; i<Size; i++) {
        //     for (int j=0; j<Size; j++) {
        //         if (i==j) { M(i, j) = diagonal_value; }
        //         else if (i==j+1) { M(i, j) = diagonal_value; }
        //         else { M(i,j) = zero; } 
        //     }
        // }
        // // std::cout << "matrix M:\n" << M << std::endl;

        
        #pragma omp parallel for
        // for every column of M
        for (int k=0; k<Size; k++) {

            // std::cout << k << "-th column of M......" << std::endl;
            // std::cout << "(a) part is processing......" << std::endl;
            // J be the set of indices j such that m_k(j) != 0
            std::vector<int> J;
            for (int j=0; j<Size; j++) {
                if (M(j,k)!=zero) { 
                    J.push_back(j); 
                    // std::cout << "the " << k << "-th column of M, adds new elements to J : " << j << std::endl;
                    }
            }
            // std::cout << "(b) part is processing......" << std::endl;
            // I be the set of indices i such that A(i, J) is not identically zero.
            std::vector<int> I;
            for (int i=0; i<Size; i++) {
                int flag = 0;
                for (const auto &j : J) { 
                    if (A(i,j)!=zero) {flag = 1;}
                }
                if (flag==1) { 
                    I.push_back(i);
                    // std::cout << "the " << k << "-th column of M, adds new elements to I : " << i << std::endl;
                }
            }
            // AIJ be the A(I,J)
            int n1 = I.size();
            int n2 = J.size();
            ASSERT((n1 >= n2), "\nn1 must be bigger than or equal to n2!\n");
            Eigen::MatrixXd AIJ(n1, n2);
            getBlockByTwoIndicesSet(A, AIJ, I, J);

            // QR decomposition of AIJ
            Eigen::MatrixXd Q;
            Eigen::MatrixXd R;
            qrDecomposition(AIJ, Q, R);

            // e_k
            Eigen::VectorXd e_k(Size);
            for (int i=0; i<Size; i++) {
                if (i==k) { e_k[i] = 1; }
                else { e_k[i]=0; }
            }
            // std::cout << "e_k :\n" << e_k << std::endl;


            // m_k
            Eigen::VectorXd m_k = Eigen::VectorXd::Zero(Size);
            // r
            Eigen::VectorXd r = Eigen::VectorXd::Zero(Size);
            computeRAndMk<decltype(A), decltype(epsilon)>(A, Size, Q, R, r, m_k, e_k, I, J, 0);

            // std::cout << "begin the loop iteration......" << std::endl;
            // while loop iteration 
            int iter = 0;
            while (r.norm()>epsilon && iter < max_iter) {
                // std::cout << "............................" << std::endl;
                // std::cout << "r :\n" << r << std::endl;
                iter++;

                // (c) 
                // std::cout << "(c) part is processing......" << std::endl;
                std::vector<int> L;
                // // Strategy 1: Set L equal to the set of indices l for which r(l) != 0
                // for (int i=0; i<Size; i++) {
                //     if (r[i]!=0) { 
                //         L.push_back(i); 
                //         // std::cout << "the " << k << "-th column of M, adds new elements to L : " << i << std::endl;
                //     }
                // }
                // // Strategy 2: according to suggestion of remarks.5, choose the largest elements in r
                int max_index = 0;
                for (int i=0; i<Size; i++) {
                    if (std::abs(r[i]) > std::abs(r[max_index])) { 
                         max_index = i;
                    }
                }
                L.push_back(max_index);
                // std::cout << "the " << k << "-th column of M, adds new elements to L : " << max_index << std::endl;

                // (d) J_triangular
                // std::cout << "(d) part is processing......" << std::endl;
                std::vector<int> J_triangular;
                int index = 0;
                for (int j=0; j<Size; j++) {
                    if (index<n2 && J[index] == j) {
                        // std::cout << "index = " << index << std::endl;
                        index++;
                    } else {
                        int flag = 0;
                        for (const auto &i : L) {
                            // std::cout << " i=" << i << " j=" << j << " A(i,j)=" << A(i,j) << std::endl; 
                            if (A(i,j)!=zero) {
                                flag = 1;
                                // std::cout << "flag=1 " << std::endl;
                            }
                        }
                        if (flag==1) { 
                            J_triangular.push_back(j);
                            // std::cout << "the " << k << "-th column of M, adds new elements to J_triangular : " << j << std::endl;
                        } 
                    }
                }
                // std::cout << "(e) part is processing......" << std::endl;
                // (e) For each j ∈ J_triangular solve the minimization problem (10).
                std::vector<Scalar> rou;
                for (const auto &j : J_triangular) {
                    std::vector<Scalar> e_j(Size, 0.0);
                    e_j[j] = 1.0;
                    std::vector<Scalar> aej = A * e_j;
                    Eigen::Map<Eigen::VectorXd> aej_ev(aej.data(), aej.size());
                    Scalar m1 = r.dot(aej_ev);
                    Scalar m2 = std::pow(aej_ev.norm(),2);
                    Scalar miu_j = m1 / m2;
                    Scalar rou_j = std::pow(r.norm(),2) + miu_j*(r.dot(aej_ev));
                    rou.push_back(rou_j);
                }
                // (f)
                // first: reserve indices j that rou_j is less than or equal to the mean value of all rou_j
                Scalar sum_of_rou = 0;
                for (const auto &r: rou) {
                    sum_of_rou += r;
                }
                Scalar mean_of_rou = sum_of_rou / rou.size();
                std::vector<int> J_triangular_first;
                std::vector<Scalar> rou_first;
                for (int j=0; j<J_triangular.size(); j++) {
                    if (rou[j] <= mean_of_rou) { 
                        J_triangular_first.push_back(J_triangular[j]); 
                        rou_first.push_back(rou[j]);
                    }
                }
                // second: From the remaining indices we keep at most s indices with smallest rou_j, and we set s equal to 5
                std::vector<int> J_triangular_second;
                std::vector<Scalar> rou_second;
                if (rou_first.size() > 5) {
                    // find the 5th smallest rou
                    std::vector<Scalar> copy_rou_first = rou_first;
                    std::partial_sort(copy_rou_first.begin(), copy_rou_first.begin() + 5, copy_rou_first.end());
                    Scalar fifth_smallest_rou = copy_rou_first[4];
                    // delete indices j that rou_j is bigger than or equal to the 5th smallest rou
                    for (int j=0; j<J_triangular_first.size(); j++) {
                        if (rou[j] <= fifth_smallest_rou) { 
                            J_triangular_second.push_back(J_triangular_first[j]); 
                            // std::cout << "the " << k << "-th column of M, adds new elements to J_triangular_second : " << J_triangular_first[j] << std::endl;
                            rou_second.push_back(rou_first[j]);
                        }
                    }
                    J_triangular.swap(J_triangular_second);
                } else {
                    J_triangular.swap(J_triangular_first);
                }
                
                // print to check
                for (const auto &j : J_triangular) { 
                    // std::cout << "the " << k << "-th column of M, adds new elements to J_triangular : " << j << std::endl;
                }

                // std::cout << "(g) part is processing......" << std::endl;
                // (g) Determine the new indices I_triangular
                std::vector<int> J_merged(J.size() + J_triangular.size());
                std::merge(J.begin(), J.end(), J_triangular.begin(), J_triangular.end(), J_merged.begin());
                std::sort(J_merged.begin(), J_merged.end());
                
                std::vector<int> I_triangular;
                index = 0;
                for (int i=0; i<Size; i++) {
                    if (index<n1 && I[index] == i) {
                        index++;
                    } else {
                        int flag = 0;
                        for (const auto &j : J_merged) { 
                            if (A(i,j)!=zero) {flag = 1;}
                        }
                        if (flag==1) { 
                            I_triangular.push_back(i);
                            // std::cout << "the " << k << "-th column of M, adds new elements to I_triangular : " << i << std::endl;
                        } 
                    }
                }
                
                int n1_triangular = I_triangular.size();
                int n2_triangular = J_triangular.size();
                
                if (n2_triangular!=0) {
                    // A_I_J_triangular
                    Eigen::MatrixXd A_I_J_triangular(n1, n2_triangular);
                    getBlockByTwoIndicesSet(A, A_I_J_triangular, I, J_triangular);
                    // A_I_triangular_J_triangular
                    Eigen::MatrixXd A_I_triangular_J_triangular(n1_triangular, n2_triangular);
                    getBlockByTwoIndicesSet(A, A_I_triangular_J_triangular, I_triangular, J_triangular);

                    // B1 and B2
                    Eigen::MatrixXd B1 = Q.transpose().topLeftCorner(n2,n1) * A_I_J_triangular;
                    Eigen::MatrixXd B2(n1_triangular+n1-n2, n2_triangular);
                    if (n1_triangular==0) {
                        // std::cout << "I_triangular is null" << std::endl;
                        B2 << Q.transpose().bottomLeftCorner(n1-n2, n1) * A_I_J_triangular;
                    } else {
                        B2 << Q.transpose().bottomLeftCorner(n1-n2, n1) * A_I_J_triangular, 
                        A_I_triangular_J_triangular;
                    }
                    
                    // std::cout << "B1 matrix:\n" << B1 << std::endl;
                    // std::cout << "B2 matrix:\n" << B2 << std::endl;
                    
                    // QR decomposition of B2
                    Eigen::MatrixXd Q_triangular;
                    Eigen::MatrixXd R_triangular;
                    qrDecomposition(B2, Q_triangular, R_triangular);

                    // Q_new and R_new is the QR decomposition of A(I+I_triangular, J+J_triangular)
                    //
                    Eigen::MatrixXd result01(n1+n1_triangular, n1+n1_triangular);
                    result01 << Q, Eigen::MatrixXd::Zero(n1, n1_triangular),
                                Eigen::MatrixXd::Zero(n1_triangular, n1), Eigen::MatrixXd::Identity(n1_triangular, n1_triangular); 
                    Eigen::MatrixXd result02(n1+n1_triangular, n1+n1_triangular);
                    result02 << Eigen::MatrixXd::Identity(n2, n2), Eigen::MatrixXd::Zero(n2, n1_triangular+n1-n2),
                                Eigen::MatrixXd::Zero(n1_triangular+n1-n2, n2), Q_triangular;
                    Eigen::MatrixXd Q_new = result01 * result02;
                    // 
                    Eigen::MatrixXd R_new(n2+n2_triangular, n2+n2_triangular);
                    R_new << R, B1,
                            Eigen::MatrixXd::Zero(n2_triangular, n2), R_triangular;
                    // 
                    Q.swap(Q_new);
                    R.swap(R_new);

                    // update I and J
                    std::vector<int> I_merged(I.size() + I_triangular.size());
                    std::merge(I.begin(), I.end(), I_triangular.begin(), I_triangular.end(), I_merged.begin());
                    std::sort(I_merged.begin(), I_merged.end());

                    I.swap(I_merged);
                    J.swap(J_merged);
                    n1 = I.size();
                    n2 = J.size();
                } else {
                    // std::cout << "J_triangular is null" << std::endl;
                }

                // repeate the squares problem process and update variables
                computeRAndMk<decltype(A), decltype(epsilon)>(A, Size, Q, R, r, m_k, e_k, I, J, 1);

                // loop end
            }
            // std::cout << "complete the loop iteration......" << std::endl;

            // k-th
            if (max_iter == iter) {
                std::cout << "the " << k << "-th column of M, complete the loop iteration, the result is " 
                << "\nepsilon=" << epsilon << "    r.norm()=" << r.norm() 
                << "\nmax_iter="<< max_iter << "    iter=" << iter << "\n\n";
            }
        
            // M
            for(int i=0; i<Size; i++) {
                M(i,k) = m_k[i];
            }
        }       

        // std::cout << "\nmatrix M:\n" << M << "\n\n";

        return 1;
    }
}


#endif