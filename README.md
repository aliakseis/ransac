### Polynomial Fitting with RANSAC and Ceres

A compact C++ example that generates synthetic 1D polynomial data with outliers and fits a polynomial using two approaches:

*   **RANSAC** robust estimator with optional least squares refinement
    
*   **Ceres Solver** nonlinear least squares with an arctan robust loss
    

The project uses OpenCV for simple matrix math and visualization, and Ceres for optimization.

### Files

*   **main.cpp** Source shown by the user. Contains:
    
    *   data generation with controllable outliers and noise
        
    *   RANSAC polynomial fitting with random sampling and SVD inversion
        
    *   Ceres fitting using a dynamic auto diff residual and ArctanLoss
        
    *   visualization with OpenCV imshow
        

### Features

*   Generates synthetic polynomial data of configurable degree and coefficients
    
*   Injects localized outliers to test robustness
    
*   RANSAC:
    
    *   Fisher Yates style sampling across data points
        
    *   SVD based pseudoinverse to solve for polynomial coefficients from minimal samples
        
    *   Consensus counting using a residual threshold
        
    *   Optional final least squares re-fit on inliers
        
*   Ceres:
    
    *   DynamicAutoDiffCostFunction for polynomial residuals
        
    *   Uses ArctanLoss to reduce the influence of outliers
        
    *   Configurable max iterations and solver options
        
*   Visual output showing data points in red and fitted curve samples in green
    

### Dependencies

*   C++17 compatible compiler
    
*   OpenCV (image display and matrix operations)
    
*   Ceres Solver (nonlinear optimization)
    
*   Standard C++ libraries: , , , ,
    

### Build and Run

1.  Configure a CMake or your build system to link against OpenCV and Ceres.
    
2.  Example compile flags
    
    *   **Include dirs** for OpenCV and Ceres
        
    *   **Link libs**: opencv\_core opencv\_highgui ceres
        
3.  Run the resulting binary. Two windows appear:
    
    *   **RANSAC** showing RANSAC fit
        
    *   **Ceres** showing Ceres fit
        
4.  Press any key in the image window to exit.
    

### Key Functions Explained

#### CalcPoly

*   Evaluates a polynomial at x using coefficients stored in a cv::Mat column vector.
    
*   Polynomial basis powers are computed iteratively for numerical stability.
    

#### RansacFitting

*   Inputs: vector of 2D points (x, y), desired polynomial degree n\_samples, residual noise sigma.
    
*   Repeats N random trials:
    
    *   Randomly select n\_samples data points using a Fisher Yates style partial shuffle.
        
    *   Build Vandermonde matrix A and right hand side B for the sample, invert A using SVD to get coefficients.
        
    *   Count inliers whose absolute residual is below threshold T.
        
    *   Retain best model by inlier count.
        
*   Optional least squares refinement:
    
    *   Collect all inliers according to the best model and recompute coefficients via SVD on the inlier set.
        
*   Returns refined coefficient vector as cv::Mat (n\_samples by 1).
    

#### CeresFitting

*   Inputs: vector of 2D points, polynomial degree n\_samples, arctanLoss scale.
    
*   Parameters are stored in a cv::Mat column vector A and wrapped as double pointers for Ceres.
    
*   For each data point adds a DynamicAutoDiffCostFunction based residual built from PolynomialResidual.
    
*   Each residual uses an ArctanLoss to attenuate the effect of outliers.
    
*   Solver options use DENSE\_QR and a limited iteration budget for speed.
    
*   Returns optimized coefficient vector as cv::Mat (n\_samples by 1).
    

#### DrawResult

*   Draws the original points in red and the polynomial evaluated at the sample x positions in green into an OpenCV image.
    
*   Displays the image under the provided window name.
    

### Parameters to Tune

*   **n\_samples** Number of polynomial coefficients to estimate. Controls polynomial degree plus one.
    
*   **noise\_sigma** Noise scale used by the RANSAC threshold T = 3 \* noise\_sigma.
    
*   **arctanLoss** Scale parameter passed to Ceres ArctanLoss.
    
*   **N** Number of RANSAC iterations in RansacFitting.
    
*   **options.max\_num\_iterations** Maximum iterations for the Ceres solver.
    

### Practical Notes and Caveats

*   The code uses cv::invert with DECOMP\_SVD for numerical stability but does not explicitly check matrix conditioning or handle degenerate samples. When A is ill conditioned the solution may be unstable.
    
*   The Fisher Yates partial shuffle uses a displaced map to avoid allocating a full permutation for large datasets; the approach is intended to uniformly sample without replacement.
    
*   Ceres fitting uses dynamic parameter blocks with one double each. For high degree polynomials prefer scaling or basis orthogonalization to reduce parameter correlations.
    
*   Visualization scales points by a fixed factor which assumes input x and y ranges fit within the image. Adjust the scaling or add coordinate transforms for general datasets.
    

### Example Usage Summary

*   Generate data and outliers by running the program.
    
*   Compare robustness of RANSAC and Ceres on the same dataset.
    
*   Tune degree and loss parameters to see behavior differences.

# ransac
ransac curve line fitting

- RANSAC을 이용한 포물선을 구하는 예제이다.
- This is an example source for getting a curvature using RANSAC algorithm

- 참고한 페이지는 http://darkpgmr.tistory.com/61 이고, 참고 페이지에 소개된 MATLAB코드를 C++코드로 변환 하였다.
- I refered http://darkpgmr.tistory.com/61, and converted Matlab code to C++ code

- Matrix계산을 위하여 OpenCV를 사용하였다.
- I used OpenCV Library for Matrix operation


# Instruction
- clone this repository
- go to ransac folder
- `make all`
- `./RansacCurvieFitting`
