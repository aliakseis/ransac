#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <ceres/ceres.h>

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <random>
#include <utility>

#include <time.h>
#include <math.h>


double CalcPoly(const cv::Mat& X, double x)
{
    double result = X.at<double>(0, 0);
    double v = 1.;
    for (int i = 1; i < X.rows; ++i)
    {
        v *= x;
        result += X.at<double>(i, 0) * v;
    }
    return result;
}

cv::Mat RansacFitting(const std::vector<cv::Point2f>& vals, int n_samples, double noise_sigma)
{
    //int n_data = vals.size();
    int N = 100;	//iterations 
    double T = 3 * noise_sigma;   // residual threshold

    //int n_sample = 3;
    int max_cnt = 0;
    cv::Mat best_model(n_samples, 1, CV_64FC1);

    std::default_random_engine dre;

    std::vector<int> k(n_samples);

    for (int n = 0; n < N; n++)
    {
        //random sampling - n_samples points
        for (int j = 0; j < n_samples; ++j)
            k[j] = j;

        std::map<int, int> displaced;

        // Fisher-Yates shuffle Algorithm
        for (int j = 0; j < n_samples; ++j)
        {
            std::uniform_int_distribution<int> di(j, vals.size() - 1);
            int idx = di(dre);

            if (idx != j)
            {
                int& to_exchange = (idx < n_samples)? k[idx] : displaced.try_emplace(idx, idx).first->second;
                std::swap(k[j], to_exchange);
            }
        }

        //printf("random sample : %d %d %d\n", k[0], k[1], k[2]);

        //model estimation
        cv::Mat AA(n_samples, n_samples, CV_64FC1);
        cv::Mat BB(n_samples, 1, CV_64FC1);
        for (int i = 0; i < n_samples; i++)
        {
            AA.at<double>(i, 0) = 1.;
            double v = 1.;
            for (int j = 1; j < n_samples; ++j)
            {
                v *= vals[k[i]].x;
                AA.at<double>(i, j) = v;
            }

            BB.at<double>(i, 0) = vals[k[i]].y;
        }

        cv::Mat AA_pinv(n_samples, n_samples, CV_64FC1);
        invert(AA, AA_pinv, cv::DECOMP_SVD);

        cv::Mat X = AA_pinv * BB;

        //evaluation 
        int cnt = 0;
        for (const auto& v : vals)
        {
            double data = std::abs(v.y - CalcPoly(X, v.x));

            if (data < T)
            {
                cnt++;
            }
        }

        if (cnt > max_cnt)
        {
            best_model = X;
            max_cnt = cnt;
        }
    }

    //------------------------------------------------------------------- optional LS fitting 
    std::vector<int> vec_index;
    for (int i = 0; i < vals.size(); i++)
    {
        const auto& v = vals[i];
        double data = std::abs(v.y - CalcPoly(best_model, v.x));
        if (data < T)
        {
            vec_index.push_back(i);
        }
    }

    cv::Mat A2(vec_index.size(), n_samples, CV_64FC1);
    cv::Mat B2(vec_index.size(), 1, CV_64FC1);

    for (int i = 0; i < vec_index.size(); i++)
    {
        A2.at<double>(i, 0) = 1.;
        double v = 1.;
        for (int j = 1; j < n_samples; ++j)
        {
            v *= vals[vec_index[i]].x;
            A2.at<double>(i, j) = v;
        }


        B2.at<double>(i, 0) = vals[vec_index[i]].y;
    }

    cv::Mat A2_pinv(n_samples, vec_index.size(), CV_64FC1);
    invert(A2, A2_pinv, cv::DECOMP_SVD);

    cv::Mat X = A2_pinv * B2;

    return X;
}

//////////////////////////////////////////////////////////////////////////////

struct PolynomialResidual {
    PolynomialResidual(double x, double y, int n_samples) : x_(x), y_(y), n_samples_(n_samples) {}

    template <typename T>
    bool operator()(T const* const* relative_poses, T* residuals) const {

        T y = *(relative_poses[0]) + *(relative_poses[1]) * x_;
        for (int i = 2; i < n_samples_; ++i) y += *(relative_poses[i]) * std::pow(x_, i);

        residuals[0] = T(y_) - y;
        return true;
    }

private:
    // Observations for a sample.
    const double x_;
    const double y_;
    int n_samples_;
};


cv::Mat CeresFitting(const std::vector<cv::Point2f>& vals, int n_samples, double arctanLoss)
{
    cv::Mat A = cv::Mat::zeros(n_samples, 1, CV_64FC1);

    std::vector<double*> params;
    params.reserve(n_samples);
    for (int i = 0; i < n_samples; ++i) params.push_back(&A.at<double>(i, 0));

    ceres::Problem problem;
    for (auto& i : vals) {
        auto cost_function =
            new ceres::DynamicAutoDiffCostFunction<PolynomialResidual>(new PolynomialResidual(i.x, i.y, n_samples));

        for (int j = 0; j < params.size(); ++j) cost_function->AddParameterBlock(1);

        cost_function->SetNumResiduals(1);

        problem.AddResidualBlock(cost_function, new ceres::ArctanLoss(arctanLoss), params);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    options.max_num_iterations = 50;

    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    return A;
}

//////////////////////////////////////////////////////////////////////////////

void DrawResult(const cv::Mat& X, const std::vector<cv::Point2f>& vals, const cv::String& name)
{
    //Drawing
    int interval = 5;
    cv::Mat imgResult(100 * interval, 100 * interval, CV_8UC3);
    imgResult = cv::Scalar(0);
    for (const auto& v : vals)
    {
        cv::circle(imgResult, cv::Point(v.x*interval, v.y*interval), 3, cv::Scalar(0, 0, 255), cv::FILLED);

        double data = CalcPoly(X, v.x);

        cv::circle(imgResult, cv::Point(v.x*interval, data*interval), 1, cv::Scalar(0, 255, 0), cv::FILLED);
    }
    cv::imshow(name, imgResult);

}


int main(void)
{
	srand(time(NULL)) ;

	//-------------------------------------------------------------- make sample data  
	/* random number in range 0 - 1 not including 1 */
	float random = 0.f;
	/* the white noise */
	float noise = 0.f;
	/* Setup constants */
	const static int q = 15;
	const static float c1 = (1 << q) - 1;
	const static float c2 = ((int)(c1 / 3)) + 1;
	const static float c3 = 1.f / c1;

	const double noise_sigma = 1;
    const double arctanLoss = 5;

    std::vector<cv::Point2f> vals(100);

    double iE = -0.000001;
    double iD = 0.00005;
    double iC = 0.005 ;
	double iB = 0.5 ;
	double iA = 0 ;
	for( int i=0 ; i<100 ; i++ )
	{
        const auto x = i;

        vals[i] = { float(i), float(iE * (x * x * x * x) + iD * (x * x * x) + iC*(x * x) + iB * x + iA) };

#if 1
		if( i > 50 && i < 70 )
		{
            vals[i].y += 0.5*abs(x);
		}
#endif
		    
		random = ((float)rand() / (float)(RAND_MAX + 1));
		noise = (2.f * ((random * c2) + (random * c2) + (random * c2)) - 3.f * (c2 - 1.f)) * c3;

		int noise_scale = 2.0 ;
		if( i > 50 && i<70 ) noise_scale = 5.0 ;
        vals[i].y += noise*noise_scale ;
	}


	//-------------------------------------------------------------- RANSAC fitting 
    DrawResult(RansacFitting(vals, 5, noise_sigma), vals, "RANSAC");
    //-------------------------------------------------------------- Ceres fitting 
    DrawResult(CeresFitting(vals, 5, arctanLoss), vals, "Ceres");

	cv::waitKey(0) ;

	return 0 ;
}
