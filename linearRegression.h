#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <vector>
using namespace std;

class SimpleLinearRegression {
private:
    double a;
    double b;
public:
    SimpleLinearRegression();
    void fit(const vector<double>& X, const vector<double>& y);
    double predict(double x) const;
};

class LinearRegression {
private:
    vector<double> W;
    double bias;
    

     // Function to generate random weights
     std::vector<double> Rand_Weights(int size);
 
     // Function to compute dot product of a row and a column (used in predictions)
     double row_into_col(const std::vector<double>& row, const std::vector<double>& col) const;
 
     // Function to compute the gradient for weights
     std::vector<double> dloss_dw(double y_pred, double y, const std::vector<double>& X);
 
     // Mean squared error loss
     double mse(double predicted, double real);
 
public:
    LinearRegression();
    void train(const vector<vector<double>>& X, const vector<double>& y, const double lr, const int iterations);
    vector<double> predict_all(const vector<vector<double>>& X) const;
    double predict(const vector<double>& X) const;
    double evaluate(const vector<vector<double>>& X, const vector<double>& y) const;
     // Function to calculate R² score
     //double r2_score(const std::vector<double>& y_true, const std::vector<double>& y_pred) const;
};

#endif
