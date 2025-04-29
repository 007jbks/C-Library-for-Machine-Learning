#ifndef LOGISTIC_H
#define LOGISTIC_H
#include<bits/stdc++.h>
using namespace std;
class logisticRegression {
    private:
    vector<double> W;
    double bias;
    public:
    logisticRegression();
    void train(vector<vector<double>>& X,vector<int>& y,double lr,int max_iter);
    vector<double> Rand_Weights(int size) ;
    double row_into_col(vector<double>& row,vector<double>& col);
    double sigmoid(double input);
    vector<double> dlossdw(vector<double>& X,double y_hat,double y);
    int predict(vector<double>& X_train);
    vector<int> predictall(vector<vector<double>>& X_train);
};

#endif
