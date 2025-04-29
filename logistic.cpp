#include<bits/stdc++.h>
#include "logistic.h"
#include<cmath>
using namespace std;

logisticRegression::logisticRegression (): bias(0.0) {}

void logisticRegression::train(vector<vector<double>>& X,vector<int>& y,double lr,int max_iter)
{
    srand(time(0));
    W = Rand_Weights(X[0].size());
    bias = 0.0;
    
    for(int i=0;i<max_iter;i++)
    {
        vector<double> dldw(W.size(),0.0);
        double dldb = 0.0;
        
        for(size_t j =0; j<X.size();j++)
        {
            double y_hat = row_into_col(X[j],W) + bias;
             y_hat = sigmoid(y_hat);
            // y_pred = y_pred>0.5?1:0;
            vector<double> grad = dlossdw(X[j],y_hat,y[j]);
            for(size_t k=0;k<grad.size();k++)
            dldw[k]+=grad[k];
            dldb+=(y_hat-y[j]); 
        }
        //Updating the weights
        for(size_t i = 0;i<W.size();i++)
        W[i]-=lr*dldw[i];
        bias-=lr*dldb;
    }
}

vector<double> logisticRegression:: Rand_Weights(int size)
{
    vector<double> weights(size);
    for (int i = 0; i < size; i++) {
        weights[i] = ((double) rand() / RAND_MAX) * 2 - 1; 
    }
    return weights;
}

double logisticRegression:: row_into_col(vector<double>& row,vector<double>& col)
{
    double res = 0;
    for(size_t i=0;i<row.size();i++) 
    {
        res+=row[i]*col[i];
    }   
    return res;
}

double  logisticRegression:: sigmoid(double input)
{
    double output;
    output = 1/(1+exp(-input));
    return output;
}

vector<double> logisticRegression::dlossdw(vector<double>& X,double y_hat,double y )
{
    vector<double> grad(X.size());
    for(size_t i=0;i<X.size();i++)
    grad[i] = (y_hat-y)*X[i];
    return grad;
}

//Prediction Function:

int logisticRegression:: predict(vector<double>& X_train)
{ 
        double y_hat = row_into_col(X_train,W) + bias;
        y_hat = sigmoid(y_hat);
        return y_hat>0.5?1:0;
}

vector<int> logisticRegression:: predictall(vector<vector<double>>& X_train)
{
    vector<int> ans;
    for(size_t i = 0;i<X_train.size();i++)
    {
        ans.push_back(predict(X_train[i]));
    }
    return ans;

}

