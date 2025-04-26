#include<bits/stdc++.h>
#include "linearRegression.h"
using namespace std;

    SimpleLinearRegression::SimpleLinearRegression() : a(0), b(0) {}

    void SimpleLinearRegression::fit(const vector<double>& X,const vector<double>& y){
        double meanX = 0;
        for(auto i:X){
                meanX+=i;
        }
        meanX/=X.size();
    
        double meany = 0;
        for(auto i:y){
            meany +=i;
        }
        meany/=y.size();
        double m = 0;
        double x = 0;
        for(size_t i=0;i<X.size();i++)
        {
                m+=((X[i]-meanX)*(y[i]-meany));
                x+=(X[i]-meanX)*(X[i]-meanX);
        }
         b = m/x;
         a = y[0]-b*X[0];
    }

    double SimpleLinearRegression::predict(double x) const{
        return b*x+a;
    }


    LinearRegression::LinearRegression(): bias(0.0){}

    void LinearRegression::train(const vector<vector<double>>& X,const vector<double>& y,const double lr,const int iterations)
    {
        srand(time(0));
        W = Rand_Weights(X[0].size());
        bias = 0.0;
      

        for(int iter=0;iter<iterations;iter++)
        {
            vector<double> dldw(W.size(), 0.0); // Initialize gradient for each weight
            double dldb = 0.0; // gradient for bias
    
            double total_loss=0;
            
            for(size_t i =0;i<X.size();i++)
            {
                double y_pred_i = 0;
                y_pred_i += row_into_col(X[i],W) +bias;
                total_loss+=mse(y_pred_i,y[i]);
                
                vector<double> grads = dloss_dw(y_pred_i, y[i], X[i]);
                for (size_t j = 0; j < grads.size(); j++)
                {
                    dldw[j] += grads[j];
                }
            
                dldb += 2 * (y_pred_i - y[i]);
            }
            
            total_loss/=X.size(); 
            
            //Updating weights and biases:
            bias = bias - lr* (dldb/X.size());
            for(size_t i=0;i<W.size();i++)
            W[i] = W[i]-lr*(dldw[i]/X.size());
        }
        
    }

    vector<double> LinearRegression:: Rand_Weights(int size)
    {
        vector<double> weights(size);
        for (int i = 0; i < size; i++) {
            weights[i] = ((double) rand() / RAND_MAX) * 2 - 1; 
        }
        return weights;
    }

    double LinearRegression::mse(double predicted,double real){
        return (1/2)*(predicted-real)*(predicted-real);
    }

    double LinearRegression:: row_into_col(const vector<double>& row,const vector<double>& col) const
    {
        double result = 0;

        for(size_t i=0;i<row.size();i++)
        {
            result+=row[i]*col[i];
        }
        return result;
    }

    vector<double> LinearRegression::dloss_dw(double y_pred,double y,const vector<double>& X)
    {
        vector<double> grads(X.size());
        for(size_t i=0;i<X.size();i++)
        {
            grads[i]=2*(y_pred-y)*X[i];
        }
        return grads;
    }

        double LinearRegression:: predict(const vector<double>& x) const{
        return row_into_col(x, W) + bias;
    }

    vector<double> LinearRegression::predict_all(const vector<vector<double>>& X) const{
        vector<double> preds;
        for (const auto& x : X) {
            preds.push_back(predict(x));
        }
        return preds;
    }
    
    // double linearRegression::r2_score(const vector<double>& y_true, const vector<double>& y_pred)  const{
    //     double mean_y = 0.0;
    //     for (double val : y_true) {
    //         mean_y += val;
    //     }
    //     mean_y /= y_true.size();
    
    //     double ss_tot = 0.0, ss_res = 0.0;
    //     for (size_t i = 0; i < y_true.size(); i++) {
    //         ss_tot += (y_true[i] - mean_y) * (y_true[i] - mean_y);
    //         ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    //     }
    
    //     return 1 - (ss_res / ss_tot);
    // }

    // double LinearRegression::evaluate(const vector<vector<double>>& X, const vector<double>& y) const {
    //     vector<double> preds = predict_all(X);
    //     return r2_score(y, preds);
    // }
    
    