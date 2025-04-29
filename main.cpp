#include <iostream>
#include <vector>
#include <chrono>
#include "linearRegression.h"

// Function to calculate R² score
double r2_score(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    double mean_y = 0.0;
    for (double val : y_true) {
        mean_y += val;
    }
    mean_y /= y_true.size();

    double ss_tot = 0.0, ss_res = 0.0;
    for (int i = 0; i < y_true.size(); i++) {
        ss_tot += (y_true[i] - mean_y) * (y_true[i] - mean_y);
        ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    }

    return 1 - (ss_res / ss_tot);
}

int main() {
    // Sample dataset (Linear Regression with multiple features)
    vector<vector<double>> X = {
        {1.0, 2.0, 4.0},  
        {2.0, 3.0, 5.0},
        {3.0, 4.0, 6.0},
        {4.0, 5.0, 7.0},
        {5.0, 6.0, 8.0},
        {6.0, 7.0, 9.0},
        {7.0, 8.0, 10.0},
        {8.0, 9.0, 11.0},
        {9.0, 10.0, 12.0},
        {10.0, 11.0, 13.0}
    };

    vector<double> y = {5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0};  

  
    LinearRegression model;

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // Train the model
    model.train(X, y, 0.001, 1000);

    // End timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken for train(): " << duration.count() << " seconds\n";
    vector<vector<double>> X_test = {
        {11.0, 12.0, 14.0},
        {12.0, 13.0, 15.0},
        {13.0, 14.0, 16.0},
        {14.0, 15.0, 17.0},
        {15.0, 16.0, 18.0}
    };
    
    vector<double> y_test = {25.0, 27.0, 29.0, 31.0, 33.0}; // Continuation of the same pattern
    
    // Get predictions for the training set
    std::vector<double> predictions = model.predict_all(X_test);

    // // Calculate and print R² (Accuracy)
    // double accuracy = model.r2_score(y, predictions);
    // std::cout << "R² Score (Model Accuracy): " << accuracy << std::endl;

    // Output predictions for each x in X
    std::cout << "Predictions: ";
    for (double pred : predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;
    double mape = 0.0;
    for(size_t i = 0; i < y_test.size(); i++) {
        mape += std::abs((y_test[i] - predictions[i]) / y_test[i]);
    }
    mape /= y_test.size();  // Calculate average
    mape *= 100;  // Convert to percentage
    std::cout << "MAPE: " << mape << "%" << std::endl;
   
    return 0;
}
