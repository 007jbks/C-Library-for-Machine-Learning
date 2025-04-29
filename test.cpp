#include <iostream>
#include <vector>
#include <chrono>
#include "logistic.h"

using namespace std;

// Function to calculate accuracy
double accuracy_score(const vector<int>& y_true, const vector<int>& y_pred) {
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / y_true.size();
}

int main() {
    vector<vector<double>> X_train = {
        {5.1, 3.5, 1.4, 0.2},  // Setosa
        {4.9, 3.0, 1.4, 0.2},  // Setosa
        {4.7, 3.2, 1.3, 0.2},  // Setosa
        {4.6, 3.1, 1.5, 0.2},  // Setosa
        {5.0, 3.6, 1.4, 0.2},  // Setosa
        {5.4, 3.9, 1.7, 0.4},  // Setosa
        {4.6, 3.4, 1.4, 0.3},  // Setosa
        {5.0, 3.4, 1.5, 0.2},  // Setosa
        {4.4, 2.9, 1.4, 0.2},  // Setosa
        {4.9, 3.1, 1.5, 0.1},  // Setosa
        {5.9, 3.0, 5.1, 1.8},  // Non-Setosa
        {6.0, 2.9, 4.5, 1.5},  // Non-Setosa
        {5.7, 3.0, 4.2, 1.2},  // Non-Setosa
        {5.5, 2.4, 3.8, 1.1},  // Non-Setosa
        {6.3, 2.8, 5.1, 1.5},  // Non-Setosa
        {5.8, 2.7, 5.1, 1.9},  // Non-Setosa
        {6.7, 3.0, 5.2, 2.3},  // Non-Setosa
        {6.9, 3.1, 5.4, 2.1},  // Non-Setosa
        {5.6, 2.5, 3.9, 1.1},  // Non-Setosa
    };
    
    vector<int> y_train = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // Setosa
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0  // Non-Setosa
    };
    
   
    vector<vector<double>> X_test = {
        {5.0, 3.5, 1.6, 0.6},  // Setosa
        {5.2, 3.4, 1.4, 0.2},  // Setosa
        {5.1, 3.5, 1.4, 0.2},  // Setosa
        {6.1, 2.8, 4.7, 1.2},  // Non-Setosa
        {5.9, 3.0, 4.2, 1.5},  // Non-Setosa
        {6.2, 2.9, 4.3, 1.3},  // Non-Setosa
        {6.5, 3.0, 5.5, 1.8},  // Non-Setosa
        {6.7, 3.1, 5.6, 2.4},  // Non-Setosa
        {5.6, 3.0, 4.1, 1.3},  // Non-Setosa
        {5.8, 2.7, 5.1, 1.9},  // Non-Setosa
    };
    
    vector<int> y_test = {
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0  // Labels for Setosa (1) vs Non-Setosa (0)
    };
    
   
    logisticRegression model;

    // Training the model
    auto start = chrono::high_resolution_clock::now();
    model.train(X_train, y_train, 0.1, 1000);  // Learning rate = 0.1, iterations = 1000
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration = end - start;
    cout << "Training Time: " << duration.count() << " seconds\n";

    // Predict on the training data
    vector<int> predictions = model.predictall(X_test);

    // Print predictions
    cout << "Predictions: ";
    for (int pred : predictions) {
        cout << pred << " ";
    }
    cout << "\n";

    // Calculate and print accuracy
    double acc = accuracy_score(y_test, predictions);
    cout << "Accuracy: " << acc * 100 << "%" << endl;

    return 0;
}
