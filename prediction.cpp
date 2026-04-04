#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include "linearRegression.h"
#include <random>
using namespace Eigen;
using namespace std;

// Function to read CSV into MatrixXd
MatrixXd readCSV(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Could not open file: " << filename << "\n";
        return MatrixXd(); // return empty matrix
    }

    vector<vector<double>> data;
    string line;

    // Read the file data
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> row;

        while (getline(ss, value, ',')) {
            row.push_back(stod(value)); // convert string to double
        }

        data.push_back(row);
    }

    file.close();

    if (data.empty()) return MatrixXd();

    // Create MatrixXd with appropriate size
    int rows = data.size();
    int cols = data[0].size();
    MatrixXd mat(rows, cols);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            mat(i, j) = data[i][j];

    return mat;
}

int main() {
    int cycles;
    float lr;
    string savefile;
    cout<<"enter lr: ";
    cin>>lr;
    cout<<"enter cycles :";
    cin>>cycles;
    cout<<"enter file save name without .csv: ";
    cin>>savefile;

    MatrixXd TrainData = readCSV("train.csv");
    MatrixXd TestData = readCSV("test.csv");

    if (TrainData.size() == 0 && TrainData.size() == 0) {
        cout << "Failed to read data." << "\n";
        return 1;
    }

    // load features and predictions seperatly
    MatrixXd features = TrainData.leftCols(TrainData.cols() - 1); // all but last column
    VectorXd predictions = TrainData.col(TrainData.cols() - 1);    // last column as target

    MatrixXd TestFeatures = TestData.leftCols(TestData.cols() - 1); // all but last column
    VectorXd TestPredictions = TestData.col(TestData.cols() - 1);    // last column as target

    // Print sizes
    cout << "Training features: " << features.rows() << " x " << features.cols() << "\n";
    cout << "Training targets: " << predictions.size() << "\n";
    cout << "Testing features: " << TestFeatures.rows() << " x " << TestFeatures.cols() << "\n";
    cout << "Testing targets: " << TestPredictions.size() << "\n";

    //initialize the model and then start its training
    linearRegression model(features,predictions);
    model.train(lr,cycles);

    //infer the model to predict
    VectorXd out=model.testAndPredict(TestFeatures);

    //display and save results in seperate log type CSV file
    ofstream Sfile("results/"+savefile+".csv");

    if (!Sfile) {
        cout << "Error opening file!" << endl;
        return 1;
    }
    Sfile<<"Predictions , Actual \n";

    for (int i = 0; i < out.size(); i++) {
        Sfile << out(i) << "," << TestPredictions(i) << "\n";
        cout << out(i) << "," << TestPredictions(i) << "\n";
}
    Sfile<<"cycles ran, "<<model.cycles<<"\n";
    Sfile<<"MSE,"<<model.MSE();
    Sfile<<"\nR2,"<<model.r2Score(TestPredictions,out)<<"\n";
    cout<<"cycles ran: "<<model.cycles<<"\n";
    cout<<"MSE:"<<model.MSE();
    cout<<"\nR2:"<<model.r2Score(TestPredictions,out)<<"\n";
    Sfile.close();
    return 0;
}

  


