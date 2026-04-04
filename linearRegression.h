#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class linearRegression{
    public:
    //matrix to store features and predictions from outside
    MatrixXd features;
    //matrix to store expected known predictions
    VectorXd ex_y_pred;
    //matrices for storing weights and made predictions
    VectorXd weights,y_predictions;
    //basically the y intercept or bias as it is called
    double bias;
    //storing mean squared error loss
    double floss;

    int cycles;

    //this cute constructor takes and stores the feature matrix address from outside
    //and sets the weights vector and y_predictions according to it.
    linearRegression(MatrixXd& features,VectorXd& ex_y_pred)
    :features(features),ex_y_pred(ex_y_pred){
        this->weights=VectorXd::Zero(this->features.cols());
        this->y_predictions=VectorXd::Zero(this->features.rows());
        this->bias=0.0;
    }

    //this function is used to train on the inputted data in the constructor
    //learningRate is what it says , maxIter is how many loops to train and tolerance
    //when to stop when reaching min improvement. 
    void train(double learningRate=0.00001,int maxIter=100,double tolarence=1e-6){
        VectorXd error,grad_w;
        double grad_c,loss;
        double prevLoss=1e9;

        cycles=0;
        //main train loop where the magic happens and weights are calculated
        for(int i=0;i<maxIter;i++){
        //make predictions against the data
        y_predictions= features * weights;
        y_predictions.array() += bias;
        
        //calculate how bad the predictions were from actual or expected predictions
        error= y_predictions - ex_y_pred;
        //calculate mean squared error , also store it in floss for later
        floss=loss=(error.array().square().sum())/error.rows();
        //checking if calculation further is useless or not
        //basically cheking if we've reached minumum improvement possiable
        if(abs(prevLoss-loss)<tolarence) break;
        //storing for comparisen next iteration
        //as a famous dude once said "for those who come after" or something
        prevLoss=loss;

        //this is a bunch of calculas stuff refer to the other doc for more on this
        grad_w=features.transpose() * error / error.rows();
        grad_c=error.sum()/error.rows();
        //set the new weights and bias
        weights = weights-learningRate*grad_w;
        bias=bias - learningRate *grad_c;
        cycles=i;
        }
    }

    //the part which allows you to actually use the trained model just give it the data to predict against
    VectorXd testAndPredict(MatrixXd& Tfeatures){
        //checking for bad input 
        if(features.cols()!=weights.rows()){cout<<"invalid features\n";return VectorXd();}
        //acutal prediction ik its kinda disappointing at this point
        VectorXd TPredictions= Tfeatures * weights;
        TPredictions.array() += bias;
        return TPredictions;
    }

    //these are evaluation matrics 
    double MSE(){
        return floss;
    }

    double r2Score(VectorXd& y, VectorXd y_pred){
        double ss_res=(y-y_pred).array().square().sum();
        double mean_y=y.mean();
        double ss_tot=(y.array()-mean_y).square().sum();
        return 1.0-(ss_res/ss_tot);
    }

};