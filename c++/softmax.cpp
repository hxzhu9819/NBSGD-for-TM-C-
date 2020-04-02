/*
 * softmax
 * softmax(x, class)[i] = exp(x[i]) / sum(exp(x[j]))
 */
#include <iostream>
#include <cmath>
using namespace std;

/*
 * This function has two inputs:
 * x: the input that will be softmaxed
 * num_class: the number of classes
 */
void softmax(float* x, int num_class, int batch_size) {
    for(int i = 0; i < batch_size; i++){
        // for each element of the batch
        // calculate denominator
        float denominator = 0;
        for(int j = 0; j < num_class; j++){
            denominator += exp(x[i * num_class + j]);
        }
        // cout << "denominator: "<<denominator << endl;
        // calculate probability for each entry
        for(int j = 0; j < num_class; j++){
            x[i * num_class + j] = exp(x[i * num_class + j]) / denominator;
        }

    }
}
