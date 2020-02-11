/*
 * Calculate cross entropy loss
 * Loss(x,class) = -log(exp(x[class])/sum_j(exp(x[j])))
 */
#include <iostream>
#include <cmath>
/*
 * This function has four inputs
 * x dimension:(batch_size,C) -> 1d array
 * target: dimension: (batch_size)
 * C: number of class
 * batch_size: as is
 */
float CrossEntropyLoss(float* x, int* target, int batch_size, int C, bool average=true){
    float batch_loss = 0.0;
    for(int i = 0; i < batch_size; i++){
        float numerator = exp(x[i * C + target[i]]); // x[i, target[i]]
        float denominator = 0;
        for(int j = 0; j < C; j++){
            denominator += exp(x[i * C + j]);
        }
        float loss = - log(numerator / denominator);
        batch_loss += loss;
    }

    return average ? batch_loss / batch_size : batch_loss;
}
