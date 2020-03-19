/*
 * Calculate cross entropy loss
 * Loss(x,class) = -log(exp(x[class])/sum_j(exp(x[j])))
 */
float CrossEntropyLoss(float* x, int* target, int batch_size, int C, bool average=true);