/*
 * softmax
 * softmax(x, class)[i] = exp(x[i]) / sum(exp(x[j]))
 */
void softmax(float* x, int num_class, int batch_size);