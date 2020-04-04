#include "forward.hpp"
#include "cross-entropy.h"
#include "softmax.h"
#include <math.h> 
using namespace std;

#define DEBUG

// def forward(self, feat_idx):
//         w = self.w(feat_idx) + self.w_adj
//         r = self.r(feat_idx)

//         x = (w * r).sum(dim=1)

//         # print("x",x.shape)
//         # print(x)
        
//         x = x / self.r_adj
//         return x
uint32_t VOCAB_SIZE = 200000;
uint32_t BATCH_SIZE = 100;
uint32_t R_DEPTH = 2;
uint32_t R_WIDTH = 1000;
uint32_t R_HEIGHT = BATCH_SIZE;
float W_ADJ = 0.4;
float R_ADJ = 10;

//inputs:
//uint32_t *feat_idx = new uint32_t[BATCH_SIZE * 1000];
uint32_t *feat_idx = feat;
float *w = w_data;
//float *w = new float[VOCAB_SIZE];
// r[2 * vocab_size + 2,1] from .hpp: in python r[vocab_size+1, 2]
// w from init [vocab_size+1, 1]: in python w[vocab_size+1, 1]

int main(){
    // Extract entries from r according to feat_idx ( r = self.r(feat_idx) )

    // One method to accomplish: [ [[1,2],[3,4]],[[5,6],[7,8]] ] -> [1,5,2,6,3,7,4,8]
    // float *chosen_r = new float[BATCH_SIZE * R_WIDTH * R_DEPTH];
    // for(uint32_t i = 0; i < BATCH_SIZE; ++i){
    //     for(uint32_t j = 0; j < R_WIDTH; ++j){
    //         for(uint32_t k = 0; k < R_DEPTH; ++k){
    //             chosen_r[i * R_WIDTH * R_DEPTH + j * R_DEPTH + k] = r[2 * feat_idx[i * R_WIDTH + j] + k];
    //         }
    //     }
    // }
    // Another method to accomplish: [ [[1,2],[3,4]],[[5,6],[7,8]] ] -> [1,2,3,4,5,6,7,8]
    float *chosen_r = new float[BATCH_SIZE * R_WIDTH * R_DEPTH];
    for(uint32_t k = 0; k < R_DEPTH; ++k){
        for(uint32_t i = 0; i < BATCH_SIZE; i++){
            for(uint32_t j = 0; j < R_WIDTH; ++j){
                chosen_r[k * BATCH_SIZE * R_WIDTH + i * R_WIDTH + j] = r[2 * feat_idx[i * R_WIDTH + j] + k];
            }
        }
    }

    #ifdef DEBUG
    cout << "printing chosen r" << endl;
    for(int i = 0; i < 10; i ++){
        cout << chosen_r[i] << endl;
    }
    #endif
    

    // Extract entries from w according to feat_idx ( w = self.w(feat_idx) + self.w_adj )
    float *chosen_w = new float[BATCH_SIZE * R_WIDTH];
    for(uint32_t i = 0; i < BATCH_SIZE; ++i){
        for(uint32_t j = 0; j < R_WIDTH; ++j){
            chosen_w[i * R_WIDTH + j] = w[feat_idx[i * R_WIDTH + j]] + W_ADJ;
        }
    }

    // Inner Product
    float *x = new float[BATCH_SIZE * R_DEPTH] ();
    for(uint32_t k = 0; k < R_DEPTH; ++k){
        for(uint32_t i = 0; i < BATCH_SIZE; ++i){
            for(uint32_t j = 0; j < R_WIDTH; ++j){
                x[i * 2 + k] += chosen_w[i * R_WIDTH + j] * chosen_r[k * BATCH_SIZE * R_WIDTH + i * R_WIDTH + j];
            }
            x[i * 2 + k] /= R_ADJ;
        }
    }

    #ifdef DEBUG
    cout << "printing x" << endl;
    for(int i = 0; i < 10; i ++){
        cout << x[i] << endl;
    }
    #endif


    /*****************************
    Calculate Loss (Cross-entropy)
    ******************************/
   
    float loss = CrossEntropyLoss(x, target, BATCH_SIZE, 2, true);
    cout << "loss: " << loss << endl;

    /*****************************
               Softmax
    ******************************/
    softmax(x, R_DEPTH, BATCH_SIZE);

    #ifdef DEBUG
    cout << "after softmax. x:" << endl;
    for(int i = 0; i < 10; i ++){
        cout << x[i] << endl;
    }
    #endif
    // end forward


    /*****************************
             Backward
    ******************************/

    // generate target
    float* target_2d = new float[BATCH_SIZE * R_DEPTH];
    for(int i = 0; i < BATCH_SIZE; i++) {
        for(int j = 0; j < R_DEPTH; j++){
            target_2d[i*R_DEPTH + j] = target[i] == j ? 1 : 0; 
            // cout << "t-j:" << target[i] << j << "target_2d[" << i*R_DEPTH + j << "]= " << target_2d[i*R_DEPTH + j] << endl;
        }
    }
    #ifdef DEBUG
    cout << "target2d:" << endl;
    for(int i = 0; i < 10; i ++){
        cout << target_2d[i] << " ";
        if (i % 2 != 0) cout << endl;
    }
    #endif

    float* dlossdy = new float[BATCH_SIZE * R_DEPTH];
    for(int i = 0; i < BATCH_SIZE * R_DEPTH; i++){
        dlossdy[i] = x[i] - target_2d[i];
    }


    #ifdef DEBUG
    cout << "dlossdy:" << endl;
    for(int i = 0; i < 10; i ++){
        cout << dlossdy[i] << " ";
        if (i % 2 != 0) cout << endl;
    }
    #endif

    //float* delta_w= new float[BATCH_SIZE * R_WIDTH];  // r is 100*1000
    // // iterate through chosen_w
    // for (uint32_t i = 0; i < BATCH_SIZE; i++) {  // Batch size
    //     for (uint32_t j = 0; j < R_WIDTH; j++) {  // R_WIDTH
    //         for (uint32_t p = 0; p < BATCH_SIZE; p++) {  // Update per batch
    //             for (uint32_t k = 0; k < R_DEPTH; k++) {
    //                 delta_w[i * R_WIDTH + j] += chosen_r[i * R_WIDTH + j] * dlossdy[p * R_DEPTH + k] / R_ADJ;
    //             }
    //         }
    //     }
    // }

    // iterate through chosen_w
    // for (uint32_t i = 0; i < BATCH_SIZE; i++) {  // Batch size
    //     for (uint32_t j = 0; j < R_WIDTH; j++) {  // R_WIDTH
    //         for (uint32_t k = 0; k < R_DEPTH; k++) {
    //             delta_w[i * R_WIDTH + j] += chosen_r[k * BATCH_SIZE * R_WIDTH + i * R_WIDTH + j] * dlossdy[i * R_DEPTH + k] / R_ADJ;
    //         }
    //     }
    // }

    float masked_r[2*(VOCAB_SIZE+1)];
    float* delta_w = new float[(VOCAB_SIZE+1)];
    for (uint32_t i = 0; i < BATCH_SIZE; i++) {  // Batch size
        for (uint32_t j = 0; j < R_DEPTH; j++) {  // R_DEPTH
                float current_dlossdy = dlossdy[i * R_DEPTH + j];
                //zero
                for(int k = 0; k < 2*(VOCAB_SIZE+1); k++){
                    masked_r[k] = 0;
                }
                //masked_r[data[i][:]] = self.r.weight[data[i][:]]
                for(int k = 0; k < R_WIDTH; k++){
                    for(int l = 0; l < R_DEPTH; l++){
                        masked_r[2* feat_idx[i * R_WIDTH + k] + l] = r[2* feat_idx[i * R_WIDTH + k] + l];
                    }
                }
                //delta_w[:, 0] += current_dlossdy*masked_r[:, j]/self.r_adj
                for(int k = 0; k < (VOCAB_SIZE+1); k++) {
                    delta_w[k] += current_dlossdy * masked_r[2*k+j] / R_ADJ;
                }
        }
    }

    for (uint32_t i = 0; i < BATCH_SIZE * R_WIDTH; i++) {
        delta_w[i] /= BATCH_SIZE;
        //delta_w[i] = abs(delta_w[i]) < 1e-10 ? 0 : delta_w[i];
    }

    cout << "delta_w:" << endl;
    for(int i = 0; i < 10; i ++){
        cout << delta_w[i] << " ";
    }
    cout << endl;


    // float* delta = new float[BATCH_SIZE * R_WIDTH];
    // for(uint32_t i = 0; i < BATCH_SIZE; ++i){
    //     float numerator = target[i] ? exp(x[i * R_DEPTH]) : exp(x[i * R_DEPTH + 1]);
    //     for(uint32_t j = 0; j < R_WIDTH; ++j){
    //         //cout << i << " " << j <<" " << chosen_r[i * R_WIDTH + j] << " "<<  x[i * R_DEPTH + 1] <<  " "<< x[i*R_DEPTH] <<endl;
    //         delta[i * R_WIDTH + j] = (chosen_r[i * R_WIDTH + j] / R_ADJ) * numerator / ( (exp(x[i*R_DEPTH + 1]) + exp(x[i*R_DEPTH])) );
    //     }
    //     //cout<<(exp(x[i*R_DEPTH + 1]) + exp(x[i*R_DEPTH]))<<" "<< numerator << " " << exp(x[i*R_DEPTH]) << " " << x[i*R_DEPTH +1] << endl;
    // } 
    // #ifdef DEBUG
    // for(int i = 0; i < 10; i ++){
    //     //cout << chosen_r[i] << endl;
    //     cout << delta[i] << " " << chosen_w[i] << endl;
    // }
    // #endif


    // //step using adam
    
    // float alpha = 0.001;
    // float beta_1 = 0.9;
    // float beta_2 = 0.999;
    // float epsilon = 1e-8;
    

    // float* new_w = new float[(VOCAB_SIZE+1)];
    // for(int i = 0; i < (VOCAB_SIZE+1); i++){
    //     new_w[i] = w[i];
    // }

    // for(int i = 0; i < (VOCAB_SIZE+1); i++){
    //     float m_t = 0;
    //     float v_t = 0;
    //     int t = 0;
    //     while(true) {
    //         t += 1;
    //         m_t = beta_1 * m_t + (1 - beta_1) * delta_w[i];	// updates the moving averages of the gradientgd
    //         v_t = beta_2 * v_t + (1 - beta_2) * (delta_w[i] * delta_w[i]);	 // updates the moving averages of the squared gradient
    //         float m_cap = m_t / (1-pow(beta_1, t));  // calculates the bias-corrected estimates
    //         float v_cap = v_t / (1-pow(beta_2, t)); // calculates the bias-corrected estimates
    //         float w_old = new_w[i];						
    //         new_w[i] = new_w[i] - (alpha * m_cap) / (sqrt(v_cap) + epsilon);  // updates the parameters
    //         // cout << i <<": " << abs(w_old - new_w[i]) << " "<< (alpha * m_cap) / (sqrt(v_cap) + epsilon) << endl;
    //         // if(abs(w_old - new_w[i]) < 1e-5){	// checks if it is converged or not
    //         //     break;
    //         // }
    //         break;
            
    //     }
    //     // cout << w[i] << " -> " << new_w[i] << endl;
    //     // cout <<"--------"<<endl;
    // }

    // cout << "new w:" << endl;
    // for(int i = 0; i < 10; i ++){
    //     cout << w[i] << " -> " << new_w[i] << endl;
    // }
    // cout << endl;

    float lr = 0.02;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-08;
    float weight_decay = 1e-06;
    

    float* new_w = new float[(VOCAB_SIZE+1)];
    float* grad  = new float[(VOCAB_SIZE+1)];
    for(int i = 0; i < (VOCAB_SIZE+1); i++){
        new_w[i] = w[i];
        grad[i] = delta_w[i];
    }

    float exp_avg[(VOCAB_SIZE+1)];
    float exp_avg_sq[(VOCAB_SIZE+1)];

    int step = 0;

    step += 1;

    double bias_correction1 = 1.0 - pow(beta1, step);
    double bias_correction2 = 1.0 - pow(beta2, step);

    // weight decay
    if(weight_decay != 0) {
        for(int i = 0; i < (VOCAB_SIZE+1); i++) {
            grad[i] += weight_decay * w[i];
        }
    }

    // decay the first and second moment rnuning average coefficient
    for(int i = 0; i < (VOCAB_SIZE+1); i++) {
        exp_avg[i] *= beta1;
        exp_avg[i] += (1 - beta1) * grad[i];
        exp_avg_sq[i] *= beta2;
        exp_avg_sq[i] += (1 - beta2) * grad[i] * grad[i];
    }

    double denom[(VOCAB_SIZE+1)];
    for(int i = 0; i < (VOCAB_SIZE+1); i++) {
        denom[i] = (sqrt(exp_avg_sq[i]) / sqrt(bias_correction2)) + eps;
    }

    float step_size = lr / bias_correction1;

    for(int i = 0; i < (VOCAB_SIZE+1); i++) {
        new_w[i] += -step_size * exp_avg[i] / denom[i];
    }

    cout << "new w:" << endl;
    for(int i = 0; i < 10; i ++){
        cout << w[i] << " -> " << new_w[i] << endl;
    }
    cout << endl;










    
    

}
