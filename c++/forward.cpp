#include "forward.hpp"
#include "cross-entropy.h"
using namespace std;

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
// r[2 * vocab_size + 2,1] from .hpp
// w from init [vocab_size+1, 1]

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

    // Test
    cout << "printing x" << endl;
    for(int i = 0; i < 10; i ++){
        cout << x[i] << endl;
    }

    // Calculate Loss (Cross-entropy)
    float loss = CrossEntropyLoss(x, target, BATCH_SIZE, 2, true);
    cout << "loss: " << loss << endl;

    //backward
    float* delta = new float[BATCH_SIZE * R_WIDTH];
    for(uint32_t i = 0; i < BATCH_SIZE; ++i){
        float numerator = target[i] ? exp(x[i * R_DEPTH]) : exp(x[i * R_DEPTH + 1]);
        for(uint32_t j = 0; j < R_WIDTH; ++j){
            //cout << i << " " << j <<" " << chosen_r[i * R_WIDTH + j] << " "<<  x[i * R_DEPTH + 1] <<  " "<< x[i*R_DEPTH] <<endl;
            delta[i * R_WIDTH + j] = (chosen_r[i * R_WIDTH + j] / R_ADJ) * numerator / ( (exp(x[i*R_DEPTH + 1]) + exp(x[i*R_DEPTH])) );
        }
        //cout<<(exp(x[i*R_DEPTH + 1]) + exp(x[i*R_DEPTH]))<<" "<< numerator << " " << exp(x[i*R_DEPTH]) << " " << x[i*R_DEPTH +1] << endl;
    }
    
    for(int i = 0; i < 10; i ++){
        //cout << chosen_r[i] << endl;
        cout << delta[i] << " " << chosen_w[i] << endl;
    }

    //step

    
    

}
