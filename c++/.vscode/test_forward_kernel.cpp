#include <iostream>
#include <cmath>
#include <stdio.h>
#include "util.hpp"

using namespace std;

uint32_t R_DEPTH = 2;
uint32_t VOCAB_SIZE = 200000;
uint32_t BATCH_SIZE = 100;
uint32_t R_DEPTH = 2;
uint32_t R_WIDTH = 1000;
uint32_t R_HEIGHT = BATCH_SIZE;

int main() {
    /*
     * w_data: weight
     * r: from forward.hpp (fixed)
     * feat_idx: the required entry indices in w and r
     * x: forward output result
     */
    uint32_t my_global_id = GPE_ID() + GPE_TILE_ID() * NUM_GPES_PER_TILE;

    // Push dimenstions & pointers
    float W_ADJ = (float) GPEQ_POP();
    float R_ADJ = (float) GPEQ_POP();
    uint32_t * feat_idx = (uint32_t *) GPEQ_POP();
    float * w_data = (float *) GPEQ_POP();
    // Receive the output array x
    float* x = (float *) GPEQ_POP();

    #ifdef DEBUG
        GPE_PRINTF("W_ADJ: %0.4f\n", W_ADJ);
        GPE_PRINTF("R_ADJ: %0.4f\n", R_ADJ);
    #endif

    // Extract entries from w according to feat_idx ( w = self.w(feat_idx) + self.w_adj )
    float *chosen_w = new float[BATCH_SIZE * R_WIDTH];
    for(uint32_t i = 0; i < BATCH_SIZE; ++i){
        for(uint32_t j = 0; j < R_WIDTH; ++j){
            chosen_w[i * R_WIDTH + j] = w[feat_idx[i * R_WIDTH + j]] + W_ADJ;
        }
    }

    // Extract entries from r according to feat_idx ( r = self.r(feat_idx) )
    float *chosen_r = new float[BATCH_SIZE * R_WIDTH * R_DEPTH];
    for(uint32_t k = 0; k < R_DEPTH; ++k){
        for(uint32_t i = 0; i < BATCH_SIZE; i++){
            for(uint32_t j = 0; j < R_WIDTH; ++j){
                chosen_r[k * BATCH_SIZE * R_WIDTH + i * R_WIDTH + j] = r[2 * feat_idx[i * R_WIDTH + j] + k];
            }
        }
    }

    // Inner Product
    //float *x = new float[BATCH_SIZE * R_DEPTH] ();
    for(uint32_t k = 0; k < R_DEPTH; ++k){
        for(uint32_t i = 0; i < BATCH_SIZE; ++i){
            for(uint32_t j = 0; j < R_WIDTH; ++j){
                x[i * 2 + k] += chosen_w[i * R_WIDTH + j] * chosen_r[k * BATCH_SIZE * R_WIDTH + i * R_WIDTH + j];
            }
            x[i * 2 + k] /= R_ADJ;
        }
    }



}