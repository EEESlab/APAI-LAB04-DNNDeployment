/*
 * layer_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */

#include "mchan_test.h"
#include "pulp_nn_kernels.h"
#include "layerConv.h"

void layerConv_Tiling_Blocking(
  void *args
) 
{
  unsigned int *real_arg = (unsigned int *) args;
  unsigned int x =(unsigned int)  real_arg[0];
  unsigned int W =(unsigned int)  real_arg[1];
  unsigned int y =(unsigned int)  real_arg[2];
  unsigned int x_L1 =(unsigned int)  real_arg[3];
  unsigned int W_L1 =(unsigned int)  real_arg[4];
  unsigned int y_L1 =(unsigned int)  real_arg[5];
  unsigned int im2col =(unsigned int)  real_arg[6];
  ////////////////////////////////////////////////////////////////
  // Add Variable Declaration --> Tile dimensions, padding, DMA //
  ////////////////////////////////////////////////////////////////
  
  ////////////////////////////
  // First Copy --> Weights //
  ////////////////////////////

  // tile loop nest
  for(int h_tile = 0; h_tile < 2; h_tile++) 
  {
    for(int w_tile = 0; w_tile < 2; w_tile++) 
    {
      /////////////////////
      // Input Tile Copy //
      /////////////////////

      ////////////////////////
      // Padding Management //
      ////////////////////////

      pi_cl_team_barrier(0);
      pulp_nn_conv(
        x_L1,
        im2col,
        NULL,
        y_L1,
        W_L1,
        OUT_SHIFT,
        x_tile_w,
        x_tile_h,
        ch_in,
        y_tile_w,
        y_tile_h,
        ch_out,
        DIM_KERNEL_X,
        DIM_KERNEL_Y,
        p_t,
        p_b,
        p_l,
        p_r,
        STRIDE_X,
        STRIDE_Y);
      pi_cl_team_barrier(0);

      //////////////////////
      // Output Tile Copy //
      //////////////////////
    }
  }
  mchan_free(dma_evt);
}
