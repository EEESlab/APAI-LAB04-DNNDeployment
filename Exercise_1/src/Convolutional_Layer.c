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

#include "pulp_nn_kernels.h"
#include "layerConv.h"

void layerConv(
  void *args
) {
  unsigned int *real_arg = (unsigned int *) args;
  unsigned int x =(unsigned int)  real_arg[0];
  unsigned int W =(unsigned int)  real_arg[1];
  unsigned int y =(unsigned int)  real_arg[2];
  unsigned int im2col =(unsigned int)  real_arg[3];

  pulp_nn_conv(
    x,
    im2col,
    NULL,
    y,
    W,
    OUT_SHIFT,
    DIM_IM_IN_X,
    DIM_IM_IN_Y,
    CH_IM_IN,
    DIM_IM_OUT_X,
    DIM_IM_OUT_Y,
    CH_IM_OUT,
    DIM_KERNEL_X,
    DIM_KERNEL_Y,
    PADDING_Y_TOP,
    PADDING_Y_BOTTOM,
    PADDING_X_LEFT,
    PADDING_X_RIGHT,
    STRIDE_X,
    STRIDE_Y);
  pi_cl_team_barrier(0);
}
