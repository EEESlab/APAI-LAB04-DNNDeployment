/*
 * test_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
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

#include "main.h"
#include "data_allocation.h"

#define DIM_IM_IN_X 16
#define DIM_IM_IN_Y 16
#define CH_IM_IN 32
#define DIM_IM_OUT_X 16
#define DIM_IM_OUT_Y 16
#define CH_IM_OUT 32
#define DIM_KERNEL_X 3
#define DIM_KERNEL_Y 3

// on cluster function execution
void cluster_main(void *arg) 
{
  int *real_arg = (int *) arg;
  pi_perf_conf(1<<PI_PERF_CYCLES); 
  pi_perf_reset();            
  pi_perf_stop();           
  pi_perf_start(); 
  layerConv_Tiling_Blocking(real_arg);

  // layerConv(real_arg);
  if (pi_core_id()==0)
  {
    int num_cycles = pi_perf_read(PI_PERF_CYCLES);
    printf("num_cycles: %d\n", num_cycles);
    int MACs = DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT * DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN;
    printf("MACs: %d\n",MACs);
    printf("MACs/cycle: %f\n", (float)MACs/num_cycles );
  }
}

// parallelization of the function given the number of cores
void pulp_parallel(void *arg)
{
  pi_cl_team_fork(NUM_CORES, (void *)cluster_main, arg);
}

// on fabric controller
int main () {

  int args[7];
  args[0] = (unsigned int) IN_INT8_L2;
  args[1] = (unsigned int) WEIGHT_INT8_L2;
  args[2] = (unsigned int) OUT_L2;
  args[3] = (unsigned int) IN_INT8_L1;
  args[4] = (unsigned int) WEIGHT_INT8_L1;
  args[5] = (unsigned int) OUT_L1;
  args[6] = (unsigned int) IM2COL_L1;
 
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_FC, 100000000);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, 100000000);
  pi_time_wait_us(10000);

  struct pi_device cluster_dev = {0};
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task = {0};
  // task parameters allocation
  pi_cluster_task(&cluster_task, pulp_parallel, args);
  cluster_task.stack_size = 4096;
  cluster_task.slave_stack_size = 3072;
  // First open the cluster
  pi_cluster_conf_init(&conf);
  conf.id=0;
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return -1;

  // Then offload an entry point, this will get executed on the cluster controller
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  // closing of the cluster
  pi_cluster_close(&cluster_dev);
  ///// CHECK ////////
  int errors = 0;
  uint8_t * ACTIVATION_CHECK = OUT_L2;
  for (int i = 0 ; i < DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT; i++)
  {
    if (ACTIVATION_CHECK[i] != OUT_INT8_L2[i])
    {
      printf("Error in activation %d: %d instead of %d\n", i, ACTIVATION_CHECK[i], OUT_INT8_L2[i]);
      errors+=1;
    }
  }
  if (errors == 0)
  {
    printf("No Errors. Convolution is OK\n");
  }
  else
  {
    printf("There are %d Errors.\n", errors);
  }
}
