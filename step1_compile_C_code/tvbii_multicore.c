// THE VIRTUAL BRAIN -- C
// A fast implementation of TVB-style brain network models based on
// DYNAMIC MEAN FIELD MODEL Deco et al. 2014 Journal of Neuroscience
//
//  m.schirner@fu-berlin.de
//  michael.schirner@charite.de
//
// MIT LICENSE
// Copyright 2020 Michael Schirner
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <stdio.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
//#include <semaphore.h>


/*
 
 Definition of pthread barrier -- since OSX doesn't seem to have it
 
 */


 
 typedef int pthread_barrierattr_t;
 typedef struct
 {
 pthread_mutex_t mutex;
 pthread_cond_t cond;
 int count;
 int tripCount;
 } pthread_barrier_t;
 
 
 int pthread_barrier_init(pthread_barrier_t *barrier, const pthread_barrierattr_t *attr, unsigned int count)
 {
 if(count == 0)
 {
 //errno = EINVAL;
 return -1;
 }
 if(pthread_mutex_init(&barrier->mutex, 0) < 0)
 {
 return -1;
 }
 if(pthread_cond_init(&barrier->cond, 0) < 0)
 {
 pthread_mutex_destroy(&barrier->mutex);
 return -1;
 }
 barrier->tripCount = count;
 barrier->count = 0;
 
 return 0;
 }
 
 int pthread_barrier_destroy(pthread_barrier_t *barrier)
 {
 pthread_cond_destroy(&barrier->cond);
 pthread_mutex_destroy(&barrier->mutex);
 return 0;
 }
 
 int pthread_barrier_wait(pthread_barrier_t *barrier)
 {
 pthread_mutex_lock(&barrier->mutex);
 ++(barrier->count);
 if(barrier->count >= barrier->tripCount)
 {
 barrier->count = 0;
 pthread_cond_broadcast(&barrier->cond);
 pthread_mutex_unlock(&barrier->mutex);
 return 1;
 }
 else
 {
 pthread_cond_wait(&barrier->cond, &(barrier->mutex));
 pthread_mutex_unlock(&barrier->mutex);
 return 0;
 }
 }



/*
 
 End: Definition of pthread barrier
 
 */

pthread_mutex_t mutex_thrcount;

struct Xi_p{
    float **Xi_elems;
};

struct SC_capS{
    float *cap;
};

struct SC_inpregS{
    int *inpreg;
};


pthread_barrier_t mybarrier_base, mybarrier1, mybarrier2, mybarrier3;

#define REAL float
//#define REAL double


int divRoundClosest(const int n, const int d)
{
    return ((n < 0) ^ (d < 0)) ? ((n - d/2)/d) : ((n + d/2)/d);
}

int divRoundUp(const int x, const int y)
{
    return (x + y - 1) / y;
}


/* Compute Pearson's correlation coefficient */
float corr(float *x, float *y, int n){
    int i;
    float mx=0, my=0;
    
    /* Calculate the mean of the two series x[], y[] */
    for (i=0; i<n; i++) {
        mx += x[i];
        my += y[i];
    }
    mx /= n;
    my /= n;
    
    /* Calculate the correlation */
    float sxy = 0, sxsq = 0, sysq = 0, tmpx, tmpy;
    for (i=0; i<n; i++) {
        tmpx = x[i] - mx;
        tmpy = y[i] - my;
        sxy += tmpx*tmpy;
        sxsq += tmpx*tmpx;
        sysq += tmpy*tmpy;
    }
    
    return (sxy / (sqrt(sxsq)*sqrt(sysq)));
}


//float gaussrand_ret()
//{
//    static double V1=0.0, V2=0.0, S=0.0, U1=0.0, U2=0.0;
//    S=0.0;
//    do {
//        U1 = (double)rand() / RAND_MAX;
//        U2 = (double)rand() / RAND_MAX;
//        V1 = 2 * U1 - 1;
//        V2 = 2 * U2 - 1;
//        S = V1 * V1 + V2 * V2;
//    } while(S >= 1 || S == 0);
//
//    return (float)(V1 * sqrt(-2 * log(S) / S));
//}
//
//static inline void gaussrand(float *randnum)
//{
//    static double V1=0.0, V2=0.0, S=0.0, U1=0.0, U2=0.0;
//
//    S=0.0;
//    do {
//        U1 = (double)rand() / RAND_MAX;
//        U2 = (double)rand() / RAND_MAX;
//        V1 = 2 * U1 - 1;
//        V2 = 2 * U2 - 1;
//        S = V1 * V1 + V2 * V2;
//    } while(S >= 1 || S == 0);
//
//    randnum[0] = (float)(V1 * sqrt(-2 * log(S) / S));
//    randnum[1] = (float)(V2 * sqrt(-2 * log(S) / S));
//
//    S=0.0;
//    do {
//        U1 = (double)rand() / RAND_MAX;
//        U2 = (double)rand() / RAND_MAX;
//        V1 = 2 * U1 - 1;
//        V2 = 2 * U2 - 1;
//        S = V1 * V1 + V2 * V2;
//    } while(S >= 1 || S == 0);
//
//    randnum[2] = (float)(V1 * sqrt(-2 * log(S) / S));
//    randnum[3] = (float)(V2 * sqrt(-2 * log(S) / S));
//}




int importGlobalConnectivity(char *SC_cap_filename, char *SC_dist_filename, char *SC_inputreg_filename, int regions, float **region_activity, struct Xi_p **reg_globinp_p, float global_trans_v, int **n_conn_table, float G_J_NMDA, struct SC_capS **SC_cap, float **SC_rowsums, struct SC_inpregS **SC_inpreg)
{
    
    int i,j,k, maxdelay=0, tmpint;
    float *region_activity_p;
    double tmp, tmp2;
    struct Xi_p *reg_globinp_pp;
    struct SC_capS      *SC_capp;
    struct SC_inpregS   *SC_inpregp;
    
    
    
    // Open SC files
    FILE *file_cap, *file_dist;
    file_cap=fopen(SC_cap_filename, "r");
    file_dist=fopen(SC_dist_filename, "r");
    if (file_cap==NULL || file_dist==NULL)
    {
        printf( "\nERROR: Could not open SC files. Terminating... \n\n");
        exit(0);
    }
    
    
    // Allocate a counter that stores number of region input connections for each region and the SCcap array
    *SC_rowsums = (float *)_mm_malloc(regions*sizeof(float),16);
    *n_conn_table = (int *)_mm_malloc(regions*sizeof(int),16);
    *SC_cap = (struct SC_capS *)_mm_malloc(regions*sizeof(struct SC_capS),16);
    SC_capp = *SC_cap;
    *SC_inpreg = (struct SC_inpregS *)_mm_malloc(regions*sizeof(struct SC_inpregS),16);
    SC_inpregp = *SC_inpreg;
    if(*n_conn_table==NULL || SC_capp==NULL || SC_rowsums==NULL || SC_inpregp==NULL){
        printf("Running out of memory. Terminating.\n");fclose(file_dist);fclose(file_cap);exit(2);
    }
    
    // Read out the maximal fiber length and the degree of each node and rewind SC files
    int n_entries = 0, curr_col = 0, curr_row = 0, curr_row_nonzero = 0;
    double tmp_max = -9999;
    while(fscanf(file_dist,"%lf",&tmp) != EOF && fscanf(file_cap,"%lf",&tmp2) != EOF){
        if (tmp_max < tmp) tmp_max = tmp;
        n_entries++;
        
        if (tmp2 > 0.0) curr_row_nonzero++;
        
        curr_col++;
        if (curr_col == regions) {
            curr_col = 0;
            (*n_conn_table)[curr_row] = curr_row_nonzero;
            curr_row_nonzero = 0;
            curr_row++;
        }
    }
    
    if (n_entries < regions * regions) {
        printf( "ERROR: Unexpected end-of-file in file. File contains less input than expected. Terminating... \n");
        exit(0);
    }
    
    rewind(file_dist);
    rewind(file_cap);
    
    maxdelay = (int)(((tmp_max/global_trans_v)*10)+0.5); // *10 for getting from m/s to 10kHz sampling, +0.5 for rounding by casting
    if (maxdelay < 1) maxdelay = 1; // Case: no time delays
    
    
    
    
    // Allocate ringbuffer that contains region activity for each past time-step until maxdelay and another ringbuffer that contains pointers to the first ringbuffer
    *region_activity = (float *)_mm_malloc(maxdelay*regions*sizeof(float),16);
    region_activity_p = *region_activity;
    *reg_globinp_p = (struct Xi_p *)_mm_malloc(maxdelay*regions*sizeof(struct Xi_p),16);
    reg_globinp_pp = *reg_globinp_p;
    if(region_activity_p==NULL || reg_globinp_p==NULL){
        printf("Running out of memory. Terminating.\n");fclose(file_dist);exit(2);
    }
    for (j=0; j<maxdelay*regions; j++) {
        region_activity_p[j]=0.001;
    }
    
    // Read SC files and set pointers for each input region and correspoding delay for each ringbuffer time-step
    int ring_buff_position;
    for (i=0; i<regions; i++) {
        
        if ((*n_conn_table)[i] > 0) {
            // SC strength and inp region numbers
            SC_capp[i].cap          = (float *)_mm_malloc(((*n_conn_table)[i])*sizeof(float),16);
            SC_inpregp[i].inpreg    = (int *)_mm_malloc(((*n_conn_table)[i])*sizeof(int),16);
            if(SC_capp[i].cap==NULL || SC_inpregp[i].inpreg==NULL){
                printf("Running out of memory. Terminating.\n");exit(2);
            }
            
            // Allocate memory for input-region-pointer arrays for each time-step in ringbuffer
            for (j=0; j<maxdelay; j++){
                reg_globinp_pp[i+j*regions].Xi_elems=(float **)_mm_malloc(((*n_conn_table)[i])*sizeof(float *),16);
                if(reg_globinp_pp[i+j*regions].Xi_elems==NULL){
                    printf("Running out of memory. Terminating.\n");exit(2);
                }
            }
            
            float sum_caps=0.0;
            // Read incoming connections and set pointers
            curr_row_nonzero = 0;
            for (j=0; j<regions; j++) {
                if(fscanf(file_cap,"%lf",&tmp) != EOF && fscanf(file_dist,"%lf",&tmp2) != EOF){
                    if (tmp > 0.0) {
                        tmpint = (int)(((tmp2/global_trans_v)*10)+0.5); //  *10 for getting from m/s or mm/ms to 10kHz sampling, +0.5 for rounding by casting
                        if (tmpint < 0 || tmpint > maxdelay){
                            printf( "\nERROR: Negative or too high (larger than maximum specified number) connection length/delay %d -> %d. Terminating... \n\n",i,j);exit(0);
                        }
                        if (tmpint <= 0) tmpint = 1; // If time delay is smaller than integration step size, than set time delay to one integration step
                        
                        SC_capp[i].cap[curr_row_nonzero] = (float)tmp * G_J_NMDA;
                        //sum_caps                += (float)tmp;
                        sum_caps                += SC_capp[i].cap[curr_row_nonzero];
                        SC_inpregp[i].inpreg[curr_row_nonzero]  =  j;
                        
                        ring_buff_position=maxdelay*regions - tmpint*regions + j;
                        for (k=0; k<maxdelay; k++) {
                            reg_globinp_pp[i+k*regions].Xi_elems[curr_row_nonzero]=&region_activity_p[ring_buff_position];
                            ring_buff_position += regions;
                            if (ring_buff_position > (maxdelay*regions-1)) ring_buff_position -= maxdelay*regions;
                        }
                        
                        curr_row_nonzero++;
                    }
                } else{
                    printf( "\nERROR: Unexpected end-of-file in file %s or %s. File contains less input than expected. Terminating... \n\n", SC_cap_filename, SC_dist_filename);
                    exit(0);
                }
                
            }
            if (sum_caps <= 0) {
                printf( "\nERROR: Sum of connection strenghts is negative or zero. sum-caps node %d = %f. Terminating... \n\n",i,sum_caps);exit(0);
            }
            (*SC_rowsums)[i] = sum_caps;
        }
    }
    
    fclose(file_dist);fclose(file_cap);
    return maxdelay;
}


/*
 Initialize thread barriers
 */
void initialize_thread_barriers(int n_threads) {
    pthread_barrier_init(&mybarrier1, NULL, n_threads);
    pthread_barrier_init(&mybarrier2, NULL, n_threads);
    pthread_barrier_init(&mybarrier3, NULL, n_threads);
    return;
}


/* create thread argument struct for thr_func() */
typedef struct _thread_data_t {
    int                 tid, rand_num_seed;
    int                 nodes, nodes_vec, fake_nodes, n_threads, vectorization_grade, time_steps, BOLD_TR, BOLD_ts_len;
    float               *J_i;
    int                 reg_act_size;
    float               *region_activity;
    float               model_dt;
    __m128              _gamma;
    __m128              _one;
    __m128              _imintau_E;
    __m128              _dt;
    __m128              _sigma_sqrt_dt;
    __m128              _sigma;
    __m128              _gamma_I;
    __m128              _imintau_I;
    __m128              _min_d_I;
    __m128              _b_I;
    __m128              _J_NMDA;
    __m128              _w_I__I_0;
    __m128              _a_I;
    __m128              _min_d_E;
    __m128              _b_E;
    __m128              _a_E;
    __m128              _w_plus_J_NMDA;
    __m128              _w_E__I_0;
    struct SC_capS      *SC_cap;
    struct SC_inpregS   *SC_inpreg;
    struct Xi_p         *reg_globinp_p;
    int                 *n_conn_table;
    float               *BOLD_ex;    
    char                *output_file;
} thread_data_t;


/*
 
 Function for multi-threaded FIC tuning
 
 */
void *run_simulation(void *arg)
{
    
    /* Local function parameters (no concurrency) */
    int j, i_node_vec, i_node_vec_local, k, int_i, ts;
    float tmpglobinput, tmpglobinput_FFI;
    __m128 _tmp_H_E, _tmp_H_I, _tmp_I_I, _tmp_I_E;
    float tmp_exp_E[4]          __attribute__((aligned(16)));
    float tmp_exp_I[4]          __attribute__((aligned(16)));
    __m128          *_tmp_exp_E         = (__m128*)tmp_exp_E;
    __m128          *_tmp_exp_I         = (__m128*)tmp_exp_I;
    float rand_number[4]        __attribute__((aligned(16)));
    __m128          *_rand_number       = (__m128*)rand_number;
    int     ring_buf_pos                = 0;


    
    
    /* Global function parameters (some with concurrency) */
    thread_data_t *thr_data = (thread_data_t *)arg;
    //printf("Hello from thread %d, process %d\n", thr_data->tid, thr_data->rank);
    int     t_id                    = thr_data->tid;
    int     rand_num_seed           = thr_data->rand_num_seed;
    int     nodes                   = thr_data->nodes;
    int     fake_nodes              = thr_data->fake_nodes;
    const   int nodes_vec           = thr_data->nodes_vec;
    int     n_threads               = thr_data->n_threads;
    int     vectorization_grade     = thr_data->vectorization_grade;
    int     reg_act_size            = thr_data->reg_act_size;
    int     *n_conn_table           = thr_data->n_conn_table;
    float   *J_i                    = thr_data->J_i;
    float   *region_activity        = thr_data->region_activity;
    float   *BOLD_ex                = thr_data->BOLD_ex;
    const   __m128 _gamma           = thr_data->_gamma;
    const   __m128 _one             = thr_data->_one;
    const   __m128 _imintau_E       = thr_data->_imintau_E;
    const   __m128 _dt              = thr_data->_dt;
    const   __m128 _sigma_sqrt_dt   = thr_data->_sigma_sqrt_dt;
    const   __m128 _gamma_I         = thr_data->_gamma_I;
    const   __m128 _imintau_I       = thr_data->_imintau_I;
    const   __m128 _min_d_I         = thr_data->_min_d_I;
    const   __m128 _b_I             = thr_data->_b_I;
    const   __m128 _J_NMDA          = thr_data->_J_NMDA;
    const   __m128 _w_I__I_0        = thr_data->_w_I__I_0;
    const   __m128 _a_I             = thr_data->_a_I;
    const   __m128 _min_d_E         = thr_data->_min_d_E;
    const   __m128 _b_E             = thr_data->_b_E;
    const   __m128 _a_E             = thr_data->_a_E;
    const   __m128 _w_plus_J_NMDA   = thr_data->_w_plus_J_NMDA;
    const   __m128 _w_E__I_0        = thr_data->_w_E__I_0;
    struct  SC_capS *SC_cap         = thr_data->SC_cap;
    struct  Xi_p *reg_globinp_p     = thr_data->reg_globinp_p;
    int     time_steps              = thr_data->time_steps;
    int     BOLD_TR                 = thr_data->BOLD_TR;
    float   model_dt                = thr_data->model_dt;
    int     BOLD_ts_len             = thr_data->BOLD_ts_len;

    
    /* Parallelism: divide work among threads */
    int nodes_vec_mt        = divRoundUp(nodes_vec, n_threads);
    int nodes_mt            = nodes_vec_mt  * vectorization_grade;
    
    int start_nodes_vec_mt  = t_id         * nodes_vec_mt;
    int start_nodes_mt      = t_id         * nodes_mt;
    int end_nodes_vec_mt    = (t_id + 1)   * nodes_vec_mt;
    int end_nodes_mt        = (t_id + 1)   * nodes_mt;
    int end_nodes_mt_glob   = end_nodes_mt; // end_nodes_mt_glob may differ from end_nodes_mt in the last thread (fake-nodes vs nodes)
    
    /* Correct settings for last thread */
    if (end_nodes_mt > nodes) {
        end_nodes_vec_mt    = nodes_vec;
        end_nodes_mt        = fake_nodes;
        end_nodes_mt_glob   = nodes;
        nodes_mt            = end_nodes_mt - start_nodes_mt;
        nodes_vec_mt        = end_nodes_vec_mt - start_nodes_vec_mt;
    }
    
    printf("thread %d: start: %d end: %d size: %d\n", thr_data->tid, start_nodes_mt, end_nodes_mt, nodes_mt);
    
    
    /* Check whether splitting makes sense (i.e. each thread has something to do). Terminate if not. */
    if (nodes_vec_mt <= 1) {
        printf("The splitting is ineffective (e.g. fewer nodes than threads or some threads have nothing to do). Use different numbers of threads. Terminating.\n");
        exit(0);
    }
    
    
   
    
    /* Set up barriers */
    if (t_id == 0) {
        initialize_thread_barriers(n_threads);
    }
    pthread_barrier_wait(&mybarrier_base);

    
    
    
    /* Initialize random number generator */
    //const gsl_rng_type * T;
    gsl_rng *r = gsl_rng_alloc (gsl_rng_mt19937);
    gsl_rng_set (r, (t_id+rand_num_seed)); // Random number seed -- otherwise every node gets the same random nums
    srand((unsigned)(t_id+rand_num_seed));
    
    
    
    
    // All threads
    float   *meanFR                 = (float *)_mm_malloc(nodes_mt * sizeof(float),16);  // summation array for mean firing rate over all threads
    float   *meanFR_INH             = (float *)_mm_malloc(nodes_mt * sizeof(float),16);  // summation array for mean firing rate over all threads
    float   *global_input           = (float *)_mm_malloc(nodes_mt * sizeof(float),16);
    float   *global_input_FFI       = (float *)_mm_malloc(nodes_mt * sizeof(float),16);
    float   *S_i_E                  = (float *)_mm_malloc(nodes_mt * sizeof(float),16);
    float   *S_i_I                  = (float *)_mm_malloc(nodes_mt * sizeof(float),16);
    //float   *r_i_E                    = (float *)_mm_malloc(nodes_mt * sizeof(float),16);
    //float   *r_i_I                    = (float *)_mm_malloc(nodes_mt * sizeof(float),16);
    float   *J_i_local              = (float *)_mm_malloc(nodes_mt * sizeof(float),16);
    
    if (meanFR == NULL || meanFR_INH == NULL || global_input == NULL || global_input_FFI == NULL || S_i_E==NULL || S_i_I==NULL || J_i_local==NULL) {
        printf( "ERROR: Running out of memory. Aborting... \n");
        exit(0);
    }
    
    // Cast to vector/SSE arrays
    __m128          *_meanFR         = (__m128*)meanFR;
    __m128          *_meanFR_INH     = (__m128*)meanFR_INH;
    __m128          *_global_input   = (__m128*)global_input;
    __m128          *_global_input_FFI = (__m128*)global_input_FFI;
    __m128          *_S_i_E          = (__m128*)S_i_E;
    __m128          *_S_i_I          = (__m128*)S_i_I;
    //__m128          *_r_i_E             = (__m128*)r_i_E;
    //__m128          *_r_i_I             = (__m128*)r_i_I;
    __m128          *_J_i_local      = (__m128*)J_i_local;
    
    
    //Balloon-Windkessel model parameters / arrays
    float rho = 0.34, alpha = 0.32, tau = 0.98, y = 1.0/0.41, kappa = 1.0/0.65;
    float V_0 = 0.02, k1 = 7 * rho, k2 = 2.0, k3 = 2 * rho - 0.2, ialpha = 1.0/alpha, itau = 1.0/tau, oneminrho = (1.0 - rho);
    float f_tmp;
    float *BOLD       = (float *)_mm_malloc(nodes_mt * BOLD_ts_len * sizeof(float),16);  // resulting BOLD data
    float *bw_x_ex    = (float *)_mm_malloc(nodes_mt * sizeof(float),16);  // State-variable 1 of BW-model (exc. pop.)
    float *bw_f_ex    = (float *)_mm_malloc(nodes_mt * sizeof(float),16);  // State-variable 2 of BW-model (exc. pop.)
    float *bw_nu_ex   = (float *)_mm_malloc(nodes_mt * sizeof(float),16);  // State-variable 3 of BW-model (exc. pop.)
    float *bw_q_ex    = (float *)_mm_malloc(nodes_mt * sizeof(float),16);  // State-variable 4 of BW-model (exc. pop.)
    //float *bw_x_in    = (float *)_mm_malloc(nodes_mt * sizeof(float),16);  // State-variable 1 of BW-model (inh. pop.)
    //float *bw_f_in    = (float *)_mm_malloc(nodes_mt * sizeof(float),16);  // State-variable 2 of BW-model (inh. pop.)
    //float *bw_nu_in   = (float *)_mm_malloc(nodes_mt * sizeof(float),16);  // State-variable 3 of BW-model (inh. pop.)
    //float *bw_q_in    = (float *)_mm_malloc(nodes_mt * sizeof(float),16);  // State-variable 4 of BW-model (inh. pop.)
    //if (bw_x_ex == NULL || bw_f_ex == NULL || bw_nu_ex==NULL || bw_q_ex==NULL || bw_x_in==NULL || bw_f_in==NULL || bw_nu_in==NULL || bw_q_in==NULL) {}
    if (bw_x_ex == NULL || bw_f_ex == NULL || bw_nu_ex==NULL || bw_q_ex==NULL) {
        printf( "ERROR: Running out of memory. Aborting... \n");
        exit(0);
    }
    
    
    

    
    // Reset arrays and variables
    ring_buf_pos     = 0;
    for (j = 0; j < nodes_mt; j++) {
        meanFR[j]           = 0.0;
        meanFR_INH[j]       = 0.0;
        global_input[j]     = 0.00;
        global_input_FFI[j] = 0.00;
        S_i_E[j]            = 0.00;
        S_i_I[j]            = 0.00;
        J_i_local[j]        = J_i[j+start_nodes_mt];
    }
    if (t_id == 0) {
        for (j=0; j<reg_act_size; j++) {
            region_activity[j]=0.00;
        }
    }
    
    // Reset Balloon-Windkessel model parameters and arrays
    for (j = 0; j < nodes_mt; j++) {
        bw_x_ex[j] = 0.0;
        bw_f_ex[j] = 1.0;
        bw_nu_ex[j] = 1.0;
        bw_q_ex[j] = 1.0;
        //bw_x_in[j] = 0.0;
        //bw_f_in[j] = 1.0;
        //bw_nu_in[j] = 1.0;
        //bw_q_in[j] = 1.0;
    }
    
    /* Barrier: only start simulating when arrays were cleared */
    pthread_barrier_wait(&mybarrier1);
    
    
    
    /*  Simulation loop */
    int ts_bold_i = 0;
    for (ts = 0; ts < time_steps; ts++) {
        
        if (t_id == 0){
            printf("%.1f %% \r", ((float)ts / (float)time_steps) * 100.0f );
        }
        
        // dt = 0.1 ms => 10 integration iterations per ms
        for (int_i = 0; int_i < 10; int_i++) {
            /* Barrier: only start integrating input when all other threads are finished copying their results to buffer */
            pthread_barrier_wait(&mybarrier2);
            
            /* Compute global coupling */
            i_node_vec_local = 0;
            for(j=start_nodes_mt; j<end_nodes_mt_glob; j++){
                tmpglobinput     = 0;
                tmpglobinput_FFI = 0;
                for (k=0; k<n_conn_table[j]; k++) {
                    tmpglobinput     += *reg_globinp_p[j+ring_buf_pos].Xi_elems[k] * SC_cap[j].cap[k];
                    //tmpglobinput_FFI += *reg_globinp_p[j+ring_buf_pos].Xi_elems[k] * SC_cap_FFI[j].cap[k];
                }
                global_input[i_node_vec_local]     = tmpglobinput;
                //global_input_FFI[i_node_vec_local] = tmpglobinput_FFI;
                
                i_node_vec_local++;
            }
            
            
            i_node_vec_local = 0;
            for (i_node_vec = start_nodes_vec_mt; i_node_vec < end_nodes_vec_mt; i_node_vec++) {
                
                // Excitatory population firing rate
                _tmp_I_E    = _mm_sub_ps(_mm_mul_ps(_a_E,_mm_add_ps(_mm_add_ps(_w_E__I_0,_mm_mul_ps(_w_plus_J_NMDA, _S_i_E[i_node_vec_local])),_mm_sub_ps(_global_input[i_node_vec_local],_mm_mul_ps(_J_i_local[i_node_vec_local], _S_i_I[i_node_vec_local])))),_b_E);
                
                *_tmp_exp_E     = _mm_mul_ps(_min_d_E, _tmp_I_E);
                tmp_exp_E[0]    = tmp_exp_E[0] != 0 ? expf(tmp_exp_E[0]) : 0.9;
                tmp_exp_E[1]    = tmp_exp_E[1] != 0 ? expf(tmp_exp_E[1]) : 0.9;
                tmp_exp_E[2]    = tmp_exp_E[2] != 0 ? expf(tmp_exp_E[2]) : 0.9;
                tmp_exp_E[3]    = tmp_exp_E[3] != 0 ? expf(tmp_exp_E[3]) : 0.9;
                _tmp_H_E        = _mm_div_ps(_tmp_I_E, _mm_sub_ps(_one, *_tmp_exp_E));
                _meanFR[i_node_vec_local] = _mm_add_ps(_meanFR[i_node_vec_local],_tmp_H_E);
                //_r_i_E[i_node_vec_local]  = _tmp_H_E;
                
                
                
                // Inhibitory population firing rate
                _tmp_I_I = _mm_sub_ps(_mm_mul_ps(_a_I,_mm_sub_ps(_mm_add_ps(_mm_add_ps(_w_I__I_0,_global_input_FFI[i_node_vec_local]),_mm_mul_ps(_J_NMDA, _S_i_E[i_node_vec_local])), _S_i_I[i_node_vec_local])),_b_I);
                *_tmp_exp_I   = _mm_mul_ps(_min_d_I, _tmp_I_I);
                tmp_exp_I[0]  = tmp_exp_I[0] != 0 ? expf(tmp_exp_I[0]) : 0.9;
                tmp_exp_I[1]  = tmp_exp_I[1] != 0 ? expf(tmp_exp_I[1]) : 0.9;
                tmp_exp_I[2]  = tmp_exp_I[2] != 0 ? expf(tmp_exp_I[2]) : 0.9;
                tmp_exp_I[3]  = tmp_exp_I[3] != 0 ? expf(tmp_exp_I[3]) : 0.9;
                _tmp_H_I  = _mm_div_ps(_tmp_I_I, _mm_sub_ps(_one, *_tmp_exp_I));
                _meanFR_INH[i_node_vec_local] = _mm_add_ps(_meanFR_INH[i_node_vec_local],_tmp_H_I);
                //_r_i_I[i_node_vec_local] = _tmp_H_I;
                
                
                
                //gaussrand(rand_number);
                rand_number[0] = (float)gsl_ran_gaussian(r, 1.0);
                rand_number[1] = (float)gsl_ran_gaussian(r, 1.0);
                rand_number[2] = (float)gsl_ran_gaussian(r, 1.0);
                rand_number[3] = (float)gsl_ran_gaussian(r, 1.0);
                
                
                _S_i_I[i_node_vec_local] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_sigma_sqrt_dt, *_rand_number),_S_i_I[i_node_vec_local]),_mm_mul_ps(_dt,_mm_add_ps(_mm_mul_ps(_imintau_I, _S_i_I[i_node_vec_local]),_mm_mul_ps(_tmp_H_I,_gamma_I))));
                
                
                //gaussrand(rand_number);
                rand_number[0] = (float)gsl_ran_gaussian(r, 1.0);
                rand_number[1] = (float)gsl_ran_gaussian(r, 1.0);
                rand_number[2] = (float)gsl_ran_gaussian(r, 1.0);
                rand_number[3] = (float)gsl_ran_gaussian(r, 1.0);
                
                
                _S_i_E[i_node_vec_local] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_sigma_sqrt_dt, *_rand_number),_S_i_E[i_node_vec_local]),_mm_mul_ps(_dt, _mm_add_ps(_mm_mul_ps(_imintau_E, _S_i_E[i_node_vec_local]),_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_one, _S_i_E[i_node_vec_local]),_gamma),_tmp_H_E))));
                
                i_node_vec_local++;
            }
            
            
            /* ensure that synaptic activity is within its boundaries */
            for(j=0; j<nodes_mt; j++){
                S_i_E[j] = S_i_E[j] >= 0.0 ? S_i_E[j] : 0.0;
                S_i_E[j] = S_i_E[j] <= 1.0 ? S_i_E[j] : 1.0;
                S_i_I[j] = S_i_I[j] >= 0.0 ? S_i_I[j] : 0.0;
                S_i_I[j] = S_i_I[j] <= 1.0 ? S_i_I[j] : 1.0;
            }
            
            /* Copy result of this thread into global region_activity buffer */
            memcpy(&region_activity[ring_buf_pos+start_nodes_mt], S_i_E, nodes_mt*sizeof( float ));
            
            
            /* Shift region_activity ring-buffer start */
            ring_buf_pos = ring_buf_pos<(reg_act_size-nodes) ? (ring_buf_pos+nodes) : 0;
            
        }
        
        /*
         Compute BOLD for that time-step (subsampled to 1 ms)
         */
        for (j = 0; j < nodes_mt; j++) {
            bw_x_ex[j]  = bw_x_ex[j]  +  model_dt * (S_i_E[j] - kappa * bw_x_ex[j] - y * (bw_f_ex[j] - 1.0));
            f_tmp       = bw_f_ex[j]  +  model_dt * bw_x_ex[j];
            bw_nu_ex[j] = bw_nu_ex[j] +  model_dt * itau * (bw_f_ex[j] - powf(bw_nu_ex[j], ialpha));
            bw_q_ex[j]  = bw_q_ex[j]  +  model_dt * itau * (bw_f_ex[j] * (1.0 - powf(oneminrho,(1.0/bw_f_ex[j]))) / rho  - powf(bw_nu_ex[j],ialpha) * bw_q_ex[j] / bw_nu_ex[j]);
            bw_f_ex[j]  = f_tmp;
            
            /*
             bw_x_in[j]  = bw_x_in[j]  +  model_dt * (S_i_I[j] - kappa * bw_x_in[j] - y * (bw_f_in[j] - 1.0));
             f_tmp       = bw_f_in[j]  +  model_dt * bw_x_in[j];
             bw_nu_in[j] = bw_nu_in[j] +  model_dt * itau * (bw_f_in[j] - powf(bw_nu_in[j], ialpha));
             bw_q_in[j]  = bw_q_in[j]  +  model_dt * itau * (bw_f_in[j] * (1.0 - powf(oneminrho,(1.0/bw_f_in[j]))) / rho  - powf(bw_nu_in[j],ialpha) * bw_q_in[j] / bw_nu_in[j]);
             bw_f_in[j]  = f_tmp;
             */
        }
        
        
        if (ts % BOLD_TR == 0) {
            for (j = 0; j < nodes_mt; j++) {
                BOLD[ts_bold_i + j * BOLD_ts_len] = 100 / rho * V_0 * (k1 * (1 - bw_q_ex[j]) + k2 * (1 - bw_q_ex[j]/bw_nu_ex[j]) + k3 * (1 - bw_nu_ex[j]));
            }
            ts_bold_i++;
        }
    }

    /* Copy results of this thread into shared memory buffer */
    memcpy(&BOLD_ex[start_nodes_mt*BOLD_ts_len], BOLD, nodes_mt * BOLD_ts_len * sizeof( float ));

    /* Wait until other threads finished before writing out result */
    pthread_barrier_wait(&mybarrier3);
    

    

    /* Write out fMRI time series */
    if (t_id == 0){
        FILE *FCout = fopen(thr_data->output_file, "w");
        for (j = 0; j < nodes; j++) {
            for (k = 0; k < ts_bold_i; k++) {
                fprintf(FCout, "%.5f ",BOLD_ex[j*BOLD_ts_len + k]);
            }
            fprintf(FCout, "\n");
        }
        fclose(FCout);
    }
    
    

    gsl_rng_free (r);
    pthread_exit(NULL);
}



/*
 Usage: tvbii <paramfile> <subject_id>
 */

int main(int argc, char *argv[])
{
    /*
     Check whether right number of arguments was passed to program call
     */
    if (argc != 4 || atoi(argv[3]) <= 0) {
        printf("\nERROR: Invalid arguments.\n\nUsage: tvbii <paramset_file> <sub_id> <#threads>\n\nTerminating... \n\n");
        printf("\nProvided arguments:\n");
        int i;
        for (i=0; i<argc; i++) {
            printf("%s\n", argv[i]);
        }
        exit(0);
    }
    
    /*
     Get current time and do some initializations
     */
    time_t  start = time(NULL);
    int     i, j;
    int     n_threads = atoi(argv[3]);

    
    /*
     Global model and integration parameters
     */
    const float dt                  = 0.1;      // Integration step length dt = 0.1 ms
    const float sqrt_dt             = sqrtf(dt);// Noise in the diffusion part scaled by sqrt of dt
    const float model_dt            = 0.001;    // Time-step of model (sampling-rate=1000 Hz)
    const int   vectorization_grade = 4;        // How many operations can be done simultaneously. Depends on CPU Architecture and available intrinsics.
    int         time_steps          = 667*1.94*1000;    // Simulation length
    int         nodes               = 84;    // Number of surface vertices; must be a multiple of vectorization grade
    int         fake_nodes          = 84;
    float       global_trans_v      = 1.0;     // Global transmission velocity (m/s); Local time-delays can be ommited since smaller than integration time-step
    float       G                   = 0.5;        // Global coupling strength
    int         BOLD_TR             = 1940;     // TR of BOLD data
    int         rand_num_seed       = 1403;
    
    
    /*
     Local model: DMF-Parameters from Deco et al. JNeuro 2014
     */
    float w_plus  = 1.4;          // local excitatory recurrence
    //float I_ext   = 0;            // External stimulation
    float J_NMDA  = 0.15;         // (nA) excitatory synaptic coupling
    //float J_i     = 1.0;          // 1 for no-FIC, !=1 for Feedback Inhibition Control
    const float a_E     = 310;          // (n/C)
    const float b_E     = 125;          // (Hz)
    const float d_E     = 0.16;         // (s)
    const float a_I     = 615;          // (n/C)
    const float b_I     = 177;          // (Hz)
    const float d_I     = 0.087;        // (s)
    const float gamma   = 0.641/1000.0; // factor 1000 for expressing everything in ms
    const float tau_E   = 100;          // (ms) Time constant of NMDA (excitatory)
    const float tau_I   = 10;           // (ms) Time constant of GABA (inhibitory)
    float       sigma   = 0.00316228;   // (nA) Noise amplitude
    const float I_0     = 0.382;        // (nA) overall effective external input
    const float w_E     = 1.0;          // scaling of external input for excitatory pool
    const float w_I     = 0.7;          // scaling of external input for inhibitory pool
    const float gamma_I = 1.0/1000.0;   // for expressing inhib. pop. in ms
    float       tmpJi   = 1.0;          // Feedback inhibition J_i
    
    

    /* Read parameters from input file. Input file is a simple text file that contains one line with parameters and white spaces in between. */
    FILE *file;
    char param_file[300];memset(param_file, 0, 300*sizeof(char));
    snprintf(param_file, sizeof(param_file), "/input/%s",argv[1]);
    file=fopen(param_file, "r");
    if (file==NULL){
        printf( "\nERROR: Could not open file %s. Terminating... \n\n", param_file);
        exit(0);
    }
    if(fscanf(file,"%d",&nodes) != EOF && fscanf(file,"%f",&G) != EOF && fscanf(file,"%f",&J_NMDA) != EOF && fscanf(file,"%f",&w_plus) != EOF && fscanf(file,"%f",&tmpJi) != EOF && fscanf(file,"%f",&sigma) != EOF && fscanf(file,"%d",&time_steps) != EOF && fscanf(file,"%d",&BOLD_TR) != EOF && fscanf(file,"%f",&global_trans_v) != EOF && fscanf(file,"%d",&rand_num_seed) != EOF){
    } else{
        printf( "\nERROR: Unexpected end-of-file in file %s. File contains less input than expected. Terminating... \n\n", param_file);
        exit(0);
    }
    fclose(file);
    
    char output_file[300];memset(output_file, 0, 300*sizeof(char));
    snprintf(output_file, sizeof(output_file), "/output/%s_%s_fMRI.txt",argv[2],argv[1]);
    
    
    /* Add fake regions to make region count a multiple of vectorization grade */
    if (nodes % vectorization_grade != 0){
        printf( "\nWarning: Specified number of nodes (%d) is not a multiple of vectorization degree (%d). Will add some fake nodes... \n\n", nodes, vectorization_grade);
        
        int remainder   = nodes%vectorization_grade;
        if (remainder > 0) {
            fake_nodes = nodes + (vectorization_grade - remainder);
        }
    } else{
        fake_nodes = nodes;
    }
    
    
    
    /* Initialize random number generator */
    srand((unsigned)rand_num_seed);
    

    
    
    /* Derived parameters */
    const float sigma_sqrt_dt = sqrt_dt * sigma;
    const int   nodes_vec     = fake_nodes/vectorization_grade;
    const float min_d_E       = -1.0 * d_E;
    const float min_d_I       = -1.0 * d_I;
    const float imintau_E     = -1.0 / tau_E;
    const float imintau_I     = -1.0 / tau_I;
    const float w_E__I_0      = w_E * I_0;
    const float w_I__I_0      = w_I * I_0;
    const float one           = 1.0;
    const float w_plus__J_NMDA= w_plus * J_NMDA;
    const float G_J_NMDA      = G * J_NMDA;
          float TR            = (float)BOLD_TR / 1000;   // (s) TR of fMRI data
          int   BOLD_ts_len   = time_steps / (TR / model_dt) + 1;  // Length of BOLD time-series written to HDD
    
    
    
    /*
     Import and setup global and local connectivity
     */
    int         *n_conn_table;
    float       *region_activity, *SC_rowsums;
    struct Xi_p *reg_globinp_p;
    struct SC_capS      *SC_cap;
    struct SC_inpregS   *SC_inpreg;
    char cap_file[300];memset(cap_file, 0, 300*sizeof(char));
    snprintf(cap_file, sizeof(cap_file), "/input/%s_SC_weights.txt",argv[2]);
    char dist_file[300];memset(dist_file, 0, 300*sizeof(char));
    snprintf(dist_file, sizeof(dist_file), "/input/%s_SC_distances.txt",argv[2]);
    char reg_file[300];memset(reg_file, 0, 300*sizeof(char));
    snprintf(reg_file, sizeof(reg_file), "/input/%s_SC_regionids.txt",argv[2]);
    
    int         maxdelay = importGlobalConnectivity(cap_file, dist_file, reg_file, nodes, &region_activity, &reg_globinp_p, global_trans_v, &n_conn_table, G_J_NMDA, &SC_cap, &SC_rowsums, &SC_inpreg);
    
    int         reg_act_size = nodes * maxdelay;
    
    
    
    /*
     Initialize and/or cast to vector-intrinsics types for variables & parameters
     */
    float *J_i      = (float *)_mm_malloc(fake_nodes * sizeof(float),16); // (nA) inhibitory synaptic coupling
    float *BOLD_ex  = (float *)_mm_malloc(nodes * BOLD_ts_len * sizeof(float),16);
    
    if (J_i == NULL || BOLD_ex == NULL) {
        printf( "ERROR: Running out of memory. Aborting... \n");
        _mm_free(J_i);_mm_free(BOLD_ex);
        return 1;
    }
    // Initialize state variables / parameters
    for (j = 0; j < fake_nodes; j++) {
        J_i[j]              = tmpJi;
    }
    const __m128    _dt                 = _mm_load1_ps(&dt);
    const __m128    _sigma_sqrt_dt      = _mm_load1_ps(&sigma_sqrt_dt);
    const __m128    _w_plus_J_NMDA      = _mm_load1_ps(&w_plus__J_NMDA);
    const __m128    _a_E                = _mm_load1_ps(&a_E);
    const __m128    _b_E                = _mm_load1_ps(&b_E);
    const __m128    _min_d_E            = _mm_load1_ps(&min_d_E);
    const __m128    _a_I                = _mm_load1_ps(&a_I);
    const __m128    _b_I                = _mm_load1_ps(&b_I);
    const __m128    _min_d_I            = _mm_load1_ps(&min_d_I);
    const __m128    _gamma              = _mm_load1_ps(&gamma);
    const __m128    _gamma_I            = _mm_load1_ps(&gamma_I);
    const __m128    _imintau_E          = _mm_load1_ps(&imintau_E);
    const __m128    _imintau_I          = _mm_load1_ps(&imintau_I);
    const __m128    _w_E__I_0           = _mm_load1_ps(&w_E__I_0);
    const __m128    _w_I__I_0           = _mm_load1_ps(&w_I__I_0);
    float           tmp_sigma           = sigma*dt;// pre-compute dt*sigma for the integration of sigma*randnumber in equations (9) and (10) of Deco2014
    const __m128    _sigma              = _mm_load1_ps(&tmp_sigma);
    const __m128    _one                = _mm_load1_ps(&one);
    const __m128    _J_NMDA             = _mm_load1_ps(&J_NMDA);

    
    
    
    
    /*
     Start multithread simulation
     */
    
    
    
    /* Create threads */
    pthread_barrier_init(&mybarrier_base, NULL, n_threads);
    pthread_t       thr[n_threads];
    thread_data_t   thr_data[n_threads];
    int rc;
    for (i = 0; i < n_threads; ++i) {
        thr_data[i].tid             =	i               ;
        thr_data[i].n_threads       =	n_threads       ;
        thr_data[i].vectorization_grade = vectorization_grade;
        thr_data[i].nodes			=	nodes			;
        thr_data[i].fake_nodes      =   fake_nodes      ;
        thr_data[i].J_i             =	J_i            	;
        thr_data[i].reg_act_size	=	reg_act_size	;
        thr_data[i].region_activity	=	region_activity	;
        thr_data[i]._gamma			=	_gamma			;
        thr_data[i]._one			=	_one			;
        thr_data[i]._imintau_E		=	_imintau_E		;
        thr_data[i]._dt             =	_dt             ;
        thr_data[i]._sigma_sqrt_dt  =   _sigma_sqrt_dt  ;
        thr_data[i]._sigma			=	_sigma			;
        thr_data[i]._gamma_I		=	_gamma_I		;
        thr_data[i]._imintau_I		=	_imintau_I		;
        thr_data[i]._min_d_I		=	_min_d_I		;
        thr_data[i]._b_I			=	_b_I			;
        thr_data[i]._J_NMDA         =	_J_NMDA         ;
        thr_data[i]._w_I__I_0		=	_w_I__I_0		;
        thr_data[i]._a_I			=	_a_I			;
        thr_data[i]._min_d_E		=	_min_d_E		;
        thr_data[i]._b_E			=	_b_E			;
        thr_data[i]._a_E			=	_a_E			;
        thr_data[i]._w_plus_J_NMDA	=	_w_plus_J_NMDA	;
        thr_data[i]._w_E__I_0		=	_w_E__I_0		;
        thr_data[i].nodes_vec		=	nodes_vec		;
        thr_data[i].SC_cap			=	SC_cap			;
        thr_data[i].SC_inpreg       =	SC_inpreg       ;
        thr_data[i].reg_globinp_p	=	reg_globinp_p	;
        thr_data[i].n_conn_table	=	n_conn_table	;
        thr_data[i].time_steps      =   time_steps      ;
        thr_data[i].BOLD_TR         =   BOLD_TR         ;
        thr_data[i].model_dt        =   model_dt        ;
        thr_data[i].BOLD_ex         =   BOLD_ex         ;
        thr_data[i].rand_num_seed   =   rand_num_seed   ;
        thr_data[i].BOLD_ts_len     =   BOLD_ts_len     ;
        thr_data[i].output_file     =   output_file     ;
        
        
        if ((rc = pthread_create(&thr[i], NULL, run_simulation, &thr_data[i]))) {
            fprintf(stderr, "error: pthread_create, rc: %d\n", rc);
            return EXIT_FAILURE;
        }
    }
    
    /* block until all threads complete */
    for (i = 0; i < n_threads; ++i) {
        pthread_join(thr[i], NULL);
    }
    printf("Threads finished. Back to main thread.\n");
    
    printf("MT-TVBii finished. Execution took %.2f s for %d nodes. Goodbye!\n", (float)(time(NULL) - start), nodes);

    return 0;
}
