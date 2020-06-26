/* compile -> gcc main_rsHRF.c -lpthread -lm -lgsl -lgslcblas -o tvbii */
/* execute -> ./tvbii <param_filename> <SC_weights_filename> <SC_tract_lengths_filename> <rsHRF_filename> <num_of_thread> */ 

/* All the inputs should be stored in a directory named C_Input and all the input files should be .txt files 
    NOTE: Do NOT add '.txt' extension to the filenames, that is appended by the program
   The output gets stored in a fMRI.txt file formed in the directory of the execution-code */

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/* Defining PI */
#ifndef PI
    # define PI	3.14159265358979323846264338327950288
#endif

/* Stores the information of the neural-activity for each region at each time-step (0 to max_delay) */
struct Xi_p 
{
    float **Xi_elems;
};

/* stores the information about the connectome weights for each region */
struct SC_capS 
{
    float *cap;
};

/* stores the information about the distances between the regions of the connectome */
struct SC_inpregS
{
    int *inpreg;
};

/* stores the retrieved rsHRF for the regions */
struct rsHRFS
{
    float* hrf;
};

/* auxiallary functions */

/* Rouding the value to the closest integer */
int divRoundClosest(const int n, const int d)
{
    return ((n < 0) ^ (d < 0)) ? ((n - d/2)/d) : ((n + d/2)/d);
}

/* Rounding up the value to the closest integer */
int divRoundup(const int x, const int y)
{
    return (x + y - 1) / y;
}

/* normalized sinc function */
float sinc(float x)
{
    x *= M_PI;

    if (x == 0.0) 
        return 1.0;

    return sin(x)/x;
}

/* interpolate a function made of input samples at point x */
float sinc_approx(float in[], size_t in_sz, float x)
{
    int   i;
    float res = 0.0;

    for (i = 0; i < in_sz; i++)
        res += in[i] * sinc(x - i);

    return res;
} 

/* resamples the signal */
void resample_sinc( float in[], size_t in_sz, float out[], size_t out_sz)
{
    int   i;
    float dx = (float) (in_sz - 1) / (out_sz - 1);
    for (i = 0; i < out_sz; i++)
        out[i] = sinc_approx(in, in_sz, i*dx);
}

/* taking the dot product (to mimic convolution) between the neural states and the HRF (which has been reversed beforehand, and is shited by 'init' to align it to the neural response) */
/* this is performed at every time-step falling at the BOLD Repetition Time */
float shifted_reversed_dot_product(float* sigA, float* sigB, int n, int init)
{
    float output = 0.;
    for(int i = 0; i < n; i++)
    {
        output = output + sigA[i]*sigB[(i + init)%n];
    }
    return output;
}

/* For thread synchronization */
pthread_mutex_t mutex_thrcount;
pthread_barrier_t mybarrier_base, mybarrier1, mybarrier2, mybarrier3;

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
    int                 nodes, nodes_vec, fake_nodes, n_threads, vectorization_grade;
    int                 BOLD_TR;
    int                 BOLD_TS_len;
    float               *J_i;
    int                 reg_act_size;
    float               *region_activity;
    float               model_dt;
    int                 stock_steps;
    int                 interim_istep;
    float               target_FR;
    int                 total_duration;
    int                 FIC_time;
    int                 HRF_samples;
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
    struct rsHRFS       *rsHRF;
    struct Xi_p         *reg_globinp_p;
    int                 *n_conn_table;
    float               *BOLD_ex;    
    char                *output_file;
} thread_data_t;

/* Function for multi-threaded FIC tuning */
void *run_simulation(void *arg)
{

    /* Local function parameters (no concurrency) */
    int                                  i, j, i_node_vec, i_node_vec_local, k, int_i, ts;
    float                                tmpglobinput, tmpglobinput_FFI;
    __m128                               _tmp_H_E, _tmp_H_I, _tmp_I_I, _tmp_I_E;
    float   tmp_exp_E[4]                __attribute__((aligned(16)));
    float   tmp_exp_I[4]                __attribute__((aligned(16)));
    float   rand_number[4]              __attribute__((aligned(16)));
    __m128  *_tmp_exp_E                 = (__m128*)tmp_exp_E;
    __m128  *_tmp_exp_I                 = (__m128*)tmp_exp_I;
    __m128  *_rand_number               = (__m128*)rand_number;
    int     ring_buf_pos                = 0;

    
    /* Global function parameters (some with concurrency) */
    thread_data_t *thr_data = (thread_data_t *)arg;

    int     t_id                    = thr_data->tid;                // thread ID
    int     rand_num_seed           = thr_data->rand_num_seed;      // random number seed
    int     nodes                   = thr_data->nodes;              // number of nodes in the brain model
    int     fake_nodes              = thr_data->fake_nodes;         // number of nodes + fake nodes 
    const   int nodes_vec           = thr_data->nodes_vec;      
    int     n_threads               = thr_data->n_threads;          // number of threads
    int     vectorization_grade     = thr_data->vectorization_grade;
    int     reg_act_size            = thr_data->reg_act_size;
    int     *n_conn_table           = thr_data->n_conn_table;       
    float   *J_i                    = thr_data->J_i;                // Feedback Inhibhition
    float   *region_activity        = thr_data->region_activity;    
    float   *BOLD_ex                = thr_data->BOLD_ex;            // BOLD output
    const   __m128 _gamma           = thr_data->_gamma;             
    const   __m128 _one             = thr_data->_one;
    const   __m128 _imintau_E       = thr_data->_imintau_E;
    const   __m128 _dt              = thr_data->_dt;
    const   __m128 _sigma           = thr_data->_sigma;
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
    struct  SC_capS *SC_cap         = thr_data->SC_cap;             // Structural connectome - weights
    struct  rsHRFS  *rsHRF          = thr_data->rsHRF;              // Region wise resting-state HRF
    struct  Xi_p *reg_globinp_p     = thr_data->reg_globinp_p;      
    int     BOLD_TR                 = thr_data->BOLD_TR;            // Repetition time for BOLD (in miliseconds)
    int     BOLD_TS_len             = thr_data->BOLD_TS_len;        // Length of the BOLD Time-Series
    float   model_dt                = thr_data->model_dt;           // Duration per sample (after temporal averaging) (in seconds)
    int     stock_steps             = thr_data->stock_steps;        // Number of samples required for each point of convolution - HRF_length / model's sampling period 
    int     interim_istep           = thr_data->interim_istep;      // Number of steps for each temporal averaging period model's sampling period / integration period
    float   target_FR               = thr_data->target_FR;          // Target firing rate
    int     total_duration          = thr_data->total_duration;     // Total duration of simulated signal (in seconds)
    int     HRF_samples             = thr_data->HRF_samples;        // Length of the HRF time-series
    int     FIC_time                = thr_data->FIC_time;           // Duration for which J_i is iteratively corrected (in seconds)
    char*   output_file             = thr_data->output_file;        // Output filename where the simulated BOLD response is stored

    /* Parallelism: divide work among threads */
    int nodes_vec_mt        = divRoundup(nodes_vec, n_threads);
    int nodes_mt            = nodes_vec_mt  * vectorization_grade;
    
    int start_nodes_vec_mt  = t_id         * nodes_vec_mt;
    int start_nodes_mt      = t_id         * nodes_mt;
    int end_nodes_vec_mt    = (t_id + 1)   * nodes_vec_mt;
    int end_nodes_mt        = (t_id + 1)   * nodes_mt;
    int end_nodes_mt_glob   = end_nodes_mt; /* end_nodes_mt_glob may differ from end_nodes_mt in the last thread (fake-nodes vs nodes) */
    
    /* Correct settings for last thread */
    if (end_nodes_mt > nodes) 
    {
        end_nodes_vec_mt    = nodes_vec;
        end_nodes_mt        = fake_nodes;
        end_nodes_mt_glob   = nodes;
        nodes_mt            = end_nodes_mt - start_nodes_mt;
        nodes_vec_mt        = end_nodes_vec_mt - start_nodes_vec_mt;
    }
    
    printf("thread %d: start: %d end: %d size: %d\n", thr_data->tid, start_nodes_mt, end_nodes_mt, nodes_mt);
    
    
    /* Check whether splitting makes sense (i.e. each thread has something to do). Terminate if not. */
    if (nodes_vec_mt <= 1) 
    {
        printf("The splitting is ineffective (e.g. fewer nodes than threads or some threads have nothing to do). Use different numbers of threads. Terminating.\n");
        exit(2);
    }
    
    /* Set up barriers */
    if (t_id == 0) 
    {
        initialize_thread_barriers(n_threads);
    }
    pthread_barrier_wait(&mybarrier_base);


    /* Initialize random number generator */
    gsl_rng *r = gsl_rng_alloc (gsl_rng_mt19937);
    gsl_rng_set (r, (t_id+rand_num_seed)); // Random number seed -- otherwise every node gets the same random nums
    srand((unsigned)(t_id+rand_num_seed));
    
    
    
    // All threads
    float   *meanFR                 = (float *)_mm_malloc(nodes_mt * sizeof(float),16);  
    float   *meanFR_INH             = (float *)_mm_malloc(nodes_mt * sizeof(float),16);  
    float   *global_input           = (float *)_mm_malloc(nodes_mt * sizeof(float),16);
    float   *S_i_E                  = (float *)_mm_malloc(nodes_mt * sizeof(float),16);
    float   *S_i_I                  = (float *)_mm_malloc(nodes_mt * sizeof(float),16);
    float   *J_i_local              = (float *)_mm_malloc(nodes_mt * sizeof(float),16);
    
    if (meanFR == NULL || meanFR_INH == NULL || global_input == NULL || S_i_E==NULL || S_i_I==NULL || J_i_local==NULL) 
    {
        printf( "ERROR: Running out of memory. Aborting... \n");
        exit(2);
    }
    
    // Cast to vector/SSE arrays
    __m128          *_meanFR            = (__m128*)meanFR;
    __m128          *_meanFR_INH        = (__m128*)meanFR_INH;
    __m128          *_global_input      = (__m128*)global_input;
    __m128          *_S_i_E             = (__m128*)S_i_E;
    __m128          *_S_i_I             = (__m128*)S_i_I;
    __m128          *_J_i_local         = (__m128*)J_i_local;
    
    // model parameters
    float mean_mean_FR        = 0.0;
    int   FIC_time_steps      = divRoundClosest(FIC_time/model_dt, 1);                                                 // Number of time steps for FIC cycle   
    int   time_steps          = FIC_time_steps + (divRoundClosest(total_duration/model_dt, 1)) + stock_steps;          // Total time-steps for the simulation loop
    int   BOLD_TR_steps       = divRoundClosest(BOLD_TR/model_dt, 1);                                                  // Number of steps per BOLD Repetition Time
    float *BOLD               = (float *)_mm_malloc(nodes_mt * BOLD_TS_len * sizeof(float),16);                        // Resulting BOLD output

    if(BOLD == NULL)
    {
        printf( "ERROR: Running out of memory. Aborting... \n");
        exit(2);
    }

    // Reset arrays and variables
    int i_meanfr            = 0;
    ring_buf_pos            = 0;

    for (j = 0; j < nodes_mt; j++) 
    {
        meanFR[j]           = 0.0;
        meanFR_INH[j]       = 0.0;
        global_input[j]     = 0.00;
        S_i_E[j]            = 0.00;
        S_i_I[j]            = 0.00;
        J_i_local[j]        = J_i[j+start_nodes_mt];
    }

    /* Allocating memory for the Simulated Neural Response */
    float **SIMULATED_signal = (float **)_mm_malloc(nodes_mt * sizeof(float*),16);
    if(SIMULATED_signal == NULL)
    {
        printf( "ERROR: Running out of memory. Aborting... \n");
        exit(2);
    }

    for (i=0; i<nodes_mt; i++)
    {
        SIMULATED_signal[i] = (float *)_mm_malloc(stock_steps * sizeof(float),16);
        if(SIMULATED_signal[i] == NULL)
        {
            printf( "ERROR: Running out of memory. Aborting... \n");
            exit(2);
        }
    }
    
    /* Allocating memory for the resting-state HRF */
    float **HRF_signal = (float **)_mm_malloc((nodes_mt) * sizeof(float*),16);
    if(HRF_signal == NULL)
    {
        printf( "ERROR: Running out of memory. Aborting... \n");
        exit(2);
    }

    for (i=0; i < (end_nodes_mt_glob - start_nodes_mt); i++)
    {
        HRF_signal[i] = (float *)_mm_malloc(stock_steps * sizeof(float),16);
        if(HRF_signal[i] == NULL)
        {
            printf( "ERROR: Running out of memory. Aborting... \n");
            exit(2);
        }
        /* resampling (from HRF_samples to stock_steps) and storing the HRF signal */
        resample_sinc(rsHRF[i+start_nodes_mt].hrf, HRF_samples, HRF_signal[i], stock_steps) ;                                
    }


    /* Initializing the region_activity once for the entire Simulation */
    if (t_id == 0) {
        for (j=0; j<reg_act_size; j++) {
            region_activity[j]=0.00;
        }
    }

    /* Barrier: only start simulating when arrays were cleared */
    pthread_barrier_wait(&mybarrier1);


    /* SIMULATION LOOP */
    int ts_bold_i = 0; // current length of the BOLD Simulation
    for (ts = 0; ts < time_steps; ts++) 
    {
        
        if (t_id == 0)
        {
            printf("%.1f %% \r", ((float)ts / (float)time_steps) * 100.0f );
        }
        // integration iterations per temporal averaging duration -> interim_istep
        for (int_i = 0; int_i < interim_istep; int_i++) 
        {
            /* Barrier: only start integrating input when all other threads are finished copying their results to buffer */
            pthread_barrier_wait(&mybarrier2);
            
            /* Compute global coupling */
            i_node_vec_local = 0;
            for(j=start_nodes_mt; j<end_nodes_mt_glob; j++){
                tmpglobinput     = 0;
                tmpglobinput_FFI = 0;
                for (k=0; k<n_conn_table[j]; k++) {
                    tmpglobinput     += *reg_globinp_p[j+ring_buf_pos].Xi_elems[k] * SC_cap[j].cap[k];
                }
                
                global_input[i_node_vec_local]     = tmpglobinput;                
                
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

                
                // Inhibitory population firing rate
                _tmp_I_I = _mm_sub_ps(_mm_mul_ps(_a_I,_mm_sub_ps(_mm_add_ps((_w_I__I_0),_mm_mul_ps(_J_NMDA, _S_i_E[i_node_vec_local])), _S_i_I[i_node_vec_local])),_b_I);
                *_tmp_exp_I   = _mm_mul_ps(_min_d_I, _tmp_I_I);
                tmp_exp_I[0]  = tmp_exp_I[0] != 0 ? expf(tmp_exp_I[0]) : 0.9;
                tmp_exp_I[1]  = tmp_exp_I[1] != 0 ? expf(tmp_exp_I[1]) : 0.9;
                tmp_exp_I[2]  = tmp_exp_I[2] != 0 ? expf(tmp_exp_I[2]) : 0.9;
                tmp_exp_I[3]  = tmp_exp_I[3] != 0 ? expf(tmp_exp_I[3]) : 0.9;
                _tmp_H_I  = _mm_div_ps(_tmp_I_I, _mm_sub_ps(_one, *_tmp_exp_I));
                _meanFR_INH[i_node_vec_local] = _mm_add_ps(_meanFR_INH[i_node_vec_local],_tmp_H_I);

                
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
            for(j=0; j<nodes_mt; j++)
            {
                S_i_E[j] = S_i_E[j] >= 0.0 ? S_i_E[j] : 0.0;
                S_i_E[j] = S_i_E[j] <= 1.0 ? S_i_E[j] : 1.0;
                S_i_I[j] = S_i_I[j] >= 0.0 ? S_i_I[j] : 0.0;
                S_i_I[j] = S_i_I[j] <= 1.0 ? S_i_I[j] : 1.0;
            }

            i_meanfr++;

            memcpy(&region_activity[ring_buf_pos+start_nodes_mt], S_i_E, nodes_mt*sizeof( float ));
            
            /* Shift region_activity ring-buffer start */
            ring_buf_pos = ring_buf_pos<(reg_act_size-nodes) ? (ring_buf_pos+nodes) : 0;
            
        }

        /* re-calculating every 10 seconds till FIC_time */
        if (ts >= divRoundClosest(10000/model_dt, 1) && ts <= (FIC_time_steps) && ts % divRoundClosest(10000/model_dt, 1) == 0) 
        {     
            /*  Compute mean firing rates */
            mean_mean_FR = 0;
            float mean_mean_FR_INH = 0;
            for (j = 0; j < nodes; j++){
                meanFR[j]     = meanFR[j] / i_meanfr;
                meanFR_INH[j] = meanFR_INH[j] / i_meanfr;
                mean_mean_FR += meanFR[j];
                mean_mean_FR_INH += meanFR_INH[j];
            }

            mean_mean_FR /= nodes;
            mean_mean_FR_INH /= nodes;

            /* Compute variance */
            float var_FR = 0.0f, tmpvar;
            for (j = 0; j < nodes; j++){
                tmpvar = meanFR[j] - mean_mean_FR;
                var_FR += (tmpvar * tmpvar);
            }

            var_FR /= nodes;
            float std_FR = sqrt(var_FR);
            printf("time (s): %d\t\tmean+/-std firing rate exc. pops.: %.2f +/- %.2f\n", ts*model_dt, mean_mean_FR, std_FR);

            /*
             #################################################
             Inhibitory synaptic plasticity from Vogels et al. Science 2011
             Eq 1: dw = eta(pre × post – r0 × pre)
             #################################################
            */

            float isp_eta = 0.001;
            float isp_r0  = target_FR;
            for (j = 0; j < nodes; j++){
                float pre  = meanFR_INH[j];
                float post = meanFR[j];
                J_i_local[j] +=  isp_eta * (pre * post - isp_r0 * pre);
                J_i_local[j] = J_i_local[j] > 0 ? J_i_local[j] : 0; // make sure that weight is always >=0
            }

            /* Reset arrays */
            i_meanfr = 0;
            for (j = 0; j < nodes; j++) {
                meanFR[j]           = 0.0;
                meanFR_INH[j]       = 0.0;
            }
        }
        
        /*Saving the Simulated Activity after the initial FIC_time_steps */
        if(ts >= FIC_time_steps)
        {
            for (j = 0; j < nodes_mt; j++) 
            {
                SIMULATED_signal[j][(ts-FIC_time_steps)%stock_steps] = S_i_E[j];   
            } 
        }

        /* after the target_firing rate has been corrected and SIMULATED_signal has been calculated for initial stock_steps, 
        we convolve the neural states with the HRF (at each time-step falling on BOLD_TR_steps) to begin obtaining the BOLD response*/
        if(ts >= (FIC_time_steps + stock_steps) && ts >= BOLD_TR_steps && ts % BOLD_TR_steps == 0)
        {
            for (j = 0; j < (end_nodes_mt_glob - start_nodes_mt); j++)
            {  
                BOLD[ts_bold_i +  j * BOLD_TS_len] = shifted_reversed_dot_product(SIMULATED_signal[j], HRF_signal[j], stock_steps, ts - FIC_time_steps);
            }
            ts_bold_i++;
        }
    } /* Simulation loop */

    /* Copying the Feedback Inhibhition Parameter Values */
    for(j = 0; j < (end_nodes_mt_glob - start_nodes_mt); j++)
    {
        J_i[start_nodes_mt + j] = J_i_local[j];
    }

    /* Copy results of this thread into shared memory buffer */
    memcpy(&BOLD_ex[start_nodes_mt*BOLD_TS_len], BOLD, (end_nodes_mt_glob - start_nodes_mt) * BOLD_TS_len * sizeof( float ));

    /* Wait until other threads finished before writing out result */
    pthread_barrier_wait(&mybarrier3);
    
    /* Writing the BOLD response in the output file */
    if (t_id == 0)
    {
        printf("\n");           
        FILE *FCout = fopen(output_file, "w+");
        for (j = 0; j < nodes; j++) 
        {
            for (k = 0; k < BOLD_TS_len; k++) 
            {
                fprintf(FCout, "%.5f ",BOLD_ex[(BOLD_TS_len *j) + k]);
            }
            fprintf(FCout, "\n");
        }
        fclose(FCout);
    }

    _mm_free(SIMULATED_signal);
    _mm_free(HRF_signal);

    gsl_rng_free (r);
    pthread_exit(NULL);
}

/*  The first line of arguments are input file names taken as input by the function, these include the names of :
        SC_cap - connectome weights
        SC_dist - connectome tract lengths
        rsHRF - retreived rsHRF
    The second line of arguments are inputs used by the function
    The third, fourth and fifth line of arguments consist of what the function modifies so as to make it available globally
    Apart from these, the function also returns max delay in transmission across the connectome
*/

int importGlobalConnectivity(char *SC_cap_filename, char *SC_dist_filename, char *rsHRF_filename, 
    int regions, float global_trans_v, int HRF_len, float G_J_NMDA, float model_dt, float dt,
    float **region_activity, struct Xi_p **reg_globinp_p,
    int **n_conn_table, struct SC_capS **SC_cap, 
    float **SC_rowsums, struct SC_inpregS **SC_inpreg, struct rsHRFS **rsHRF)
{
    
    int                 i,j,k, maxdelay=0, tmpint;
    float               *region_activity_p;
    double              tmp_, tmp, tmp2;
    struct Xi_p         *reg_globinp_pp;
    struct SC_capS      *SC_capp;
    struct rsHRFS       *rsHRFp;
    struct SC_inpregS   *SC_inpregp;

    /* Open SC and rsHRF files */
    FILE *file_cap, *file_dist, *file_rsHRF;

    file_cap        =fopen(SC_cap_filename, "r");
    file_dist       =fopen(SC_dist_filename, "r");
    file_rsHRF      =fopen(rsHRF_filename, "r");
    

    if (file_cap==NULL || file_dist==NULL || file_rsHRF==NULL)
    {
        printf( "\nERROR: Could not open SC and/or rsHRF files. Terminating... \n\n");
        exit(2);
    }
    
    
    /* Allocate a counter that stores number of region input connections for each region, the SCcap array and the rsHRF struct */
    *SC_rowsums         = (float *)_mm_malloc(regions*sizeof(float),16);
    *n_conn_table       = (int *)_mm_malloc(regions*sizeof(int),16);
    *SC_cap             = (struct SC_capS *)_mm_malloc(regions*sizeof(struct SC_capS),16);
    SC_capp             = *SC_cap;
    *SC_inpreg          = (struct SC_inpregS *)_mm_malloc(regions*sizeof(struct SC_inpregS),16);
    SC_inpregp          = *SC_inpreg;
    *rsHRF              = (struct rsHRFS *)_mm_malloc(regions*sizeof(struct SC_inpregS),16);
    rsHRFp              = *rsHRF;
    
    if(*n_conn_table==NULL || SC_capp==NULL || SC_rowsums==NULL || SC_inpregp==NULL || rsHRFp==NULL)
    {
        printf("Running out of memory. Terminating.\n");fclose(file_dist);fclose(file_cap);exit(2);
    }

    /* Importing the rsHRF */
    int n_entries_  = 0;
    int counter_    = 0;
    tmp             = 0;

    for(int i = 0; i < regions; i++)
    {
        rsHRFp[i].hrf      = (float *)_mm_malloc((HRF_len)*sizeof(float),16);
        if(rsHRFp[counter_].hrf == NULL){
            printf("Running out of memory. Terminating.\n");exit(2);
        }
    }

    counter_ = 0;
    int t = 0;

    while(fscanf(file_rsHRF,"%lf",&tmp_) != EOF)
    {
        // reading the HRF in the reverse order, as that is what we need for the convolution
        rsHRFp[counter_].hrf[(HRF_len - t) - 1] = tmp_;
        n_entries_++;
        t++;
        if(t == HRF_len)
        {
            counter_++;
            t = 0;
        }
    }

    if (n_entries_ < regions * HRF_len) 
    {
        printf( "ERROR: Unexpected end-of-file in file. File contains less input than expected. Terminating... \n");
        exit(2);
    }

    fclose(file_rsHRF);

    
    /* Read out the maximal fiber length and the degree of each node and rewind SC files */
    int n_entries = 0, curr_col = 0, curr_row = 0, curr_row_nonzero = 0;
    double tmp_max = -9999;

    while(fscanf(file_dist,"%lf",&tmp) != EOF && fscanf(file_cap,"%lf",&tmp2) != EOF)
    {
        if (tmp_max < tmp) tmp_max = tmp;
            n_entries++;
        
        if (tmp2 > 0.0) curr_row_nonzero++;
        
        curr_col++;
        if (curr_col == regions) 
        {
            curr_col = 0;
            (*n_conn_table)[curr_row] = curr_row_nonzero;
            curr_row_nonzero = 0;
            curr_row++;
        }
    }
    
    if (n_entries < regions * regions) 
    {
        printf( "ERROR: Unexpected end-of-file in file. File contains less input than expected. Terminating... \n");
        exit(2);
    }
    
    rewind(file_dist);
    rewind(file_cap);
    
    maxdelay = (int)(((tmp_max/global_trans_v)/(model_dt*dt))+0.5); // for getting from m/s to (model_dt * dt) sampling, +0.5 for rounding by casting
    if (maxdelay < 1) maxdelay = 1; // Case: no time delays
    
    
    
    
    /* Allocate ringbuffer that contains region activity for each past time-step until maxdelay and another ringbuffer that contains pointers to the first ringbuffer */
    *region_activity    = (float *)_mm_malloc(maxdelay*regions*sizeof(float),16);
    region_activity_p   = *region_activity;
    *reg_globinp_p      = (struct Xi_p *)_mm_malloc(maxdelay*regions*sizeof(struct Xi_p),16);
    reg_globinp_pp      = *reg_globinp_p;

    if(region_activity_p==NULL || reg_globinp_p==NULL)
    {
        printf("Running out of memory. Terminating.\n");fclose(file_dist);exit(2);
    }
    for (j=0; j<maxdelay*regions; j++) 
    {
        region_activity_p[j]=0.001;
    }
    
    /* Read SC files and set pointers for each input region and correspoding delay for each ringbuffer time-step */
    int ring_buff_position;
    for (i=0; i<regions; i++)
    {
        
        if ((*n_conn_table)[i] > 0) 
        {
            // SC strength and inp region numbers
            SC_capp[i].cap          = (float *)_mm_malloc(((*n_conn_table)[i])*sizeof(float),16);
            SC_inpregp[i].inpreg    = (int *)_mm_malloc(((*n_conn_table)[i])*sizeof(int),16);
            if(SC_capp[i].cap==NULL || SC_inpregp[i].inpreg==NULL)
            {
                printf("Running out of memory. Terminating.\n");exit(2);
            }
            
            // Allocate memory for input-region-pointer arrays for each time-step in ringbuffer
            for (j=0; j<maxdelay; j++)
            {
                reg_globinp_pp[i+j*regions].Xi_elems=(float **)_mm_malloc(((*n_conn_table)[i])*sizeof(float *),16);
                if(reg_globinp_pp[i+j*regions].Xi_elems==NULL)
                {
                    printf("Running out of memory. Terminating.\n");exit(2);
                }
            }
            
            float sum_caps=0.0;
            // Read incoming connections and set pointers
            curr_row_nonzero = 0;
            for (j=0; j<regions; j++) 
            {
                if(fscanf(file_cap,"%lf",&tmp) != EOF && fscanf(file_dist,"%lf",&tmp2) != EOF)
                {
                    if (tmp > 0.0) 
                    {
                        tmpint = (int)(((tmp2/global_trans_v)/(model_dt*dt))+0.5); //  *10 for getting from m/s or mm/ms to 10kHz sampling, +0.5 for rounding by casting
                        if (tmpint < 0 || tmpint > maxdelay)
                        {
                            printf("Delay: %d \t Max delay: %d\n", tmpint, maxdelay);
                            printf( "\nERROR: Negative or too high (larger than maximum specified number) connection length/delay %d -> %d. Terminating... \n\n",i,j);exit(2);
                        }
                        if (tmpint <= 0) tmpint = 1; // If time delay is smaller than integration step size, than set time delay to one integration step
                        
                        SC_capp[i].cap[curr_row_nonzero] = (float)tmp * G_J_NMDA;
                        sum_caps                += SC_capp[i].cap[curr_row_nonzero];
                        SC_inpregp[i].inpreg[curr_row_nonzero]  =  j;
                        
                        ring_buff_position=maxdelay*regions - tmpint*regions + j;
                        for (k=0; k<maxdelay; k++)
                        {
                            reg_globinp_pp[i+k*regions].Xi_elems[curr_row_nonzero]=&region_activity_p[ring_buff_position];
                            ring_buff_position += regions;
                            if (ring_buff_position > (maxdelay*regions-1)) ring_buff_position -= maxdelay*regions;
                        }
                        
                        curr_row_nonzero++;
                    }
                } 
                else
                {
                    printf( "\nERROR: Unexpected end-of-file in file %s or %s. File contains less input than expected. Terminating... \n\n", SC_cap_filename, SC_dist_filename);
                    exit(2);
                }
                
            }
            if (sum_caps <= 0) 
            {
                printf( "\nERROR: Sum of connection strenghts is negative or zero. sum-caps node %d = %f. Terminating... \n\n",i,sum_caps);exit(2);
            }
            (*SC_rowsums)[i] = sum_caps;
        }
    }
    
    fclose(file_dist);fclose(file_cap);
    return maxdelay;

}

int main(int argc, char* argv[])
{

    /* checking whether the right number of parameters were passed, if not returns with an error */
    if (argc != 6 || atoi(argv[5]) <= 0) 
    {
        printf("\nERROR: Invalid arguments.\n\nUsage: tvbii <paramset_file> <SC_distances> <SC_weights> <rsHRF> <#threads>\n\nTerminating... \n\n");
        printf("\nProvided arguments:\n");
        int i;
        for (i=0; i<argc; i++) 
        {
            printf("%s\n", argv[i]);
        }
        exit(2);
    }

    /*
     Get current time and do some initializations
     */
    time_t  start = time(NULL);
    int     i, j;
    int     n_threads = atoi(argv[5]);

    /*
     Global model and integration parameters
     */
    
    const float dt                  = 0.1;              // Integration step length dt = 0.1 ms
    const float sqrt_dt             = sqrtf(dt);        // Noise in the diffusion part scaled by sqrt of dt
    const float model_dt            = 1;                // Period of model (in ms) (sampling-rate=1000 Hz)
    const int   vectorization_grade = 4;                // How many operations can be done simultaneously. Depends on CPU Architecture and available intrinsics.
    int         FIC_time            = 10000;            // FIC - in miliseconds
    int         total_duration      = 120000;           // Length of the signal in miliseconds (here, it corresponds to 2 minutes)
    int         nodes               = 84;               // Number of surface vertices; must be a multiple of vectorization grade
    int         fake_nodes          = 84;               // Added nodes in-case the total number of nodes % vectorization grade != 0
    float       global_trans_v      = 1.0;              // Global transmission velocity (m/s); Local time-delays can be ommited since smaller than integration time-step
    float       G                   = 0.5;              // Global coupling strength
    int         BOLD_TR             = 2000;             // TR of BOLD data
    float       target_FR           = 3.0f;             // Target firing rate for exc. pops during inhibitory plasticity
    int         rand_num_seed       = 1403;             // Random Number Seed
    int         HRF_length          = 25;               // Duration of HRF (in s)
    int         HRF_samples         = 11;               // Number of sampled in the rsHRF

    /*
     Local model: DMF-Parameters from Deco et al. JNeuro 2014
     */
    float w_plus        = 1.4;          // Local excitatory recurrence synaptic weight
    float J_NMDA        = 0.15;         // (nA) excitatory synaptic coupling
    const float a_E     = 310;          // (n/C)
    const float b_E     = 125;          // (Hz)
    const float d_E     = 0.16;         // (s)
    const float a_I     = 615;          // (n/C)
    const float b_I     = 177;          // (Hz)
    const float d_I     = 0.087;        // (s)
    const float gamma   = 0.641/1000.0; // Factor 1000 for expressing everything in ms
    const float tau_E   = 100;          // (ms) Time constant of NMDA (excitatory)
    const float tau_I   = 10;           // (ms) Time constant of GABA (inhibitory)
    float       sigma   = 0.00316228;   // (nA) Noise amplitude
    const float I_0     = 0.382;        // (nA) overall effective external input
    const float w_E     = 1.0;          // Scaling of external input for excitatory pool
    const float w_I     = 0.7;          // Scaling of external input for inhibitory pool
    const float gamma_I = 1.0/1000.0;   // For expressing inhib. pop. in ms
    float       tmpJi   = 1.0;          // Feedback inhibition J_i
    
    /* Read parameters from input file. Input file is a simple text file that contains one line with parameters and white spaces in between. */
    FILE *file;
    char param_file[300];memset(param_file, 0, 300*sizeof(char));
    snprintf(param_file, sizeof(param_file), "./Input/%s.txt",argv[1]);
    file=fopen(param_file, "r");
    if (file==NULL){
        printf( "\nERROR: Could not open file %s. Terminating... \n\n", param_file);
        exit(2);
    }

    /* Reading the values for : 

        Number of nodes/regions
        Global coupling strength
        Excitatory synaptic coupling
        Local excitatory recurrence
        Feedback inhibition
        Noise amplitude
        FIC_time - in milliseconds
        Simulation length - in milliseconds
        HRF sample length 
        BOLD Repetition Time - in milliseconds
        Global transmission velocity
        Random Number Seed
        
    */
    if(fscanf(file,"%d",&nodes) != EOF && fscanf(file,"%f",&G) != EOF && fscanf(file,"%f",&J_NMDA) != EOF && fscanf(file,"%f",&w_plus) != EOF && fscanf(file,"%f",&tmpJi) != EOF && fscanf(file,"%f",&sigma) != EOF && fscanf(file,"%d",&FIC_time) != EOF && fscanf(file,"%d",&total_duration) != EOF && fscanf(file,"%d",&HRF_samples) != EOF && fscanf(file,"%d",&BOLD_TR) != EOF && fscanf(file,"%f",&global_trans_v) != EOF && fscanf(file,"%d",&rand_num_seed) != EOF){} 
    else
    {
        printf( "\nERROR: Unexpected end-of-file in file %s. File contains less input than expected. Terminating... \n\n", param_file);
        exit(2);
    }
    fclose(file);

    /* file name where the fMRI output gets stored */
    char output_file[300]; memset(output_file, 0, 300*sizeof(char));
    strcpy(output_file, "fMRI.txt");

    /* Add fake regions to make region count a multiple of vectorization grade */
    if (nodes % vectorization_grade != 0)
    {
        printf( "\nWarning: Specified number of nodes (%d) is not a multiple of vectorization degree (%d). Will add some fake nodes... \n\n", nodes, vectorization_grade);
        
        int remainder   = nodes%vectorization_grade;
        if (remainder > 0)
        {
            fake_nodes = nodes + (vectorization_grade - remainder);
        }
    } else
    {
        fake_nodes = nodes;
    }

    /* Initialize random number generator */
    srand((unsigned)rand_num_seed);

    /* Derived parameters */
    const float sigma_sqrt_dt    = sqrt_dt * sigma;
    const int   nodes_vec        = fake_nodes/vectorization_grade;
    const float min_d_E          = -1.0 * d_E;
    const float min_d_I          = -1.0 * d_I;
    const float imintau_E        = -1.0 / tau_E;
    const float imintau_I        = -1.0 / tau_I;
    const float w_E__I_0         = w_E * I_0;
    const float w_I__I_0         = w_I * I_0;
    const float one              = 1.0;
    const float w_plus__J_NMDA   = w_plus * J_NMDA;
    const float G_J_NMDA         = G * J_NMDA;

    const int stock_steps        = divRoundClosest((HRF_length*1000)/model_dt, 1); // length of time-series required for each dot product                       
    const int interim_istep      = divRoundClosest(model_dt/dt, 1);                // number of steps required per model period
    const int BOLD_TS_len        = divRoundup(total_duration/BOLD_TR, 1);          // length of the BOLD response
    
    /*
     Import and setup global and local connectivity
     */
    int                 *n_conn_table;
    float               *region_activity, *SC_rowsums;
    struct Xi_p         *reg_globinp_p;
    struct SC_capS      *SC_cap;
    struct rsHRFS       *rsHRF;
    struct SC_inpregS   *SC_inpreg;

    char cap_file[300];memset(cap_file, 0, 300*sizeof(char));
    strcat(strcat(strcat(cap_file,"./Input/"), argv[2]),".txt");

    char dist_file[300];memset(dist_file, 0, 300*sizeof(char));
    strcat(strcat(strcat(dist_file,"./Input/"), argv[3]),".txt");

    char rsHRF_file[300]; memset(rsHRF_file, 0, sizeof(rsHRF_file));
    strcat(strcat(strcat(rsHRF_file,"./Input/"), argv[4]),".txt");
    
    int         maxdelay = importGlobalConnectivity(cap_file, dist_file, rsHRF_file, nodes, global_trans_v, HRF_samples, G_J_NMDA, model_dt, dt, &region_activity, &reg_globinp_p, &n_conn_table, &SC_cap, &SC_rowsums, &SC_inpreg, &rsHRF);
    
    int         reg_act_size = nodes * maxdelay;

    /*
     Initialize and/or cast to vector-intrinsics types for variables & int
    */
    float *J_i      = (float *)_mm_malloc(fake_nodes * sizeof(float),16); // (nA) inhibitory synaptic coupling
    float *BOLD_ex  = (float *)_mm_malloc(nodes * BOLD_TS_len * sizeof(float),16);
    
    if (J_i == NULL || BOLD_ex == NULL) 
    {
        printf( "ERROR: Running out of memory. Terminating... \n\n");
        _mm_free(J_i);_mm_free(BOLD_ex);
        exit(2);
    }

    // Initialize state variables / parameters
    for (j = 0; j < fake_nodes; j++) 
    {
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
    float           tmp_sigma           = sigma*dt;                         // pre-compute dt*sigma for the integration of sigma*randnumber in equations (9) and (10) of Deco2014
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
    for (i = 0; i < n_threads; ++i) 
    {
        thr_data[i].tid                 =	i                       ;
        thr_data[i].n_threads           =	n_threads               ;
        thr_data[i].vectorization_grade =   vectorization_grade     ;
        thr_data[i].nodes			    =	nodes			        ;
        thr_data[i].fake_nodes          =   fake_nodes              ;
        thr_data[i].J_i                 =	J_i            	        ;
        thr_data[i].reg_act_size	    =	reg_act_size	        ;
        thr_data[i].region_activity	    =	region_activity	        ;
        thr_data[i]._gamma			    =	_gamma			        ;
        thr_data[i]._one			    =	_one			        ;
        thr_data[i]._imintau_E		    =	_imintau_E		        ;
        thr_data[i]._dt                 =	_dt                     ;
        thr_data[i]._sigma_sqrt_dt      =   _sigma_sqrt_dt          ;
        thr_data[i]._sigma			    =	_sigma			        ;
        thr_data[i]._gamma_I		    =	_gamma_I		        ;
        thr_data[i]._imintau_I		    =	_imintau_I		        ;
        thr_data[i]._min_d_I		    =	_min_d_I		        ;
        thr_data[i]._b_I			    =	_b_I			        ;
        thr_data[i]._J_NMDA             =	_J_NMDA                 ;
        thr_data[i]._w_I__I_0		    =	_w_I__I_0		        ;
        thr_data[i]._a_I			    =	_a_I			        ;
        thr_data[i]._min_d_E		    =	_min_d_E		        ;
        thr_data[i]._b_E			    =	_b_E			        ;
        thr_data[i]._a_E			    =	_a_E			        ;
        thr_data[i]._w_plus_J_NMDA	    =	_w_plus_J_NMDA	        ;
        thr_data[i]._w_E__I_0		    =	_w_E__I_0		        ;
        thr_data[i].nodes_vec		    =	nodes_vec		        ;
        thr_data[i].SC_cap			    =	SC_cap			        ;
        thr_data[i].SC_inpreg           =	SC_inpreg               ;
        thr_data[i].rsHRF               =   rsHRF                   ;
        thr_data[i].reg_globinp_p	    =	reg_globinp_p	        ;
        thr_data[i].n_conn_table	    =	n_conn_table	        ;
        thr_data[i].BOLD_TR             =   BOLD_TR                 ;
        thr_data[i].BOLD_TS_len         =   BOLD_TS_len             ;
        thr_data[i].target_FR           =   target_FR               ;
        thr_data[i].total_duration      =   total_duration          ;
        thr_data[i].model_dt            =   model_dt                ;
        thr_data[i].stock_steps         =   stock_steps             ;
        thr_data[i].interim_istep       =   interim_istep           ;
        thr_data[i].BOLD_ex             =   BOLD_ex                 ;
        thr_data[i].rand_num_seed       =   rand_num_seed           ;
        thr_data[i].FIC_time            =   FIC_time                ;
        thr_data[i].HRF_samples         =   HRF_samples             ;
        thr_data[i].output_file         =   output_file             ;
        
        if ((rc = pthread_create(&thr[i], NULL, run_simulation, &thr_data[i]))) 
        {
            fprintf(stderr, "error: pthread_create, rc: %d\n", rc);
            return EXIT_FAILURE;
        }
    }
    
    /* block until all threads complete */
    for (i = 0; i < n_threads; ++i) 
    {
        pthread_join(thr[i], NULL);
    }
    printf("Threads finished. Back to main thread.\n"); 
    printf("MT-TVBii finished. Execution took %.2f s for %d nodes. Goodbye!\n", (float)(time(NULL) - start), nodes);

    return 0;
}