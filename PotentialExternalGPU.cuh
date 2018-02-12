// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

#include <assert.h>

/*! \file PotentialExternalGPU.cuh
    \brief Defines templated GPU kernel code for calculating the external forces.
*/

#ifndef __POTENTIAL_EXTERNAL_GPU_CUH__
#define __POTENTIAL_EXTERNAL_GPU_CUH__

//! Wraps arguments to gpu_cpef
struct external_potential_args_t
    {
    //! Construct a external_potential_args_t
    external_potential_args_t(Scalar4 *_d_force,
              Scalar *_d_virial,
              const unsigned int _virial_pitch,
              const unsigned int _N,
              const Scalar4 *_d_pos,
              const Scalar *_d_diameter,
              const Scalar *_d_charge,
              const BoxDim& _box,
              const unsigned int _block_size)
                : d_force(_d_force),
                  d_virial(_d_virial),
                  virial_pitch(_virial_pitch),
                  box(_box),
                  N(_N),
                  d_pos(_d_pos),
                  d_diameter(_d_diameter),
                  d_charge(_d_charge),
                  block_size(_block_size)
        {
        };

    Scalar4 *d_force;                //!< Force to write out
    Scalar *d_virial;                //!< Virial to write out
    const unsigned int virial_pitch; //!< The pitch of the 2D array of virial matrix elements
    const BoxDim& box;         //!< Simulation box in GPU format
    const unsigned int N;           //!< Number of particles
    const Scalar4 *d_pos;           //!< Device array of particle positions
    const Scalar *d_diameter;       //!< particle diameters
    const Scalar *d_charge;         //!< particle charges
    const unsigned int block_size;  //!< Block size to execute
    };

//! Driver function for compute external field kernel
/*!
 * \param external_potential_args External potential parameters
 * \param d_params External evaluator parameters
 * \param d_field External field parameters
 * \tparam Evaluator functor
 */
template< class evaluator >
cudaError_t gpu_cpef(const external_potential_args_t& external_potential_args,
                     const typename evaluator::param_type *d_params,
                     const typename evaluator::field_type *d_field);

#ifdef NVCC

#if (__CUDA_ARCH__ >= 300)
// need this wrapper here for CUDA toolkit versions (<6.5) which do not provide a
// double specialization
__device__ inline
double __my_shfl_down(double var, unsigned int srcLane, int width=32)
    {
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, srcLane, width);
    a.y = __shfl_down(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
    }

__device__ inline
float __my_shfl_down(float var, unsigned int srcLane, int width=32)
    {
    return __shfl_down(var, srcLane, width);
    }
#endif

//! CTA reduce, returns result in first thread
template<typename T>
__device__ static T warp_reduce(unsigned int NT, int tid, T x, volatile T* shared)
    {
    #if (__CUDA_ARCH__ < 300)
    shared[tid] = x;
    __syncthreads();
    #endif

    for (int dest_count = NT/2; dest_count >= 1; dest_count /= 2)
        {
        #if (__CUDA_ARCH__ < 300)
        if (tid < dest_count)
            {
            shared[tid] += shared[dest_count + tid];
            }
        __syncthreads();
        #else
        x += __my_shfl_down(x, dest_count, NT);
        #endif
        }

    #if (__CUDA_ARCH__ < 300)
    T total;
    if (tid == 0)
        {
        total = shared[0];
        }
    __syncthreads();
    return total;
    #else
    return x;
    #endif
    }


//! Kernel for calculating external forces
/*! This kernel is called to calculate the external forces on all N particles. Actual evaluation of the potentials and
    forces for each particle is handled via the template class \a evaluator.

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos device array of particle positions
    \param box Box dimensions used to implement periodic boundary conditions
    \param params per-type array of parameters for the potential

*/
template< class evaluator >
__global__ void gpu_compute_external_forces_kernel(Scalar4 *d_force,
                                               Scalar *d_virial,
                                               const unsigned int virial_pitch,
                                               const unsigned int N,
                                               const Scalar4 *d_pos,
                                               const Scalar *d_diameter,
                                               const Scalar *d_charge,
                                               const BoxDim box,
                                               const typename evaluator::param_type *params,
                                               const typename evaluator::field_type *d_field)
    {
    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // read in field data cooperatively
    extern __shared__ char s_data[];
    typename evaluator::field_type *s_field = (typename evaluator::field_type *)(&s_data[0]);

        {
        unsigned int tidx = threadIdx.x;
        unsigned int block_size = blockDim.x;
        unsigned int field_size = sizeof(typename evaluator::field_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < field_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < field_size)
                {
                ((int *)s_field)[cur_offset + tidx] = ((int *)d_field)[cur_offset + tidx];
                }
            }
        }
    const typename evaluator::field_type& field = *s_field;

    __syncthreads();

    bool active = true;
    if (idx >= N)
       active = false;


    // initialize the force to 0
    Scalar3 force = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar virial[6];
    for (unsigned int k = 0; k < 6; k++)
        virial[k] = Scalar(0.0);
    Scalar energy = Scalar(0.0);

    if (active)
        {
        // read in the position of our particle.
        // (MEM TRANSFER: 16 bytes)
        Scalar4 posi = d_pos[idx];
        Scalar di;
        Scalar qi;
        if (evaluator::needsDiameter())
            di = d_diameter[idx];
        else
            di += Scalar(1.0); // shutup compiler warning

        if (evaluator::needsCharge())
            qi = d_charge[idx];
        else
            qi = Scalar(0.0); // shutup compiler warning



        unsigned int typei = __scalar_as_int(posi.w);
        Scalar3 Xi = make_scalar3(posi.x, posi.y, posi.z);
        evaluator eval(Xi, box, params[typei], field);

        if (evaluator::needsDiameter())
            eval.setDiameter(di);
        if (evaluator::needsCharge())
            eval.setCharge(qi);

        eval.evalForceEnergyAndVirial(force, energy, virial);

        // now that the force calculation is complete, write out the result)
        d_force[idx].x = force.x;
        d_force[idx].y = force.y;
        d_force[idx].z = force.z;
        d_force[idx].w = energy;

        for (unsigned int k = 0; k < 6; k++)
            d_virial[k*virial_pitch+idx] = virial[k];
       }

   // Values to stores warp_reduced forces acting on External Partile
    unsigned int tpp = 32;
    Scalar4  f_force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

   // we need to access a separate portion of shared memory to avoid race conditions
    const unsigned int shared_bytes =(sizeof(typename evaluator::field_type)/sizeof(int)+1)*sizeof(int) ;

    // need to declare as volatile, because we are using warp-synchronous programming
    volatile Scalar *sh = (Scalar *) &s_data[shared_bytes];

    unsigned int cta_offs = (threadIdx.x/tpp)*tpp;

    // reduce force over threads in cta

    f_force.x = warp_reduce(tpp, threadIdx.x % tpp, force.x , &sh[cta_offs]);
    f_force.y = warp_reduce(tpp, threadIdx.x % tpp, force.y , &sh[cta_offs]);
    f_force.z = warp_reduce(tpp, threadIdx.x % tpp, force.z , &sh[cta_offs]);
    f_force.w  = warp_reduce(tpp, threadIdx.x % tpp, energy, &sh[cta_offs]);


    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    if (field.tag >=0)
       {
       if (active && threadIdx.x % tpp == 0)
          {
          Scalar old_force_x = atomicAdd(&(d_force[field.tag].x),-f_force.x);
          Scalar old_force_y = atomicAdd(&(d_force[field.tag].y),-f_force.y);
          Scalar old_force_z = atomicAdd(&(d_force[field.tag].z),-f_force.z);
          Scalar old_force_w = atomicAdd(&(d_force[field.tag].w),-f_force.w);
          }
       }

    }

/*!
 * This implements the templated kernel driver. The template must be explicitly
 * instantiated per potential in a cu file.
 */
template< class evaluator >
cudaError_t gpu_cpef(const external_potential_args_t& external_potential_args,
                     const typename evaluator::param_type *d_params,
                     const typename evaluator::field_type *d_field)
    {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, gpu_compute_external_forces_kernel<evaluator>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        unsigned int run_block_size = min(external_potential_args.block_size, max_block_size);
        run_block_size -= run_block_size % 32;

        // setup the grid to run the kernel
        dim3 grid( external_potential_args.N / run_block_size + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);
        unsigned int bytes = (sizeof(typename evaluator::field_type)/sizeof(int)+1)*sizeof(int) + run_block_size * sizeof(Scalar);

        // run the kernel
        gpu_compute_external_forces_kernel<evaluator><<<grid, threads, bytes>>>(external_potential_args.d_force,
                                                                                external_potential_args.d_virial,
                                                                                external_potential_args.virial_pitch,
                                                                                external_potential_args.N,
                                                                                external_potential_args.d_pos,
                                                                                external_potential_args.d_diameter,
                                                                                external_potential_args.d_charge,
                                                                                external_potential_args.box,
                                                                                d_params,
                                                                                d_field);

        return cudaSuccess;
    };
#endif // NVCC
#endif // __POTENTIAL_PAIR_GPU_CUH__
