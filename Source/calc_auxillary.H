#ifndef CALC_AUXILLARY_H_
#define CALC_AUXILLARY_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <fctconstants.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void CalcAuxillary(amrex::Box const& bx, amrex::Array4<amrex::Real> const& state, amrex::Real pmin)
{
    using namespace amrex;
    const auto lo = lbound(bx);
    const auto hi = ubound(bx);
    for(int k = lo.z; k <= hi.z; ++k){
        for(int j = lo.y; j <= hi.y; ++j){
            for(int i = lo.x; i <= hi.x; ++i){
#if AMREX_SPACEDIM==2
                Real velmod =  sqrt( pow(state(i,j,k,rou),2.0) + pow(state(i,j,k,rov),2.0) )/state(i,j,k,ro); 
#endif
#if AMREX_SPACEDIM==3
                Real velmod =  sqrt( pow(state(i,j,k,rou),2.0) + pow(state(i,j,k,rov),2.0) + pow(state(i,j,k,row),2.0) )/state(i,j,k,ro);
#endif
                state(i,j,k,pre) = (gamma-1)*( state(i,j,k,roE) -  half*(state(i,j,k,ro)*pow(velmod,2.0)));
                if(state(i,j,k,pre) < pmin){
                    state(i,j,k,pre) = pmin;
                    state(i,j,k,roE) = c1*state(i,j,k,roE) + half*(state(i,j,k,ro)*pow(velmod,2.0));
#if AMREX_SPACEDIM==2
                    velmod =  sqrt( pow(state(i,j,k,rou),2.0) + pow(state(i,j,k,rov),2.0) )/state(i,j,k,ro); 
#endif
#if AMREX_SPACEDIM==3
                    velmod =  sqrt( pow(state(i,j,k,rou),2.0) + pow(state(i,j,k,rov),2.0) + pow(state(i,j,k,row),2.0) )/state(i,j,k,ro);
#endif
                }
                Real ss = sqrt(gamma*state(i,j,k,pre)/state(i,j,k,ro));
                state(i,j,k,mac) = velmod/ss;
            }
        }
    }

}

#endif