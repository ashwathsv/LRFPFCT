#ifndef FACE_VELOCITY_H_
#define FACE_VELOCITY_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <fctconstants.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void get_face_velocity_psi(amrex::Box const& bx,
                           const amrex::Real time,
                           amrex::Array4<amrex::Real> const& psi,
                           amrex::GeometryData const& geomdata)
{
    using namespace amrex;
    constexpr Real PI = 3.1415926535897932384626;

    const auto lo  = lbound(bx);
    const auto hi  = ubound(bx);

    const Real* AMREX_RESTRICT prob_lo = geomdata.ProbLo();
    const Real* AMREX_RESTRICT dx      = geomdata.CellSize(); 

    for     (int j = lo.y; j <= hi.y; ++j) {
        Real y = dx[1]*(0.5+j) + prob_lo[1]; 
        AMREX_PRAGMA_SIMD
        for (int i = lo.x; i <= hi.x; ++i) {
            Real x = dx[0]*(0.5+i) + prob_lo[0];
            psi(i,j,0) = std::pow(std::sin(PI*x), 2) * std::pow(std::sin(PI*y), 2)
                       * std::cos(PI*time/2.0) * 1.0/PI;
        }
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void get_face_velocity_x_dt(int i, int j, int k,
                         amrex::Array4<amrex::Real> const& vx,
                         amrex::Array4<amrex::Real> const& fab)
{
    vx(i,j,k) = half*( ( sqrt(gamma*fab(i-1,j,k,pre)/fab(i-1,j,k,ro)) + fabs(fab(i-1,j,k,rou)/fab(i-1,j,k,ro)) ) 
              + ( sqrt(gamma*fab(i,j,k,pre)/fab(i,j,k,ro)) + fabs(fab(i,j,k,rou)/fab(i,j,k,ro)) ) );
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void get_face_velocity_y_dt(int i, int j, int k,
                         amrex::Array4<amrex::Real> const& vy,
                         amrex::Array4<amrex::Real> const& fab)
{
    vy(i,j,k) = half*( ( sqrt(gamma*fab(i,j-1,k,pre)/fab(i,j-1,k,ro)) + fabs(fab(i,j-1,k,rov)/fab(i,j-1,k,ro)) ) 
              + ( sqrt(gamma*fab(i,j,k,pre)/fab(i,j,k,ro)) + fabs(fab(i,j,k,rov)/fab(i,j,k,ro)) ) );
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void get_face_velocity_z_dt(int i, int j, int k,
                         amrex::Array4<amrex::Real> const& vz,
                         amrex::Array4<amrex::Real> const& fab)
{
#if AMREX_SPACEDIM==3
    vz(i,j,k) = half*( ( sqrt(gamma*fab(i,j,k-1,pre)/fab(i,j,k-1,ro)) + fabs(fab(i,j,k-1,row)/fab(i,j,k-1,ro)) ) 
              + ( sqrt(gamma*fab(i,j,k,pre)/fab(i,j,k,ro)) + fabs(fab(i,j,k,row)/fab(i,j,k,ro)) ) );
#endif
}

#endif
