#ifndef TAGGING_H
#define TAGGING_H

#include <AMReX_Array4.H>
#include <Kernels.H>

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
state_error (amrex::Box const& bx,
			 amrex::Array4<char> const& tag,
             amrex::Array4<amrex::Real const> const& state,
             AMREX_D_DECL(amrex::Real maxgradpx, amrex::Real maxgradpy, amrex::Real maxgradpz),
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx, 
             amrex::Real tagfrac, char tagval)
{
	const auto lo = lbound(bx);
    const auto hi = ubound(bx);
    AMREX_D_TERM(amrex::Real gpx_lim = tagfrac*maxgradpx;, 
                 amrex::Real gpy_lim = tagfrac*maxgradpy;,
                 amrex::Real gpz_lim = tagfrac*maxgradpz);
    
    for(int k = lo.z; k <= hi.z; ++k){
    	for(int j = lo.y; j <= hi.y; ++j){
    		for(int i = lo.x; i <= hi.x; ++i){
    			AMREX_D_TERM(amrex::Real gradpx = half*(state(i+1,j,k,pre) - state(i-1,j,k,pre))/dx[0];,
    			             amrex::Real gradpy = half*(state(i,j+1,k,pre) - state(i,j-1,k,pre))/dx[1];,
                             amrex::Real gradpz = half*(state(i,j,k+1,pre) - state(i,j,k-1,pre))/dx[2]);
#if AMREX_SPACEDIM==2
    			if(fabs(gradpx) >= gpx_lim || fabs(gradpy) >= gpy_lim)
#endif
#if AMREX_SPACEDIM==3
                if(fabs(gradpx) >= gpx_lim || fabs(gradpy) >= gpy_lim || fabs(gradpz) >= gpz_lim)   
#endif
    				tag(i,j,k) = tagval;
    		}
    	}
    }
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
get_gradp_x(amrex::Box const& bx, amrex::Array4<amrex::Real> const& phi,
         amrex::Array4<amrex::Real> const& gradp,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx)
{
    const auto lo = lbound(bx);
    const auto hi = ubound(bx);

    for(int k = lo.z; k <= hi.z; ++k){
        for(int j = lo.y; j <= hi.y; ++j){
            for(int i = lo.x; i <= hi.x; ++i){
                gradp(i,j,k,0) = half*(phi(i+1,j,k,pre) - phi(i-1,j,k,pre))/dx[0];
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
get_gradp_y(amrex::Box const& bx, amrex::Array4<amrex::Real> const& phi,
         amrex::Array4<amrex::Real> const& gradp,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx)
{
    const auto lo = lbound(bx);
    const auto hi = ubound(bx);

    for(int k = lo.z; k <= hi.z; ++k){
        for(int j = lo.y; j <= hi.y; ++j){
            for(int i = lo.x; i <= hi.x; ++i){
                gradp(i,j,k,1) = half*(phi(i,j+1,k,pre) - phi(i,j-1,k,pre))/dx[1];
            }
        }
    }
}

#if AMREX_SPACEDIM==3
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
get_gradp_z(amrex::Box const& bx, amrex::Array4<amrex::Real> const& phi,
         amrex::Array4<amrex::Real> const& gradp,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx)
{
    const auto lo = lbound(bx);
    const auto hi = ubound(bx);

    for(int k = lo.z; k <= hi.z; ++k){
        for(int j = lo.y; j <= hi.y; ++j){
            for(int i = lo.x; i <= hi.x; ++i){
                gradp(i,j,k,2) = half*(phi(i,j,k+1,pre) - phi(i,j,k-1,pre))/dx[2];
            }
        }
    }
}
#endif
#endif
