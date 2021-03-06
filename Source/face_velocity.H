#ifndef FACE_VELOCITY_H_
#define FACE_VELOCITY_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <fctconstants.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void get_face_velocity_x_dt(int i, int j, int k,
                         amrex::Array4<amrex::Real> const& vx,
                         amrex::Array4<amrex::Real> const& fab)
{
    vx(i,j,k) = half*( ( sqrt(gamma*fab(i-1,j,k,pre)/fab(i-1,j,k,ro)) + fabs(fab(i-1,j,k,rou)/fab(i-1,j,k,ro)) ) 
              + ( sqrt(gamma*fab(i,j,k,pre)/fab(i,j,k,ro)) + fabs(fab(i,j,k,rou)/fab(i,j,k,ro)) ) );
//	if (vx(i,j,k) < 0.0){
//		AMREX_ASSERT_WITH_MESSAGE(vx(i,j,k) > 0.0,"vx < 0.0 (face_velocity.H, L18)");	}
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void get_face_velocity_y_dt(int i, int j, int k,
                         amrex::Array4<amrex::Real> const& vy,
                         amrex::Array4<amrex::Real> const& fab)
{
    vy(i,j,k) = half*( ( sqrt(gamma*fab(i,j-1,k,pre)/fab(i,j-1,k,ro)) + fabs(fab(i,j-1,k,rov)/fab(i,j-1,k,ro)) ) 
              + ( sqrt(gamma*fab(i,j,k,pre)/fab(i,j,k,ro)) + fabs(fab(i,j,k,rov)/fab(i,j,k,ro)) ) );
//	if (vy(i,j,k) < 0.0){
//		AMREX_ASSERT_WITH_MESSAGE(vy(i,j,k) > 0.0,"vy < 0.0 (face_velocity.H, L30)");   }
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

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void get_face_velocity_x(int i, int j, int k,
                         amrex::Array4<amrex::Real> const& vx,
                         amrex::Array4<amrex::Real> const& fab,
                         int vx_lox, int vx_hix)
{
    if(i > vx_lox && i < vx_hix){
        vx(i,j,k) = half*( ( fab(i-1,j,k,rou)/fab(i-1,j,k,ro) ) + ( fab(i,j,k,rou)/fab(i,j,k,ro) ) );
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void get_face_velocity_y(int i, int j, int k,
                         amrex::Array4<amrex::Real> const& vy,
                         amrex::Array4<amrex::Real> const& fab,
                         int vy_loy, int vy_hiy)
{
    if(j > vy_loy && j < vy_hiy){
        vy(i,j,k) = half*( ( fab(i,j-1,k,rov)/fab(i,j-1,k,ro) ) + ( fab(i,j,k,rov)/fab(i,j,k,ro) ) );
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void get_face_velocity_z(int i, int j, int k,
                         amrex::Array4<amrex::Real> const& vz,
                         amrex::Array4<amrex::Real> const& fab,
                         int vz_loz, int vz_hiz)
{
#if AMREX_SPACEDIM==3
    if(k > vz_loz && k < vz_hiz){
        vz(i,j,k) = half*( ( fab(i,j,k-1,row)/fab(i,j,k-1,ro) ) + ( fab(i,j,k,row)/fab(i,j,k,ro) ) );
    }
#endif
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void get_face_velocity_x_bc(int i, int j, int k,
                         amrex::Array4<amrex::Real> const& vx,
                         // amrex::Array4<amrex::Real> const& fab,
                         int vx_lox, int vx_hix)
{
    if(i == vx_lox){
        vx(i,j,k) = vx(i+1,j,k);
    }else if(i == vx_hix){
        vx(i,j,k) = vx(i-1,j,k);    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void get_face_velocity_y_bc(int i, int j, int k,
                         amrex::Array4<amrex::Real> const& vy,
                         // amrex::Array4<amrex::Real> const& fab,
                         int vy_loy, int vy_hiy)
{
    if(j == vy_loy){
        vy(i,j,k) = vy(i,j+1,k);
    }else if(j == vy_hiy){
        vy(i,j,k) = vy(i,j-1,k);    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void get_face_velocity_z_bc(int i, int j, int k,
                         amrex::Array4<amrex::Real> const& vz,
                         // amrex::Array4<amrex::Real> const& fab,
                         int vz_loz, int vz_hiz)
{
    if(k == vz_loz){
        vz(i,j,k) = vz(i,j,k+1); 
    }else if(k == vz_hiz){
        vz(i,j,k) = vz(i,j,k-1);    }
}

#endif
