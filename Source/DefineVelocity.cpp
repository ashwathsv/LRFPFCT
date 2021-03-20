#include <LRFPFCT.H>
#include <Kernels.H>

#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void
LRFPFCT::DefineVelocityAllLevels (Real time)
{
    for (int lev = 0; lev <= finest_level; ++lev)
        DefineVelocityAtLevel(lev,time);
}

void
LRFPFCT::DefineVelocityAtLevel (int lev, Real /*time*/)
{
    MultiFab& state = phi_new[lev];
    const int ngrow = nghost;

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {

        // ======== GET FACE VELOCITY =========
            GpuArray<Box, AMREX_SPACEDIM> nbx;
            AMREX_D_TERM(nbx[0] = mfi.nodaltilebox(0);,
                         nbx[1] = mfi.nodaltilebox(1);,
                         nbx[2] = mfi.nodaltilebox(2););

            AMREX_D_TERM(const Box& ngbxx = amrex::grow(mfi.nodaltilebox(0),ngrow);,
                         const Box& ngbxy = amrex::grow(mfi.nodaltilebox(1),ngrow);,
                         const Box& ngbxz = amrex::grow(mfi.nodaltilebox(2),ngrow));
            
            // Print(myproc) << "rank= " << myproc << "lo(x)= " << ngbxx.smallEnd(0) << 
            // " " << ngbxx.smallEnd(1) << ", hi(x)= " << ngbxx.bigEnd(0) << " " << ngbxx.bigEnd(1) << "\n";

            GpuArray<Array4<Real>, AMREX_SPACEDIM> vel{ AMREX_D_DECL( facevel[lev][0].array(mfi),
                                                                      facevel[lev][1].array(mfi),
                                                                      facevel[lev][2].array(mfi)) };

            int vx_lox = lbound(vel[0]).x; 
            int vx_hix = ubound(vel[0]).x;

            int vy_loy = lbound(vel[1]).y;
            int vy_hiy = ubound(vel[1]).y;

#if AMREX_SPACEDIM==3
            int vz_loz = lbound(vel[2]).z;
            int vz_hiz = ubound(vel[2]).z;            
#endif
            // Print(myproc) << "rank= " << myproc << "lo(velx)= " << lo.x << " " << lo.y << ", hi(velx)= " << hi.x << " " << hi.y << "\n";
            // Print(myproc) << "rank= " << myproc << "lo(vely)= " << lbound(vel[1]) << ", hi(vely)= " << ubound(vel[1]) << "\n";

            Array4<Real> fab = state[mfi].array();
            // GeometryData geomdata = geom[lev].data();

            amrex::ParallelFor(AMREX_D_DECL(ngbxx,ngbxy,ngbxz),
                 AMREX_D_DECL(
                     [=] AMREX_GPU_DEVICE (int i, int j, int k)
                     {  get_face_velocity_x(i, j, k, vel[0], fab, vx_lox, vx_hix);  },
                     [=] AMREX_GPU_DEVICE (int i, int j, int k)
                     {  get_face_velocity_y(i, j, k, vel[1], fab, vy_loy, vy_hiy);  },
                     [=] AMREX_GPU_DEVICE (int i, int j, int k)
                     {  get_face_velocity_z(i, j, k, vel[2], fab, vz_loz, vz_hiz);  }));

            amrex::ParallelFor(ngbxx,
                     [=] AMREX_GPU_DEVICE (int i, int j, int k)
                     {  get_face_velocity_x_bc(i, j, k, vel[0], vx_lox, vx_hix);  });

            amrex::ParallelFor(ngbxy,
                     [=] AMREX_GPU_DEVICE (int i, int j, int k)
                     {  get_face_velocity_y_bc(i, j, k, vel[1], vy_loy, vy_hiy);  });
#if AMREX_SPACEDIM==3
            amrex::ParallelFor(ngbxz,
                     [=] AMREX_GPU_DEVICE (int i, int j, int k)
                     {  get_face_velocity_z_bc(i, j, k, vel[2], vz_loz, vz_hiz);  });
#endif
        }
    }
}

void
LRFPFCT::DefineVelocityAtLevelDt (int lev, Real /*time*/)
{
    MultiFab& state = phi_new[lev];

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        FArrayBox tmpfab;
        for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {

        // ======== GET FACE VELOCITY =========
            GpuArray<Box, AMREX_SPACEDIM> nbx;
            AMREX_D_TERM(nbx[0] = mfi.nodaltilebox(0);,
                         nbx[1] = mfi.nodaltilebox(1);,
                         nbx[2] = mfi.nodaltilebox(2););

            AMREX_D_TERM(const Box& ngbxx = nbx[0];,
                         const Box& ngbxy = nbx[1];,
                         const Box& ngbxz = nbx[2];);

            GpuArray<Array4<Real>, AMREX_SPACEDIM> vel{ AMREX_D_DECL( facevel[lev][0].array(mfi),
                                                                      facevel[lev][1].array(mfi),
                                                                      facevel[lev][2].array(mfi)) };

            Array4<Real> fab = state[mfi].array();

            amrex::ParallelFor
                (AMREX_D_DECL(ngbxx,ngbxy,ngbxz),
                 AMREX_D_DECL(
                     [=] AMREX_GPU_DEVICE (int i, int j, int k)
                     {
                         get_face_velocity_x_dt(i, j, k, vel[0], fab);
                     },
                     [=] AMREX_GPU_DEVICE (int i, int j, int k)
                     {
                         get_face_velocity_y_dt(i, j, k, vel[1], fab);
                     },
                     [=] AMREX_GPU_DEVICE (int i, int j, int k)
                     {
                         get_face_velocity_z_dt(i, j, k, vel[2], fab);
                     }));

        }
    }
//for(int idim = 0; idim < AMREX_SPACEDIM; ++idim){
//	AMREX_ASSERT_WITH_MESSAGE(facevel[lev][idim].min(0,0,true) >= 0.0, "min < 0.0, aborting (DefineVelocityDT()");
//}
ParallelDescriptor::Barrier();
}
