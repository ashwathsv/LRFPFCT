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
LRFPFCT::DefineVelocityAtLevel (int lev, Real time)
{
    int myproc = ParallelDescriptor::MyProc();
    // Print(myproc) << "rank= " << myproc << ", reached DefineVelocityAtLevel()" << "\n";
    const auto dx = geom[lev].CellSizeArray();
    MultiFab& state = phi_new[lev];

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

            AMREX_D_TERM(const Box& ngbxx = amrex::grow(mfi.nodaltilebox(0),1);,
                         const Box& ngbxy = amrex::grow(mfi.nodaltilebox(1),1);,
                         const Box& ngbxz = amrex::grow(mfi.nodaltilebox(2),1););
            
            Print(myproc) << "rank= " << myproc << "lo(x)= " << ngbxx.smallEnd(0) << 
            " " << ngbxx.smallEnd(1) << ", hi(x)= " << ngbxx.bigEnd(0) << " " << ngbxx.bigEnd(1) << "\n";

            GpuArray<Array4<Real>, AMREX_SPACEDIM> vel{ AMREX_D_DECL( facevel[lev][0].array(mfi),
                                                                      facevel[lev][1].array(mfi),
                                                                      facevel[lev][2].array(mfi)) };

            // const auto lo = lbound(vel[0]);
            // const auto hi = ubound(vel[0]);
            // Print(myproc) << "rank= " << myproc << "lo(velx)= " << lo.x << " " << lo.y << ", hi(velx)= " << hi.x << " " << hi.y << "\n";
            // Print(myproc) << "rank= " << myproc << "lo(vely)= " << lbound(vel[1]) << ", hi(vely)= " << ubound(vel[1]) << "\n";

            Array4<Real> fab = state[mfi].array();
            GeometryData geomdata = geom[lev].data();

            // amrex::ParallelFor
            //     (AMREX_D_DECL(ngbxx,ngbxy,ngbxz),
            //      AMREX_D_DECL(
            //          [=] AMREX_GPU_DEVICE (int i, int j, int k)
            //          {
            //              get_face_velocity_x(i, j, k, vel[0], fab, dx[0]);
            //          },
            //          [=] AMREX_GPU_DEVICE (int i, int j, int k)
            //          {
            //              get_face_velocity_y(i, j, k, vel[1], fab, dx[1]);
            //          },
            //          [=] AMREX_GPU_DEVICE (int i, int j, int k)
            //          {
            //              get_face_velocity_z(i, j, k, vel[2]);
            //          }));
        }
    }
}

void
LRFPFCT::DefineVelocityAtLevelDt (int lev, Real time)
{
    int myproc = ParallelDescriptor::MyProc();
    // Print(myproc) << "rank= " << myproc << ", reached DefineVelocityAtLevel()" << "\n";
    const auto dx = geom[lev].CellSizeArray();
    MultiFab& state = phi_new[lev];

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

            AMREX_D_TERM(const Box& ngbxx = nbx[0];,
                         const Box& ngbxy = nbx[1];,
                         const Box& ngbxz = nbx[2];);

            // Print(myproc) << "rank= " << myproc << "lo(y)= " << lbound(ngbxy) << ", hi(y)= " << ubound(ngbxy) << " " << "\n";

            // Print(myproc) << "rank= " << myproc << "lo(y)= " << ngbxy.smallEnd(0) << 
            // " " << ngbxy.smallEnd(1) << ", hi(y)= " << ngbxy.bigEnd(0) << " " << ngbxy.bigEnd(1) << "\n";

            GpuArray<Array4<Real>, AMREX_SPACEDIM> vel{ AMREX_D_DECL( facevel[lev][0].array(mfi),
                                                                      facevel[lev][1].array(mfi),
                                                                      facevel[lev][2].array(mfi)) };

            // Print(myproc) << "rank= " << myproc << "lo(velx)= " << lo.x << " " << lo.y << ", hi(velx)= " << hi.x << " " << hi.y << "\n";
            // Print(myproc) << "rank= " << myproc << "lo(vely)= " << lbound(vel[1]) << ", hi(vely)= " << ubound(vel[1]) << "\n";

            Array4<Real> fab = state[mfi].array();
            GeometryData geomdata = geom[lev].data();

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
}
