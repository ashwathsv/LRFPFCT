#ifndef PROB_H_
#define PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <fctconstants.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
initdata(amrex::Box const& bx, amrex::Array4<amrex::Real> const& phi,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_hi,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
         int probtag, amrex::Real ro2, amrex::Real ro1, amrex::Real p2, amrex::Real p1,
         AMREX_D_DECL(amrex::Real u2, amrex::Real v2, amrex::Real w2),
         AMREX_D_DECL(amrex::Real u1, amrex::Real v1, amrex::Real w1), 
         AMREX_D_DECL(amrex::Real xcm, amrex::Real ycm, amrex::Real zcm),
         amrex::Real rad_bw)
{
    using namespace amrex;


    const auto lo = lbound(bx);
    const auto hi = ubound(bx);
   
    // int myproc = ParallelDescriptor::MyProc();
    // Print(myproc) << "rank= " << myproc << ", lo= " << lo << ", hi= " << hi << "\n";                   

    if(probtag == 1){
        AMREX_D_TERM(xcm = prob_lo[0];, 
                     ycm = 0.5*(prob_lo[1] + prob_hi[1]);,
                     zcm = 0.5*(prob_lo[2] + prob_hi[2]));
        }
    for         (int k = lo.z; k <= hi.z; ++k) {
        for     (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                AMREX_D_TERM(Real x = prob_lo[0] + (0.5+i) * dx[0];,
                             Real y = prob_lo[1] + (0.5+j) * dx[1];,
                             Real z = prob_lo[2] + (0.5+k) * dx[2]);

                Real dist = 0.0;
                if(probtag <= 5){
#if AMREX_SPACEDIM==2
                    dist = std::pow(x-xcm,2.0) + std::pow(y-ycm,2.0) - std::pow(rad_bw,2.0);
#endif
#if AMREX_SPACEDIM==3
                    dist = std::pow(x-xcm,2.0) + std::pow(y-ycm,2.0) + std::pow(z-zcm,2.0) - std::pow(rad_bw,2.0);
#endif
                }
                if(dist <= 0.0){
                    phi(i,j,k,ro) = ro2;
                    AMREX_D_TERM(phi(i,j,k,rou) = ro2*(u2);, 
                                 phi(i,j,k,rov) = ro2*(v2);, 
                                 phi(i,j,k,row) = ro2*(w2));                    
                    phi(i,j,k,pre) = p2;
                }else{
                    phi(i,j,k,ro) = ro1;
                    AMREX_D_TERM(phi(i,j,k,rou) = ro1*(u1);, 
                                 phi(i,j,k,rov) = ro1*(v1);, 
                                 phi(i,j,k,row) = ro1*(w1));
                    phi(i,j,k,pre) = p1;
                }
                Real ss = std::sqrt(gamma*phi(i,j,k,pre)/phi(i,j,k,ro));
#if AMREX_SPACEDIM==2            
            phi(i,j,k,roE) = phi(i,j,k,pre)*c1 + 0.5*( ( std::pow(phi(i,j,k,rou),2.0) 
                   + std::pow(phi(i,j,k,rov),2.0) )/phi(i,j,k,ro) );
            Real velmod = std::sqrt( std::pow(phi(i,j,k,rou)/phi(i,j,k,ro),2.0) 
                        + std::pow(phi(i,j,k,rov)/phi(i,j,k,ro),2.0) );
#endif
#if AMREX_SPACEDIM==3            
            phi(i,j,k,roE) = phi(i,j,k,pre)*c1 + 0.5*( ( std::pow(phi(i,j,k,rou),2.0) 
                   + std::pow(phi(i,j,k,rov),2.0) + std::pow(phi(i,j,k,row), 2.0) )/phi(i,j,k,ro) );
            Real velmod = std::sqrt( std::pow(phi(i,j,k,rou)/phi(i,j,k,ro),2.0) 
                        + std::pow(phi(i,j,k,rov)/phi(i,j,k,ro),2.0)
                        + std::pow(phi(i,j,k,row)/phi(i,j,k,ro),2.0) );
#endif
            phi(i,j,k,mac) = velmod/ss;

            // Print() << "i= " << i << ", j " << j << ", ro= " << phi(i,j,k,ro) << ", rou= " << phi(i,j,k,rou)
            // << ", rov= " << phi(i,j,k,rov) << ", roE= " << phi(i,j,k,roE) << ", mac= " << phi(i,j,k,mac) << "\n";
            }
        }
    }
}

#endif
