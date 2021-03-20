#ifndef BCFILL_H
#define BCFILL_H

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>
// #include <Kernels.H>

struct AmrCoreFill
{
    AMREX_GPU_DEVICE
    void operator() (const amrex::IntVect& iv, amrex::Array4<amrex::Real> const& data,
                     const int /*dcomp*/, const int numcomp,
                     amrex::GeometryData const& geom, const amrex::Real /*time*/,
                     const amrex::BCRec* bcr, const int /* bcomp */,
                     const int /*orig_comp*/) const
        {
            using namespace amrex;
            const Box& domain = geom.Domain();

            const auto domlo = amrex::lbound(domain);
            const auto domhi = amrex::ubound(domain);

            const int ilo = domlo.x;
            const int ihi = domhi.x;

#if AMREX_SPACEDIM >= 2
            const int jlo = domlo.y;
            const int jhi = domhi.y;
#if AMREX_SPACEDIM==2
            const int k = domlo.z;
#endif
#if AMREX_SPACEDIM==3
            const int k = iv[2];
            const int klo = domlo.z;
            const int khi = domhi.z;
#endif
#endif

    for (int n = 0; n < numcomp; ++n){
        Array4<Real> q(data,n);
        BCRec const& bc = bcr[n];
        if (iv[0] < ilo) {
            if (bc.lo(0) == BCType::ext_dir) {
                // Do nothing.
            } else if (bc.lo(0) == BCType::foextrap) {
                q(iv[0],iv[1],k) = q(ilo,iv[1],k);
            } else if (bc.lo(0) == BCType::reflect_even) {
                q(iv[0],iv[1],k) =  q(ilo+(ilo-iv[0])-1,iv[1],k);
            } else if (bc.lo(0) == BCType::reflect_odd) {
                q(iv[0],iv[1],k) = -q(ilo+(ilo-iv[0])-1,iv[1],k);
            }
        }
        if (iv[0] > ihi) {
            if (bc.hi(0) == BCType::ext_dir) {
                // Do nothing.
            } else if (bc.hi(0) == BCType::foextrap) {
                q(iv[0],iv[1],k) = q(ihi,iv[1],k);
            } else if (bc.hi(0) == BCType::reflect_even) {
                q(iv[0],iv[1],k) = q(ihi-(iv[0]-ihi)+1,iv[1],k);
            } else if (bc.hi(0) == BCType::reflect_odd) {
                q(iv[0],iv[1],k) = -q(ihi-(iv[0]-ihi)+1,iv[1],k);
            }
        }
#if AMREX_SPACEDIM >= 2
        if (iv[1] < jlo) {
            if (bc.lo(1) == BCType::ext_dir) {
                // Do nothing.
            } else if (bc.lo(1) == BCType::foextrap) {
                q(iv[0],iv[1],k) = q(iv[0],jlo,k);
            } else if (bc.lo(1) == BCType::reflect_even) {
                q(iv[0],iv[1],k) = q(iv[0],jlo+(jlo-iv[1])-1,k);
            } else if (bc.lo(1) == BCType::reflect_odd) {
                q(iv[0],iv[1],k) = -q(iv[0],jlo+(jlo-iv[1])-1,k);
            }           
        }

        if (iv[1] > jhi) {
            if (bc.hi(1) == BCType::ext_dir) {
                // Do nothing.
            } else if (bc.hi(1) == BCType::foextrap) {
                q(iv[0],iv[1],k) = q(iv[0],jhi,k);
            } else if (bc.hi(1) == BCType::reflect_even) {
                q(iv[0],iv[1],k) = q(iv[0],jhi-(iv[1]-jhi)+1,k);
            } else if (bc.hi(1) == BCType::reflect_odd) {
                q(iv[0],iv[1],k) = -q(iv[0],jhi-(iv[1]-jhi)+1,k);
            }
        }

#if AMREX_SPACEDIM==3
        if (k < klo) {
            if (bc.lo(2) == BCType::ext_dir) {
                // Do nothing.
            } else if (bc.lo(2) == BCType::foextrap) {
                q(iv[0],iv[1],k) = q(iv[0],iv[1],klo);
            } else if (bc.lo(2) == BCType::reflect_even) {
                q(iv[0],iv[1],k) = q(iv[0],iv[1],klo+(klo-k)-1);
            } else if (bc.lo(2) == BCType::reflect_odd) {
                q(iv[0],iv[1],k) = -q(iv[0],iv[1],klo+(klo-k)-1);
            }           
        }

        if (k > khi) {
            if (bc.hi(2) == BCType::ext_dir) {
                // Do nothing.
            } else if (bc.hi(2) == BCType::foextrap) {
                q(iv[0],iv[1],k) = q(iv[0],iv[1],khi);
            } else if (bc.hi(2) == BCType::reflect_even) {
                q(iv[0],iv[1],k) = q(iv[0],iv[1],khi-(k-khi)+1);
            } else if (bc.hi(2) == BCType::reflect_odd) {
                q(iv[0],iv[1],k) = -q(iv[0],iv[1],khi-(k-khi)+1);
            }
        }
#endif
#endif              
    }
//             // do something for external Dirichlet (BCType::ext_dir)
//             // amrex::Print() << "entered AmrCoreFill{}" << "\n";
        }
};

AMREX_FORCE_INLINE
void outletBC_partialwall (amrex::Box const& bx, amrex::Array4<amrex::Real> const& qn,
                            const int /*dcomp*/, const int numcomp,
                            amrex::GeometryData const& geom, const amrex::Real /*time*/,
                            const amrex::BCRec* bcr, const int /*bcomp*/,
                            const int /*orig_comp*/)
{
	using namespace amrex;

	amrex::Print() << "entered outletBC_partialwall()" << "\n";

    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);
    const Box qbx(qn);
    const auto qlo = amrex::lbound(qbx);
    const auto qhi = amrex::ubound(qbx);

    const Box& domain = geom.Domain();
    const auto domlo = amrex::lbound(domain);
    const auto domhi = amrex::ubound(domain);

    // const int is = std::max(qlo.x, domlo.x);
    // const int ie = std::min(qhi.x, domhi.x);
    const int ilo = domlo.x;
    const int ihi = domhi.x;

#if AMREX_SPACEDIM >= 2
    // const int js = std::max(qlo.y, domlo.y);
    // const int je = std::min(qhi.y, domhi.y);
    const int jlo = domlo.y;
    const int jhi = domhi.y;
#endif

#if AMREX_SPACEDIM == 3
    const int ks = std::max(qlo.z, domlo.z);
    const int ke = std::min(qhi.z, domhi.z);
    const int klo = domlo.z;
    const int khi = domhi.z;
#endif

	for (int n = 0; n < numcomp; ++n){
        Array4<Real> q(qn,n);
        BCRec const& bc = bcr[n];
        if (lo.x < ilo) {
           	const int imin = lo.x;
           	const int imax = ilo-1;
        	if (bc.lo(0) == BCType::ext_dir) {
            	// Do nothing.
        	} else if (bc.lo(0) == BCType::foextrap) {
               for (int k = lo.z; k <= hi.z; ++k) {
               for (int j = lo.y; j <= hi.y; ++j) {
               for (int i = imin; i <= imax; ++i) {
                   q(i,j,k) = q(ilo,j,k);
        	}}}
        	} else if (bc.lo(0) == BCType::reflect_even) {
               for (int k = lo.z; k <= hi.z; ++k) {
               for (int j = lo.y; j <= hi.y; ++j) {
               for (int i = imin; i <= imax; ++i) {
                   q(i,j,k) = q(ilo+(ilo-i)-1,j,k);
        	}}}
        	} else if (bc.lo(0) == BCType::reflect_odd) {
               for (int k = lo.z; k <= hi.z; ++k) {
               for (int j = lo.y; j <= hi.y; ++j) {
               for (int i = imin; i <= imax; ++i) {
                   q(i,j,k) = -q(ilo+(ilo-i)-1,j,k);
        	}}}
        	}
        }
        if (hi.x > ihi) {
            const int imin = ihi+1;
            const int imax = hi.x;
        	if (bc.hi(0) == BCType::ext_dir) {
        		// Do nothing.
        	} else if (bc.hi(0) == BCType::foextrap) {
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = imin; i <= imax; ++i) {
                    q(i,j,k) = q(ihi,j,k);
                }}}
            } else if (bc.hi(0) == BCType::reflect_even) {
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = imin; i <= imax; ++i) {
                    q(i,j,k) = q(ihi-(i-ihi)+1,j,k);
                }}}
            } else if (bc.hi(0) == BCType::reflect_odd) {
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = imin; i <= imax; ++i) {
                    q(i,j,k) = -q(ihi-(i-ihi)+1,j,k);
                }}}
            }
        }
#if AMREX_SPACEDIM >= 2
        if (lo.y < jlo) {
            const int jmin = lo.y;
            const int jmax = jlo-1;
            if (bc.lo(1) == BCType::ext_dir) {
                // Do nothing.
            } else if (bc.lo(1) == BCType::foextrap) {
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = jmin; j <= jmax; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    q(i,j,k) = q(i,jlo,k);
                }}}
            } else if (bc.lo(1) == BCType::reflect_even) {
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = jmin; j <= jmax; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    q(i,j,k) = q(i,jlo+(jlo-j)-1,k);
                }}}
            } else if (bc.lo(1) == BCType::reflect_odd) {
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = jmin; j <= jmax; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    q(i,j,k) = -q(i,jlo+(jlo-j)-1,k);
                }}}
            }         	
        }

        if (hi.y > jhi) {
            const int jmin = jhi+1;
            const int jmax = hi.y;
            if (bc.hi(1) == BCType::ext_dir) {
                // Do nothing.
        } else if (bc.hi(1) == BCType::foextrap) {
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = jmin; j <= jmax; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    q(i,j,k) = q(i,jhi,k);
                }}}
        } else if (bc.hi(1) == BCType::reflect_even) {
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = jmin; j <= jmax; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    q(i,j,k) = q(i,jhi-(j-jhi)+1,k);
                }}}
        } else if (bc.hi(1) == BCType::reflect_odd) {
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = jmin; j <= jmax; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    q(i,j,k) = -q(i,jhi-(j-jhi)+1,k);
                }}}
        }
    }
#endif        		
	}

}


// Function to fill physical domain boundary (fill ghost cells)
AMREX_FORCE_INLINE
void 
FillDomBoundary (amrex::MultiFab& phi, const amrex::Geometry& geom, const amrex::Vector<amrex::BCRec>& bc, amrex::Real cur_time)
{
    using namespace amrex;
    BL_PROFILE_VAR("FillDomainBoundary()", dombndry);
    // int myproc = ParallelDescriptor::MyProc();
    // Print(myproc) << "rank= " << myproc << ", entered LRFPFCT::FillDomainBoundary()" << "\n";
    if (geom.isAllPeriodic()) return;
    if (phi.nGrow() == 0) return;

    AMREX_ALWAYS_ASSERT(phi.ixType().cellCentered());

    // Print() << " entered FillDomBoundary() " << "\n";

#if !(defined(AMREX_USE_CUDA) && defined(AMREX_USE_GPU_PRAGMA) && defined(AMREX_GPU_PRAGMA_NO_HOST))
    if (Gpu::inLaunchRegion())
    {
#endif  
        GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
        PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > physbcf
            (geom, bc, gpu_bndry_func);
        physbcf(phi, 0, phi.nComp(), phi.nGrowVect(), cur_time, 0);
        // Print(myproc) << "rank= " << myproc << ", reached GpuBndryFuncFab()" << "\n";
#if !(defined(AMREX_USE_CUDA) && defined(AMREX_USE_GPU_PRAGMA) && defined(AMREX_GPU_PRAGMA_NO_HOST))
    }
    else
    {
        CpuBndryFuncFab cpu_bndry_func(nullptr);;
        PhysBCFunct<CpuBndryFuncFab> physbcf(geom, bc, cpu_bndry_func);
        physbcf(phi, 0, phi.nComp(), phi.nGrowVect(), cur_time, 0);
        // Print(myproc) << "rank= " << myproc << ", reached CpuBndryFuncFab()" << "\n";
    }
#endif
}
#endif
