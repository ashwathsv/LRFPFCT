
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_BCUtil.H>

#ifdef AMREX_MEM_PROFILING
#include <AMReX_MemProfiler.H>
#endif

#include <LRFPFCT.H>
#include <Kernels.H>
#include <bc_fill.H>

using namespace amrex;

// constructor - reads in parameters from inputs file
//             - sizes multilevel arrays and data structures
//             - initializes BCRec boundary condition object
LRFPFCT::LRFPFCT ()
{
#ifdef _OPENMP
    #pragma omp parallel
    #pragma omp master
    {
        Print(0) << "Running with " <<  omp_get_num_threads() << " thread(s)\n";
    }
#endif
    ReadParameters();

    // Geometry on all levels has been defined already.

    // No valid BoxArray and DistributionMapping have been defined.
    // But the arrays for them have been resized.

    int nlevs_max = max_level + 1;

    istep.resize(nlevs_max, 0);
    nsubsteps.resize(nlevs_max, 1);
    if (do_subcycle) {
        for (int lev = 1; lev <= max_level; ++lev) {
            nsubsteps[lev] = MaxRefRatio(lev-1);
            if(do_fixeddt){
                nsubsteps[lev] = 1;
            }
        }
    }

    t_new.resize(nlevs_max, 0.0);
    t_old.resize(nlevs_max, -1.e100);
    dt.resize(nlevs_max, 1.e100);

    phi_new.resize(nlevs_max);
    phi_old.resize(nlevs_max);

    facevel.resize(nlevs_max);

    bcs.resize(ncomp);

#if AMREX_SPACEDIM>=2
    if(probtag == 1){
        for(int n = 0; n < ncomp; ++n){
            bcs[n].setLo(0, BCType::reflect_even);
            bcs[n].setHi(0, BCType::foextrap);
            for(int i = 1; i < AMREX_SPACEDIM; ++i){
                bcs[n].setLo(i, BCType::foextrap);
                bcs[n].setHi(i, BCType::foextrap);
            }
        }
        bcs[rou].setLo(0, BCType::reflect_odd);
    }else if(probtag == 2){
        for(int n = 0; n < ncomp; ++n){
            bcs[n].setLo(0, BCType::reflect_even);
            bcs[n].setLo(1, BCType::reflect_even);
            bcs[n].setHi(0, BCType::reflect_even);
            bcs[n].setHi(1, BCType::reflect_even);            
        }
        bcs[rou].setLo(0, BCType::reflect_odd);
        bcs[rou].setHi(0, BCType::reflect_odd);
        bcs[rov].setLo(1, BCType::reflect_odd);
        bcs[rov].setHi(1, BCType::reflect_odd);
    }
#endif

    // stores fluxes at coarse-fine interface for synchronization
    // this will be sized "nlevs_max+1"
    // NOTE: the flux register associated with flux_reg[lev] is associated
    // with the lev/lev-1 interface (and has grid spacing associated with lev-1)
    // therefore flux_reg[0] is never actually used in the reflux operation
    flux_reg.resize(nlevs_max+1);

	ParallelDescriptor::Barrier();
}

LRFPFCT::~LRFPFCT ()
{
}

// read in some parameters from inputs file
void
LRFPFCT::ReadParameters ()
{
    {
    ParmParse pp;  // Traditionally, max_step and stop_time do not have prefix.
    pp.query("max_step", max_step);
    pp.query("stop_time", stop_time);
    pp.query("max_rk", max_rk);
    }

    {
    ParmParse pp("amr"); // Traditionally, these have prefix, amr.

    pp.query("regrid_int", regrid_int);
    pp.query("plot_file", plot_file);
    pp.query("plot_int", plot_int);
    pp.query("chk_file", chk_file);
    pp.query("chk_int", chk_int);
    pp.query("restart",restart_chkfile);
    }

    {
    ParmParse pp("adv");

    pp.query("cfl", cfl);
    pp.query("do_reflux", do_reflux);
    pp.query("do_subcycle", do_subcycle);
    pp.query("do_fixeddt", do_fixeddt);
    pp.query("diff1", diff1);
    }

    {
        ParmParse pp("amr");

        pp.get("allow_level",lev_allow);
    }

    {
        ParmParse pp("prob");

        pp.get("probtag",probtag);
    }

#if AMREX_SPACEDIM==2
    {
        ParmParse pp("aux");
        
        pp.query("nprobes",nprobes);
        if(nprobes > 0){
            pp.getarr("iprobes", iprobe);
            pp.getarr("jprobes", jprobe);
            if(iprobe.size() != nprobes || jprobe.size() != nprobes){
                amrex::Error("Coorodinate vector of probes does not equal number of probes");
            }
        }               
    }
#endif
#if AMREX_SPACEDIM==3
    {
        ParmParse pp("aux");
        
        pp.query("nprobes",nprobes);
        if(nprobes > 0){
            pp.getarr("iprobes", iprobe);
            pp.getarr("jprobes", jprobe);
            pp.getarr("kprobes", kprobe);
            if(iprobe.size() != nprobes || jprobe.size() != nprobes || kprobe.size() != nprobes){
                amrex::Error("Coorodinate vector of probes does not equal number of probes");
            }
        }               
    }
#endif

}

// initializes multilevel data
void
LRFPFCT::InitData ()
{
    // int myproc = ParallelDescriptor::MyProc();
    if (restart_chkfile == "") {
        // start simulation from the beginning
        const Real time = 0.0;
        InitFromScratch(time);
        // ParallelDescriptor::Barrier();
        AverageDown();
        ParallelDescriptor::Barrier();

        if (chk_int > 0) {
            WriteCheckpointFile();
        }

        if (plot_int > 0) {
            WritePlotFile();
        }

        if(nprobes > 0){
            // LRFPFCT::GetProbeDets();
            // LRFPFCT::WriteProbeFile(0, 0.0, 0);
        }
    }
    else {
        // restart from a checkpoint
        ReadCheckpointFile();

        if(nprobes > 0){
            // LRFPFCT::GetProbeDets();
        }
    }

    ParallelDescriptor::Barrier();
}

// Make a new level using provided BoxArray and DistributionMapping and 
// fill with interpolated coarse level data.
// overrides the pure virtual function in AmrCore
void
LRFPFCT::MakeNewLevelFromCoarse (int lev, Real time, const BoxArray& ba,
				    const DistributionMapping& dm)
{
    const int numcomp = phi_new[lev-1].nComp();
    const int numghost = phi_new[lev-1].nGrow();
    
    phi_new[lev].define(ba, dm, numcomp, numghost);
    phi_old[lev].define(ba, dm, numcomp, numghost);

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    // This clears the old MultiFab and allocates the new one
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
    {
	facevel[lev][idim] = MultiFab(amrex::convert(ba,IntVect::TheDimensionVector(idim)), dm, 1, numghost);
    }

    if (lev > 0 && do_reflux) {
	flux_reg[lev].reset(new FluxRegister(ba, dm, refRatio(lev-1), lev, conscomp));
    }

    FillCoarsePatch(lev, time, phi_new[lev], 0, numcomp);

    phi_new[lev].FillBoundary();
    phi_new[lev].FillBoundary(geom[lev].periodicity());
    FillDomBoundary(phi_new[lev],geom[lev],bcs,time);

    phi_old[lev].FillBoundary();
    phi_old[lev].FillBoundary(geom[lev].periodicity());
    FillDomBoundary(phi_old[lev],geom[lev],bcs,time);

	ParallelDescriptor::Barrier();
}

// Remake an existing level using provided BoxArray and DistributionMapping and 
// fill with existing fine and coarse data.
// overrides the pure virtual function in AmrCore
void
LRFPFCT::RemakeLevel (int lev, Real time, const BoxArray& ba,
			 const DistributionMapping& dm)
{
    const int numcomp = phi_new[lev].nComp();
    const int numghost = phi_new[lev].nGrow();

    MultiFab new_state(ba, dm, numcomp, numghost);
    MultiFab old_state(ba, dm, numcomp, numghost);

    FillPatch(lev, time, new_state, 0, numcomp);

    new_state.FillBoundary();
    new_state.FillBoundary(geom[lev].periodicity());
    FillDomBoundary(new_state,geom[lev],bcs,time);

    std::swap(new_state, phi_new[lev]);
    std::swap(old_state, phi_old[lev]);

    phi_new[lev].FillBoundary();
    phi_new[lev].FillBoundary(geom[lev].periodicity());
    FillDomBoundary(phi_new[lev],geom[lev],bcs,time);

    phi_old[lev].FillBoundary();
    phi_old[lev].FillBoundary(geom[lev].periodicity());
    FillDomBoundary(phi_old[lev],geom[lev],bcs,time);

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    // This clears the old MultiFab and allocates the new one
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
    {
	facevel[lev][idim] = MultiFab(amrex::convert(ba,IntVect::TheDimensionVector(idim)), dm, 1, numghost);
    }

    if (lev > 0 && do_reflux) {
	flux_reg[lev].reset(new FluxRegister(ba, dm, refRatio(lev-1), lev, conscomp));
    }

	ParallelDescriptor::Barrier();    
}

// Delete level data
// overrides the pure virtual function in AmrCore
void
LRFPFCT::ClearLevel (int lev)
{
    phi_new[lev].clear();
    phi_old[lev].clear();
    flux_reg[lev].reset(nullptr);
}

// Make a new level from scratch using provided BoxArray and DistributionMapping.
// Only used during initialization.
// overrides the pure virtual function in AmrCore
void LRFPFCT::MakeNewLevelFromScratch (int lev, Real time, const BoxArray& ba,
					  const DistributionMapping& dm)
{
    phi_new[lev].define(ba, dm, ncomp, nghost);
    phi_old[lev].define(ba, dm, ncomp, nghost);

    // int nc = phi_new[lev].nComp();

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    Real pminfrac = 0.01, rominfrac = 0.01;
    Real rad_bw = 0.01;
    Real p2, p1, ro2, ro1;
    AMREX_D_TERM(Real xcm = 0.0;, Real ycm = 0.0;, Real zcm = 0.0;);
    AMREX_D_TERM(Real u1;, Real v1;, Real w1;);
    AMREX_D_TERM(Real u2;, Real v2;, Real w2;);

    const auto problo = Geom(lev).ProbLoArray();
    const auto probhi = Geom(lev).ProbHiArray();
    const auto dx     = Geom(lev).CellSizeArray();

    int prob_tag;
    {
        ParmParse pp("prob");
        pp.query("rad_blast",rad_bw);
        pp.get("probtag",prob_tag);

        AMREX_D_TERM(pp.query("xcm", xcm);, pp.query("ycm", ycm);, pp.query("zcm", zcm));
        
        pp.get("p2",p2);
        pp.get("p1",p1);
        pp.get("ro2",ro2);
        pp.get("ro1",ro1);

        AMREX_D_TERM(pp.get("u2",u2);, pp.get("v2",v2);, pp.get("w2",w2));
        AMREX_D_TERM(pp.get("u1",u1);, pp.get("v1",v1);, pp.get("w1",w1));

        pp.query("pminfrac",pminfrac);
        pp.query("rominfrac",rominfrac);
    }

    // This clears the old MultiFab and allocates the new one
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
    {
	facevel[lev][idim] = MultiFab(amrex::convert(ba,IntVect::TheDimensionVector(idim)), dm, 1, nghost);
    }

    if (lev > 0 && do_reflux) {
	flux_reg[lev].reset(new FluxRegister(ba, dm, refRatio(lev-1), lev, conscomp));
    }

    MultiFab& state = phi_new[lev];

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Array4<Real> fab = state[mfi].array();
        const Box& box = mfi.tilebox();

        amrex::launch(box,
        [=] AMREX_GPU_DEVICE (Box const& tbx)
        {
            initdata(tbx, fab, problo, probhi, dx, prob_tag, ro2, ro1, p2, p1, AMREX_D_DECL(u2, v2, w2), AMREX_D_DECL(u1, v1, w1), AMREX_D_DECL(xcm, ycm, zcm), rad_bw);
        });
    }

	ParallelDescriptor::Barrier();
    
    FillPatch(lev, time, phi_new[lev], 0, ncomp);
    phi_new[lev].FillBoundary();
    phi_new[lev].FillBoundary(Geom(lev).periodicity());

    pmin = pminfrac*phi_new[lev].min(pre);
    romin = rominfrac*phi_new[lev].min(ro);

    // if(ParallelDescriptor::MyProc() == 0){
        Print() << "min(ro)= " << phi_new[lev].min(ro,4) << ", max(pre)= " << phi_new[lev].max(pre,4)
        << ", min(roE)= " << phi_new[lev].min(roE,4) << ", min(mach)= " << phi_new[lev].min(mach,4) << "\n";
    // }

    // ParallelDescriptor::Barrier();

    if(phi_new[lev].min(ro,4) < 0.0 || phi_new[lev].min(pre,4) < 0.0 || phi_new[lev].min(mach,4) < 0.0 || phi_new[lev].min(roE,4) < 0.0){
        Print() << "Level (after FillPatch) = " << lev << "\n";
        amrex::Error("Pressure/density/Mach number is negative, aborting...");
    }

    if(phi_new[lev].contains_nan()){
        Print() << "End of MakeNewLevelFromScratch(), lev = " << lev << ", contains NaN() " << "\n";
        amrex::Error("NaN value found in initial conditions, aborting...");
    }

    ParallelDescriptor::Barrier();
}

// tag all cells for refinement
// overrides the pure virtual function in AmrCore
void
LRFPFCT::ErrorEst (int lev, TagBoxArray& tags, Real /*time*/, int /*ngrow*/)
{
    static bool first = true;
    static Real tagfrac;

    // only do this during the first call to ErrorEst
    if (first)
    {
	   first = false;
        // read in an array of "phierr", which is the tagging threshold
        // in this example, we tag values of "phi" which are greater than phierr
        // for that particular level
        // in subroutine state_error, you could use more elaborate tagging, such
        // as more advanced logical expressions, or gradients, etc.
	   ParmParse pp("adv");
        pp.get("tagfrac", tagfrac);
    }

    // if (lev >= phierr.size()) return;

   const int clearval = TagBox::CLEAR;
   const int   tagval = TagBox::SET;

    MultiFab& state = phi_new[lev];
    const auto dx   = Geom(lev).CellSizeArray();
    AMREX_D_TERM(Real maxgradpx;, Real maxgradpy;, Real maxgradpz);
// First, get the pressure gradients in x and y directions
if(lev < lev_allow){
{
        const BoxArray& ba = phi_new[lev].boxArray();
        const DistributionMapping& dm = phi_new[lev].DistributionMap();
        MultiFab gradp(ba, dm, AMREX_SPACEDIM, 0);
        MultiFab& gradp_state = gradp;
#ifdef _OPENMP
#pragma omp parallel if(Gpu::notInLaunchRegion())
#endif
    {
    
    for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
            const Box& bx  = mfi.tilebox();
            Array4<Real> statefab = state[mfi].array();
            Array4<Real> gradpfab = gradp_state[mfi].array();

            amrex::launch(bx,
            [=] AMREX_GPU_DEVICE (Box const& tbx)
            {
                get_gradp_x(tbx, statefab, gradpfab, dx);
            });

            amrex::launch(bx,
            [=] AMREX_GPU_DEVICE (Box const& tbx)
            {
                get_gradp_y(tbx, statefab, gradpfab, dx);
            });

#if AMREX_SPACEDIM==3
            amrex::launch(bx,
            [=] AMREX_GPU_DEVICE (Box const& tbx)
            {
                get_gradp_z(tbx, statefab, gradpfab, dx);
            });
#endif
    }
    }
	ParallelDescriptor::Barrier();
    AMREX_D_TERM(maxgradpx = gradp.norm0(0);, maxgradpy = gradp.norm0(1);, maxgradpz = gradp.norm0(2));
    // Print() << "maxgradpx= " << maxgradpx << ", maxgradpy= " << maxgradpy << "\n";

}

#ifdef _OPENMP
#pragma omp parallel if(Gpu::notInLaunchRegion())
#endif
    {
	
	for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
	{
	        const Box& bx  = mfi.tilebox();
            const auto statefab = state.array(mfi);
            const auto tagfab  = tags.array(mfi);
            Real tag_frac = tagfrac;
	    
            amrex::launch(bx,
            [=] AMREX_GPU_DEVICE (Box const& tbx)
            {
                state_error(tbx, tagfab, statefab, AMREX_D_DECL(maxgradpx, maxgradpy, maxgradpz), dx, tag_frac, tagval);
            });
	}
    }
}
ParallelDescriptor::Barrier();
}

// set covered coarse cells to be the average of overlying fine cells
void
LRFPFCT::AverageDown ()
{
    for (int lev = finest_level-1; lev >= 0; --lev)
    {
	amrex::average_down(phi_new[lev+1], phi_new[lev],
                            geom[lev+1], geom[lev],
                            0, phi_new[lev].nComp(), refRatio(lev));
    }
	ParallelDescriptor::Barrier();
}

// more flexible version of AverageDown() that lets you average down across multiple levels
void
LRFPFCT::AverageDownTo (int crse_lev)
{
    amrex::average_down(phi_new[crse_lev+1], phi_new[crse_lev],
                        geom[crse_lev+1], geom[crse_lev],
                        0, phi_new[crse_lev].nComp(), refRatio(crse_lev));
	ParallelDescriptor::Barrier();
}

// compute a new multifab by coping in phi from valid region and filling ghost cells
// works for single level and 2-level cases (fill fine grid ghost by interpolating from coarse)
void
LRFPFCT::FillPatch (int lev, Real time, MultiFab& mf, int icomp, int numcomp)
{
    if (lev == 0)
    {
	Vector<MultiFab*> smf;
	Vector<Real> stime;
	GetData(0, time, smf, stime);

        if(Gpu::inLaunchRegion())
        {
            GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
            PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > physbc(geom[lev],bcs,gpu_bndry_func);
            amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, numcomp, 
                                        geom[lev], physbc, 0);
        }
        else
        {
            CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
            PhysBCFunct<CpuBndryFuncFab> physbc(geom[lev],bcs,bndry_func);
            amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, numcomp, 
                                        geom[lev], physbc, 0);
        }
    }
    else
    {
	Vector<MultiFab*> cmf, fmf;
	Vector<Real> ctime, ftime;
	GetData(lev-1, time, cmf, ctime);
	GetData(lev  , time, fmf, ftime);

	Interpolater* mapper = &cell_cons_interp;

        if(Gpu::inLaunchRegion())
        {
            GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
            PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > cphysbc(geom[lev-1],bcs,gpu_bndry_func);
            PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > fphysbc(geom[lev],bcs,gpu_bndry_func);

            amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
                                      0, icomp, numcomp, geom[lev-1], geom[lev],
                                      cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                      mapper, bcs, 0);
        }
        else
        {
            CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
            PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev-1],bcs,bndry_func);
            PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev],bcs,bndry_func);

            amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
                                      0, icomp, numcomp, geom[lev-1], geom[lev],
                                      cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                      mapper, bcs, 0);
        }
    }
	ParallelDescriptor::Barrier();
}

// fill an entire multifab by interpolating from the coarser level
// this comes into play when a new level of refinement appears
void
LRFPFCT::FillCoarsePatch (int lev, Real time, MultiFab& mf, int icomp, int numcomp)
{
    BL_ASSERT(lev > 0);

    Vector<MultiFab*> cmf;
    Vector<Real> ctime;
    GetData(lev-1, time, cmf, ctime);
    Interpolater* mapper = &cell_cons_interp;
    
 //    if (cmf.size() != 1) {
	// amrex::Abort("FillCoarsePatch: how did this happen?");
 //    }

    if(Gpu::inLaunchRegion())
    {
        GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
        PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > cphysbc(geom[lev-1],bcs,gpu_bndry_func);
        PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > fphysbc(geom[lev],bcs,gpu_bndry_func);

        amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, numcomp, geom[lev-1], geom[lev],
                                     cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                     mapper, bcs, 0);
    }
    else
    {
        CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
        PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev-1],bcs,bndry_func);
        PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev],bcs,bndry_func);

        amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, numcomp, geom[lev-1], geom[lev],
                                     cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                     mapper, bcs, 0);
    }
	ParallelDescriptor::Barrier();
}

// utility to copy in data from phi_old and/or phi_new into another multifab
void
LRFPFCT::GetData (int lev, Real time, Vector<MultiFab*>& data, Vector<Real>& datatime)
{
    data.clear();
    datatime.clear();

    const Real teps = (t_new[lev] - t_old[lev]) * 1.e-3;

    if (time > t_new[lev] - teps && time < t_new[lev] + teps)
    {
	data.push_back(&phi_new[lev]);
	datatime.push_back(t_new[lev]);
    }
    else if (time > t_old[lev] - teps && time < t_old[lev] + teps)
    {
	data.push_back(&phi_old[lev]);
	datatime.push_back(t_old[lev]);
    }
    else
    {
	data.push_back(&phi_old[lev]);
	data.push_back(&phi_new[lev]);
	datatime.push_back(t_old[lev]);
	datatime.push_back(t_new[lev]);
    }
	ParallelDescriptor::Barrier();
}

// advance solution to final time
void
LRFPFCT::Evolve ()
{
    Real cur_time = t_new[0];
    int last_plot_file_step = 0;

    for (int step = istep[0]; step < max_step && cur_time < stop_time; ++step)
    {
        amrex::Print() << "\nCoarse STEP " << step+1 << " starts ..." << std::endl;

        ComputeDt();

        int lev = 0;
        int iteration = 1;
        if (do_subcycle)
            timeStepWithSubcycling(lev, cur_time, iteration);
        else
            timeStepNoSubcycling(cur_time, iteration);

        ParallelDescriptor::Barrier();

        cur_time += dt[0];

        amrex::Print() << "Coarse STEP " << step+1 << " ends." << " TIME = " << cur_time
                       << " DT = " << dt[0] << " max(pre) = " << phi_new[0].max(pre,4) << 
                       ", max(mach)= " << phi_new[0].max(mach,4) << std::endl;

        // sync up time
        for (lev = 0; lev <= finest_level; ++lev) {
            t_new[lev] = cur_time;
        }

        if (plot_int > 0 && (step+1) % plot_int == 0) {
            last_plot_file_step = step+1;
            WritePlotFile();
        }

        if (chk_int > 0 && (step+1) % chk_int == 0) {
            WriteCheckpointFile();
        }

#ifdef AMREX_MEM_PROFILING
        {
            std::ostringstream ss;
            ss << "[STEP " << step+1 << "]";
            MemProfiler::report(ss.str());
        }
#endif

        if (cur_time >= stop_time - 1.e-6*dt[0]) break;
    }

    if (chk_int > 0) {
       WriteCheckpointFile();
    }

    if(plot_int > 0){
        WritePlotFile();
    }
}

// a wrapper for EstTimeStep
void
LRFPFCT::ComputeDt ()
{
    Vector<Real> dt_tmp(finest_level+1);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        dt_tmp[lev] = EstTimeStep(lev, t_new[lev]);
    }
    ParallelDescriptor::ReduceRealMin(&dt_tmp[0], dt_tmp.size());

    constexpr Real change_max = 1.1;
    Real dt_0 = dt_tmp[0];
    int n_factor = 1;

    for (int lev = 0; lev <= finest_level; ++lev) {
        dt_tmp[lev] = std::min(dt_tmp[lev], change_max*dt[lev]);
        n_factor *= nsubsteps[lev];
        dt_0 = std::min(dt_0, n_factor*dt_tmp[lev]);
    }

    // Limit dt's by the value of stop_time.
    const Real eps = 1.e-3*dt_0;

    if (t_new[0] + dt_0 > stop_time - eps) {
        dt_0 = stop_time - t_new[0];
    }

    dt[0] = dt_0;
	if(dt[0] < 0.0){	
		AMREX_ASSERT_WITH_MESSAGE(dt[0] > 0.0, "dt[0] < 0 (LRFPFCT::ComputeDt())");
	}
    // Print() << "lev= 0" << ", dt= " << dt[0] << ", nsubsteps= " << nsubsteps[0] << "\n";
    for (int lev = 1; lev <= finest_level; ++lev) {
        dt[lev] = dt[lev-1] / nsubsteps[lev];
    }

    if(do_subcycle && do_fixeddt){
        for (int lev = 0; lev <= finest_level; ++lev) {
            dt[lev] = dt[finest_level];
            if(lev_allow < finest_level){
                dt[lev] = dt[lev_allow];
            }
        }           
    }

    ParallelDescriptor::Barrier();
}

// compute dt from CFL considerations
Real
LRFPFCT::EstTimeStep (int lev, Real time)
{
    BL_PROFILE("LRFPFCT::EstTimeStep()");

    Real dt_est = std::numeric_limits<Real>::max();

    const Real* dx  =  geom[lev].CellSize();

    if (time == 0.0) {
       DefineVelocityAtLevelDt(lev,time);
    } else {
       Real t_nph_predicted = time + 0.5 * dt[lev];
       DefineVelocityAtLevelDt(lev,t_nph_predicted);
    }
	ParallelDescriptor::Barrier();

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
		Real est = facevel[lev][idim].norminf(0,0,false);
    //  Real est = facevel[lev][idim].max(0,0,true);
		dt_est = amrex::min(dt_est, dx[idim]/est);
	}

    dt_est *= cfl;

	ParallelDescriptor::Barrier();

    return dt_est;
}


// Advance a level by dt
// (includes a recursive call for finer levels)
void
LRFPFCT::timeStepWithSubcycling (int lev, Real time, int iteration)
{
    if (regrid_int > 0)  // We may need to regrid
    {

        // help keep track of whether a level was already regridded
        // from a coarser level call to regrid
        static Vector<int> last_regrid_step(max_level+1, 0);

        // regrid changes level "lev+1" so we don't regrid on max_level
        // also make sure we don't regrid fine levels again if 
        // it was taken care of during a coarser regrid

        if (lev < max_level && istep[lev] > last_regrid_step[lev]) 
        {
            if (istep[lev] % regrid_int == 0)
            {
                // regrid could add newly refine levels (if finest_level < max_level)
                // so we save the previous finest level index
                int old_finest = finest_level; 
                regrid(lev, time);

                // mark that we have regridded this level already
                for (int k = lev; k <= finest_level; ++k) {
                    last_regrid_step[k] = istep[k];
                }

                // if there are newly created levels, set the time step
                for (int k = old_finest+1; k <= finest_level; ++k) {
                    dt[k] = dt[k-1] / MaxRefRatio(k-1);
                }
                if(do_fixeddt){
                    for (int k = 0; k <= finest_level; ++k){
                        dt[k] = dt[finest_level];
                    }                    
                }

            }
        }
    }

    if (Verbose()) {
        amrex::Print() << "[Level " << lev << " step " << istep[lev]+1 << "] ";
        amrex::Print() << "ADVANCE with time = " << t_new[lev] 
                       << " dt = " << dt[lev] << std::endl;
    }

    // Advance a single level for a single time step, and update flux registers

    t_old[lev] = t_new[lev];
    t_new[lev] += dt[lev];

    // Real t_nph = t_old[lev] + 0.5*dt[lev]; 

    // DefineVelocityAtLevel(lev, t_nph);
    AdvancePhiAtLevel(lev, time, dt[lev], iteration, nsubsteps[lev]);

    ++istep[lev];

    if (Verbose())
    {
        amrex::Print() << "[Level " << lev << " step " << istep[lev] << "] ";
        amrex::Print() << "Advanced " << CountCells(lev) << " cells" << std::endl;
    }

    if (lev < finest_level)
    {
        // recursive call for next-finer level
        for (int i = 1; i <= nsubsteps[lev+1]; ++i)
        {
            timeStepWithSubcycling(lev+1, time+(i-1)*dt[lev+1], i);
        }

        ParallelDescriptor::Barrier();
        phi_new[lev].FillBoundary();
        phi_new[lev].FillBoundary(geom[lev].periodicity());
        FillDomBoundary(phi_new[lev],geom[lev],bcs,time);

        if (do_reflux)
        {
            // update lev based on coarse-fine flux mismatch
            flux_reg[lev+1]->Reflux(phi_new[lev], 1.0, 0, 0, conscomp, geom[lev]);
            
            ParallelDescriptor::Barrier();

            CalcAuxillaryWrapper(lev);
            phi_new[lev].FillBoundary();
            FillDomBoundary(phi_new[lev],geom[lev],bcs,time);
            phi_new[lev].FillBoundary(geom[lev].periodicity());
        }

        AverageDownTo(lev); // average lev+1 down to lev
        ParallelDescriptor::Barrier();

        CalcAuxillaryWrapper(lev);

        phi_new[lev].FillBoundary();
        phi_new[lev].FillBoundary(geom[lev].periodicity());
        FillDomBoundary(phi_new[lev],geom[lev],bcs,time);    
    }
 ParallelDescriptor::Barrier();   
}

// Advance all the levels with the same dt
void
LRFPFCT::timeStepNoSubcycling (Real time, int iteration)
{
    if (max_level > 0 && regrid_int > 0)  // We may need to regrid
    {
        if (istep[0] % regrid_int == 0)
        {
            regrid(0, time);
        }
    }

    if (Verbose()) {
        for (int lev = 0; lev <= finest_level; lev++)
        {
           amrex::Print() << "[Level " << lev << " step " << istep[lev]+1 << "] ";
           amrex::Print() << "ADVANCE with time = " << t_new[lev] 
                          << " dt = " << dt[0] << std::endl;
        }
    }

    AdvancePhiAllLevels (time, dt[0], iteration);

    // Make sure the coarser levels are consistent with the finer levels
    AverageDown ();

    for (int lev = 0; lev <= finest_level; lev++)
        ++istep[lev];

    if (Verbose())
    {
        for (int lev = 0; lev <= finest_level; lev++)
        {
            amrex::Print() << "[Level " << lev << " step " << istep[lev] << "] ";
            amrex::Print() << "Advanced " << CountCells(lev) << " cells" << std::endl;
        }
    }
	ParallelDescriptor::Barrier();
}

// get plotfile name
std::string
LRFPFCT::PlotFileName (int lev) const
{
    return amrex::Concatenate(plot_file, lev, 7);
}

// put together an array of multifabs for writing
Vector<const MultiFab*>
LRFPFCT::PlotFileMF () const
{
    Vector<const MultiFab*> r;
    for (int i = 0; i <= finest_level; ++i) {
	r.push_back(&phi_new[i]);
    }
    return r;
}

// set plotfile variable names
Vector<std::string>
LRFPFCT::PlotFileVarNames () const
{
#if AMREX_SPACEDIM==2
    return {"ro","rou","rov","roE","pre","Mach"};
#endif
#if AMREX_SPACEDIM==3
    return {"ro","rou","rov","row","roE","pre","Mach"};
#endif
}

// write plotfile to disk
void
LRFPFCT::WritePlotFile () const
{
    const std::string& plotfilename = PlotFileName(istep[0]);
    const auto& mf = PlotFileMF();
    const auto& varnames = PlotFileVarNames();
    
    amrex::Print() << "Writing plotfile " << plotfilename << "\n";
    amrex::Print() << "ncomp= " << mf[0]->nComp() << ", varsize= " << varnames.size() << "\n";

    amrex::WriteMultiLevelPlotfile(plotfilename, finest_level+1, mf, varnames,
				   Geom(), t_new[0], istep, refRatio());
}

void
LRFPFCT::WriteCheckpointFile () const
{

    // chk00010            write a checkpoint file with this root directory
    // chk00010/Header     this contains information you need to save (e.g., finest_level, t_new, etc.) and also
    //                     the BoxArrays at each level
    // chk00010/Level_0/
    // chk00010/Level_1/
    // etc.                these subdirectories will hold the MultiFab data at each level of refinement

    // checkpoint file name, e.g., chk00010
    const std::string& checkpointname = amrex::Concatenate(chk_file,istep[0],7);

    amrex::Print() << "Writing checkpoint " << checkpointname << "\n";

    const int nlevels = finest_level+1;

    // ---- prebuild a hierarchy of directories
    // ---- dirName is built first.  if dirName exists, it is renamed.  then build
    // ---- dirName/subDirPrefix_0 .. dirName/subDirPrefix_nlevels-1
    // ---- if callBarrier is true, call ParallelDescriptor::Barrier()
    // ---- after all directories are built
    // ---- ParallelDescriptor::IOProcessor() creates the directories
    amrex::PreBuildDirectorHierarchy(checkpointname, "Level_", nlevels, true);

    // write Header file
   if (ParallelDescriptor::IOProcessor()) {

       std::string HeaderFileName(checkpointname + "/Header");
       VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);
       std::ofstream HeaderFile;
       HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
       HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out   |
		                               std::ofstream::trunc |
                                               std::ofstream::binary);
       if( ! HeaderFile.good()) {
           amrex::FileOpenFailed(HeaderFileName);
       }

       HeaderFile.precision(17);

       // write out title line
       HeaderFile << "Checkpoint file for LRFPFCT\n";

       // write out finest_level
       HeaderFile << finest_level << "\n";

       // write out array of istep
       for (int i = 0; i < istep.size(); ++i) {
           HeaderFile << istep[i] << " ";
       }
       HeaderFile << "\n";

       // write out array of dt
       for (int i = 0; i < dt.size(); ++i) {
           HeaderFile << dt[i] << " ";
       }
       HeaderFile << "\n";

       // write out array of t_new
       for (int i = 0; i < t_new.size(); ++i) {
           HeaderFile << t_new[i] << " ";
       }
       HeaderFile << "\n";

       // write the BoxArray at each level
       for (int lev = 0; lev <= finest_level; ++lev) {
           boxArray(lev).writeOn(HeaderFile);
           HeaderFile << '\n';
       }
   }

   // write the MultiFab data to, e.g., chk00010/Level_0/
   for (int lev = 0; lev <= finest_level; ++lev) {
       VisMF::Write(phi_new[lev],
                    amrex::MultiFabFileFullPrefix(lev, checkpointname, "Level_", "phi"));
   }

}

namespace {
// utility to skip to next line in Header
void GotoNextLine (std::istream& is)
{
    constexpr std::streamsize bl_ignore_max { 100000 };
    is.ignore(bl_ignore_max, '\n');
}
}

void
LRFPFCT::ReadCheckpointFile ()
{

    amrex::Print() << "Restart from checkpoint " << restart_chkfile << "\n";

    // Header
    std::string File(restart_chkfile + "/Header");

    VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    std::string line, word;

    // read in title line
    std::getline(is, line);

    // read in finest_level
    is >> finest_level;
    GotoNextLine(is);

    // read in array of istep
    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0;
        while (lis >> word) {
            istep[i++] = std::stoi(word);
        }
    }

    // read in array of dt
    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0;
        while (lis >> word) {
            dt[i++] = std::stod(word);
        }
    }

    // read in array of t_new
    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0;
        while (lis >> word) {
            t_new[i++] = std::stod(word);
        }
    }

    for (int lev = 0; lev <= finest_level; ++lev) {

        // read in level 'lev' BoxArray from Header
        BoxArray ba;
        ba.readFrom(is);
        GotoNextLine(is);

        // create a distribution mapping
        DistributionMapping dm { ba, ParallelDescriptor::NProcs() };

        // set BoxArray grids and DistributionMapping dmap in AMReX_AmrMesh.H class
        SetBoxArray(lev, ba);
        SetDistributionMap(lev, dm);

        // build MultiFab and FluxRegister data
        int numcomp = ncomp;
        int numghost = nghost;
        phi_old[lev].define(grids[lev], dmap[lev], numcomp, numghost);
        phi_new[lev].define(grids[lev], dmap[lev], numcomp, numghost);

        if (lev > 0 && do_reflux) {
            flux_reg[lev].reset(new FluxRegister(grids[lev], dmap[lev], refRatio(lev-1), lev, conscomp));
        }

        // build face velocity MultiFabs
        for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
        {
	    facevel[lev][idim] = MultiFab(amrex::convert(ba,IntVect::TheDimensionVector(idim)), dm, 1, numghost);
        }
    }

    // read in the MultiFab data
    for (int lev = 0; lev <= finest_level; ++lev) {
        VisMF::Read(phi_new[lev],
                    amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "phi"));
    }

}
