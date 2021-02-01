
#include <iostream>

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>

#include <LRFPFCT.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    {
        // timer for profiling
        BL_PROFILE("main()");

        // wallclock time
        const Real strt_total = amrex::second();

        // constructor - reads in parameters from inputs file
        //             - sizes multilevel arrays and data structures
        LRFPFCT LRFPFCT_adv;
	
        // initialize AMR data
	LRFPFCT_adv.InitData();

        // advance solution to final time
	// LRFPFCT_adv.Evolve();
	
        // wallclock time
	Real end_total = amrex::second() - strt_total;
	
	if (LRFPFCT_adv.Verbose()) {
            // print wallclock time
            ParallelDescriptor::ReduceRealMax(end_total ,ParallelDescriptor::IOProcessorNumber());
            amrex::Print() << "\nTotal Time: " << end_total << '\n';
	}
    }

    amrex::Finalize();
}
