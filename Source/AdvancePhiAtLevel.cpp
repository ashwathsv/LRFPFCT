#include <LRFPFCT.H>
#include <Kernels.H>

using namespace amrex;

// Advance a single level for a single time step, updates flux registers
void
LRFPFCT::AdvancePhiAtLevel (int lev, Real time, Real dt_lev, int /*iteration*/, int /*ncycle*/)
{
	const int num_grow = nghost;

	std::swap(phi_old[lev], phi_new[lev]);

	MultiFab& S_new = phi_new[lev];
	MultiFab& S_old = phi_old[lev];

	int rk_max = max_rk;
	Real coeff = 1.0;
	Real minp = pmin;
	Real minro = romin;

	const Real dx = geom[lev].CellSize(0);
	const Real dy = geom[lev].CellSize(1);
	const Real dz = (AMREX_SPACEDIM == 2) ? Real(1.0) : geom[lev].CellSize(2);

	AMREX_D_TERM(Real dtdx = dt_lev/dx;,
				 Real dtdy = dt_lev/dy;,
				 Real dtdz = dt_lev/dz);

	// Define MultiFabs for partially convected quantities and Fill the MultiFabs
	AMREX_D_TERM(MultiFab Sconvx(grids[lev], dmap[lev], conscomp, num_grow);,
				 MultiFab Sconvy(grids[lev], dmap[lev], conscomp, num_grow);,
				 MultiFab Sconvz(grids[lev], dmap[lev], conscomp, num_grow));

	FillPatch(lev, time, S_new, 0, conscomp);
	S_new.FillBoundary();
	S_new.FillBoundary(geom[lev].periodicity());
	FillDomBoundary(S_new,geom[lev],bcs,time);

	AMREX_D_TERM(FillPatch(lev, time, Sconvx, 0, Sconvx.nComp());,
				 FillPatch(lev, time, Sconvy, 0, Sconvy.nComp());,
				 FillPatch(lev, time, Sconvz, 0, Sconvz.nComp()));
	AMREX_D_TERM(Sconvx.FillBoundary();, Sconvy.FillBoundary();, Sconvz.FillBoundary());
	AMREX_D_TERM(Sconvx.FillBoundary(geom[lev].periodicity());, Sconvy.FillBoundary(geom[lev].periodicity());,
				 Sconvz.FillBoundary(geom[lev].periodicity()));
	AMREX_D_TERM(FillDomBoundary(Sconvx,geom[lev],bcs,time);,FillDomBoundary(Sconvy,geom[lev],bcs,time);,
				 FillDomBoundary(Sconvz,geom[lev],bcs,time));

// Calculate pressure and Mach number for 
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
	{
		for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
		{
			const Box& box = amrex::grow(mfi.tilebox(),num_grow);
			Array4<Real> state = S_new[mfi].array();
			amrex::launch(box,
					 [=] AMREX_GPU_DEVICE (Box const& tbx)
					 {  CalcAuxillary(tbx, state, minp);  });
		}
	}

	if(S_new.min(ro,num_grow) < 0.0 || S_new.min(pre,num_grow) < 0.0 || S_new.min(mach,num_grow) < 0.0 || S_new.min(roE,num_grow) < 0.0){
		Print() << "Level (after FillPatch) = " << lev << "\n";
		Print() << "min ro= " << S_new.min(ro,num_grow) 
				<< ", min pre= " << S_new.min(pre,num_grow) << ", min roE= " << S_new.min(roE,num_grow)
				<< ", min mach= " << S_new.min(mach,num_grow) <<"\n";
		WritePlotFile();
		amrex::Error("Pressure/density/Mach number is negative, aborting...");
	}  

// Define MultiFabs for fluxes
	MultiFab fluxes[AMREX_SPACEDIM];
	if (do_reflux)
	{
		for (int i = 0; i < AMREX_SPACEDIM; ++i)
		{
			BoxArray ba = grids[lev];
			ba.surroundingNodes(i);
			fluxes[i].define(ba, dmap[lev], conscomp, num_grow);
		}
	}

  for(int rk = 1; rk <= rk_max; ++rk)
  {
	if(rk == 1 && rk_max > 1){ coeff = half;  }
	else{ coeff = 1.0;  }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
	{
		FArrayBox fluxc[BL_SPACEDIM], phitemp, fluxad[BL_SPACEDIM], fluxd[BL_SPACEDIM], fldiff[BL_SPACEDIM], frin, frout, prefab;
		for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
		{
// ==== Define nodal boxes to define iterators for calculating fluxes===
		AMREX_D_TERM(const Box& ngbxx = amrex::grow(mfi.nodaltilebox(0),num_grow);,
					 const Box& ngbxy = amrex::grow(mfi.nodaltilebox(1),num_grow);,
					 const Box& ngbxz = amrex::grow(mfi.nodaltilebox(2),num_grow););
		
		AMREX_D_TERM(Array4<Real> fluxx = fluxes[0].array(mfi);,
					 Array4<Real> fluxy = fluxes[1].array(mfi);,
					 Array4<Real> fluxz = fluxes[2].array(mfi));
// Declare array for face velocities
		GpuArray<Array4<Real>, AMREX_SPACEDIM> vel{ AMREX_D_DECL( facevel[lev][0].array(mfi),
											   facevel[lev][1].array(mfi), facevel[lev][2].array(mfi)) };

// Define integers for lower and upper bounds of face velocity array
		AMREX_D_TERM(int vx_lox = lbound(vel[0]).x;, int vy_loy = lbound(vel[1]).y;, int vz_loz = lbound(vel[2]).z;);
		AMREX_D_TERM(int vx_hix = ubound(vel[0]).x;, int vy_hiy = ubound(vel[1]).y;, int vz_hiz = ubound(vel[2]).z;);

// Define arrays for S_new(fabnew), S_old(fabold), Sconvx(fabcx), Sconvy(fabcy), Sconvz(fabcz)
		Array4<Real> fabold = S_old[mfi].array();
		Array4<Real> fabnew = S_new[mfi].array();
		AMREX_D_TERM(Array4<Real> fabcx  = Sconvx[mfi].array();,
					 Array4<Real> fabcy  = Sconvy[mfi].array();,
					 Array4<Real> fabcz  = Sconvz[mfi].array(););
		
		// Define integers for lower and upper bounds of fabnew array 
		AMREX_D_TERM(int philox = lbound(fabnew).x;, int philoy = lbound(fabnew).y;, int philoz = lbound(fabnew).z;);
		AMREX_D_TERM(int phihix = ubound(fabnew).x;, int phihiy = ubound(fabnew).y;, int phihiz = ubound(fabnew).z;);

// Get face velocity at the beginning of the FCT solution (before FCT diffusion step)
		if(rk == 1){
		  amrex::ParallelFor(AMREX_D_DECL(ngbxx,ngbxy,ngbxz),
				  AMREX_D_DECL(
				  [=] AMREX_GPU_DEVICE (int i, int j, int k)
				  {  get_face_velocity_x(i, j, k, vel[0], fabold, vx_lox, vx_hix);
					 get_face_velocity_x_bc(i, j, k, vel[0], vx_lox, vx_hix);   },
				  [=] AMREX_GPU_DEVICE (int i, int j, int k)
				  {  get_face_velocity_y(i, j, k, vel[1], fabold, vy_loy, vy_hiy);
					 get_face_velocity_y_bc(i, j, k, vel[1], vy_loy, vy_hiy);   },
				  [=] AMREX_GPU_DEVICE (int i, int j, int k)
				  {  get_face_velocity_z(i, j, k, vel[2], fabold, vz_loz, vz_hiz);   }));                    
		}else{
		  amrex::ParallelFor(AMREX_D_DECL(ngbxx,ngbxy,ngbxz),
				  AMREX_D_DECL(
				  [=] AMREX_GPU_DEVICE (int i, int j, int k)
				  {  get_face_velocity_x(i, j, k, vel[0], fabnew, vx_lox, vx_hix);
					 get_face_velocity_x_bc(i, j, k, vel[0], vx_lox, vx_hix);   },
				  [=] AMREX_GPU_DEVICE (int i, int j, int k)
				  {  get_face_velocity_y(i, j, k, vel[1], fabnew, vy_loy, vy_hiy);
					 get_face_velocity_y_bc(i, j, k, vel[1], vy_loy, vy_hiy);   },
				  [=] AMREX_GPU_DEVICE (int i, int j, int k)
				  {  get_face_velocity_z(i, j, k, vel[2], fabnew, vz_loz, vz_hiz);   }));                    
		}

		const Box& bx = mfi.tilebox();
		Elixir fluxeli[BL_SPACEDIM];

		for (int i = 0; i < BL_SPACEDIM ; i++) {
			const Box& bxtmp = amrex::grow(mfi.nodaltilebox(i), num_grow);
			fluxc[i].resize(bxtmp,conscomp);  
			fluxeli[i] = fluxc[i].elixir();
		}

		// Temporary pressure fab to store the right pressure for calculating fluxes
		// if rk == 1, use pressure from fabold, else use fabnew to calculate pressure
		prefab.resize(amrex::grow(mfi.tilebox(),num_grow),1);
		Array4<Real> prearr = prefab.array();
		if(rk == 1){
		  amrex::ParallelFor(amrex::grow(bx,num_grow),
			[=] AMREX_GPU_DEVICE (int i, int j, int k)
			{   prearr(i,j,k) = fabold(i,j,k,pre); });                    
		}else{
		  amrex::ParallelFor(amrex::grow(bx,num_grow),
			[=] AMREX_GPU_DEVICE (int i, int j, int k)
			{   prearr(i,j,k) = fabnew(i,j,k,pre); });                       
		}

		GpuArray<Array4<Real>, AMREX_SPACEDIM> flx{ AMREX_D_DECL( fluxc[0].array(),
												fluxc[1].array(), fluxc[2].array()) };

		AMREX_D_TERM(int fx_lox=lbound(flx[0]).x;, int fy_loy=lbound(flx[1]).y;, int fz_loz=lbound(flx[2]).z;);
		AMREX_D_TERM(int fx_hix=ubound(flx[0]).x;, int fy_hiy=ubound(flx[1]).y;, int fz_hiz=ubound(flx[2]).z;);        

		amrex::ParallelFor(AMREX_D_DECL(ngbxx,ngbxy,ngbxz),
		 AMREX_D_DECL(
			[=] AMREX_GPU_DEVICE (int i, int j, int k)
			{  compute_conv_flux_x(i, j, k, flx[0], fabold, prearr, vel[0], fx_lox, fx_hix);   },
			[=] AMREX_GPU_DEVICE (int i, int j, int k)
			{  compute_conv_flux_y(i, j, k, flx[1], fabold, prearr, vel[1], fy_loy, fy_hiy);   },
			[=] AMREX_GPU_DEVICE (int i, int j, int k)
			{  compute_conv_flux_z(i, j, k, flx[2], fabold, prearr, vel[2], fz_loz, fz_hiz);   }));
		amrex::ParallelFor(ngbxx, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  flux_x_bc(i, j, k, n, flx[0], fx_lox, fx_hix);   });
		amrex::ParallelFor(ngbxy, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  flux_y_bc(i, j, k, n, flx[1], fy_loy, fy_hiy);   });
#if AMREX_SPACEDIM==3
		amrex::ParallelFor(ngbxz, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  flux_z_bc(i, j, k, n, flx[2], fz_loz, fz_hiz);   });
#endif
		if(rk > 1){
			amrex::ParallelFor(AMREX_D_DECL(ngbxx,ngbxy,ngbxz),
				AMREX_D_DECL(
				[=] AMREX_GPU_DEVICE (int i, int j, int k)
				{	get_face_velocity_x(i, j, k, vel[0], fabold, vx_lox, vx_hix);
					get_face_velocity_x_bc(i, j, k, vel[0], vx_lox, vx_hix);   },
				[=] AMREX_GPU_DEVICE (int i, int j, int k)
				{	get_face_velocity_y(i, j, k, vel[1], fabold, vy_loy, vy_hiy);
					get_face_velocity_y_bc(i, j, k, vel[1], vy_loy, vy_hiy);   },
				[=] AMREX_GPU_DEVICE (int i, int j, int k)
				{	get_face_velocity_z(i, j, k, vel[2], fabold, vz_loz, vz_hiz);   }));                      
		}

// do a conservative update (RK step 1)
		int ngrowbx = 0;
		if(lev > 0){ ngrowbx = num_grow-1; }
		const Box& iterbx = amrex::grow(mfi.tilebox(),ngrowbx);
		amrex::ParallelFor(iterbx, conscomp,
		[=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
		{   conv_updatephi(i, j, k, n, fabnew, fabold, AMREX_D_DECL(flx[0],flx[1],flx[2]), 
							AMREX_D_DECL(fabcx, fabcy, fabcz), 
							AMREX_D_DECL(coeff*dtdx, coeff*dtdy, coeff*dtdz)); });
		if(lev > 0){
			amrex::ParallelFor(amrex::grow(bx,num_grow), conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  phiconv_bc(i, j, k, n, fabnew, AMREX_D_DECL(fabcx, fabcy, fabcz), 
				AMREX_D_DECL(philox, philoy, philoz), AMREX_D_DECL(phihix, phihiy, phihiz)); });
		}

// proceeed with diffusion step
// allocate memory for temporary FArrayBox to store diffusion fluxes
		Elixir fluxeli2[BL_SPACEDIM];
		for (int i = 0; i < BL_SPACEDIM ; i++) {
			const Box& bxnd = amrex::surroundingNodes(bx,i);
			fluxd[i].resize(amrex::grow(bxnd,num_grow),conscomp);
			fluxeli2[i] = fluxd[i].elixir();
		}
		GpuArray<Array4<Real>, AMREX_SPACEDIM> fltx{ AMREX_D_DECL( fluxd[0].array(),
													fluxd[1].array(), fluxd[2].array()) };
		AMREX_D_TERM(int ftx_lox=lbound(fltx[0]).x;, int fty_loy=lbound(fltx[1]).y;, int ftz_loz=lbound(fltx[2]).z;);
		AMREX_D_TERM(int ftx_hix=ubound(fltx[0]).x;, int fty_hiy=ubound(fltx[1]).y;, int ftz_hiz=ubound(fltx[2]).z;);

// compute diffusive fluxes in the coordinate directions
		amrex::ParallelFor(ngbxx, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  compute_diff_flux_x(i, j, k, n, fltx[0], vel[0], fabold, ftx_lox, ftx_hix, coeff*dtdx); });
		amrex::ParallelFor(ngbxy, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  compute_diff_flux_y(i, j, k, n, fltx[1], vel[1], fabold, fty_loy, fty_hiy, coeff*dtdy); });
#if AMREX_SPACEDIM==3
		amrex::ParallelFor(ngbxz,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  compute_diff_flux_z(i, j, k, n, fltx[2], vel[2], fabold, ftz_loz, ftz_hiz, coeff*dtdz); });
#endif

		amrex::ParallelFor(ngbxx, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  flux_x_bc(i, j, k, n, fltx[0], ftx_lox, ftx_hix);   });
		amrex::ParallelFor(ngbxy, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  flux_y_bc(i, j, k, n, fltx[1], fty_loy, fty_hiy);   });
#if AMREX_SPACEDIM==3
		amrex::ParallelFor(ngbxz, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  flux_z_bc(i, j, k, n, fltx[2], ftz_loz, ftz_hiz);   });
#endif

// update conserved variables for diffusion step 
		ngrowbx = 0;
		if(lev > 0){ ngrowbx = num_grow-1; }
		amrex::ParallelFor(amrex::grow(mfi.tilebox(),ngrowbx), conscomp,
			[=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
			{   diff_updatephi(i, j, k, n, fabnew, AMREX_D_DECL(fltx[0],fltx[1],fltx[2]), 
							   AMREX_D_DECL(coeff*dtdx, coeff*dtdy, coeff*dtdz)); });

			if(lev > 0){
				amrex::ParallelFor(amrex::grow(bx,num_grow), conscomp,
				[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
				{  phidiff_bc(i, j, k, n, fabnew, AMREX_D_DECL(philox, philoy, philoz), 
				   AMREX_D_DECL(phihix, phihiy, phihz)); });
			}   

// add diffusive flux for scaling (shift this to rk = 2 step)
		if(rk == rk_max){
			amrex::ParallelFor(mfi.nodaltilebox(0), conscomp,
				[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
				{  fluxx(i,j,k,n) = flx[0](i,j,k,n) + fltx[0](i,j,k,n);    });
			amrex::ParallelFor(mfi.nodaltilebox(1), conscomp,
				[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
				{  fluxy(i,j,k,n) = flx[1](i,j,k,n) + fltx[1](i,j,k,n);    });
#if AMREX_SPACEDIM==3
			amrex::ParallelFor(mfi.nodaltilebox(2), conscomp,
				[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
				{  fluxz(i,j,k,n) = flx[2](i,j,k,n) + fltx[2](i,j,k,n); });
#endif                 
		}
	}
	// Print() << "reached end of FCT step 1, RK= " << rk << "\n";
// fill boundaries
#ifdef _OPENMP
		#pragma omp master
#endif
		{
			S_new.FillBoundary();
			FillDomBoundary(S_new,geom[lev],bcs,time);
			S_new.FillBoundary(geom[lev].periodicity());

			AMREX_D_TERM(Sconvx.FillBoundary();,Sconvy.FillBoundary();,Sconvz.FillBoundary());
			AMREX_D_TERM(Sconvx.FillBoundary(geom[lev].periodicity());, Sconvy.FillBoundary(geom[lev].periodicity());,
					 Sconvz.FillBoundary(geom[lev].periodicity()));
			AMREX_D_TERM(FillDomBoundary(Sconvx,geom[lev],bcs,time);,FillDomBoundary(Sconvy,geom[lev],bcs,time);,
					 FillDomBoundary(Sconvz,geom[lev],bcs,time));
		}
	// Print() << "about to stary FCT step 2, RK= " << rk << "\n";
//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------
// FCT step 2 (flux correction), RK = 1
	for (MFIter mfi(phi_new[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi){

		const Box& bx = mfi.tilebox();
		// ======== GET FACE VELOCITY =========
		AMREX_D_TERM(const Box& ngbxx = amrex::grow(mfi.nodaltilebox(0),num_grow);,
					 const Box& ngbxy = amrex::grow(mfi.nodaltilebox(1),num_grow);,
					 const Box& ngbxz = amrex::grow(mfi.nodaltilebox(2),num_grow));

// Define arrays for the different multifabs
		phitemp.resize(amrex::grow(bx,num_grow),S_new.nComp()-1);
		Array4<Real> fabtemp = phitemp.array();
		Array4<Real> fabold = S_old[mfi].array();
		Array4<Real> fabnew = S_new[mfi].array();

		AMREX_D_TERM(int philox = lbound(fabnew).x;, int philoy = lbound(fabnew).y;, int philoz = lbound(fabnew).z);
		AMREX_D_TERM(int phihix = ubound(fabnew).x;, int phihiy = ubound(fabnew).y;, int phihiz = ubound(fabnew).z);

		AMREX_D_TERM(Array4<Real> fabcx=Sconvx[mfi].array();, Array4<Real> fabcy=Sconvy[mfi].array();, Array4<Real> fabcz=Sconvz[mfi].array());

// Copy data from fabnew into fabtemp
		amrex::ParallelFor(amrex::grow(bx,num_grow), S_new.nComp()-1,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{   fabtemp(i,j,k,n) = fabnew(i,j,k,n); });
		
		GpuArray<Array4<Real>, AMREX_SPACEDIM> vel{ AMREX_D_DECL( facevel[lev][0].array(mfi),
													facevel[lev][1].array(mfi),facevel[lev][2].array(mfi)) };
				  
		Elixir fluxeli2[BL_SPACEDIM];
		for (int i = 0; i < BL_SPACEDIM ; i++) 
		{   fluxad[i].resize(amrex::grow(mfi.nodaltilebox(i),num_grow),conscomp);
			fluxeli2[i] = fluxad[i].elixir();  }

		GpuArray<Array4<Real>, AMREX_SPACEDIM> fltx{ AMREX_D_DECL( fluxad[0].array(),
													 fluxad[1].array(), fluxad[2].array()) };

// Define lower and upper bounds of fltmp
		AMREX_D_TERM(int ftx_lox = lbound(fltx[0]).x;,int ftx_loy = lbound(fltx[0]).y;,int ftx_loz = lbound(fltx[0]).z);
		AMREX_D_TERM(int ftx_hix = ubound(fltx[0]).x;,int ftx_hiy = ubound(fltx[0]).y;,int ftx_hiz = ubound(fltx[0]).z);

		AMREX_D_TERM(int fty_lox = lbound(fltx[1]).x;,int fty_loy = lbound(fltx[1]).y;,int fty_loz = lbound(fltx[1]).z);
		AMREX_D_TERM(int fty_hix = ubound(fltx[1]).x;,int fty_hiy = ubound(fltx[1]).y;,int fty_hiz = ubound(fltx[1]).z);

#if AMREX_SPACEDIM==3
		AMREX_D_TERM(int ftz_lox = lbound(fltx[2]).x;,int ftz_loy = lbound(fltx[2]).y;,int ftz_loz = lbound(fltx[2]).z);
		AMREX_D_TERM(int ftz_hix = ubound(fltx[2]).x;,int ftz_hiy = ubound(fltx[2]).y;,int ftz_hiz = ubound(fltx[2]).z);           
#endif
		Real diffc = diff1;
		amrex::ParallelFor(ngbxx, conscomp,
		  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
		  {  compute_ad_flux_x(i, j, k, n, fltx[0], vel[0], vel[1], fabold, fabcx, ftx_lox, ftx_hix, ftx_loy, ftx_hiy, coeff*dtdx, coeff*dtdy, diffc); });
		amrex::ParallelFor(ngbxy, conscomp,
		  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
		  {  compute_ad_flux_y(i, j, k, n, fltx[1], vel[0], vel[1], fabold, fabcy, fty_lox, fty_hix, fty_loy, fty_hiy, coeff*dtdx, coeff*dtdy, diffc); });
#if AMREX_SPACEDIM==3
		amrex::ParallelFor(ngbxz,
		[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
		{  compute_ad_flux_z(i, j, k, n, fltx[2], vel[2], fabold, fabcz, ftz_loz, ftz_hiz, coeff*dtdz); });
#endif
		amrex::ParallelFor(ngbxx, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  flux_x_bc(i, j, k, n, fltx[0], ftx_lox, ftx_hix);   });
		amrex::ParallelFor(ngbxy, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  flux_y_bc(i, j, k, n, fltx[1], fty_loy, fty_hiy);   });
#if AMREX_SPACEDIM==3
		amrex::ParallelFor(ngbxz, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  flux_z_bc(i, j, k, n, fltx[2], ftz_loz, ftz_hiz);   });
#endif
		Elixir fldxeli[BL_SPACEDIM];

		for (int i = 0; i < BL_SPACEDIM ; i++) {
			const Box& bxtmp = amrex::surroundingNodes(bx,i);
			fldiff[i].resize(amrex::grow(bxtmp,num_grow),conscomp);
			fldxeli[i] = fldiff[i].elixir();
		}
		GpuArray<Array4<Real>, AMREX_SPACEDIM> fldx{ AMREX_D_DECL( fldiff[0].array(),
													 fldiff[1].array(), fldiff[2].array()) };

		AMREX_D_TERM(int fdx_lox = lbound(fldx[0]).x;, int fdy_loy = lbound(fldx[1]).y;, int fdz_loz = lbound(fldx[2]).z);
		AMREX_D_TERM(int fdx_hix = ubound(fldx[0]).x;, int fdy_hiy = ubound(fldx[1]).y;, int fdz_hiz = ubound(fldx[2]).z);

// compute diffusive fluxes in the coordinate directions
		amrex::ParallelFor(ngbxx, conscomp,
		[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
		{  compute_diff_flux_x(i, j, k, n, fldx[0], vel[0], fabold, fdx_lox, fdx_hix, coeff*dtdx); });
		amrex::ParallelFor(ngbxy, conscomp,
		[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
		{  compute_diff_flux_y(i, j, k, n, fldx[1], vel[1], fabold, fdy_loy, fdy_hiy, coeff*dtdy); });
#if AMREX_SPACEDIM==3
		amrex::ParallelFor(ngbxz, conscomp
		[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
		{  compute_diff_flux_z(i, j, k, n, fldx[2], vel[2], fabold, fdz_loz, fdz_hiz, coeff*dtdz); });
#endif
		amrex::ParallelFor(ngbxx, conscomp,
		[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
		{  flux_x_bc(i, j, k, n, fldx[0], fdx_lox, fdx_hix);   });
		amrex::ParallelFor(ngbxy, conscomp,
		[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
		{  flux_y_bc(i, j, k, n, fldx[1], fdy_loy, fdy_hiy);   });
#if AMREX_SPACEDIM==3
		amrex::ParallelFor(ngbxz, conscomp,
		[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
		{  flux_z_bc(i, j, k, n, fldx[2], fdz_loz, fdz_hiz);   });
#endif

// update Sconvx, Sconvy and Sconvz to partially diffused values
		int ngrowbx = 1;
		if(lev > 0) ngrowbx = num_grow-1;
		const Box& iterbx = amrex::grow(mfi.tilebox(),ngrowbx);
		amrex::ParallelFor(iterbx, conscomp,
		[=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
		{   diff_updatephiconv(i, j, k, n, AMREX_D_DECL(fldx[0],fldx[1],fldx[2]), 
				AMREX_D_DECL(fabcx, fabcy, fabcz), AMREX_D_DECL(coeff*dtdx, coeff*dtdy, coeff*dtdz)); });
		if(lev > 0){
			amrex::ParallelFor(amrex::grow(bx,num_grow),conscomp,
				[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
				{  phidiffconv_bc(i, j, k, n, AMREX_D_DECL(fabcx, fabcy, fabcz), 
					AMREX_D_DECL(philox, philoy, philoz), AMREX_D_DECL(phihix, phihiy, phihiz)); });
		}

// prelimit the antidiffusive fluxes
// prelimit antidiffusive fluxes in x-direction
		AMREX_D_TERM(int ilo;, int jlo;, int klo); AMREX_D_TERM(int ihi;, int jhi;, int khi);
		if(lev == 0){   const Box& tempbx0 = mfi.tilebox();
			AMREX_D_TERM(ilo= lbound(tempbx0).x-1;, jlo= lbound(tempbx0).y;, klo= lbound(tempbx0).z); 
			AMREX_D_TERM(ihi= ubound(tempbx0).x+1;, jhi= lbound(tempbx0).y;, klo= ubound(tempbx0).z); }
		else{   const Box& tempbx1 = amrex::grow(mfi.tilebox(),num_grow-1);
			AMREX_D_TERM(ilo= lbound(tempbx1).x+1;, jlo= lbound(tempbx1).y;, klo= lbound(tempbx1).z); 
			AMREX_D_TERM(ihi= ubound(tempbx1).x;,   jhi= lbound(tempbx1).y;, klo= ubound(tempbx1).z);   }
		amrex::ParallelFor(ngbxx, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  prelimit_ad_flux_x(i, j, k, n, fltx[0], fabcx, AMREX_D_DECL(ilo,jlo,klo),AMREX_D_DECL(ihi,jhi,khi)); });
// prelimit antidiffusive fluxes in y-direction
		if(lev == 0){   const Box& tempby0 = mfi.tilebox();
			AMREX_D_TERM(ilo= lbound(tempby0).x;, jlo= lbound(tempby0).y-1;, klo= lbound(tempby0).z); 
			AMREX_D_TERM(ihi= ubound(tempby0).x;, jhi= lbound(tempby0).y+1;, klo= ubound(tempby0).z); }
		else{   const Box& tempby1 = amrex::grow(mfi.tilebox(),num_grow-1);
			 AMREX_D_TERM(ilo= lbound(tempby1).x;, jlo= lbound(tempby1).y+1;, klo= lbound(tempby1).z); 
			 AMREX_D_TERM(ihi= ubound(tempby1).x;, jhi= lbound(tempby1).y;,   klo= ubound(tempby1).z);   }
		amrex::ParallelFor(ngbxy, conscomp,
		   [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
		   {  prelimit_ad_flux_y(i, j, k, n, fltx[1], fabcy, AMREX_D_DECL(ilo,jlo,klo),AMREX_D_DECL(ihi,jhi,khi)); });
#if AMREX_SPACEDIM==3
		amrex::ParallelFor(ngbxz,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  prelimit_ad_flux_z(i, j, k, n, fltx[2], fabcz, fz_loz, fz_hiz); });
#endif
		if(lev > 0){
			amrex::ParallelFor(ngbxx, conscomp,
				[=] AMREX_GPU_DEVICE (int i, int j, int k, int n){  flux_x_bc(i,j,k,n,fltx[0],fdx_lox+1,fdx_hix-1);   });  
			amrex::ParallelFor(ngbxy, conscomp,
				[=] AMREX_GPU_DEVICE (int i, int j, int k, int n){  flux_y_bc(i,j,k,n,fltx[1],fdy_loy+1,fdy_hiy-1);   });
			amrex::ParallelFor(ngbxx, conscomp,
				[=] AMREX_GPU_DEVICE (int i, int j, int k, int n){  flux_x_bc(i,j,k,n,fltx[0],fdx_lox,fdx_hix);   });
			amrex::ParallelFor(ngbxy, conscomp,
				[=] AMREX_GPU_DEVICE (int i, int j, int k, int n){  flux_y_bc(i,j,k,n,fltx[1],fdy_loy,fdy_hiy);   });                      
		}
// resize frin and frout and define elixir on them (allocate memory for rk=1)
		Elixir frinel, froutel;
		ngrowbx = 1;
		if(lev > 0){ ngrowbx = 3;   }

		frin.resize(amrex::grow(mfi.tilebox(),ngrowbx),conscomp);
		frout.resize(amrex::grow(mfi.tilebox(),ngrowbx),conscomp); 

		frinel = frin.elixir(); froutel = frout.elixir();
		Array4<Real> arrin = frin.array(); Array4<Real> arrou = frout.array();
// perform flux correction steps by calculating fraction of fluxes entering and leaving cells
		amrex::ParallelFor(iterbx, conscomp,
			[=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
			{   get_flux_fracs(i, j, k, n, AMREX_D_DECL(fltx[0],fltx[1],fltx[2]), fabtemp, arrin, arrou); }); 
// compute the corrected fluxes before updating the solution
		// corrected fluxes in x-direction
		if(lev == 0){   const Box& tmpbox0 = mfi.tilebox();
			AMREX_D_TERM(ilo = lbound(tmpbox0).x;, jlo = lbound(tmpbox0).y;, lbound(tmpbox0).z); 
			AMREX_D_TERM(ihi = ubound(tmpbox0).x+1;, jhi = lbound(tmpbox0).y;, ubound(tmpbox0).z); }
		else{   const Box& tmpbox1 = amrex::grow(mfi.tilebox(),num_grow-1);
			AMREX_D_TERM(ilo = lbound(tmpbox1).x+1;, jlo = lbound(tmpbox1).y;, lbound(tmpbox1).z); 
			AMREX_D_TERM(ihi = ubound(tmpbox1).x;, jhi = lbound(tmpbox1).y;, ubound(tmpbox1).z);   }
		amrex::ParallelFor(ngbxx, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  corrected_flux_x(i, j, k, n, fltx[0], arrin, arrou, AMREX_D_DECL(ilo,jlo,klo), AMREX_D_DECL(ihi,jhi,khi)); });
		if(lev > 0){
				amrex::ParallelFor(ngbxx, conscomp,
				[=] AMREX_GPU_DEVICE (int i, int j, int k, int n){  flux_x_bc(i,j,k,n,fltx[0],ftx_lox+1,ftx_hix-1);   });
				amrex::ParallelFor(ngbxx, conscomp,
				[=] AMREX_GPU_DEVICE (int i, int j, int k, int n){  flux_x_bc(i,j,k,n,fltx[0],ftx_lox,ftx_hix);   });                      
		}

		// corrected fluxes in y-direction
		if(lev == 0){   const Box& bxxtemp0 = mfi.tilebox();
			AMREX_D_TERM(ilo = lbound(bxxtemp0).x;, jlo = lbound(bxxtemp0).y;, lbound(bxxtemp0).z); 
			AMREX_D_TERM(ihi = ubound(bxxtemp0).x;, jhi = lbound(bxxtemp0).y+1;, ubound(bxxtemp0).z); }
		else{   const Box& bxxtemp1 = amrex::grow(mfi.tilebox(),num_grow-1);
			AMREX_D_TERM(ilo = lbound(bxxtemp1).x;, jlo = lbound(bxxtemp1).y+1;, lbound(bxxtemp1).z); 
			AMREX_D_TERM(ihi = ubound(bxxtemp1).x;, jhi = lbound(bxxtemp1).y;, ubound(bxxtemp1).z);   }
		amrex::ParallelFor(ngbxy, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
			{  corrected_flux_y(i, j, k, n, fltx[1], arrin, arrou, AMREX_D_DECL(ilo,jlo,klo), AMREX_D_DECL(ihi,jhi,khi)); });
		if(lev > 0){
			amrex::ParallelFor(ngbxy, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n){  flux_y_bc(i,j,k,n,fltx[1],fty_loy+1,fty_hiy-1);   });
			amrex::ParallelFor(ngbxy, conscomp,
			[=] AMREX_GPU_DEVICE (int i, int j, int k, int n){  flux_y_bc(i,j,k,n,fltx[1],fty_loy,fty_hiy);   });
		}

// carry out flux correction step to get final solution
		if(lev == 0){ ngrowbx = 0;   } else{ ngrowbx = 3;  } 

		const Box& iterbx2 = amrex::grow(mfi.tilebox(),ngrowbx);

		amrex::ParallelFor(iterbx2, conscomp,
			[=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
			{   correct_phi(i, j, k, n, fabnew, fabtemp, AMREX_D_DECL(fltx[0],fltx[1],fltx[2]), minro); });

// calculate pressure and mach number (and other auxillary quantities defined)
        amrex::launch(iterbx2,
            [=] AMREX_GPU_DEVICE (Box const& tbx)
            {  CalcAuxillary(tbx, fabnew, minp);  });

        if(lev > 0){
            amrex::ParallelFor(amrex::grow(bx,num_grow),S_new.nComp(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {  phicorr_bc(i, j, k, n, fabnew,
                    AMREX_D_DECL(philox, philoy, philoz), AMREX_D_DECL(phihix, phihiy, phihiz)); });    
        }
// scale the fluxes for refluxing operation
        // Copy into Flux MultiFab
        GpuArray<Array4<Real>, AMREX_SPACEDIM> fluxx{ AMREX_D_DECL( fluxes[0].array(mfi),
                                                      fluxes[1].array(mfi),fluxes[2].array(mfi)) };

        if(do_reflux && rk == rk_max){

            amrex::ParallelFor(mfi.nodaltilebox(0), conscomp,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { scale_x_flux(i, j, k, n, fluxx[0], fltx[0], 1.0/dtdx, dt_lev, dy); });
            amrex::ParallelFor(mfi.nodaltilebox(1), conscomp,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { scale_y_flux(i, j, k, n, fluxx[1], fltx[1], 1.0/dtdy, dt_lev, dx); });

            // for(int dir = 0; dir < BL_SPACEDIM; dir++){
            // Print() << "i= " << dir << "fluxx_lim= " << lbound(fluxx[i]) << ", " << ubound(fluxx[i]) << "\n";
            // Print() << "i= " << dir << "flx_lim= " << lbound(flx[i]) << ", " << ubound(flx[i]) << "\n";
            // Print() << "ncomp= " << fluxx[dir].nComp() << ", " << flx[dir].nComp() << "\n";
            //     amrex::ParallelFor(mfi.nodaltilebox(dir), conscomp,
            //         [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            //         {   fluxx[dir](i,j,k,n) = flx[dir](i,j,k,n); });                        
            // }
        }                                                 
	  }// end(MFIter)
// fill boundaries
#ifdef _OPENMP
        #pragma omp master
#endif
        {
            S_new.FillBoundary();
            FillDomBoundary(S_new,geom[lev],bcs,time);
            S_new.FillBoundary(geom[lev].periodicity());

            AMREX_D_TERM(Sconvx.FillBoundary();,Sconvy.FillBoundary();,Sconvz.FillBoundary());
            AMREX_D_TERM(Sconvx.FillBoundary(geom[lev].periodicity());, Sconvy.FillBoundary(geom[lev].periodicity());,
                Sconvz.FillBoundary(geom[lev].periodicity()));
            AMREX_D_TERM(FillDomBoundary(Sconvx,geom[lev],bcs,time);,FillDomBoundary(Sconvy,geom[lev],bcs,time);,
                FillDomBoundary(Sconvz,geom[lev],bcs,time));
        }

	}//end(omp)
  }//end(rk)

	// ======== CFL CHECK, MOVED OUTSIDE MFITER LOOP =========

	AMREX_D_TERM(Real umax = facevel[lev][0].norminf(0,0,true);,
				 Real vmax = facevel[lev][1].norminf(0,0,true);,
				 Real wmax = facevel[lev][2].norminf(0,0,true));

	if (AMREX_D_TERM(umax*dt_lev > dx, ||
					 vmax*dt_lev > dy, ||
					 wmax*dt_lev > dz))
	{
#if (AMREX_SPACEDIM > 2)
		amrex::AllPrint() << "umax = " << umax << ", vmax = " << vmax << ", wmax = " << wmax
						  << ", dt = " << dt_lev << " dx = " << dx << " " << dy << " " << dz << std::endl;
#else
		amrex::AllPrint() << "umax = " << umax << ", vmax = " << vmax
						  << ", dt = " << dt_lev << " dx = " << dx << " " << dy << " " << dz << std::endl;
#endif
		amrex::Abort("CFL violation. use smaller adv.cfl.");
	}

	// increment or decrement the flux registers by area and time-weighted fluxes
	// Note that the fluxes have already been scaled by dt and area
	// In this example we are solving phi_t = -div(+F)
	// The fluxes contain, e.g., F_{i+1/2,j} = (phi*u)_{i+1/2,j}
	// Keep this in mind when considering the different sign convention for updating
	// the flux registers from the coarse or fine grid perspective
	// NOTE: the flux register associated with flux_reg[lev] is associated
	// with the lev/lev-1 interface (and has grid spacing associated with lev-1)
	if (do_reflux) { 
	if (flux_reg[lev+1]) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// update the lev+1/lev flux register (index lev+1)   
			flux_reg[lev+1]->CrseInit(fluxes[i],i,0,0,conscomp, -1.0);
		}       
	}
	if (flux_reg[lev]) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// update the lev/lev-1 flux register (index lev) 
		flux_reg[lev]->FineAdd(fluxes[i],i,0,0,conscomp, 1.0);
		}
	}
	}
	ParallelDescriptor::Barrier();
}
