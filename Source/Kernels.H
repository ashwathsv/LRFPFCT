#ifndef Kernels_H_
#define Kernels_H_

#include <face_velocity.H>
#include <Prob.H>
#include <Adv_K.H>
#include <updatephi_K.H>
#include <Tagging.H>
#include <bc_fill.H>
#include <fctconstants.H>

#if (AMREX_SPACEDIM == 2)
#include <compute_flux_2D_K.H>
#else
#include <compute_flux_3D_K.H>
#endif

#endif
