/*
 * David Frey
 * University of Michigan Medicine Department of Radiology
 * Functional MRI Laboratory, PI: Luis Hernandez-Garcia
 *
 * GE Medical Systems
 * Copyright (C) 1996-2003 The General Electric Company
 *
 * File Name : umvsasl.e
 * Language  : EPIC/ANSI C
 * Date      : 10-May-2023
 *
 * a velocity selective ASL-prepped turboFLASH sequence (umvsasl),
 * built on grass.e
 */

@inline epic.h
@inline intwave.h

@global
/*********************************************************************
 *                   UMVSASL.E GLOBAL SECTION                        *
 *                                                                   *
 * Common code shared between the Host and IPG PSD processes.  This  *
 * section contains all the #define's, global variables and function *
 * declarations (prototypes).                                        *
 *********************************************************************/
#include <stdio.h>
#include <string.h>

#include "em_psd_ermes.in"
#include "grad_rf_umvsasl.globals.h"

#include "stddef_ep.h"
#include "epicconf.h"
#include "pulsegen.h"
#include "epic_error.h"
#include "epic_loadcvs.h"
#include "InitAdvisories.h"
#include "psdiopt.h"
#ifdef psdutil
#include "psdutil.h"
#endif
#include "psd_proto.h"
#include "epic_iopt_util.h"
#include "filter.h"

#include "umvsasl.h"

/* Define important values */
#define MAXWAVELEN 50000 /* Maximum wave length for gradients */
#define MAXNSHOTS 512 /* Maximum number of echo trains per frame */
#define MAXNECHOES 512 /* Maximum number of echoes per echo train */
#define MAXNFRAMES 1000 /* Maximum number of temporal frames */
#define MAXITR 50 /* Maximum number of iterations for iterative processes */
#define GAMMA 26754 /* Gyromagnetic ratio (rad/s/G) */
#define TIMESSI 120 /* SSP instruction time */
#define SPOIL_SEED 21001 /* rf spoiler seed */

@inline Prescan.e PSglobal
int debugstate = 1;

@ipgexport
/*********************************************************************
 *                 UMVSASL.E IPGEXPORT SECTION                       *
 *                                                                   *
 * Standard C variables of _any_ type common for both the Host and   *
 * IPG PSD processes. Declare here all the complex type, e.g.,       *
 * structures, arrays, files, etc.                                   *
 *                                                                   *
 * NOTE FOR Lx:                                                      *
 * Since the architectures between the Host and the IPG schedule_ides are    *
 * different, the memory alignment for certain types varies. Hence,  *
 * the following types are "forbidden": short, char, and double.     *
 *********************************************************************/
@inline Prescan.e PSipgexport
RF_PULSE_INFO rfpulseInfo[RF_FREE] = { {0,0} };

/* Define temporary error message string */
char tmpstr[200];

/* Declare sequencer hardware limit variables */
float XGRAD_max;
float YGRAD_max;
float ZGRAD_max;
float RHO_max;
float THETA_max;
int ZGRAD_risetime;
int ZGRAD_falltime;

/* Declare readout gradient waveform arrays */
int Gx[MAXWAVELEN];
int Gy[MAXWAVELEN];
int grad_len = 5000;
int acq_len = 4000;
int acq_offset = 50;

/* Declare table of readout gradient transformation matrices */
long tmtxtbl[MAXNSHOTS*MAXNECHOES][9];

/* Declare ASL prep pulse variables */
int prep1_len = 5000;
int prep1_rho_lbl[MAXWAVELEN];
int prep1_theta_lbl[MAXWAVELEN];
int prep1_grad_lbl[MAXWAVELEN];
int prep1_rho_ctl[MAXWAVELEN];
int prep1_theta_ctl[MAXWAVELEN];
int prep1_grad_ctl[MAXWAVELEN];

int prep2_len = 5000;
int prep2_rho_lbl[MAXWAVELEN];
int prep2_theta_lbl[MAXWAVELEN];
int prep2_grad_lbl[MAXWAVELEN];
int prep2_rho_ctl[MAXWAVELEN];
int prep2_theta_ctl[MAXWAVELEN];
int prep2_grad_ctl[MAXWAVELEN];

/* Declare receiver and Tx frequencies */
float recfreq;
float xmitfreq;

@cv
/*********************************************************************
 *                      UMVSASL.E CV SECTION                         *
 *                                                                   *
 * Standard C variables of _limited_ types common for both the Host  *
 * and IPG PSD processes. Declare here all the simple types, e.g,    *
 * int, float, and C structures containing the min and max values,   *
 * and ID description, etc.                                          *
 *                                                                   *
 * NOTE FOR Lx:                                                      *
 * Since the architectures between the Host and the IPG schedule_ides are    *
 * different, the memory alignment for certain types varies. Hence,  *
 * the following types are "forbidden": short, char, and double.     *
 *********************************************************************/
@inline loadrheader.e rheadercv
@inline vmx.e SysCVs

@inline Prescan.e PScvs

int numdda = 4;			/* For Prescan: # of disdaqs ps2*/

float SLEWMAX = 12500.0 with {1000, 25000.0, 12500.0, VIS, "Maximum allowed slew rate (G/cm/s)",};
float GMAX = 4.0 with {0.5, 5.0, 4.0, VIS, "Maximum allowed gradient (G/cm)",};
float RFMAX = 300 with {0, 500, 300, VIS, "Maximum allowed RF amplitude (mG)",};

/* readout cvs */
int nframes = 2 with {1, , 2, VIS, "Number of frames",};
int ndisdaqtrains = 2 with {0, , 2, VIS, "Number of disdaq echo trains at beginning of scan loop",};
int ndisdaqechoes = 0 with {0, , 0, VIS, "Number of disdaq echos at beginning of echo train",};

int echo_mode = 0 with {0, 1, 0, VIS, "SPGR (0) or FSE (1)",};
int fatsat_flag = 0 with {0, 1, 0, VIS, "Option to do play a fat saturation pulse/crusher before the readout",};
int rfspoil_flag = 0 with {0, 1, 0, VIS, "Option to do RF phase cycling (117deg increments to tipdown phase)",};
int flowcomp_flag = 0 with {0, 1, 0, VIS, "Option to use flow-compensated slice select gradients",};

int pgbuffertime = 248 with {100, , 248, INVIS, "Gradient IPG buffer time (us)",};
float crushfac = 3.0 with {0, 10, 0, VIS, "Crusher amplitude factor (a.k.a. cycles of phase/vox; dk_crush = crushfac*kmax)",};
int kill_grads = 0 with {0, 1, 0, VIS, "Option to turn off readout gradients",};

int tr_deadtime = 0 with {0, , 0, INVIS, "TR deadtime (us)",};

/* Trajectory cvs */
int nnav = 250 with {0, 1000, 250, VIS, "Number of navigator points in spiral",};
int narms = 1 with {1, 1000, 1, VIS, "Number of spiral arms",};
int spi_mode = 0 with {0, 2, 0, VIS, "SPI mode (0 = SOS, 1 = TGA, 2 = 3DTGA)",};
float kz_acc = 1.0 with {1, 100.0, 1.0, VIS, "kz acceleration (SENSE) factor (for SOS only)",};
float vds_acc0 = 1.0 with {0.001, 50.0, 1.0, VIS, "Spiral center oversampling factor",};
float vds_acc1 = 1.0 with {0.001, 50.0, 1.0, VIS, "Spiral edge oversampling factor",};
float F0 = 0 with { , , 0, INVIS, "vds fov coefficient 0",};
float F1 = 0 with { , , 0, INVIS, "vds fov coefficient 1",};
float F2 = 0 with { , , 0, INVIS, "vds fov coefficient 2",};

/* ASL prep pulse cvs */
int presat_flag = 0 with {0, 1, 0, VIS, "Option to play asl pre-saturation pulse at beginning of each tr",};
int presat_delay = 1000000 with {0, , 1000000, VIS, "ASL pre-saturation delay (us)",};

int zero_ctl_grads = 0 with {0, 1, 0, VIS, "option to zero out control gradients for asl prep pulses",};

int prep1_id = 0 with {0, , 0, VIS, "ASL prep pulse 1: ID number (0 = no pulse)",};
int prep1_pld = 0 with {0, , 0, VIS, "ASL prep pulse 1: post-labeling delay (us; includes background suppression)",};
int prep1_ncycles = 1 with {1, , 1, VIS, "ASL prep pulse 1: number of cycles",};
float prep1_rfmax = 234 with {0, , 0, VIS, "ASL prep pulse 1: maximum RF amplitude",};
float prep1_gmax = 1.5 with {0, , 3, VIS, "ASL prep pulse 1: maximum gradient amplitude",};
int prep1_mod = 1 with {1, 4, 1, VIS, "ASL prep pulse 1: labeling modulation scheme (1 = label/control, 2 = control/label, 3 = always label, 4 = always control)",};
int prep1_tbgs1 = 0 with {0, , 0, VIS, "ASL prep pulse 1: 1st background suppression delay (0 = no pulse)",};
int prep1_tbgs2 = 0 with {0, , 0, VIS, "ASL prep pulse 1: 2nd background suppression delay (0 = no pulse)",};
int prep1_tbgs3 = 0 with {0, , 0, VIS, "ASL prep pulse 1: 3rd background suppression delay (0 = no pulse)",};

int prep2_id = 0 with {0, , 0, VIS, "ASL prep pulse 2: ID number (0 = no pulse)",};
int prep2_pld = 0 with {0, , 0, VIS, "ASL prep pulse 2: post-labeling delay (us; includes background suppression)",};
int prep2_ncycles = 1 with {1, , 1, VIS, "ASL prep pulse 2: number of cycles",};
float prep2_rfmax = 234 with {0, , 0, VIS, "ASL prep pulse 2: maximum RF amplitude",};
float prep2_gmax = 1.5 with {0, , 1.5, VIS, "ASL prep pulse 2: maximum gradient amplitude",};
int prep2_mod = 1 with {1, 4, 1, VIS, "ASL prep pulse 2: labeling modulation scheme (1 = label/control, 2 = control/label, 3 = always label, 4 = always control)",};
int prep2_tbgs1 = 0 with {0, , 0, VIS, "ASL prep pulse 2: 1st background suppression delay (0 = no pulse)",};
int prep2_tbgs2 = 0 with {0, , 0, VIS, "ASL prep pulse 2: 2nd background suppression delay (0 = no pulse)",};
int prep2_tbgs3 = 0 with {0, , 0, VIS, "ASL prep pulse 2: 3rd background suppression delay (0 = no pulse)",};

/* Declare core duration variables */
int dur_presatcore = 0 with {0, , 0, INVIS, "Duration of the ASL pre-saturation core (us)",};
int dur_prep1core = 0 with {0, , 0, INVIS, "Duration of the ASL prep 1 cores (us)",};
int dur_prep2core = 0 with {0, , 0, INVIS, "Duration of the ASL prep 2 cores (us)",};
int dur_bkgsupcore = 0 with {0, , 0, INVIS, "Duration of the background suppression core (us)",};
int dur_fatsatcore = 0 with {0, , 0, INVIS, "Duration of the fat saturation core (us)",};
int dur_tipdowncore = 0 with {0, , 0, INVIS, "Duration of the slice selective tipdown core (us)",};
int dur_seqcore = 0 with {0, , 0, INVIS, "Duration of the spiral readout core (us)",};
int deadtime1_seqcore = 0 with {0, , 0, INVIS, "Pre-readout deadtime within core (us)",};
int deadtime2_seqcore = 0 with {0, , 0, INVIS, "Post-readout deadtime within core (us)",};

/* inhereted from grass.e: */
float xmtaddScan;
int obl_debug = 0 with {0, 1, 0, INVIS, "On(=1) to print messages for obloptimize",};
int obl_method = 0 with {0, 1, 0, INVIS, "On(=1) to optimize the targets based on actual rotation matrices",};
int debug = 0 with {0,1,0,INVIS,"1 if debug is on ",};
float echo1bw = 16 with {,,,INVIS,"Echo1 filter bw.in KHz",};

@host
/*********************************************************************
 *                     UMVSASL.E HOST SECTION                        *
 *                                                                   *
 * Write here the code unique to the Host PSD process. The following *
 * functions must be declared here: cvinit(), cveval(), cvcheck(),   *
 * and predownload().                                                *
 *                                                                   *
 *********************************************************************/
#include <math.h>
#include <stdlib.h>
#include "grad_rf_umvsasl.h"
#include "psdopt.h"
#include "sar_pm.h"
#include "support_func.host.h"
#include "helperfuns.h"
#include "vds.c"

/* fec : Field strength dependency library */
#include <sysDep.h>
#include <sysDepSupport.h>      /* FEC : fieldStrength dependency libraries */

@inline loadrheader.e rheaderhost

/** Load PSD Header **/
abstract("umvsasl sequence");
psdname("umvsasl");

int num_conc_grad = 3;          /* always three for grass 	*/
int entry;

/* peak B1 amplitudes */
float maxB1[MAX_ENTRY_POINTS], maxB1Seq;

/* This will point to a structure defining parameters of the filter
   used for the 1st echo */
FILTER_INFO *echo1_filt; 

/* Use real time filters, so allocate space for them instead of trying
   to point to an infinite number of structures in filter.h. */
FILTER_INFO echo1_rtfilt;

/* Golden ratio numbers */
float PHI = (1.0 + sqrt(5.0)) / 2.0; /* 1d golden ratio */
float phi1 = 0.4656; /* 2d golden ratio 1 */
float phi2 = 0.6823; /* 2d golden ratio 2 */

/* Declare trajectory generation function prototypes */
int genspiral();
int genviews();

/* Declare function prototypes from aslprep.h */
int readprep(int id, int *len,
		int *rho_lbl, int *theta_lbl, int *grad_lbl,
		int *rho_ctl, int *theta_ctl, int *grad_ctl); 
float calc_sinc_B1(float cyc_rf, int pw_rf, float flip_rf);
float calc_hard_B1(int pw_rf, float flip_rf);

@inline Prescan.e PShostVars            /* added with new filter calcs */

static char supfailfmt[] = "Support routine %s failed";


/************************************************************************/
/*       			CVINIT    				*/
/* Invoked once (& only once) when the PSD host process	is started up.	*/
/* Code which is independent of any OPIO button operation is put here.	*/
/************************************************************************/
STATUS cvinit( void )
{

	/* turn off bandwidth option */
	cvdef(oprbw, 500.0 / (float)GRAD_UPDATE_TIME);
	cvmin(oprbw, 500.0 / (float)GRAD_UPDATE_TIME);
	cvmax(oprbw, 500.0 / (float)GRAD_UPDATE_TIME);
	oprbw = 500.0 / (float)GRAD_UPDATE_TIME;
	pircbnub = 0;

	/* fov */
	opfov = 240;
	pifovnub = 5;
	pifovval2 = 200;
	pifovval3 = 220;
	pifovval4 = 240;
	pifovval5 = 260;
	pifovval6 = 280;

	/* tr */
	opautotr = PSD_MINIMUMTR;
	pitrnub = 2;
	pitrval2 = PSD_MINIMUMTR;
	cvmax(optr,50s);

	/* te */
	opautote = PSD_MINTE;	
	pite1nub = 3;
	pite1val2 = PSD_MINTE;
	cvmin(opte, 0);
	cvmax(opte, 500ms);

	/* esp */
	esp = 100ms;
	cvmin(esp, 0);
	cvmax(esp, 500ms);

	/* rhrecon */
	rhrecon = 2327;

	/* frequency (xres) */
	opxres = 64;
	cvmin(opxres, 16);
	cvmax(opxres, 512);
	pixresnub = 15;
	pixresval2 = 32;
	pixresval3 = 64;
	pixresval4 = 128;

	/* flip angle */
	cvmin(opflip, 0.0);
	cvmax(opflip, 360.0);
	pifanub = 2;
	pifaval2 = 90.0;

	/* echo train length */
	cvmin(opetl, 1);
	cvmax(opetl, MAXNECHOES);
	pietlnub = 7;
	pietlval2 = 1;
	pietlval3 = 16;

	/* nshots */
	cvmin(opnshots, 1);
	cvmax(opnshots, MAXNSHOTS);
	pishotnub = 2;
	pishotval2 = 1;	

	/* hide phase (yres) option */
	piyresnub = 0;

	/* Hide inversion time */
	pitinub = 0;

	/* hide second bandwidth option */
	pircb2nub = 0;

	/* hide nex stuff */
	piechnub = 0;
	pinexnub = 0;

#ifdef ERMES_DEBUG
	use_ermes = 0;
#else /* !ERMES_DEBUG */
	use_ermes = 1;
#endif /* ERMES_DEBUG */

	configSystem();
	EpicConf();
	inittargets(&loggrd, &phygrd);

	/* Init filter slots */
	initfilter();
	
	if (_psd_rf_wait.fixedflag == 0)  { /* sets psd_grd_wait and psd_rf_wait */
		if (setsysparms() == FAILURE)  {
			epic_error(use_ermes,"Support routine setsysparams failed",
					EM_PSD_SUPPORT_FAILURE,1, STRING_ARG,"setsysparms");
			return FAILURE;
		}
	}

	if( obloptimize( &loggrd, &phygrd, scan_info, exist(opslquant),
				exist(opplane), exist(opcoax), obl_method, obl_debug,
				&opnewgeo, cfsrmode ) == FAILURE )
	{
		return FAILURE;
	}
	
	/* Get sequencer hardware limits */
	gettarget(&XGRAD_max, XGRAD, &loggrd);
	gettarget(&YGRAD_max, YGRAD, &loggrd);
	gettarget(&ZGRAD_max, ZGRAD, &loggrd);
	gettarget(&RHO_max, RHO, &loggrd);
	gettarget(&THETA_max, THETA, &loggrd);
	getramptime(&ZGRAD_risetime, &ZGRAD_falltime, ZGRAD, &loggrd);	
	ZGRAD_risetime *= 2; /* extra fluffy */
	fprintf(stderr, "ZGRAD_risetime = %d\n", ZGRAD_risetime);	

@inline Prescan.e PScvinit

#include "cvinit.in"	/* Runs the code generated by macros in preproc.*/

	return SUCCESS;
}   /* end cvinit() */

@inline InitAdvisories.e InitAdvPnlCVs

/************************************************************************/
/*       			CVEVAL    				*/
/* Called w/ every OPIO button push which has a corresponding CV. 	*/
/* CVEVAL should only contain code which impacts the advisory panel--	*/
/* put other code in cvinit or predownload				*/
/************************************************************************/
STATUS cveval( void )
{
	configSystem();
	InitAdvPnlCVs();

	pititle = 1;
	cvdesc(pititle, "Advanced pulse sequence parameters");
	piuset = 0;

	/* Add opuser fields to the Adv. pulse sequence parameters interface */	
	piuset += use1;
	cvdesc(opuser1, "Echo spacing (ms)");
	cvdef(opuser1, esp*1e-3);
	opuser1 = esp*1e-3;
	cvmin(opuser1, 0);
	cvmax(opuser1, 1000);	
	esp = opuser1*1e3;

	piuset += use2;
	cvdesc(opuser2, "Number of frames to acquire");
	cvdef(opuser2, nframes);
	opuser2 = nframes;
	cvmin(opuser2, 1);
	cvmax(opuser2, MAXNFRAMES);
	nframes = opuser2;
	
	piuset += use3;
	cvdesc(opuser3, "Number of disdaq trains");
	cvdef(opuser3, ndisdaqtrains);
	opuser3 = ndisdaqtrains;
	cvmin(opuser3, 0);
	cvmax(opuser3, 100);
	ndisdaqtrains = opuser3;
	
	piuset += use4;
	cvdesc(opuser4, "Number of disdaq echoes");
	cvdef(opuser4, ndisdaqechoes);
	opuser4 = ndisdaqechoes;
	cvmin(opuser4, 0);
	cvmax(opuser4, 100);
	ndisdaqechoes = opuser4;
	
	piuset += use5;
	cvdesc(opuser5, "Number of spiral arms");
	cvdef(opuser5, narms);
	opuser5 = narms;
	cvmin(opuser5, 1);
	cvmax(opuser5, opxres);
	narms = opuser5;

	piuset += use7;
	cvdesc(opuser7, "kz acceleration factor (SOS only)");
	cvdef(opuser7, kz_acc);
	opuser7 = kz_acc;
	cvmin(opuser7, 1.0);
	cvmax(opuser7, 100.0);
	kz_acc = opuser7;

	piuset += use8;
	cvdesc(opuser8, "VD-spiral center oversampling factor");
	cvdef(opuser8, vds_acc0);
	opuser8 = vds_acc0;
	cvmin(opuser8, 0.001);
	cvmax(opuser8, 50.0);
	vds_acc0 = opuser8;

	piuset += use9;
	cvdesc(opuser9, "VD-spiral edge oversampling factor");
	cvdef(opuser9, vds_acc1);
	opuser9 = vds_acc1;
	cvmin(opuser9, 0.001);
	cvmax(opuser9, 50.0);
	vds_acc1 = opuser9;

	piuset += use10;
	cvdesc(opuser10, "Recon script ID #");
	cvdef(opuser10, rhrecon);
	opuser10 = rhrecon;
	cvmin(opuser10, 0);
	cvmax(opuser10, 9999);
	rhrecon = opuser10;

@inline Prescan.e PScveval

	return SUCCESS;
}   /* end cveval() */

void getAPxParam(optval   *min,
		optval   *max,
		optdelta *delta,
		optfix   *fix,
		float    coverage,
		int      algorithm)
{
	/* Need to be filled when APx is supported in this PSD */
}

int getAPxAlgorithm(optparam *optflag, int *algorithm)
{
	return APX_CORE_NONE;
}

/************************************************************************/
/*       			CVCHECK    				*/
/* Executed on each 'next page' to ensure prescription can proceed 	*/
/* to the next page. 							*/
/************************************************************************/
STATUS cvcheck( void )
{
	return SUCCESS;
}   /* end cvcheck() */


/************************************************************************/
/*             		    PRE-DOWNLOAD           		        */
/* Executed prior to a download--all operations not needed for the 	*/
/* advisory panel results.  Execute the	pulsegen macro expansions for	*/
/* the predownload section here.  All internal amps, slice ordering,  	*/
/* prescan slice calc., and SAT placement calculations are performed 	*/
/* in this section.  Time anchor settings for pulsegen are done in this */
/* section too.  				 			*/
/************************************************************************/
STATUS predownload( void )
{
	FILE* finfo;
	int echo1_freq[opslquant], rf1_freq[opslquant];
	int slice;
	float kzmax;
	int minesp, minte, maxte, absmintr;	
	float rf1_b1;
	float rfps1_b1, rfps2_b1, rfps3_b1, rfps4_b1;
	float rffs_b1, rfbs_b1;
	float prep1_b1, prep2_b1;
	int tmp_pwa, tmp_pw, tmp_pwd;
	float tmp_a, tmp_area;

	/*********************************************************************/
#include "predownload.in"	/* include 'canned' predownload code */
	/*********************************************************************/
	
	/* Read in asl prep pulses */
	fprintf(stderr, "predownload(): calling readprep() to read in ASL prep 1 pulse\n");
	if (readprep(prep1_id, &prep1_len,
		prep1_rho_lbl, prep1_theta_lbl, prep1_grad_lbl,
		prep1_rho_ctl, prep1_theta_ctl, prep1_grad_ctl) == 0)
	{
		epic_error(use_ermes,"failure to read in ASL prep 1 pulse", EM_PSD_SUPPORT_FAILURE, EE_ARGS(0));
		return FAILURE;
	}
	
	fprintf(stderr, "predownload(): calling readprep() to read in ASL prep 2 pulse\n");
	if (readprep(prep2_id, &prep2_len,
		prep2_rho_lbl, prep2_theta_lbl, prep2_grad_lbl,
		prep2_rho_ctl, prep2_theta_ctl, prep2_grad_ctl) == 0)
	{
		epic_error(use_ermes,"failure to read in ASL prep 2 pulse", EM_PSD_SUPPORT_FAILURE, EE_ARGS(0));
		return FAILURE;
	}
	
	/* update presat pulse parameters */
	pw_rfps1 = 1ms; /* 1ms hard pulse */
	pw_rfps1a = 0; /* hard pulse - no ramp */
	pw_rfps1d = 0;
	pw_rfps2 = 2ms; /* 2ms hard pulse */
	pw_rfps2a = 0; /* hard pulse - no ramp */
	pw_rfps2d = 0;
	pw_rfps3 = 3ms; /* 3ms hard pulse */
	pw_rfps3a = 0; /* hard pulse - no ramp */
	pw_rfps3d = 0;
	pw_rfps4 = 4ms; /* 4ms hard pulse */
	pw_rfps4a = 0; /* hard pulse - no ramp */
	pw_rfps4d = 0;
	
	/* update sinc pulse parameters */
	pw_rf1 = 3200;
	pw_rffs = 3200;

	/* Update the background suppression pulse parameters */
	res_rfbs_rho = 500;
	pw_rfbs_rho = 5000;
	a_rfbs_theta = 1.0;
	res_rfbs_theta = res_rfbs_rho;
	pw_rfbs_theta = pw_rfbs_rho;

	/* Set the parameters for the fat sat pulse */
	a_rffs = 0.5 * 440 / 1250;	
	pw_rffs = 4 * round(cyc_rffs*1e6 / 440);
	res_rffs = pw_rffs / 2;	
	
	/* Set the parameters for the spin echo tipdown kspace rewinder */
	tmp_area = a_gzrf1 * (pw_gzrf1 + (pw_gzrf1a + pw_gzrf1d)/2.0);
	amppwgrad(tmp_area, GMAX, 0, 0, ZGRAD_risetime, 0, &tmp_a, &tmp_pwa, &tmp_pw, &tmp_pwd); 
	tmp_a *= -0.5;
	pw_gzrf1r = tmp_pw;
	pw_gzrf1ra = tmp_pwa;
	pw_gzrf1rd = tmp_pwd;
	a_gzrf1r = tmp_a;

	/* Set the parameters for the crusher gradients */
	tmp_area = crushfac * 2*M_PI/GAMMA * opxres/(opfov/10.0) * 1e6; /* Area under crusher s.t. dk = crushfac*kmax (G/cm*us) */
	amppwgrad(tmp_area, GMAX, 0, 0, ZGRAD_risetime, 0, &tmp_a, &tmp_pwa, &tmp_pw, &tmp_pwd); 	

	pw_gzrffsspoil = tmp_pw;
	pw_gzrffsspoila = tmp_pwa;
	pw_gzrffsspoild = tmp_pwd;
	a_gzrffsspoil = tmp_a;

	pw_gzrf1spoil = tmp_pw;
	pw_gzrf1spoila = tmp_pwa;
	pw_gzrf1spoild = tmp_pwd;
	a_gzrf1spoil = tmp_a;

	/* set parameters for tipdown kspace rewinder */
	tmp_area = a_gzrf1 * (pw_gzrf1 + (pw_gzrf1a + pw_gzrf1d)/2.0);
	amppwgrad(tmp_area, GMAX, 0, 0, ZGRAD_risetime, 0, &tmp_a, &tmp_pwa, &tmp_pw, &tmp_pwd); 	
	tmp_a *= -0.5;
	pw_gzrf1r = tmp_pw;
	pw_gzrf1ra = tmp_pwa;
	pw_gzrf1rd = tmp_pwd;
	a_gzrf1r = tmp_a;
	
	/* set parameters for flow compensated kz-encode (pre-scaled to kzmax) */
	kzmax = (float)(kz_acc * opetl * opnshots) / ((float)opfov/10.0) / 2.0;
	tmp_area = 2*M_PI/(GAMMA*1e-6) * kzmax * (1 + flowcomp_flag); /* multiply by 2 if flow compensated */
	amppwgrad(tmp_area, GMAX, 0, 0, ZGRAD_risetime, 0, &tmp_a, &tmp_pwa, &tmp_pw, &tmp_pwd); 	
	pw_gzw = tmp_pw;
	pw_gzwa = tmp_pwa;
	pw_gzwd = tmp_pwd;
	a_gzw = tmp_a;

	/* set parameters for flowcomp pre-phaser */
	tmp_area = 2*M_PI/(GAMMA*1e-6) * kzmax; /* multiply by 2 if flow compensated */
	amppwgrad(tmp_area, GMAX, 0, 0, ZGRAD_risetime, 0, &tmp_a, &tmp_pwa, &tmp_pw, &tmp_pwd); 
	pw_gzfc = tmp_pw;
	pw_gzfca = tmp_pwa;
	pw_gzfcd = tmp_pwd;
	a_gzfc = -tmp_a;
	
	/* generate initial spiral trajectory */
	fprintf(stderr, "predownload(): calculating spiral gradients...\n");
	if (genspiral() == 0) {
		epic_error(use_ermes,"failure to generate spiral waveform", EM_PSD_SUPPORT_FAILURE, EE_ARGS(0));
		return FAILURE;
	}
	a_gxw = XGRAD_max;
	a_gyw = YGRAD_max;
	ia_gxw = MAX_PG_WAMP;
	ia_gyw = MAX_PG_WAMP;
	res_gxw = grad_len;
	res_gyw = grad_len;
	pw_gxw = GRAD_UPDATE_TIME*res_gxw;
	pw_gyw = GRAD_UPDATE_TIME*res_gyw;
	
	/* Generate view transformations */
	if (genviews() == 0) {
		epic_error(use_ermes,"failure to generate view transformation matrices", EM_PSD_SUPPORT_FAILURE, EE_ARGS(0));
		return FAILURE;
	}
	scalerotmats(tmtxtbl, &loggrd, &phygrd, opetl*opnshots*narms, 0);

	/* calculate minimum ESP as beginning of tipdowncore to end of seqcore without deadtimes*/
	fprintf(stderr, "predownload(): calculating minesp...\n");
	minesp = 1ms; /* fudge factor */
	
	/* add tipdowncore object durations */
	minesp += pgbuffertime; /* buffer */
	minesp += pw_gzrf1spoila + pw_gzrf1spoil + pw_gzrf1spoild; /* rf1 spoiler */
	minesp += pgbuffertime; /* buffer */
	minesp += pw_gzrf1a + pw_gzrf1 + pw_gzrf1d; /* rf1 and s.s. gradients */
	minesp += pgbuffertime; /* buffer */
	minesp += pw_gzrf1ra + pw_gzrf1r + pw_gzrf1rd; /* refocuser gradient */
	minesp += pgbuffertime; /* buffer */

	/* add intercore time */
	minesp += TIMESSI;

	/* add seqcore object durations */
	minesp += pgbuffertime; /* buffer */
	if (flowcomp_flag) {
		minesp += pw_gzfca + pw_gzfc + pw_gzfcd; /* flow comp prephaser gradient */
		minesp += pgbuffertime; /* buffer */
	}
	minesp += pw_gzwa + pw_gzw + pw_gzwd; /* kz blip gradient */
	minesp += pgbuffertime; /* buffer */
	minesp += pw_gxw; /* spiral readout gradients */
	minesp += pgbuffertime; /* buffer */
	
	/* add intercore time */
	minesp += TIMESSI;
	
	minesp = GRAD_UPDATE_TIME * ceil((float)minesp / (float)GRAD_UPDATE_TIME); /* round to nearest sampling interval */
	fprintf(stderr, "predownload(): minesp = %dus...\n", minesp);
	
	/* calculate minimum TE as center of tipdown rf to beginning of spiral readout without deadtimes */
	fprintf(stderr, "predownload(): calculating minte...\n");
	minte = 200us; /* fudge factor */

	/* add tipdowncore object durations */
	minte = (pw_gzrf1/2 + pw_gzrf1d); /* 2nd half of rf1 pulse & s.s. gradients */
	minte += pgbuffertime; /* buffer */
	minte += pw_gzrf1ra + pw_gzrf1r + pw_gzrf1rd; /* rewinder */
	minte += pgbuffertime; /* buffer */

	/* add intercore time */
	minte += TIMESSI;

	/* add seqcore object durations */
	minte += pgbuffertime; /* buffer */
	if (flowcomp_flag) {
		minte += pw_gzfca + pw_gzfc + pw_gzfcd; /* flowcomp prephaser */
		minte += pgbuffertime; /* buffer */
	}
	minte += pw_gzwa + pw_gzw + pw_gzwd; /* z encode + flowcomp rephaser */
	minte += pgbuffertime; /* buffer */
	
	minte = GRAD_UPDATE_TIME * ceil((float)minte / (float)GRAD_UPDATE_TIME); /* round to nearest sampling interval */
	fprintf(stderr, "predownload(): minte = %dus...\n", minte);

	/* calculate max TE as ESP minus minte */
	fprintf(stderr, "predownload(): calculating maxte...\n");
	maxte = esp - minte; /* echo spacing */
	fprintf(stderr, "predownload(): maxte = %dus...\n", maxte);

	fprintf(stderr, "predownload(): calculating seqcore deadtimes...\n");
	deadtime1_seqcore = opte - minte;
	minesp += deadtime1_seqcore;
	deadtime2_seqcore = esp - minesp; 
	fprintf(stderr, "predownload(): deadtime1_seqcore = %dus, deadtime2_seqcore = %dus...\n", deadtime1_seqcore, deadtime2_seqcore);
	
	cvmin(esp, minesp);
	cvmin(opuser1, minesp*1e-3);	
	if ((exist(opautote) == PSD_MINTE)||(exist(opautote) == PSD_MINTEFULL))
		opte = minte;
	cvmin(opte, minte);
	cvmax(opte, maxte);
	
	/* Update the asl prep pulse parameters */
	a_prep1gradlbl = (prep1_id > 0) ? (prep1_gmax) : (0);
	ia_prep1gradlbl = (int)ceil(a_prep1gradlbl / ZGRAD_max * (float)MAX_PG_WAMP);
	a_prep1gradctl = (prep1_id > 0) ? (prep1_gmax) : (0); 
	ia_prep1gradctl = (int)ceil(a_prep1gradctl / ZGRAD_max * (float)MAX_PG_WAMP);
	a_prep2gradlbl = (prep2_id > 0) ? (prep2_gmax) : (0);
	ia_prep2gradlbl = (int)ceil(a_prep2gradlbl / ZGRAD_max * (float)MAX_PG_WAMP);
	a_prep2gradctl = (prep2_id > 0) ? (prep2_gmax) : (0); 
	ia_prep2gradctl = (int)ceil(a_prep2gradctl / ZGRAD_max * (float)MAX_PG_WAMP);

	/* First, find the peak B1 for all entry points (other than L_SCAN) */
	for( entry=0; entry < MAX_ENTRY_POINTS; ++entry )
	{
		if( peakB1( &maxB1[entry], entry, RF_FREE, rfpulse ) == FAILURE )
		{
			epic_error( use_ermes, "peakB1 failed.", EM_PSD_SUPPORT_FAILURE,
					EE_ARGS(1), STRING_ARG, "peakB1" );
			return FAILURE;
		}
	}

	fprintf(stderr, "opflip = %f\n", (float)opflip);
	rf1_b1 = calc_sinc_B1(cyc_rf1, pw_rf1, opflip);
	fprintf(stderr, "predownload(): maximum B1 for rf1 pulse: %f\n", rf1_b1);
	if (rf1_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = rf1_b1;
	
	rfps1_b1 = calc_hard_B1(pw_rfps1, 72.0); /* flip angle of 72 degrees */
	fprintf(stderr, "predownload(): maximum B1 for presat pulse 1: %f Gauss \n", rfps1_b1);
	if (rfps1_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = rfps1_b1;
	
	rfps2_b1 = calc_hard_B1(pw_rfps2, 92.0); /* flip angle of 92 degrees */
	fprintf(stderr, "predownload(): maximum B1 for presat pulse 2: %f Gauss \n", rfps2_b1);
	if (rfps2_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = rfps2_b1;
	
	rfps3_b1 = calc_hard_B1(pw_rfps3, 126.0); /* flip angle of 126 degrees */
	fprintf(stderr, "predownload(): maximum B1 for presat pulse 3: %f Gauss \n", rfps3_b1);
	if (rfps3_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = rfps3_b1;
	
	rfps4_b1 = calc_hard_B1(pw_rfps4, 193.0); /* flip angle of 193 degrees */
	fprintf(stderr, "predownload(): maximum B1 for presat pulse 4: %f Gauss \n", rfps4_b1);
	if (rfps4_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = rfps4_b1;

	rfbs_b1 = 0.234;
	fprintf(stderr, "predownload(): maximum B1 for background suppression prep pulse: %f Gauss \n", rfbs_b1);
	if (rfbs_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = rfbs_b1;
	
	rffs_b1 = calc_sinc_B1(cyc_rffs, pw_rffs, 90.0);
	fprintf(stderr, "predownload(): maximum B1 for fatsat pulse: %f Gauss\n", rffs_b1);
	if (rffs_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = rffs_b1;
	
	prep1_b1 = (prep1_id > 0) ? (prep1_rfmax*1e-3) : (0);
	fprintf(stderr, "predownload(): maximum B1 for prep1 pulse: %f Gauss\n", prep1_b1);
	if (prep1_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = prep1_b1;

	prep2_b1 = (prep2_id > 0) ? (prep2_rfmax*1e-3) : (0);
	fprintf(stderr, "predownload(): maximum B1 for prep2 pulse: %f Gauss\n", prep2_b1);
	if (prep2_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = prep2_b1;

	/* Determine peak B1 across all entry points */
	maxB1Seq = RFMAX * 1e-3;  /* units: maxB1Seq is in  Gauss */

	/* Set xmtadd according to maximum B1 and rescale for powermon,
	   adding additional (audio) scaling if xmtadd is too big.
	   Add in coilatten, too. */
	xmtaddScan = -200 * log10( maxB1[L_SCAN] / maxB1Seq ) + getCoilAtten(); 

	if( xmtaddScan > cfdbmax )
	{
		extraScale = (float)pow( 10.0, (cfdbmax - xmtaddScan) / 200.0 );
		xmtaddScan = cfdbmax;
	} 
	else
	{
		extraScale = 1.0;
	}
	
	/* Update all the rf amplitudes */
	a_rf1 = rf1_b1 / maxB1Seq;
	ia_rf1 = a_rf1 * MAX_PG_WAMP;

	a_rfps1 = rfps1_b1 / maxB1Seq;
	ia_rfps1 = a_rfps1 * MAX_PG_WAMP;
	a_rfps2 = rfps2_b1 / maxB1Seq;
	ia_rfps2 = a_rfps2 * MAX_PG_WAMP;
	a_rfps3 = rfps3_b1 / maxB1Seq;
	ia_rfps3 = a_rfps3 * MAX_PG_WAMP;
	a_rfps4 = rfps4_b1 / maxB1Seq;
	ia_rfps4 = a_rfps4 * MAX_PG_WAMP;
	a_rfbs_rho = rfbs_b1 / maxB1Seq;
	ia_rfbs_rho = a_rfbs_rho * MAX_PG_WAMP;
	
	a_rffs = rffs_b1 / maxB1Seq;
	ia_rffs = a_rffs * MAX_PG_WAMP;
	
	a_prep1rholbl = prep1_b1 / maxB1Seq;
	ia_prep1rholbl = a_prep1rholbl * MAX_PG_WAMP;
	
	a_prep1rhoctl = prep1_b1 / maxB1Seq;
	ia_prep1rhoctl = a_prep1rhoctl * MAX_PG_WAMP;
	
	a_prep2rholbl = prep2_b1 / maxB1Seq;
	ia_prep2rholbl = a_prep2rholbl * MAX_PG_WAMP;
	
	a_prep2rhoctl = prep2_b1 / maxB1Seq;
	ia_prep2rhoctl = a_prep2rhoctl * MAX_PG_WAMP;

	/* Calculate the duration of presatcore */
	dur_presatcore = 0;
	dur_presatcore += pgbuffertime;
	dur_presatcore += 4*4000;
	dur_presatcore += pgbuffertime;

	/* Calcualte the duration of prep1core */
	dur_prep1core = 0;
	dur_prep1core += pgbuffertime;
	dur_prep1core += GRAD_UPDATE_TIME*prep1_len;
	dur_prep1core += pgbuffertime;

	/* Calculate the duration of prep2core */
	dur_prep2core = 0;
	dur_prep2core += pgbuffertime;
	dur_prep2core += GRAD_UPDATE_TIME*prep2_len;
	dur_prep2core += pgbuffertime;

	/* Calculate the duration of bkgsupcore */
	dur_bkgsupcore = 0;
	dur_bkgsupcore += pgbuffertime;
	dur_bkgsupcore += pw_rfbs_rho;
	dur_bkgsupcore += pgbuffertime;	

	/* Calculate the duration of fatsatcore */
	dur_fatsatcore = 0;
	dur_fatsatcore += pgbuffertime;
	dur_fatsatcore += pw_rffs;
	dur_fatsatcore += pgbuffertime;
	dur_fatsatcore += pw_gzrffsspoila + pw_gzrffsspoil + pw_gzrffsspoild;
	dur_fatsatcore += pgbuffertime;
	
	/* calculate duration of tipdowncore */
	dur_tipdowncore = 0;
	dur_tipdowncore += pgbuffertime;
	dur_tipdowncore += pw_gzrf1spoila + pw_gzrf1spoil + pw_gzrf1spoild;
	dur_tipdowncore += pgbuffertime;
	dur_tipdowncore += pw_gzrf1a + pw_gzrf1 + pw_gzrf1d;
	dur_tipdowncore += pgbuffertime;
	dur_tipdowncore += pw_gzrf1ra + pw_gzrf1r + pw_gzrf1rd;
	dur_tipdowncore += pgbuffertime; 

	/* calculate duration of seqcore */
	dur_seqcore = 0;
	dur_seqcore += deadtime1_seqcore + pgbuffertime;
	if (flowcomp_flag) {
		dur_seqcore += pw_gzfca + pw_gzfc + pw_gzfcd;
		dur_seqcore += pgbuffertime;
	}
	dur_seqcore += pw_gzwa + pw_gzw + pw_gzwd;
	dur_seqcore += pgbuffertime;	
	dur_seqcore += pw_gxw;
	dur_seqcore += pgbuffertime + deadtime2_seqcore;

	/* calculate minimum TR */
	absmintr = presat_flag*(dur_presatcore + TIMESSI + presat_delay);
	absmintr += (prep1_id > 0)*(dur_prep1core + TIMESSI + prep1_pld + TIMESSI);
	absmintr += (prep2_id > 0)*(dur_prep2core + TIMESSI + prep2_pld + TIMESSI);
	absmintr += fatsat_flag*(dur_fatsatcore + TIMESSI);
	absmintr += (opetl + ndisdaqechoes) * (dur_tipdowncore + TIMESSI + dur_seqcore + TIMESSI);
	if (exist(opautotr) == PSD_MINIMUMTR)
		optr = absmintr;	
	cvmin(optr, absmintr);

	/* calculate TR deadtime */
	tr_deadtime = optr - absmintr;
	
	/* 
	 * Calculate RF filter and update RBW:
	 *   &echo1_rtfilt: I: all the filter parameters.
	 *   exist(oprbw): I/O: desired and final allowable bw.
	 *   exist(opxres): I: output pts generated by filter.
	 *   OVERWRITE_OPRBW: oprbw will be updated.
	 */
	if( calcfilter( &echo1_rtfilt,
				exist(oprbw),
				acq_len,
				OVERWRITE_OPRBW ) == FAILURE)
	{
		epic_error( use_ermes, supfailfmt, EM_PSD_SUPPORT_FAILURE,
				EE_ARGS(1), STRING_ARG, "calcfilter:echo1" );
		return FAILURE;
	}

	echo1_filt = &echo1_rtfilt;

	/* Divide by 0 protection */
	if( (echo1_filt->tdaq == 0) || 
			floatsAlmostEqualEpsilons(echo1_filt->decimation, 0.0f, 2) ) 
	{
		epic_error( use_ermes, "echo1 tdaq or decimation = 0",
				EM_PSD_BAD_FILTER, EE_ARGS(0) );
		return FAILURE;
	}

	/* For use on the RSP schedule_ide */
	echo1bw = echo1_filt->bw;

@inline Prescan.e PSfilter

	/* For Prescan: Inform 'Auto' Prescan about prescan parameters 	*/
	pislquant = 10;	/* # of 2nd pass slices */

	/* For Prescan: Declare the entry point table 	*/
	if( entrytabinit( entry_point_table, (int)ENTRY_POINT_MAX ) == FAILURE ) 
	{
		epic_error( use_ermes, supfailfmt, EM_PSD_SUPPORT_FAILURE,
				EE_ARGS(1), STRING_ARG, "entrytabinit" );
		return FAILURE;
	}

	/* For Prescan: Define the entry points in the table */
	/* Scan Entry Point */
	(void)strcpy( entry_point_table[L_SCAN].epname, "scan" );
	entry_point_table[L_SCAN].epfilter = (unsigned char)echo1_filt->fslot;
	entry_point_table[L_SCAN].epprexres = acq_len;

	(void)strcpy( entry_point_table[L_APS2].epname, "aps2" );
	entry_point_table[L_APS2].epfilter = (unsigned char)echo1_filt->fslot;
	entry_point_table[L_APS2].epprexres = acq_len;

	(void)strcpy( entry_point_table[L_MPS2].epname, "mps2" );
	entry_point_table[L_MPS2].epfilter = (unsigned char)echo1_filt->fslot;
	entry_point_table[L_MPS2].epprexres = acq_len;

	/* set sequence clock */
	pidmode = PSD_CLOCK_NORM;
	pitslice = optr;
	pitscan = (nframes*narms*opnshots + ndisdaqtrains) * optr; /* pitscan controls the clock time on the interface */	
	
	/* Set up the filter structures to be downloaded for realtime 
	   filter generation. Get the slot number of the filter in the filter rack 
	   and assign to the appropriate acquisition pulse for the right 
	   filter selection - LxMGD, RJF */
	setfilter( echo1_filt, SCAN );
	filter_echo1 = echo1_filt->fslot;
	entry_point_table[L_SCAN].epxmtadd = (short)rint( (double)xmtaddScan );

	/* APS2 & MPS2 */
	entry_point_table[L_APS2] = entry_point_table[L_MPS2] = entry_point_table[L_SCAN];	/* copy scan into APS2 & MPS2 */
	(void)strcpy( entry_point_table[L_APS2].epname, "aps2" );
	(void)strcpy( entry_point_table[L_MPS2].epname, "mps2" );

	/* Set up Tx/Rx frequencies */
	for (slice = 0; slice < opslquant; slice++) rsp_info[slice].rsprloc = 0;
	setupslices(rf1_freq, rsp_info, opslquant, a_gzrf1, 1.0, opfov, TYPTRANSMIT);
	setupslices(echo1_freq, rsp_info, opslquant, 0.0, 1.0, 2.0, TYPREC);

	/* Average together all slice frequencies */
	xmitfreq = 0;
	recfreq = 0;	
	for (slice = 0; slice < opslquant; slice++) {
		xmitfreq += (float)rf1_freq[slice] / (float)opslquant;
		recfreq += (float)echo1_freq[slice] / (float)opslquant;
	}

	if( orderslice( TYPNORMORDER, MAXNFRAMES+1, MAXNFRAMES+1, TRIG_INTERN ) == FAILURE )
	{
		epic_error( use_ermes, supfailfmt, EM_PSD_SUPPORT_FAILURE,
				EE_ARGS(1), STRING_ARG, "orderslice" );
	}

	/* nex, exnex, acqs and acq_type are used in the rhheaderinit routine */
	/* -- to initialize recon header variables */
	if( floatsAlmostEqualEpsilons(opnex, 1.0, 2) )
	{
		baseline = 8;
		nex = 1;
		exnex = 1;
	}
	else
	{
		baseline = 0;
		nex = opnex;
		exnex = opnex;
	}

@inline loadrheader.e rheaderinit   /* Recon variables */
	
	/* Set recon header variables:
	 *   rhptsize: number of bytes per data point
	 *   rhfrsize: number of data points per acquisition
	 *   rhrawsize: total number of bytes to allocate
	 *   rhrcctrl: recon image control (bitmap)
	 *   rhexecctrl: recon executive control (bitmap)
	 */ 
	cvmax(rhfrsize, 32767);
	cvmax(rhnframes, 32767);
	cvmax(rhnslices, 32767);

	rhfrsize = acq_len;
	rhnframes = 2*ceil((float)(opetl * narms * opnshots + 1) / 2.0);
	rhnecho = 1;
	rhnslices = nframes + 1;
	rhrawsize = 2*rhptsize*rhfrsize * (rhnframes + 1) * rhnslices * rhnecho;
	
	rhrcctrl = 1; /* bit 7 (2^7 = 128) skips all recon */
	rhexecctrl = 2; /* bit 1 (2^1 = 2) sets autolock of raw files + bit 3 (2^3 = 8) transfers images to disk */
	
	/* Print scan info to a file */
	finfo = fopen("scaninfo.txt", "w");

	fprintf(finfo, "cvs:\n");
	fprintf(finfo, "\t%-50s%20f\n", "opfov:", opfov);
	fprintf(finfo, "\t%-50s%20f\n", "opflip:", opflip);
	fprintf(finfo, "\t%-50s%20d\n", "opslquant:", opslquant);
	fprintf(finfo, "\t%-50s%20f\n", "opslthick:", opslthick);
	fprintf(finfo, "\t%-50s%20f\n", "optr:", (float)optr);
	fprintf(finfo, "\t%-50s%20d\n", "opte:", opte);	
	fprintf(finfo, "\t%-50s%20d\n", "opnshots:", opnshots);
	fprintf(finfo, "\t%-50s%20d\n", "opetl:", opetl);	

	fprintf(finfo, "hardware cvs:\n");
	fprintf(finfo, "\t%-50s%20f\n", "SLEWMAX:", SLEWMAX);
	fprintf(finfo, "\t%-50s%20f\n", "GMAX:", GMAX);
	fprintf(finfo, "\t%-50s%20f\n", "RFMAX:", RFMAX);

	fprintf(finfo, "readout cvs:\n");
	fprintf(finfo, "\t%-50s%20d\n", "nframes:", nframes);
	fprintf(finfo, "\t%-50s%20d\n", "ndisdaqtrains", ndisdaqtrains);
	fprintf(finfo, "\t%-50s%20d\n", "ndisdaqechoes", ndisdaqechoes);
	fprintf(finfo, "\t%-50s%20d\n", "fatsat_flag:", fatsat_flag);
	fprintf(finfo, "\t%-50s%20d\n", "pgbuffertime:", pgbuffertime);
	fprintf(finfo, "\t%-50s%20d\n", "kill_grads:", kill_grads);
	fprintf(finfo, "\t%-50s%20d\n", "rfspoil_flag:", rfspoil_flag);	
	fprintf(finfo, "\t%-50s%20d\n", "flowcomp_flag:", flowcomp_flag);	

	fprintf(finfo, "trajectory cvs:\n");
	fprintf(finfo, "\t%-50s%20d\n", "narms:", narms);
	fprintf(finfo, "\t%-50s%20d\n", "nnav:", nnav);
	fprintf(finfo, "\t%-50s%20f\n", "vds_acc0:", vds_acc0);
	fprintf(finfo, "\t%-50s%20f\n", "vds_acc1:", vds_acc1);
	fprintf(finfo, "\t%-50s%20f\n", "kz_acc:", kz_acc);
	
	fprintf(finfo, "ASL prep cvs:\n");
	fprintf(finfo, "\t%-50s%20d\n", "presat_flag:", presat_flag);
	fprintf(finfo, "\t%-50s%20d\n", "presat_delay:", presat_delay);	
	fprintf(finfo, "\t%-50s%20d\n", "prep1_id:", prep1_id);
	fprintf(finfo, "\t%-50s%20d\n", "prep1_pld:", prep1_pld);
	fprintf(finfo, "\t%-50s%20d\n", "prep1_ncycles", prep1_ncycles);
	fprintf(finfo, "\t%-50s%20f\n", "prep1_rfmax:", prep1_rfmax);
	fprintf(finfo, "\t%-50s%20f\n", "prep1_gmax:", prep1_gmax);
	fprintf(finfo, "\t%-50s%20d\n", "prep1_mod", prep1_mod);
	fprintf(finfo, "\t%-50s%20d\n", "prep1_tbgs1", prep1_tbgs1);
	fprintf(finfo, "\t%-50s%20d\n", "prep1_tbgs2", prep1_tbgs2);
	fprintf(finfo, "\t%-50s%20d\n", "prep1_tbgs3", prep1_tbgs3);
	fprintf(finfo, "\t%-50s%20d\n", "prep2_id:", prep2_id);
	fprintf(finfo, "\t%-50s%20d\n", "prep2_pld:", prep2_pld);
	fprintf(finfo, "\t%-50s%20d\n", "prep2_ncycles", prep2_ncycles);
	fprintf(finfo, "\t%-50s%20f\n", "prep2_rfmax:", prep2_rfmax);
	fprintf(finfo, "\t%-50s%20f\n", "prep2_gmax:", prep2_gmax);
	fprintf(finfo, "\t%-50s%20d\n", "prep2_mod", prep2_mod);
	fprintf(finfo, "\t%-50s%20d\n", "prep2_tbgs1", prep2_tbgs1);
	fprintf(finfo, "\t%-50s%20d\n", "prep2_tbgs2", prep2_tbgs2);
	fprintf(finfo, "\t%-50s%20d\n", "prep2_tbgs3", prep2_tbgs3);

	fprintf(finfo, "\ntime cvs:\n");
	fprintf(finfo, "\t%-50s%20f\n", "pitscan:", pitscan);
	fprintf(finfo, "\t%-50s%20d\n", "dur_presatcore:", dur_presatcore);
	fprintf(finfo, "\t%-50s%20d\n", "dur_prep1core:", dur_prep1core);
	fprintf(finfo, "\t%-50s%20d\n", "dur_prep2core:", dur_prep2core);
	fprintf(finfo, "\t%-50s%20d\n", "dur_bkgsupcore:", dur_bkgsupcore);
	fprintf(finfo, "\t%-50s%20d\n", "dur_fatsatcore:", dur_fatsatcore);
	fprintf(finfo, "\t%-50s%20d\n", "dur_tipdowncore:", dur_tipdowncore);
	fprintf(finfo, "\t%-50s%20d\n", "dur_seqcore:", dur_seqcore);
	fprintf(finfo, "\t%-50s%20d\n", "tr_deadtime:", tr_deadtime);
	
	fclose(finfo);

@inline Prescan.e PSpredownload	

	return SUCCESS;
}   /* end predownload() */


@inline Prescan.e PShost


@pg
/*********************************************************************
 *                  UMVSASL.E PULSEGEN SECTION                       *
 *                                                                   *
 * Write here the functional code that loads hardware sequencer      *
 * memory with data that will allow it to play out the sequence.     *
 * These functions call pulse generation macros previously defined   *
 * with @pulsedef, and must return SUCCESS or FAILURE.               *
 *********************************************************************/
#include "support_func.h"
#include "epicfuns.h"


STATUS pulsegen( void )
{
	sspinit(psd_board_type);
	int tmploc;	


	/*************************/
	/* generate readout core */
	/*************************/
	fprintf(stderr, "pulsegen(): beginning pulse generation of readout core (seqcore)\n");
	tmploc = 0;
	tmploc += deadtime1_seqcore + pgbuffertime; /* add pre-readout deadtime + buffer */

	if (flowcomp_flag) {
		fprintf(stderr, "pulsegen(): generating gzfc... (flow comp dephaser gradient\n");
		TRAPEZOID(ZGRAD, gzfc, tmploc + pw_gzfca, 1ms, 0, loggrd);	
		fprintf(stderr, "\tstart: %dus, ", tmploc);
		tmploc += pw_gzfca + pw_gzfc + pw_gzfcd; /* end time for gzfc */
		fprintf(stderr, " end: %dus\n", tmploc);
		tmploc += pgbuffertime; /* add some buffer */
	}	

	fprintf(stderr, "pulsegen(): generating gzw... (z encode + flow comp rephase gradient\n");
	TRAPEZOID(ZGRAD, gzw, tmploc + pw_gzwa, 1ms, 0, loggrd);	
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzwa + pw_gzw + pw_gzwd; /* end time for gzw */
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime; /* add some buffer */

	fprintf(stderr, "pulsegen(): generating gxw, gyw (spiral readout gradients) and echo1 (data acquisition window)...\n");
	INTWAVE(XGRAD, gxw, tmploc, XGRAD_max, grad_len, GRAD_UPDATE_TIME*grad_len, Gx, 1, loggrd);
	INTWAVE(YGRAD, gyw, tmploc, YGRAD_max, grad_len, GRAD_UPDATE_TIME*grad_len, Gy, 1, loggrd);
	ACQUIREDATA(echo1, tmploc + psd_grd_wait + GRAD_UPDATE_TIME*acq_offset,,,);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gxw; /* end time for readout */
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime; /* add some buffer */

	tmploc += deadtime2_seqcore; /* add post-readout deadtime */

	fprintf(stderr, "pulsegen(): finalizing spiral readout core...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_seqcore, tmploc);
	SEQLENGTH(seqcore, dur_seqcore, seqcore);
	fprintf(stderr, "\tDone.\n");


	/************************/
	/* generate presat core */
	/************************/	
	fprintf(stderr, "pulsegen(): beginning pulse generation of presatcore (asl pre-saturation core)\n");
	tmploc = 0;
	
	fprintf(stderr, "pulsegen(): generating rfps1 (presat rf pulse 1)...\n");
	tmploc += pgbuffertime; /* start time for rfps1 pulse */
	TRAPEZOID(RHO, rfps1, tmploc + psd_rf_wait, 1ms, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rfps1;
	fprintf(stderr, " end: %dus\n", tmploc);
	
	fprintf(stderr, "pulsegen(): generating rfps1c (presat rf crusher 1)...\n");
	tmploc += pgbuffertime; /*start time for gzrf1spoil */
	TRAPEZOID(ZGRAD, rfps1c, tmploc + pw_rfps1ca, 1ms, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rfps1ca + pw_rfps1c + pw_rfps1cd; /* end time for gzrf1spoil */
	fprintf(stderr, " end: %dus\n", tmploc);

	fprintf(stderr, "pulsegen(): generating rfps2 (presat rf pulse 2)...\n");
	tmploc = 4000 + pgbuffertime; /* start time for rfps2 pulse */
	TRAPEZOID(RHO, rfps2, tmploc + psd_rf_wait, 1ms, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rfps2;
	fprintf(stderr, " end: %dus\n", tmploc);
	
	fprintf(stderr, "pulsegen(): generating rfps2c (presat rf crusher 2)...\n");
	tmploc += pgbuffertime; /*start time for gzrf1spoil */
	TRAPEZOID(ZGRAD, rfps2c, tmploc + pw_rfps1ca, 1ms, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rfps2ca + pw_rfps1c + pw_rfps1cd; /* end time for gzrf1spoil */
	fprintf(stderr, " end: %dus\n", tmploc);
	
	fprintf(stderr, "pulsegen(): generating rfps3 (presat rf pulse 3)...\n");
	tmploc = 8000 + pgbuffertime; /* start time for rfps3 pulse */
	TRAPEZOID(RHO, rfps3, tmploc + psd_rf_wait, 1ms, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rfps3;
	fprintf(stderr, " end: %dus\n", tmploc);
	
	fprintf(stderr, "pulsegen(): generating rfps3c (presat rf crusher 3)...\n");
	tmploc += pgbuffertime;
	TRAPEZOID(ZGRAD, rfps3c, tmploc + pw_rfps1ca, 1ms, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rfps3ca + pw_rfps1c + pw_rfps1cd;
	fprintf(stderr, " end: %dus\n", tmploc);
	
	fprintf(stderr, "pulsegen(): generating rfps4 (presat rf pulse 4)...\n");
	tmploc = 12000 + pgbuffertime;
	TRAPEZOID(RHO, rfps4, tmploc + psd_rf_wait, 1ms, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rfps4;
	fprintf(stderr, " end: %dus\n", tmploc);
	
	fprintf(stderr, "pulsegen(): generating rfps4c (presat rf crusher 4)...\n");
	tmploc += pgbuffertime;
	TRAPEZOID(ZGRAD, rfps4c, tmploc + pw_rfps1ca, 1ms, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rfps4ca + pw_rfps1c + pw_rfps1cd;
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime; /* add some buffer time */

	fprintf(stderr, "pulsegen(): finalizing presatcore...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_presatcore, tmploc);
	SEQLENGTH(presatcore, dur_presatcore, presatcore);
	fprintf(stderr, "\tDone.\n");


	/**************************/
	/* generate prep1lbl core */
	/**************************/	
	fprintf(stderr, "pulsegen(): beginning pulse generation of prep1lblcore (prep1 label core)\n");
	tmploc = 0;
	
	fprintf(stderr, "pulsegen(): generating prep1rholbl, prep1thetalbl & prep1gradlbl (prep1 label rf & gradients)...\n");
	tmploc += pgbuffertime; /* start time for prep1 pulse */
	INTWAVE(RHO, prep1rholbl, tmploc + psd_rf_wait, 0.0, prep1_len, GRAD_UPDATE_TIME*prep1_len, prep1_rho_lbl, 1, loggrd); 
	INTWAVE(THETA, prep1thetalbl, tmploc + psd_rf_wait, 1.0, prep1_len, GRAD_UPDATE_TIME*prep1_len, prep1_theta_lbl, 1, loggrd); 
	INTWAVE(ZGRAD, prep1gradlbl, tmploc, 0.0, prep1_len, GRAD_UPDATE_TIME*prep1_len, prep1_grad_lbl, 1, loggrd); 
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += GRAD_UPDATE_TIME*prep1_len; /* end time for prep1 pulse */
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime; /* add some buffer */

	fprintf(stderr, "pulsegen(): finalizing prep1lblcore...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_prep1core, tmploc);
	SEQLENGTH(prep1lblcore, dur_prep1core, prep1lblcore);
	fprintf(stderr, "\tDone.\n");


	/**************************/
	/* generate prep1ctl core */
	/**************************/	
	fprintf(stderr, "pulsegen(): beginning pulse generation of prep1ctlcore (prep1 control core)\n");
	tmploc = 0;
	
	fprintf(stderr, "pulsegen(): generating prep1rhoctl, prep1thetactl & prep1gradctl (prep1 control rf & gradients)...\n");
	tmploc += pgbuffertime; /* start time for prep1 pulse */
	INTWAVE(RHO, prep1rhoctl, tmploc + psd_rf_wait, 0.0, prep1_len, GRAD_UPDATE_TIME*prep1_len, prep1_rho_ctl, 1, loggrd); 
	INTWAVE(THETA, prep1thetactl, tmploc + psd_rf_wait, 1.0, prep1_len, GRAD_UPDATE_TIME*prep1_len, prep1_theta_ctl, 1, loggrd); 
	INTWAVE(ZGRAD, prep1gradctl, tmploc, 0.0, prep1_len, GRAD_UPDATE_TIME*prep1_len, prep1_grad_ctl, 1, loggrd); 
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += GRAD_UPDATE_TIME*prep1_len; /* end time for prep1 pulse */
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime; /* add some buffer */

	fprintf(stderr, "pulsegen(): finalizing prep1ctlcore...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_prep1core, tmploc);
	SEQLENGTH(prep1ctlcore, dur_prep1core, prep1ctlcore);
	fprintf(stderr, "\tDone.\n");
	

	/**************************/
	/* Generate prep2lbl core */
	/**************************/	
	fprintf(stderr, "pulsegen(): beginning pulse generation of prep2lblcore (prep2 label core)\n");
	tmploc = 0;
	
	fprintf(stderr, "pulsegen(): generating prep2rholbl, prep2thetalbl & prep2gradlbl (prep2 label rf & gradients)...\n");
	tmploc += pgbuffertime; /* start time for prep2 pulse */
	INTWAVE(RHO, prep2rholbl, tmploc + psd_rf_wait, 0.0, prep2_len, GRAD_UPDATE_TIME*prep2_len, prep2_rho_lbl, 1, loggrd); 
	INTWAVE(THETA, prep2thetalbl, tmploc + psd_rf_wait, 1.0, prep2_len, GRAD_UPDATE_TIME*prep2_len, prep2_theta_lbl, 1, loggrd); 
	INTWAVE(ZGRAD, prep2gradlbl, tmploc, 0.0, prep2_len, GRAD_UPDATE_TIME*prep2_len, prep2_grad_lbl, 1, loggrd); 
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += GRAD_UPDATE_TIME*prep2_len; /* end time for prep2 pulse */
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime; /* add some buffer */

	fprintf(stderr, "pulsegen(): finalizing prep2lblcore...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_prep2core, tmploc);
	SEQLENGTH(prep2lblcore, dur_prep2core, prep2lblcore);
	fprintf(stderr, "\tDone.\n");


	/**************************/
	/* Generate prep2ctl core */
	/**************************/	
	fprintf(stderr, "pulsegen(): beginning pulse generation of prep2ctlcore (prep2 control core)\n");
	tmploc = 0;
	
	fprintf(stderr, "pulsegen(): generating prep2rhoctl, prep2thetactl & prep2gradctl (prep2 control rf & gradients)...\n");
	tmploc += pgbuffertime; /* start time for prep2 pulse */
	INTWAVE(RHO, prep2rhoctl, tmploc + psd_rf_wait, 0.0, prep2_len, GRAD_UPDATE_TIME*prep2_len, prep2_rho_ctl, 1, loggrd); 
	INTWAVE(THETA, prep2thetactl, tmploc + psd_rf_wait, 1.0, prep2_len, GRAD_UPDATE_TIME*prep2_len, prep2_theta_ctl, 1, loggrd); 
	INTWAVE(ZGRAD, prep2gradctl, tmploc, 0.0, prep2_len, GRAD_UPDATE_TIME*prep2_len, prep2_grad_ctl, 1, loggrd); 
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += GRAD_UPDATE_TIME*prep2_len; /* end time for prep2 pulse */
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime; /* add some buffer */

	fprintf(stderr, "pulsegen(): finalizing prep2ctlcore...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_prep2core, tmploc);
	SEQLENGTH(prep2ctlcore, dur_prep2core, prep2ctlcore);
	fprintf(stderr, "\tDone.\n");
		
	
	/************************/
	/* generate bkgsup core */
	/************************/
	fprintf(stderr, "pulsegen(): beginning pulse generation of bkgsupcore (background suppression core)\n");
	tmploc = 0;	

	fprintf(stderr, "pulsegen(): generating rfbs_rho & rfbs_theta (background suppression rf)...\n");
	tmploc += pgbuffertime; /* start time for bkgsup rf */
	EXTWAVE(RHO, rfbs_rho, tmploc, 5000, 1.0, 500, sech_7360.rho, , loggrd);
	EXTWAVE(THETA, rfbs_theta, tmploc, 5000, 1.0, 500, sech_7360.theta, , loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rfbs_rho; /* end time for bkg sup rf */
	fprintf(stderr, " end: %dus\n", tmploc);	
	tmploc += pgbuffertime; /* add some buffer */

	fprintf(stderr, "pulsegen(): finalizing bkgsupcore...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_bkgsupcore, tmploc);
	SEQLENGTH(bkgsupcore, dur_bkgsupcore, bkgsupcore);
	fprintf(stderr, "\tDone.\n");


	/************************/
	/* generate fatsat core */
	/************************/
	fprintf(stderr, "pulsegen(): beginning pulse generation of fatsatcore (fat saturation core)\n");	
	tmploc = 0;
	
	fprintf(stderr, "pulsegen(): generating rffs (fat saturation rf pulse)...\n");
	tmploc += pgbuffertime; /* start time for rffs */
	SINC2(RHO, rffs, tmploc + psd_rf_wait, 3200, 1.0, ,0.5, , , loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rffs; /* end time for rffs */
	fprintf(stderr, " end: %dus\n", tmploc);	
 
	fprintf(stderr, "pulsegen(): generating gzrffsspoil (fat saturation crusher gradients)...\n");
	tmploc += pgbuffertime; /* start time for gzrffsspoil */
	TRAPEZOID(ZGRAD, gzrffsspoil, tmploc + pw_gzrffsspoila, GRAD_UPDATE_TIME*1000, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzrffsspoila + pw_gzrffsspoil + pw_gzrffsspoild; /* end time for gzrffsspoil */
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime; /* add some buffer */

	fprintf(stderr, "pulsegen(): finalizing fatsatcore...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_fatsatcore, tmploc);
	SEQLENGTH(fatsatcore, dur_fatsatcore, fatsatcore);
	fprintf(stderr, "\tDone.\n");

	
	/*************************/
	/* generate tipdown core */
	/*************************/
	fprintf(stderr, "pulsegen(): beginning pulse generation of tipdown core (GRE tipdown core)\n");
	tmploc = 0;

	fprintf(stderr, "pulsegen(): generating gzrf1spoil (pre-rf1 crusher)...\n");
	tmploc += pgbuffertime; /* start time for gzrf1spoil */
	TRAPEZOID(ZGRAD, gzrf1spoil, tmploc + pw_gzrf1spoila, 3200, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzrf1spoila + pw_gzrf1spoil + pw_gzrf1spoild; /* end time for gzrf1spoil */
	fprintf(stderr, " end: %dus\n", tmploc);

	fprintf(stderr, "pulsegen(): generating rf1 (tipdown pulse)...\n");
	tmploc += pgbuffertime; /* start time for rf1 */
	SLICESELZ(rf1, tmploc + pw_gzrf1a, 3200, (opslthick + opslspace)*opslquant, opflip, 2, 1, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzrf1a + pw_gzrf1 + pw_gzrf1d; /* end time for rf2 pulse */
	fprintf(stderr, " end: %dus\n", tmploc);

	fprintf(stderr, "pulsegen(): generating gzrf1r (rf1 tipdown kspace rewinder)...\n");
	tmploc += pgbuffertime; /* start time for gzrf1r */
	TRAPEZOID(ZGRAD, gzrf1r, tmploc + pw_gzrf1ra, 3200, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzrf1ra + pw_gzrf1r + pw_gzrf1rd; /* end time for gzrf1r pulse */
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime;

	fprintf(stderr, "pulsegen(): finalizing tipdown core...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_tipdowncore, tmploc);
	SEQLENGTH(tipdowncore, dur_tipdowncore, tipdowncore);
	fprintf(stderr, "\tDone.\n");


	/**********************************/
	/* generate deadtime (empty) core */
	/**********************************/
	fprintf(stderr, "pulsegen(): beginning pulse generation of empty core (emptycore)\n");

	fprintf(stderr, "pulsegen(): finalizing empty core...\n");
	SEQLENGTH(emptycore, 1000, emptycore);
	fprintf(stderr, "\tDone.\n");


@inline Prescan.e PSpulsegen

	PASSPACK(endpass, 49ms);   /* tell Signa system we're done */
	SEQLENGTH(pass, 50ms, pass);

	buildinstr();              /* load the sequencer memory       */
	fprintf(stderr, "\tDone with pulsegen().\n");

	return SUCCESS;
}   /* end pulsegen() */


/* For Prescan: Pulse Generation functions */
@inline Prescan.e PSipg


@rspvar
/*********************************************************************
 *                    UMVSASL.E RSPVAR SECTION                       *
 *                                                                   *
 * Declare here the real time variables that can be viewed and modi- *
 * fied while the IPG PSD process is running. Only limited standard  *
 * C types are provided: short, int, long, float, double, and 1D     *
 * arrays of those types.                                            *
 *                                                                   *
 * NOTE: Do not declare all real-time variables here because of the  *
 *       overhead required for viewing and modifying them.           *
 *********************************************************************/
extern PSD_EXIT_ARG psdexitarg;

/* Declare rsps */
int echon;
int shotn;
int armn;
int framen;
int disdaqn;
int n;
int rspfct;
int rspsct;
int view;
int slice;
int echo;

/* Inherited from grass.e: */
int dabop;
int excitation;
int rspent;
int rspdda;
int rspbas;
int rspvus;
int rspgy1;
int rspasl;
int rspesl;
int rspchp;
int rspnex;
int rspslq;

/* For Prescan: K */
int seqCount;

@inline Prescan.e PSrspvar 


@rsp
/*********************************************************************
 *                    UMVSASL.E RSP SECTION                          *
 *                                                                   *
 * Write here the functional code for the real time processing (IPG  *
 * schedule_ide). You may declare standard C variables, but of limited types *
 * short, int, long, float, double, and 1D arrays of those types.    *
 *********************************************************************/
#include <math.h>

/* For IPG Simulator: will generate the entry point list in the IPG tool */
const CHAR *entry_name_list[ENTRY_POINT_MAX] = {
	"scan", 
	"aps2",
	"mps2",
@inline Prescan.e PSeplist
};

/* Do not move the line above and do not insert any code or blank
   lines before the line above.  The code inline'd from Prescan.e
   adds more entry points and closes the list. */

long tmtx0[9]; /* Initial transformation matrix */
long zmtx[9] = {0};

STATUS psdinit( void )
{
	/* Initialize everything to a known state */
	setrfconfig( ENBL_RHO1 + ENBL_THETA );
	setssitime( TIMESSI/GRAD_UPDATE_TIME );
	rspqueueinit( 200 );	/* Initialize to 200 entries */
	scopeon( &seqcore );	/* Activate scope for core */
	syncon( &seqcore );		/* Activate sync for core */
	syncoff( &pass );		/* Deactivate sync during pass */
	seqCount = 0;		/* Set SPGR sequence counter */
	setrotatearray( 1, rsprot[0] );
	settriggerarray( 1, rsptrigger );
	setrfltrs( (int)filter_echo1, &echo1 );
			
	/* Set rf1, rf2 tx and rx frequency */
	setfrequency((int)xmitfreq, &rf1, 0);
	setfrequency((int)recfreq, &echo1, 0);

	/* Set fat sat frequency */
	setfrequency( (int)(-520 / TARDIS_FREQ_RES), &rffs, 0);
	
	/* Get the original rotation matrix */
	getrotate( tmtx0, 0 );

	return SUCCESS;
}   /* end psdinit() */


@inline Prescan.e PScore


/* PLAY_DEADTIME() Function for playing TR deadtime */
int play_deadtime(int deadtime) {
	int ttotal = 0;
	fprintf(stderr, "\tplay_deadtime(): playing deadtime (%d us)...\n", deadtime);

	/* Play empty core */
	setperiod(deadtime - TIMESSI, &emptycore, 0);
	boffset(off_emptycore);
	startseq(0, MAY_PAUSE);
	settrigger(TRIG_INTERN, 0);
	ttotal += deadtime;

	fprintf(stderr, "\tplay_deadtime(): Done.\n");	
	
	return ttotal;
}
/* function for playing asl pre-saturation pulse */
int play_presat() {

	/* Play bulk saturation pulse */	
	fprintf(stderr, "\tplay_presat(): playing asl pre-saturation pulse (%d us)...\n", dur_presatcore + TIMESSI);

	boffset(off_presatcore);
	startseq(0, MAY_PAUSE);
	settrigger(TRIG_INTERN, 0);

	/* play the pre-saturation delay */
	fprintf(stderr, "scan(): playing asl pre-saturation delay (%d us)...\n", presat_delay);
	play_deadtime(presat_delay);		

	fprintf(stderr, "\tplay_presat(): Done.\n");

	return dur_presatcore + TIMESSI + presat_delay;
}

/* function for playing asl prep pulses & delays */
int play_aslprep(s32* off_ctlcore, s32* off_lblcore, int mod, int dur, int pld, int tbgs1, int tbgs2, int tbgs3) {
	int ttotal = 0;
	int ttmp;
	int type;
	
	/* determine current mode */		
	switch (mod) {
		case 1: /* label, control... */
			type = (framen + 1) % 2; /* 1, 0, 1, 0 */
			break;
		case 2: /* control, label... */
			type = framen % 2; /* 0, 1, 0, 1 */
			break;
		case 3: /* label */
			type = 1;
			break;
		case 4: /* control */
			type = 0;
			break;
	}

	/* play the asl prep pulse */	
	switch (type) {
		case 0: /* control */
			fprintf(stderr, "\tplay_aslprep(): playing control pulse (%d us)...\n", dur + TIMESSI);
			boffset(off_ctlcore);
			break;
		case 1: /* label */
			fprintf(stderr, "\tplay_aslprep(): playing label pulse (%d us)...\n", dur + TIMESSI);
			boffset(off_lblcore);
			break;
		case -1: /* off */
			ttotal = dur + TIMESSI + pld + TIMESSI;
			fprintf(stderr, "\tplay_aslprep(): playing deadtime in place of asl prep pulse (%d us)...\n", ttotal);
			play_deadtime(ttotal);
			return ttotal;
		default: /* invalid */
			fprintf(stderr, "\tplay_aslprep(): ERROR - invalid type (%d)\n", type);
			rspexit();
			return -1;
	}	
	ttotal += dur + TIMESSI;
	startseq(0, MAY_PAUSE);
	settrigger(TRIG_INTERN, 0);

	/* play pld and background suppression */
	if (pld > 0) {

		/* initialize pld before subtracting out tbgs timing */
		ttmp = pld;

		if (tbgs1 > 0) {
			/* play first background suppression delay/pulse */
			fprintf(stderr, "\tplay_aslprep(): playing bkg suppression pulse 1 delay (%d us)...\n", tbgs1 + TIMESSI);		
			setperiod(tbgs1, &emptycore, 0);
			ttmp -= (tbgs1 + TIMESSI);
			boffset(off_emptycore);
			startseq(0, MAY_PAUSE);
			settrigger(TRIG_INTERN, 0);
			ttotal += tbgs1 + TIMESSI;

			fprintf(stderr, "\tplay_aslprep(): playing bkg suppression pulse 1 (%d us)...\n", dur_bkgsupcore + TIMESSI);
			ttmp -= (dur_bkgsupcore + TIMESSI);
			boffset(off_bkgsupcore);
			startseq(0, MAY_PAUSE);
			settrigger(TRIG_INTERN, 0);
			ttotal += dur_bkgsupcore + TIMESSI;
		}
		
		if (tbgs2 > 0) {
			/* play second background suppression delay/pulse */
			fprintf(stderr, "\tplay_aslprep(): playing bkg suppression pulse 2 delay (%d us)...\n", tbgs2 + TIMESSI);		
			setperiod(tbgs2, &emptycore, 0);
			ttmp -= (tbgs2 + TIMESSI);
			boffset(off_emptycore);
			startseq(0, MAY_PAUSE);
			settrigger(TRIG_INTERN, 0);
			ttotal += tbgs2 + TIMESSI;

			fprintf(stderr, "\tplay_aslprep(): playing bkg suppression pulse 2 (%d us)...\n", dur_bkgsupcore + TIMESSI);
			ttmp -= (dur_bkgsupcore + TIMESSI);
			boffset(off_bkgsupcore);
			startseq(0, MAY_PAUSE);
			settrigger(TRIG_INTERN, 0);
			ttotal += dur_bkgsupcore + TIMESSI;
		}
		
		if (tbgs3 > 0) {
			/* play second background suppression delay/pulse */
			fprintf(stderr, "\tplay_aslprep(): playing bkg suppression pulse 2 delay (%d us)...\n", tbgs3 + TIMESSI);		
			setperiod(tbgs3, &emptycore, 0);
			ttmp -= (tbgs3 + TIMESSI);
			boffset(off_emptycore);
			startseq(0, MAY_PAUSE);
			settrigger(TRIG_INTERN, 0);
			ttotal += tbgs3 + TIMESSI;

			fprintf(stderr, "\tplay_aslprep(): playing bkg suppression pulse 2 (%d us)...\n", dur_bkgsupcore + TIMESSI);
			ttmp -= (dur_bkgsupcore + TIMESSI);
			boffset(off_bkgsupcore);
			startseq(0, MAY_PAUSE);
			settrigger(TRIG_INTERN, 0);
			ttotal += dur_bkgsupcore + TIMESSI;
		}

		/* check that ttmp is non-negative */
		if (ttmp < 0) {
			fprintf(stderr, "\tplay_aslprep(): ERROR: sum of background supression pulse delays must not exceed the PLD!\n");
			rspexit();
		}

		/* play remaining PLD deadtime */
		fprintf(stderr, "\tplay_aslprep(): playing post-label delay (%d us), total end delay = %d us...\n", pld, ttmp);
		setperiod(ttmp - TIMESSI, &emptycore, 0);
		boffset(off_emptycore);
		startseq(0, MAY_PAUSE);
		settrigger(TRIG_INTERN, 0);
		ttotal += ttmp;
	}

	return ttotal;
}

/* function for playing fat sat pulse */
int play_fatsat() {
	int ttotal = 0;
	fprintf(stderr, "\tplay_fatsat(): playing fat sat pulse (%d us)...\n", dur_fatsatcore + TIMESSI);

	/* Play fatsat core */
	boffset(off_fatsatcore);
	startseq(0, MAY_PAUSE);
	settrigger(TRIG_INTERN, 0);
	ttotal += dur_fatsatcore + TIMESSI;

	fprintf(stderr, "\tplay_fatsat(): Done.\n");
	return ttotal;
}

/* function for playing GRE tipdown pulse */
int play_tipdown(float phs) {
	int ttotal = 0;

	/* set rx and tx phase */
	setphase(phs, &rf1, 0);
	setphase(phs, &echo1, 0);

	/* Play the tipdown */
	fprintf(stderr, "\tplay_tipdown(): playing tipdowncore (%d us)...\n", dur_tipdowncore);
	boffset(off_tipdowncore);
	startseq(0, MAY_PAUSE);
	settrigger(TRIG_INTERN, 0);
	ttotal += dur_tipdowncore + TIMESSI;

	return ttotal;	
}

/* function for playing the acquisition window */
int play_readout() {
	int ttotal = 0;
	fprintf(stderr, "\tplay_readout(): playing seqcore (%d us)...\n", dur_seqcore);

	/* play the seqcore */
	boffset(off_seqcore);
	startseq(0, MAY_PAUSE);
	settrigger(TRIG_INTERN, 0);
	ttotal += dur_seqcore + TIMESSI; 

	fprintf(stderr, "\tplay_readout(): Done.\n");
	return ttotal;
}

/* function for sending endpass packet at end of sequence */
STATUS play_endscan() {
	fprintf(stderr, "\tplay_endscan(): sending endpass packet...\n");
	
	/* send SSP packet to end scan */
	boffset( off_pass );
	setwamp(SSPD + DABPASS + DABSCAN, &endpass, 2);
	settrigger(TRIG_INTERN, 0);
	startseq(0, MAY_PAUSE);  

	fprintf(stderr, "\tplay_endscan(): Done.\n");
	return SUCCESS;
}

/* function for playing prescan sequence */
STATUS prescanCore() {

	/* initialize the rotation matrix */
	setrotate( tmtx0, 0 );
	
	for (view = 1 - rspdda; view < rspvus + 1; view++) {

		fprintf(stderr, "prescanCore(): Playing flip pulse for prescan iteration %d...\n", view);
		play_tipdown(0);

		/* Load the DAB */	
		if (view < 1) {
			fprintf(stderr, "prescanCore(): loaddab(&echo1, 0, 0, 0, 0, DABOFF, PSD_LOAD_DAB_ALL)...\n");
			loaddab(&echo1, 0, 0, 0, 0, DABOFF, PSD_LOAD_DAB_ALL);
		}
		else {
			fprintf(stderr, "prescanCore(): loaddab(&echo1, 0, 0, 0, %d, DABON, PSD_LOAD_DAB_ALL)...\n", view);
			loaddab(&echo1, 0, 0, 0, view, DABON, PSD_LOAD_DAB_ALL);
		}
		
		/* kill gradients */				
		setrotate( zmtx, 0 );

		fprintf(stderr, "prescanCore(): playing readout for prescan iteration %d...\n", view);
		play_readout();
		
		/* restore gradients */				
		setrotate( tmtx0, 0 );

		fprintf(stderr, "prescanCore(): playing deadtime for prescan iteration %d...\n", view);
		play_deadtime(50000);

	}

	rspexit();

	return SUCCESS;
}

/* For Prescan: MPS2 Function */
STATUS mps2( void )
{
	if( psdinit() == FAILURE )
	{
		return rspexit();
	}

	rspent = L_MPS2;
	rspvus = 30000;
	rspdda = 0;
	prescanCore();
	rspexit();

	return SUCCESS;
}   /* end mps2() */


/* For Prescan: APS2 Function */
STATUS aps2( void )
{   
	if( psdinit() == FAILURE )
	{
		return rspexit();
	}

	rspent = L_APS2;
	rspvus = 1026;
	rspdda = 2;
	prescanCore();
	rspexit();

	return SUCCESS;
}   /* end aps2() */

STATUS scan( void )
{ 
	if( psdinit() == FAILURE )
	{
		return rspexit();
	}

	int ttotal = 0;
	int rotidx;
	fprintf(stderr, "scan(): beginning scan (t = %d / %.0f us)...\n", ttotal, pitscan);	
	
	/* Play an empty acquisition to reset the DAB after prescan */
	if (disdaqn == 0) {
		/* Turn the DABOFF */
		loaddab(&echo1, 0, 0, DABSTORE, 0, DABOFF, PSD_LOAD_DAB_ALL);
		/* kill gradients */				
		setrotate( zmtx, 0 );

		play_readout();
		
		/* restore gradients */				
		setrotate( tmtx0, 0 );
	}

	/* Play disdaqs */
	for (disdaqn = 0; disdaqn < ndisdaqtrains; disdaqn++) {
		
		/* Calculate and play deadtime */
		fprintf(stderr, "scan(): playing TR deadtime for disdaq train %d (t = %d / %.0f us)...\n", disdaqn, ttotal, pitscan);
		ttotal += play_deadtime(optr - opetl * (dur_tipdowncore + TIMESSI + dur_seqcore + TIMESSI));
		
		/* Loop through echoes */
		for (echon = 0; echon < opetl; echon++) {
			fprintf(stderr, "scan(): playing flip pulse for disdaq train %d (t = %d / %.0f us)...\n", disdaqn, ttotal, pitscan);
			ttotal += play_tipdown(rfspoil_flag*117*echon);

			/* Load the DAB */		
			fprintf(stderr, "scan(): loaddab(&echo1, %d, 0, DABSTORE, 0, DABOFF, PSD_LOAD_DAB_ALL)...\n", echon+1);
			loaddab(&echo1,
					0,
					0,
					DABSTORE,
					0,
					DABOFF,
					PSD_LOAD_DAB_ALL);		

			fprintf(stderr, "scan(): playing readout for disdaq train %d (%d us)...\n", disdaqn, dur_seqcore);
			
			/* kill gradients */				
			setrotate( zmtx, 0 );

			ttotal += play_readout();
		
			/* restore gradients */				
			setrotate( tmtx0, 0 );
		}
	}


	/* loop through frames and shots */
	for (framen = 0; framen < nframes; framen++) {
		for (armn = 0; armn < narms; armn++) {
			for (shotn = 0; shotn < opnshots; shotn++) {

				/* play TR deadtime */
				ttotal += play_deadtime(tr_deadtime);

				fprintf(stderr, "scan(): ************* beginning loop for frame %d, arm %d, shot %d *************\n", framen, shotn, armn);

				/* play the ASL pre-saturation pulse for background suppression */
				if (presat_flag) {
					fprintf(stderr, "scan(): playing asl pre-saturation pulse for frame %d, arm %d, shot %d (t = %d / %.0f us)...\n", framen, armn, shotn, ttotal, pitscan);
					ttotal += play_presat();
				}

				if (prep1_id > 0 ) {
					fprintf(stderr, "scan(): playing prep1 pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
					ttotal += play_aslprep(off_prep1ctlcore, off_prep1lblcore, prep1_mod, dur_prep1core, prep1_pld, prep1_tbgs1, prep1_tbgs2, prep1_tbgs3);
				}

				if (prep2_id > 0 ) {
					fprintf(stderr, "scan(): playing prep2 pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
					ttotal += play_aslprep(off_prep2ctlcore, off_prep2lblcore, prep2_mod, dur_prep2core, prep2_pld, prep2_tbgs1, prep2_tbgs2, prep2_tbgs3);
				}

				/* fat sat pulse */
				if (fatsat_flag) {
					fprintf(stderr, "scan(): playing fat sat pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
					ttotal += play_fatsat();
				}

				/* play disdaq echoes */
				for (echon = 0; echon < ndisdaqechoes; echon++) {
					fprintf(stderr, "scan(): playing flip pulse for frame %d, shot %d, disdaq echo %d (t = %d / %.0f us)...\n", framen, shotn, echon, ttotal, pitscan);
					ttotal += play_tipdown(rfspoil_flag*117*echon);


					fprintf(stderr, "scan(): playing deadtime in place of readout for frame %d, shot %d, disdaq echo %d (%d us)...\n", framen, shotn, echon, dur_seqcore);
					ttotal += play_deadtime(dur_seqcore);
				};

				for (echon = 0; echon < opetl; echon++) {
					fprintf(stderr, "scan(): playing flip pulse for frame %d, shot %d, echo %d (t = %d / %.0f us)...\n", framen, shotn, echon, ttotal, pitscan);
					ttotal += play_tipdown(rfspoil_flag*117*(echon + ndisdaqechoes));

					/* load the DAB */
					slice = framen+1;
					view = 	armn*opnshots*opetl + shotn*opetl + echon + 1;
					echo = 0;
					fprintf(stderr, "scan(): loaddab(&echo1, %d, %d, DABSTORE, %d, DABON, PSD_LOAD_DAB_ALL)...\n", slice, echo, view);
					loaddab(&echo1,
							slice,
							echo,
							DABSTORE,
							view,
							DABON,
							PSD_LOAD_DAB_ALL);		

					/* Set the view transformation matrix */
					rotidx = armn*opnshots*opetl + shotn*opetl + echon;
					if (kill_grads)
						setrotate( zmtx, 0 );
					else
						setrotate( tmtxtbl[rotidx], 0 );

					fprintf(stderr, "scan(): playing readout for frame %d, shot %d, echo %d (%d us)...\n", framen, shotn, echon, dur_seqcore);
					ttotal += play_readout();

					/* Reset the rotation matrix */
					setrotate( tmtx0, 0 );
				}

			}
		}
	}

	fprintf(stderr, "scan(): reached end of scan, sending endpass packet (t = %d / %.0f us)...\n", ttotal, pitscan);
	play_endscan();

	rspexit();

	return SUCCESS;
}


/********************************************
 * dummylinks
 *
 * This routine just pulls in routines from
 * the archive files by making a dummy call.
 ********************************************/
void dummylinks( void )
{
	epic_loadcvs( "thefile" );            /* for downloading CVs */
}


@host
/******************************************************
* Define the functions that will run on the host 
* during predownload operations
*****************************************************/
int genspiral() {

	FILE *fID_ktraj = fopen("ktraj.txt", "w");

	/* declare waveform sizes */
	int n_vds, n_rmp, n_rwd; /* spiral-out, ramp-down, rewind */
	int n_sprl; /* total pts in spiral */
	int n;

	/* declare gradient waveforms */
	float *gx_vds, *gx_rmp, *gx_rwd;
	float *gy_vds, *gy_rmp, *gy_rwd;
	float *gx_sprli, *gx_sprlo, *gx_tmp;
	float *gy_sprli, *gy_sprlo, *gy_tmp;
	float *gx, *gy;

	/* declare constants */
	float F[3]; /* FOV coefficients (cm, cm^2, cm^3) */
	float dt = GRAD_UPDATE_TIME*1e-6; /* raster time (s) */
	float gam = 4258; /* gyromagnetic ratio (Hz/G) */
	float kxymax = (float)opxres / ((float)opfov/10.0) / 2.0; /* max kxy radius (1/cm) */
	float kzmax = (spi_mode == 0) * (float)(kz_acc * opetl * opnshots) / ((float)opfov/10.0); /* max kz radius (1/cm), 0 if SPI */

	/* declare temporary variables */
	float gx_area, gy_area;
	float tmp_area, tmp_a;
	int tmp_pwa, tmp_pw, tmp_pwd;
	float kxn, kyn;
	/* calculate FOV coefficients */
	F0 = vds_acc1 / (float)narms * (float)opfov / 10.0;
	F1 = 2*pow((float)opfov/10.0,2)/opxres *(vds_acc1 - vds_acc0)/(float)narms;
	F2 = 0;
	if (echo_mode == 1) { /* FSE */
		F0 /= 2;
		F1 /= 2;
		F2 /= 2;
	}
	F[0] = F0;
	F[1] = F1;
	F[2] = F2;
	
	/* generate the vd-spiral out gradients */	
	fprintf(stderr, "genspiral(): calculating vds waveform...\n");
	calc_vds(SLEWMAX, GMAX, dt, dt, 1, F, 2, kxymax, MAXWAVELEN, &gx_vds, &gy_vds, &n_vds);
	fprintf(stderr, "genspiral(): n_vds = %d...\n", n_vds);

	/* calculate gradient ramp-down */
	fprintf(stderr, "genspiral(): calculating gradient ramp-down...\n");
	n_rmp = ceil(fmax(fabs(gx_vds[n_vds - 1]), fabs(gy_vds[n_vds - 1])) / SLEWMAX / dt);
	gx_rmp = (float *)malloc(n_rmp*sizeof(float));
	gy_rmp = (float *)malloc(n_rmp*sizeof(float));
	for (n = 0; n < n_rmp; n++) {
		gx_rmp[n] = gx_vds[n_vds - 1]*(1 - (float)n/(float)n_rmp);
		gy_rmp[n] = gy_vds[n_vds - 1]*(1 - (float)n/(float)n_rmp);
	}
	fprintf(stderr, "genspiral(): n_rmp = %d...\n", n_rmp);

	fprintf(stderr, "genspiral(): calculating area under gradients...\n");
	gx_area = 1e6 * dt * (fsumarr(gx_vds, n_vds) + fsumarr(gx_rmp, n_rmp));
	gy_area = 1e6 * dt * (fsumarr(gy_vds, n_vds) + fsumarr(gy_rmp, n_rmp));
	tmp_area = fmax(fabs(gx_area), fabs(gy_area)); /* get max abs area */
	fprintf(stderr, "genspiral(): gx_area = %f, gy_area = %f...\n", gx_area, gy_area);
	fprintf(stderr, "genspiral(): tmp_area = %f...\n", tmp_area);

	/* calculate optimal trapezoid kspace rewinder */
	fprintf(stderr, "genspiral(): calculating kspace rewinders...\n");
	amppwgrad(tmp_area, GMAX, 0, 0, ZGRAD_risetime, 0, &tmp_a, &tmp_pwa, &tmp_pw, &tmp_pwd);
	n_rwd = ceil((float)(tmp_pwa + tmp_pw + tmp_pwd)/(float)GRAD_UPDATE_TIME);
	gx_rwd = (float *)malloc(n_rwd*sizeof(float));
	gy_rwd = (float *)malloc(n_rwd*sizeof(float));
	for (n = 0; n < n_rwd; n++) {
		gx_rwd[n] = -gx_area/tmp_area*tmp_a*trap(n*1e6*dt,0.0,tmp_pwa,tmp_pw);
		gy_rwd[n] = -gy_area/tmp_area*tmp_a*trap(n*1e6*dt,0.0,tmp_pwa,tmp_pw);
		fprintf(stderr, "genspiral(): gx_rwd[%d] = %f, gy_rwd[%d] = %f...\n", n, gx_rwd[n], n, gy_rwd[n]);
	}
	fprintf(stderr, "genspiral(): n_rwd = %d...\n", n_rwd);

	/* calculate total points in spiral + rewinder */
	fprintf(stderr, "genspiral(): calculating points in spiral...\n");
	n_sprl = n_vds + n_rmp + n_rwd;
	gx_sprlo = (float *)malloc(n_sprl*sizeof(float));
	gy_sprlo = (float *)malloc(n_sprl*sizeof(float));
	gx_sprli = (float *)malloc(n_sprl*sizeof(float));
	gy_sprli = (float *)malloc(n_sprl*sizeof(float));
	fprintf(stderr, "genspiral(): n_sprl = %d...\n", n_sprl);

	/* concatenate gradients to form spiral out */
	fprintf(stderr, "genspiral(): forming spiral-out...\n");
	gx_tmp = (float *)malloc((n_vds + n_rmp)*sizeof(float));
	gy_tmp = (float *)malloc((n_vds + n_rmp)*sizeof(float));
	catArray(gx_vds, n_vds, gx_rmp, n_rmp, 0, gx_tmp);
	catArray(gy_vds, n_vds, gy_rmp, n_rmp, 0, gy_tmp);
	catArray(gx_tmp, n_vds + n_rmp, gx_rwd, n_rwd, 0, gx_sprlo);
	catArray(gy_tmp, n_vds + n_rmp, gy_rwd, n_rwd, 0, gy_sprlo);
	free(gx_tmp);
	free(gy_tmp);

	/* reverse the gradients to form spiral in */
	fprintf(stderr, "genspiral(): forming spiral-in...\n");
	reverseArray(gx_sprlo, n_sprl, gx_sprli);
	reverseArray(gy_sprlo, n_sprl, gy_sprli);

	fprintf(stderr, "genspiral(): combining spirals...\n");
	if (echo_mode == 0) { /* SPGR - spiral out */
		/* calculate window lengths */
		grad_len = nnav + n_sprl;
		acq_len = nnav + n_vds;
		acq_offset = 0;

		gx = (float *)malloc(grad_len*sizeof(float));
		gy = (float *)malloc(grad_len*sizeof(float));
	
		/* zero-pad with navigators */
		catArray(gx_sprlo, 0, gx_sprlo, n_sprl, nnav, gx);
		catArray(gy_sprlo, 0, gy_sprlo, n_sprl, nnav, gy);
	}
	else { /* FSE - spiral in-out */
		/* calculate window lengths */
		grad_len = 2*(n_rmp + n_rwd + n_vds) + nnav;
		acq_len = 2*n_vds + nnav;
		acq_offset = n_rwd + n_rmp;
		
		gx = (float *)malloc(grad_len*sizeof(float));
		gy = (float *)malloc(grad_len*sizeof(float));
		
		/* concatenate and zero-pad the spiral in & out waveforms */
		catArray(gx_sprli, n_sprl, gx_sprlo, n_sprl, nnav, gx);
		catArray(gy_sprli, n_sprl, gy_sprlo, n_sprl, nnav, gy);
	}

	/* integrate gradients to calculate kspace */
	fprintf(stderr, "genspiral(): converting trajectory and gradients...\n");
	kxn = 0.0;
	kyn = 0.0;
	for (n = 0; n < grad_len; n++) {
		/* integrate gradients */
		kxn += gam * gx[n] * dt;
		kyn += gam * gy[n] * dt;
		if (n > acq_offset-1 && n < acq_offset + acq_len)
			fprintf(fID_ktraj, "%f \t%f \t%f\n", kxn, kyn, kzmax);
		
		/* convert gradients to integer units */
		Gx[n] = 2*round(MAX_PG_WAMP/XGRAD_max * gx[n] / 2.0);
		Gy[n] = 2*round(MAX_PG_WAMP/YGRAD_max * gy[n] / 2.0);
	}

	fclose(fID_ktraj);

	return SUCCESS;
}

int genviews() {

	/* Declare values and matrices */
	FILE* fID_kviews = fopen("kviews.txt","w");
	int rotidx, armn, shotn, echon, n;
	float rz, dz;
	float Rz[9], Tz[9];
	float T_0[9], T[9];

	/* Initialize z translation to identity matrix */
	eye(Tz, 3);

        /* Get original transformation matrix */
        for (n = 0; n < 9; n++) T_0[n] = (float)rsprot[0][n] / MAX_PG_WAMP;
        orthonormalize(T_0, 3, 3);

	/* Loop through all views */
	for (armn = 0; armn < narms; armn++) {
		for (shotn = 0; shotn < opnshots; shotn++) {
			for (echon = 0; echon < opetl; echon++) {

				/* calculate view index */
				rotidx = armn*opnshots*opetl + shotn*opetl + echon;

				/* Set the rotation angle and kz step (as a fraction of kzmax) */ 
				rz = 2.0*M_PI * (float)armn / (float)narms;
				dz = 2.0/(float)opetl * (center_out_idx(opetl,echon) - 1.0/(float)opnshots*center_out_idx(opnshots,shotn)) - 1.0;

				/* Calculate the transformation matrices */
				Tz[8] = dz;
				genrotmat('z', rz, Rz);

				/* Multiply the transformation matrices */
				multmat(3,3,3,T_0,Tz,T); /* T = T_0 * Tz */
				multmat(3,3,3,Rz,T,T); /* T = Rz * T */

				/* Save the matrix to the table of matrices */
				fprintf(fID_kviews, "%d \t%d \t%d \t%f \t%f \t", armn, shotn, echon, rz, dz);	
				for (n = 0; n < 9; n++) {
					fprintf(fID_kviews, "%f \t", T[n]);
					tmtxtbl[rotidx][n] = (long)round(MAX_PG_WAMP*T[n]);
				}
				fprintf(fID_kviews, "\n");
			}
		}
	}

	/* Close the files */
	fclose(fID_kviews);

	return 1;
};

int readprep(int id, int *len,
		int *rho_lbl, int *theta_lbl, int *grad_lbl,
		int *rho_ctl, int *theta_ctl, int *grad_ctl)
{

	/* Declare variables */
	char fname[80];
	FILE *fID;
	char buff[200];
	int i, tmplen;
	double lblval, ctlval;
	
	if (id == 0) {
		/* Set all values to zero and return */
		for (i = 0; i < *len; i++) {
			rho_lbl[i] = 0;
			theta_lbl[i] = 0;
			grad_lbl[i] = 0;
			rho_ctl[i] = 0;
			theta_ctl[i] = 0;
			grad_ctl[i] = 0;
		}
		return 1;
	}

	/* Read in RF magnitude from rho file */
	sprintf(fname, "./aslprep/pulses/%05d/rho.txt", id);
	fprintf(stderr, "readprep(): opening %s...\n", fname);
	fID = fopen(fname, "r");

	/* Check if rho file was read successfully */
	if (fID == 0) {
		fprintf(stderr, "readprep(): failure opening %s\n", fname);
		return 0;
	}

	/* Loop through points in rho file */
	i = 0;
	while (fgets(buff, 200, fID)) {
		sscanf(buff, "%lf %lf", &lblval, &ctlval);
		rho_lbl[i] = (int)lblval;
		rho_ctl[i] = (int)ctlval;
		i++;
	}
	fclose(fID);
	tmplen = i;
	
	/* Read in RF phase from theta file */
	sprintf(fname, "./aslprep/pulses/%05d/theta.txt", id);
	fID = fopen(fname, "r");

	/* Check if theta file was read successfully */
	if (fID == 0) {
		fprintf(stderr, "readprep(): failure opening %s\n", fname);
		return 0;
	}

	/* Loop through points in theta file */
	i = 0;
	while (fgets(buff, 200, fID)) {
		sscanf(buff, "%lf %lf", &lblval, &ctlval);
		theta_lbl[i] = (int)lblval;
		theta_ctl[i] = (int)ctlval;
		i++;
	}
	fclose(fID);

	/* Check that length is consistent */
	if (tmplen != i) {
		fprintf(stderr, "readprep(): length of theta file (%d) is not consistent with rho file length (%d)\n", i, tmplen);
		return 0;
	}
	
	/* Read in RF phase from theta file */
	sprintf(fname, "./aslprep/pulses/%05d/grad.txt", id);
	fID = fopen(fname, "r");

	/* Check if theta file was read successfully */
	if (fID == 0) {
		fprintf(stderr, "readprep(): failure opening %s\n", fname);
		return 0;
	}

	/* Loop through points in theta file */
	i = 0;
	while (fgets(buff, 200, fID)) {
		sscanf(buff, "%lf %lf", &lblval, &ctlval);
		grad_lbl[i] = (int)lblval;
		grad_ctl[i] = (int)ctlval*(!zero_ctl_grads);
		i++;
	}
	fclose(fID);

	/* Check that length is consistent */
	if (tmplen != i) {
		fprintf(stderr, "readprep(): length of grad file (%d) is not consistent with rho/theta file length (%d)\n", i, tmplen);
		return 0;
	}
	
	*len = tmplen;

	return 1;
}

float calc_sinc_B1(float cyc_rf, int pw_rf, float flip_rf) {

	int M = 1001;
	int n;
	float w[M], x[M];
	float area = 0.0;

	/* Create an M-point symmetrical Hamming window */
	for (n = 0; n < M; n++)
		w[n] = 0.54 - 0.46*cos( 2*M_PI*n / (M-1) );
	
	/* Create a sinc pulse */
	for (n = -(M-1)/2; n < (M-1)/2; n++) {
		if (n == 0)
			x[n + (M-1)/2] = 1.0;
		else
			x[n + (M-1)/2] = sin( 4 * M_PI * cyc_rf * n / (M-1) ) / ( 4 * M_PI * cyc_rf * n / (M-1) );
	}
	
	/* Calculate the area (abswidth) */
	for (n = 0; n < M; n++) {
		area += x[n] * w[n] / M;
	}

	/* Return the B1 (derived from eq. 1 on page 2-31 in EPIC manual) */
	return (SAR_ASINC1/area * 3200/pw_rf * flip_rf/90.0 * MAX_B1_SINC1_90);
}

float calc_hard_B1(int pw_rf, float flip_rf) {
	return (flip_rf / 180.0 * M_PI / GAMMA / (float)(pw_rf*1e-6));
}

/************************ END OF UMVSASL.E ******************************/

