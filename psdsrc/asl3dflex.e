/*
 * David Frey & Luis Hernandez-Garcia
 * University of Michigan Department of Radiology
 * Functional MRI Laboratory
 *
 * GE Medical Systems
 * Copyright (C) 1996-2003 The General Electric Company
 *
 * File Name : asl3dflex.e
 * Language  : EPIC/ANSI C
 * Date      : 10-May-2023
 *
 * An ASL-prepped flexible spiral fse readout sequence (ASL3DFLEX),
 * built up from grass.e
 */

@inline epic.h
@inline intwave.h

@global
/*********************************************************************
 *                  ASL3DFLEX.E GLOBAL SECTION                       *
 *                                                                   *
 * Common code shared between the Host and IPG PSD processes.  This  *
 * section contains all the #define's, global variables and function *
 * declarations (prototypes).                                        *
 *********************************************************************/
#include <stdio.h>
#include <string.h>

#include "em_psd_ermes.in"
#include "grad_rf_asl3dflex.globals.h"

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

#include "asl3dflex.h"

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
 *                ASL3DFLEX.E IPGEXPORT SECTION                      *
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
int Gz[MAXWAVELEN];
int grad_len = 5000;

/* Declare table of readout gradient transformation matrices */
long tmtxtbl[MAXNSHOTS*MAXNECHOES][9];

/* Declare table of refocuser flip angles */
float flipfactbl[MAXNECHOES];
float flipphstbl[MAXNECHOES];

/* Declare ASL prep 1 pulse variables */
int prep1_len = 5000;
int prep1_lbltbl[MAXNFRAMES];
int prep1_pldtbl[MAXNFRAMES];
int prep1_tbgs1tbl[MAXNFRAMES];
int prep1_tbgs2tbl[MAXNFRAMES];
int prep1_rho_lbl[MAXWAVELEN];
int prep1_theta_lbl[MAXWAVELEN];
int prep1_grad_lbl[MAXWAVELEN];
int prep1_rho_ctl[MAXWAVELEN];
int prep1_theta_ctl[MAXWAVELEN];
int prep1_grad_ctl[MAXWAVELEN];

/* Declare ASL prep 2 pulse variables */
int prep2_len = 5000;
int prep2_lbltbl[MAXNFRAMES];
int prep2_pldtbl[MAXNFRAMES];
int prep2_tbgs1tbl[MAXNFRAMES];
int prep2_tbgs2tbl[MAXNFRAMES];
int prep2_rho_lbl[MAXWAVELEN];
int prep2_theta_lbl[MAXWAVELEN];
int prep2_grad_lbl[MAXWAVELEN];
int prep2_rho_ctl[MAXWAVELEN];
int prep2_theta_ctl[MAXWAVELEN];
int prep2_grad_ctl[MAXWAVELEN];

/* other schedule arrays */
int tadjusttbl[MAXNFRAMES];
int doblksattbl[MAXNFRAMES];

/* Declare receiver and Tx frequencies */
float recfreq;
float xmitfreq1;
float xmitfreq2;

@cv
/*********************************************************************
 *                     ASL3DFLEX.E CV SECTION                        *
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

float xmtaddScan;
int obl_debug = 0 with {0, 1, 0, INVIS, "On(=1) to print messages for obloptimize",};
int obl_method = 0 with {0, 1, 0, INVIS, "On(=1) to optimize the targets based on actual rotation matrices",};
int debug = 0 with {0,1,0,INVIS,"1 if debug is on ",};
float echo1bw = 16 with {,,,INVIS,"Echo1 filter bw.in KHz",};

float SLEWMAX = 12500.0 with {1000, 25000.0, 12500.0, VIS, "Maximum allowed slew rate (G/cm/s)",};
float GMAX = 4.0 with {0.5, 5.0, 4.0, VIS, "Maximum allowed gradient (G/cm)",};
float RFMAX = 300 with {0, 500, 300, VIS, "Maximum allowed RF amplitude (mG)",};

/* readout cvs */
int nframes = 2 with {1, , 2, VIS, "Number of frames",};
int ndisdaqtrains = 2 with {0, , 2, VIS, "Number of disdaq echo trains at beginning of scan loop",};
int ndisdaqechoes = 0 with {0, , 0, VIS, "Number of disdaq echos at beginning of echo train",};
int dofatsat = 1 with {0, 1, 0, VIS, "Option to do play a fat saturation pulse/crusher before the readout",};
int ro_mode = 0 with {0, 1, 0, VIS, "Readout mode (0 = GRE, 1 = FSE)",};
int pgbuffertime = 248 with {100, , 248, INVIS, "Gradient IPG buffer time (us)",};
float phs_tip = 0.0 with { , , 0.0, VIS, "Initial transmitter phase for tipdown pulse",};
float phs_inv = M_PI/2 with { , , M_PI/2, VIS, "Transmitter phase for inversion pulse",};
float phs_rx = 0.0 with { , , 0.0, VIS, "Receiever phase",};
int phscyc_fse = 0 with {0, 1, 0, VIS, "Option to do rf phase cycling for FSE sequence",};
float spgr_phsinc = 117 with {0, , 117, VIS, "Phase increment (deg) for RF spoiling in GRE mode",};
float crushfac = 3.0 with {0, 10, 0, VIS, "Crusher amplitude factor (a.k.a. cycles of phase/vox; dk_crush = crushfac*kmax)",};
float varflipfac = 1 with {0, 1, 0, VIS, "Scaling factor for variable flip angle schedule (1 = constant fa)",};
int kill_grads = 0 with {0, 1, 0, VIS, "Option to turn off readout gradients",};

/* Trajectory cvs */
int nnav = 250 with {0, 1000, 250, VIS, "Number of navigator points in spiral",};
float kz_acc = 1.0 with {1, 100.0, 1.0, VIS, "kz acceleration (SENSE) factor (for SOS only)",};
float spvd0 = 1.0 with {0.001, 50.0, 1.0, VIS, "Spiral center oversampling factor",};
float spvd1 = 1.0 with {0.001, 50.0, 1.0, VIS, "Spiral edge oversampling factor",};
int sptype2d = 4 with {1, 4, 1, VIS, "1 = spiral out, 2 = spiral in, 3 = spiral out-in, 4 = spiral in-out",};
int sptype3d = 3 with {0, 4, 1, VIS, "0 = 2D (shotxshot rot only) 1 = stack of spirals, 2 = rotating spirals (single axis), 3 = rotating spirals (2 axes), 4 = debug mode",};
float F0 = 0 with { , , 0, INVIS, "vds fov coefficient 0",};
float F1 = 0 with { , , 0, INVIS, "vds fov coefficient 1",};
float F2 = 0 with { , , 0, INVIS, "vds fov coefficient 2",};

/* ASL prep pulse cvs */
int nm0frames = 2 with {0, , 2, VIS, "Number of M0 frames (no prep pulses are played)",};
int schedule_id = 0 with {0, , 0, VIS, "ASL labeling schedule (0 = no external schedule)",};
int doblksat = 1 with {0, 1, 1, VIS, "Option to do bulk spin saturation at end of each readout",};

int zero_CTL_grads = 0 with {0, 1, 0, VIS, "option to zero out control gradients",};
int prep1_id = 0 with {0, , 0, VIS, "ASL prep pulse 1: ID number (0 = no pulse)",};
int prep1_pld = 0 with {0, , 0, VIS, "ASL prep pulse 1: post-labeling delay (us; includes background suppression)",};
int prep1_ncycles = 1 with {1, , 1, VIS, "ASL prep pulse 1: number of cycles",};
float prep1_rfmax = 234 with {0, , 0, VIS, "ASL prep pulse 1: maximum RF amplitude",};
float prep1_gmax = 1.5 with {0, , 3, VIS, "ASL prep pulse 1: maximum gradient amplitude",};
int prep1_mod = 1 with {1, 4, 1, VIS, "ASL prep pulse 1: labeling modulation scheme (1 = label/control, 2 = control/label, 3 = always label, 4 = always control)",};
int prep1_tbgs1 = 0 with {0, , 0, VIS, "ASL prep pulse 1: 1st background suppression delay (0 = no pulse)",};
int prep1_tbgs2 = 0 with {0, , 0, VIS, "ASL prep pulse 1: 2nd background suppression delay (0 = no pulse)",};

int prep2_id = 0 with {0, , 0, VIS, "ASL prep pulse 2: ID number (0 = no pulse)",};
int prep2_pld = 0 with {0, , 0, VIS, "ASL prep pulse 2: post-labeling delay (us; includes background suppression)",};
int prep2_ncycles = 1 with {1, , 1, VIS, "ASL prep pulse 2: number of cycles",};
float prep2_rfmax = 234 with {0, , 0, VIS, "ASL prep pulse 2: maximum RF amplitude",};
float prep2_gmax = 1.5 with {0, , 1.5, VIS, "ASL prep pulse 2: maximum gradient amplitude",};
int prep2_mod = 1 with {1, 4, 1, VIS, "ASL prep pulse 2: labeling modulation scheme (1 = label/control, 2 = control/label, 3 = always label, 4 = always control)",};
int prep2_tbgs1 = 0 with {0, , 0, VIS, "ASL prep pulse 2: 1st background suppression delay (0 = no pulse)",};
int prep2_tbgs2 = 0 with {0, , 0, VIS, "ASL prep pulse 2: 2nd background suppression delay (0 = no pulse)",};

/* Declare core duration variables */
int dur_blksatcore = 0 with {0, , 0, INVIS, "Duration of the bulk saturation core (us)",};
int dur_prep1core = 0 with {0, , 0, INVIS, "Duration of the ASL prep 1 cores (us)",};
int dur_prep2core = 0 with {0, , 0, INVIS, "Duration of the ASL prep 2 cores (us)",};
int dur_bkgsupcore = 0 with {0, , 0, INVIS, "Duration of the background suppression core (us)",};
int dur_fatsatcore = 0 with {0, , 0, INVIS, "Duration of the fat saturation core (us)",};
int dur_tipcore = 0 with {0, , 0, INVIS, "Duration of the tipdown core (us)",};
int dur_flipcore = 0 with {0, , 0, INVIS, "Duration of the refocus core (us)",};
int dur_seqcore = 0 with {0, , 0, INVIS, "Duration of the spiral readout core (us)",};
int dur_readout = 0 with {0, , 0, INVIS, "Duration of the whole readout section (us, including TIMESSI's)",};
int deadtime1_seqcore = 0 with {0, , 0, INVIS, "Pre-readout deadtime inschedule_ide seqcore (us)",};
int deadtime2_seqcore = 0 with {0, , 0, INVIS, "Post-readout deadtime inschedule_ide seqcore (us)",};
int deadtime_tipcore = 0 with {0, , 0, INVIS, "Deadtime inschedule_ide tipcore for FSE readout (us)",};

@host
/*********************************************************************
 *                    ASL3DFLEX.E HOST SECTION                       *
 *                                                                   *
 * Write here the code unique to the Host PSD process. The following *
 * functions must be declared here: cvinit(), cveval(), cvcheck(),   *
 * and predownload().                                                *
 *                                                                   *
 *********************************************************************/
#include <math.h>
#include <stdlib.h>
#include "grad_rf_asl3dflex.h"
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
abstract("asl3dflex sequence");
psdname("asl3dflex");

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
int genspiral(FILE* fID_partitions);
int genviews(FILE* fID_partitions);

/* Declare function prototypes from aslprep.h */
int readprep(int id, int *len,
		int *rho_lbl, int *theta_lbl, int *grad_lbl,
		int *rho_ctl, int *theta_ctl, int *grad_ctl); 
int readschedule(int id, int* var, char* varname, int lines);
int readschedulef(int id, float* var, char* varname, int lines);
int gentadjusttbl();
int genlbltbl(int mod, int* lbltbl); 
float calc_sinc_B1(float cyc_rf, int pw_rf, float flip_rf);

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

	/* Add opuser fields to the Adv. pulse sequence parameters interface */	
	piuset = use0;
	cvdesc(opuser0, "Readout mode (0 = GRE, 1 = FSE)");
	cvdef(opuser0, ro_mode);
	opuser0 = 1;
	cvmin(opuser0, 0);
	cvmax(opuser0, 1);
	ro_mode = opuser0;

	piuset += use1;
	cvdesc(opuser1, "Echo spacing (ms)");
	cvdef(opuser1, esp);
	if (ro_mode) {/* FSE */
		cvmin(opuser1, opte*1e-3);
		cvmax(opuser1, opte*1e-3);
	}
	else { /* GRE */
		cvmin(opuser1, 0);
		cvmax(opuser1, 1000);	
	}
	opuser1 = esp;
	esp = opuser1*1e3;

	piuset += use2;
	cvdesc(opuser2, "Number of frames to acquire");
	cvdef(opuser2, nframes);
	opuser2 = nframes;
	cvmin(opuser2, 1);
	cvmax(opuser2, MAXNFRAMES);
	nframes = opuser2;
	
	piuset += use5;
	cvdesc(opuser5, "Number of disdaq trains");
	cvdef(opuser5, ndisdaqtrains);
	opuser5 = ndisdaqtrains;
	cvmin(opuser5, 0);
	cvmax(opuser5, 100);
	ndisdaqtrains = opuser5;
	
	piuset += use6;
	cvdesc(opuser6, "Number of disdaq echoes");
	cvdef(opuser6, ndisdaqechoes);
	opuser6 = ndisdaqechoes;
	cvmin(opuser6, 0);
	cvmax(opuser6, 100);
	ndisdaqechoes = opuser6;

	piuset += use7;
	cvdesc(opuser7, "2D spiral: 1=out 2=in 3=out-in 4=in-out");
	cvdef(opuser7, sptype2d);
	opuser7 = sptype2d;
	cvmin(opuser7, 1);
	cvmax(opuser7, 4);
	sptype2d = opuser7;

	piuset += use8;
	cvdesc(opuser8, "3D spiral: 0=2D 1=stack 2=1-ax-rots 3=2-ax-rots");
	cvdef(opuser8, sptype3d);
	opuser8 = sptype3d;
	cvmin(opuser8, 0);
	cvmax(opuser8, 4);
	sptype3d = opuser8;

	piuset += use9;
	cvdesc(opuser9, "kz acceleration factor (SOS only)");
	cvdef(opuser9, kz_acc);
	opuser9 = kz_acc;
	cvmin(opuser9, 1.0);
	cvmax(opuser9, 100.0);
	kz_acc = opuser9;

	piuset += use10;
	cvdesc(opuser10, "VD-spiral center oversampling factor");
	cvdef(opuser10, spvd0);
	opuser10 = spvd0;
	cvmin(opuser10, 0.001);
	cvmax(opuser10, 50.0);
	spvd0 = opuser10;

	piuset += use11;
	cvdesc(opuser11, "VD-spiral edge oversampling factor");
	cvdef(opuser11, spvd1);
	opuser11 = spvd1;
	cvmin(opuser11, 0.001);
	cvmax(opuser11, 50.0);
	spvd1 = opuser11;

	piuset += use12;
	cvdesc(opuser12, "Variable refocuser flip angle attenuation factor");
	cvdef(opuser12, varflipfac);
	opuser11 = varflipfac;
	cvmin(opuser12, 0.1);
	cvmax(opuser12, 1.0);
	varflipfac = opuser12;

	piuset += use13;
	cvdesc(opuser13, "Recon script ID #");
	cvdef(opuser13, rhrecon);
	opuser13 = rhrecon;
	cvmin(opuser13, 0);
	cvmax(opuser13, 9999);
	rhrecon = opuser13;
	
	piuset += use14;
	cvdesc(opuser14, "ASL prep schedule ID #");
	cvdef(opuser14, schedule_id);
	opuser14 = schedule_id;
	cvmin(opuser14, 0);
	cvmax(opuser14, 9999);
	schedule_id = opuser14;

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
	FILE* fseq;
	FILE* fschedule_id;
	int receive_freq[opslquant], rf1_freq[opslquant], rf2_freq[opslquant];
	int slice, ddan, framen, shotn, echon;
	FILE* fID_partitions;
	int minesp, minte, maxte, absmintr;	
	float rf1_b1, rf2_b1;
	float fatsat_b1, blksat_b1, bkgsup_b1;
	float prep1_b1, prep2_b1;
	int tmp_pwa, tmp_pw, tmp_pwd;
	float tmp_a, tmp_area;

	/*********************************************************************/
#include "predownload.in"	/* include 'canned' predownload code */
	/*********************************************************************/
	
	/* Create a file containing the schedule id number */
	if (schedule_id > 0) {
		fschedule_id = fopen("asl3dflex_scheduleidnum.txt", "w");
		fprintf(fschedule_id, "%05d", schedule_id);
		fclose(fschedule_id);
	}

	/* Read in flip angle schedule */
	sprintf(tmpstr, "flipfactbl");
	fprintf(stderr, "predownload(): reading in %s using readschedule(), schedule_id = %d\n", tmpstr, schedule_id);
	switch (readschedulef(schedule_id, flipfactbl, tmpstr, opetl)) {
		case 0:
			fprintf(stderr, "predownload(): generating schedule for %s...\n", tmpstr);
			for (echon = 0; echon < opetl; echon++) {
				if (echon == 0)
					flipfactbl[echon] = 1.0;
				else
					flipfactbl[echon] = varflipfac + (float)(echon - 1) / (float)(opetl - 1) * (1.0 - varflipfac);
			}
			
			break;
		case -1:
			return FAILURE;
	}
	
	/* Read in rf phase schedule */
	sprintf(tmpstr, "flipphstbl");
	fprintf(stderr, "predownload(): reading in %s using readschedule(), schedule_id = %d\n", tmpstr, schedule_id);
	switch (readschedulef(schedule_id, flipphstbl, tmpstr, opetl)) {
		case 0:
			fprintf(stderr, "predownload(): generating schedule for %s...\n", tmpstr);
			for (echon = 0; echon < opetl; echon++) {
				if (ro_mode == 0) /* GRE mode */
					flipphstbl[echon] = phs_tip + M_PI * spgr_phsinc/180 * pow(echon,2.0);
				else if (ro_mode == 1 && phscyc_fse == 0) /* phase cycled FSE */
					flipphstbl[echon] = phs_inv;
				else /* phase cycled FSE */
					flipphstbl[echon] = phs_inv * pow(-1.0, (float)echon);
			}
			break;
		case -1:
			return FAILURE;
	}

	/* Read in prep1_id (scalar) */
	sprintf(tmpstr, "prep1_id");
	fprintf(stderr, "predownload(): reading in %s using readschedule(), schedule_id = %d\n", tmpstr, schedule_id);	
	readschedule(schedule_id, &prep1_id, tmpstr, 1);


	/* Read in prep1_pldtbl schedule */
	sprintf(tmpstr, "prep1_pldtbl");
	fprintf(stderr, "predownload(): reading in %s using readschedule(), schedule_id = %d\n", tmpstr, schedule_id);	
	switch (readschedule(schedule_id, prep1_pldtbl, tmpstr, nframes)) {
		case 0:
			fprintf(stderr, "predownload(): generating schedule for %s...\n", tmpstr);	
			for (framen = 0; framen < nframes; framen++)
				prep1_pldtbl[framen] = (framen >= nm0frames) * prep1_pld;
			break;
		case -1:
			return FAILURE;
	}
	
	/* Read in prep1_lbltbl schedule */
	sprintf(tmpstr, "prep1_lbltbl");
	fprintf(stderr, "predownload(): reading in %s using readschedule(), schedule_id = %d\n", tmpstr, schedule_id);	
	switch (readschedule(schedule_id, prep1_lbltbl, tmpstr, nframes)) {
		case 0:
			fprintf(stderr, "predownload(): generating schedule for %s...\n", tmpstr);	
			genlbltbl(prep1_mod, prep1_lbltbl);
			break;
		case -1:
			return FAILURE;
	}
	
	/* Read in prep1_tbgs1tbl schedule */
	sprintf(tmpstr, "prep1_tbgs1tbl");
	fprintf(stderr, "predownload(): reading in %s using readschedule(), schedule_id = %d\n", tmpstr, schedule_id);	
	switch (readschedule(schedule_id, prep1_tbgs1tbl, tmpstr, nframes)) {
		case 0:
			fprintf(stderr, "predownload(): generating schedule for %s...\n", tmpstr);	
			for (framen = 0; framen < nframes; framen++)
				prep1_tbgs1tbl[framen] = (framen >= nm0frames) * prep1_tbgs1;
			break;
		case -1:
			return FAILURE;
	}
	
	/* Read in prep1_tbgs2tbl schedule */
	sprintf(tmpstr, "prep1_tbgs2tbl");
	fprintf(stderr, "predownload(): reading in %s using readschedule(), schedule_id = %d\n", tmpstr, schedule_id);	
	switch (readschedule(schedule_id, prep1_tbgs2tbl, tmpstr, nframes)) {
		case 0:
			fprintf(stderr, "predownload(): generating schedule for %s...\n", tmpstr);	
			for (framen = 0; framen < nframes; framen++)
				prep1_tbgs2tbl[framen] = (framen >= nm0frames) * prep1_tbgs2;
			break;
		case -1:
			return FAILURE;
	}
	
	/* Read in prep2_id (scalar) */
	sprintf(tmpstr, "prep2_id");
	fprintf(stderr, "predownload(): reading in %s using readschedule(), schedule_id = %d\n", tmpstr, schedule_id);	
	readschedule(schedule_id, &prep2_id, tmpstr, 1);
	
	/* Read in prep2_pldtbl schedule */
	sprintf(tmpstr, "prep2_pldtbl");
	fprintf(stderr, "predownload(): reading in %s using readschedule(), schedule_id = %d\n", tmpstr, schedule_id);	
	switch (readschedule(schedule_id, prep2_pldtbl, tmpstr, nframes)) {
		case 0:
			fprintf(stderr, "predownload(): generating schedule for %s...\n", tmpstr);	
			for (framen = 0; framen < nframes; framen++)
				prep2_pldtbl[framen] = (framen >= nm0frames) * prep2_pld;
			break;
		case -1:
			return FAILURE;
	}
	
	/* Read in prep2_lbltbl schedule */
	sprintf(tmpstr, "prep2_lbltbl");
	fprintf(stderr, "predownload(): reading in %s using readschedule(), schedule_id = %d\n", tmpstr, schedule_id);	
	switch (readschedule(schedule_id, prep2_lbltbl, tmpstr, nframes)) {
		case 0:
			fprintf(stderr, "predownload(): generating schedule for %s...\n", tmpstr);	
			genlbltbl(prep2_mod, prep2_lbltbl);
			break;
		case -1:
			return FAILURE;
	}
	
	/* Read in prep2_tbgs1tbl schedule */
	sprintf(tmpstr, "prep2_tbgs1tbl");
	fprintf(stderr, "predownload(): reading in %s using readschedule(), schedule_id = %d\n", tmpstr, schedule_id);	
	switch (readschedule(schedule_id, prep2_tbgs1tbl, tmpstr, nframes)) {
		case 0:
			fprintf(stderr, "predownload(): generating schedule for %s...\n", tmpstr);	
			for (framen = 0; framen < nframes; framen++)
				prep2_tbgs1tbl[framen] = (framen >= nm0frames) * prep2_tbgs1;
			break;
		case -1:
			return FAILURE;
	}
	
	/* Read in prep2_tbgs2tbl schedule */
	sprintf(tmpstr, "prep2_tbgs2tbl");
	fprintf(stderr, "predownload(): reading in %s using readschedule(), schedule_id = %d\n", tmpstr, schedule_id);	
	switch (readschedule(schedule_id, prep2_tbgs2tbl, tmpstr, nframes)) {
		case 0:
			fprintf(stderr, "predownload(): generating schedule for %s...\n", tmpstr);	
			for (framen = 0; framen < nframes; framen++)
				prep2_tbgs2tbl[framen] = (framen >= nm0frames) * prep2_tbgs2;
			break;
		case -1:
			return FAILURE;
	}
	
	/* Read in doblksat */
	sprintf(tmpstr, "doblksat");
	fprintf(stderr, "predownload(): reading in %s using readschedule(), schedule_id = %d\n", tmpstr, schedule_id);	
	readschedule(schedule_id, &doblksat, tmpstr, 1);
	
	/* Read in doblksattbl schedule */
	sprintf(tmpstr, "doblksattbl");
	fprintf(stderr, "predownload(): reading in %s using readschedule(), schedule_id = %d\n", tmpstr, schedule_id);	
	switch (readschedule(schedule_id, doblksattbl, tmpstr, nframes)) {
		case 0:
			fprintf(stderr, "predownload(): generating schedule for %s...\n", tmpstr);	
			for (framen = 0; framen < nframes; framen++)
				doblksattbl[framen] = (framen >= nm0frames) * doblksat;
			break;
		case -1:
			return FAILURE;
	}

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
	
	/* Update the background suppression pulse parameters */
	res_bkgsuprho = 500;
	pw_bkgsuprho = 5000;
	a_bkgsuptheta = 1.0;
	res_bkgsuptheta = res_bkgsuprho;
	pw_bkgsuptheta = pw_bkgsuprho;
	
	/* Update the bulk saturation pulse parameters */
	res_blksatrho = 250;
	pw_blksatrho = 2500;
	a_blksattheta = 1.0;
	res_blksattheta = res_blksatrho;
	pw_blksattheta = pw_blksatrho;

	/* Set the parameters for the fat sat pulse */
	a_fatsatrho = 0.5 * 440 / 1250;	
	pw_fatsatrho = 4 * round(cyc_fatsatrho*1e6 / 440);
	res_fatsatrho = pw_fatsatrho / 2;	
	
	/* Set the parameters for the spin echo tipdown refocuser gradients */
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
	pw_gzblksatcrush = tmp_pw;
	pw_gzblksatcrusha = tmp_pwa;
	pw_gzblksatcrushd = tmp_pwd;
	a_gzblksatcrush = tmp_a;

	pw_gzfatsatcrush = tmp_pw;
	pw_gzfatsatcrusha = tmp_pwa;
	pw_gzfatsatcrushd = tmp_pwd;
	a_gzfatsatcrush = tmp_a;

	pw_gzrf2crush1 = tmp_pw;
	pw_gzrf2crush1a = tmp_pwa;
	pw_gzrf2crush1d = tmp_pwd;
	a_gzrf2crush1 = tmp_a;

	if (ro_mode == 0) { /* GRE - crusher acts as a refocuser for rf2 slice select gradients */
		tmp_area = a_gzrf2 * (pw_gzrf2 + (pw_gzrf2a + pw_gzrf2d)/2.0);
		amppwgrad(tmp_area, GMAX, 0, 0, ZGRAD_risetime, 0, &tmp_a, &tmp_pwa, &tmp_pw, &tmp_pwd); 	
		tmp_a *= -0.5;
	} /* otherwise, FSE - crusher is just a crusher */
	pw_gzrf2crush2 = tmp_pw;
	pw_gzrf2crush2a = tmp_pwa;
	pw_gzrf2crush2d = tmp_pwd;
	a_gzrf2crush2 = tmp_a;
		
	/* Open the transformations schedule file */
	sprintf(tmpstr, "./aslprep/schedules/%05d/partitions.txt", schedule_id);
	fprintf(stderr, "predownload(): opening %s...\n", tmpstr);
	fID_partitions = fopen(tmpstr, "r");	

	/* Generate initial spiral trajectory */
	fprintf(stderr, "predownload(): calling genspiral()\n");
	F0 = spvd0/(float)opnshots * (float)opfov / 10.0;
	F1 = 2*pow((float)opfov/10.0,2)/opxres *(spvd1 - spvd0)/(float)opnshots;
	F2 = 0;
	if (genspiral(fID_partitions) == 0) {
		epic_error(use_ermes,"failure to generate spiral waveform", EM_PSD_SUPPORT_FAILURE, EE_ARGS(0));
		return FAILURE;
	}
	
	/* Generate view transformations */
	fprintf(stderr, "predownload(): calling genviews()\n");
	if (genviews(fID_partitions) == 0) {
		epic_error(use_ermes,"failure to generate view transformation matrices", EM_PSD_SUPPORT_FAILURE, EE_ARGS(0));
		return FAILURE;
	}

	/* Close the transformations file */
	if (fID_partitions != 0)
		fclose(fID_partitions);

	/* Scale the rotation matrices */
	scalerotmats(tmtxtbl, &loggrd, &phygrd, opetl*opnshots*nframes, 0);
	
	/* Update the readout pulse parameters */
	a_gxw = XGRAD_max;
	a_gyw = YGRAD_max;
	a_gzw = ZGRAD_max;
	ia_gxw = MAX_PG_WAMP;
	ia_gyw = MAX_PG_WAMP;
	ia_gzw = MAX_PG_WAMP;
	res_gxw = grad_len;
	res_gyw = grad_len;
	res_gzw = grad_len;
	pw_gxw = GRAD_UPDATE_TIME*grad_len;
	pw_gyw = GRAD_UPDATE_TIME*grad_len;
	pw_gzw = GRAD_UPDATE_TIME*grad_len;

	/* Calculate minimum ESP */
	minesp = 1ms; /* fudge factor */
	minesp += pw_gzrf2/2 + pw_gzrf2d/2; /* 2nd half of the rf2 pulse */
	minesp += pgbuffertime; /* buffer */
	minesp += pw_gzrf2crush2a + pw_gzrf2crush2 + pw_gzrf2crush2d; /* crush2 pulse */
	minesp += pgbuffertime; /* buffer */
	minesp += TIMESSI; /* inter-core time */
	minesp += pgbuffertime; /* buffer */
	minesp += pw_gxw; /* readout window length */
	minesp += pgbuffertime; /* buffer */
	minesp += TIMESSI; /* inter-core time */
	minesp += pgbuffertime; /* buffer */
	minesp += pw_gzrf2crush1a + pw_gzrf2crush1 + pw_gzrf2crush1d; /* crush1 pulse */
	minesp += pgbuffertime; /* buffer */
	minesp += pw_gzrf2a + pw_gzrf2/2; /* 1st half of rf2 pulse */
	minesp = GRAD_UPDATE_TIME*ceil((float)minesp/(float)GRAD_UPDATE_TIME); /* round up to gradient sampling interval */
	
	/* Calculate tipcore deadtime */
	deadtime_tipcore = esp/2;
	deadtime_tipcore -= (pw_gzrf1/2 + pw_gzrf1d); /* 2nd half of rf1 pulse */
	deadtime_tipcore -= pgbuffertime; /* buffer */
	deadtime_tipcore -= (pw_gzrf1ra + pw_gzrf1r + pw_gzrf1rd); /* rf1 rewinder pulse */
	deadtime_tipcore -= pgbuffertime; /* buffer */
	deadtime_tipcore -= TIMESSI; /* inter-core time */
	deadtime_tipcore -= pgbuffertime; /* buffer */
	deadtime_tipcore -= (pw_gzrf2crush1a + pw_gzrf2crush1 + pw_gzrf2crush1d); /* crush1 pulse */
	deadtime_tipcore -= pgbuffertime; /* buffer */
	deadtime_tipcore -= (pw_gzrf2a + pw_gzrf2/2); /* 1st half of rf2 pulse */	

	/* Set echo time constraints */
	if (ro_mode == 1) { /* FSE */
		cvdef(opte,esp);
		minte = esp;
		maxte = esp;
		
		deadtime1_seqcore = (esp - minesp) / 2 + pgbuffertime;
		deadtime2_seqcore = (esp - minesp) / 2 + pgbuffertime;
	}
	else { /* GRE */
		minte = (pw_gzrf2/2 + pw_gzrf2d); /* 2nd half of rf2 pulse */
		minte += pgbuffertime; /* buffer */
		minte += pw_gzrf2crush2a + pw_gzrf2crush2 + pw_gzrf2crush2d; /* rewinder */
		minte += pgbuffertime; /* buffer */
		minte += TIMESSI; /* inter-core time */
		minte += pgbuffertime; /* buffer */
		
		maxte = esp; /* echo spacing */
		maxte -= (pw_gzrf2/2 + pw_gzrf2a); /* 1st half of crusher at end of readout */
		maxte -= pgbuffertime; /* buffer */
		maxte -= (pw_gzrf2crush2a + pw_gzrf2crush2 + pw_gzrf2crush2d); /* crusher */
		maxte -= pgbuffertime; /* buffer */
		maxte -= TIMESSI; /* inter-core time */
		maxte -= pgbuffertime; /* buffer */
		maxte -= pw_gxw; /* readout */
		
		deadtime1_seqcore = opte - minte + pgbuffertime;
		minesp += deadtime1_seqcore;
		deadtime2_seqcore = esp - minesp + pgbuffertime; 
	}
	cvmin(esp, minesp);
	cvmin(opuser1, minesp*1e-3);	
	if ((exist(opautote) == PSD_MINTE)||(exist(opautote) == PSD_MINTEFULL))
		opte = minte;
	cvmin(opte, minte);
	cvmax(opte, maxte);

	/* Round deadtimes to nearest sampling interval */
	deadtime1_seqcore += GRAD_UPDATE_TIME - (deadtime1_seqcore % GRAD_UPDATE_TIME);
	deadtime2_seqcore -= (deadtime2_seqcore % GRAD_UPDATE_TIME);
	
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

	/* Determine max B1 for the rest of the pulses in L_SCAN */
	rf1_b1 = calc_sinc_B1(cyc_rf1, pw_rf1, 90.0);
	fprintf(stderr, "predownload(): maximum B1 for rf1 pulse: %f\n", rf1_b1);
	if (rf1_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = rf1_b1;

	rf2_b1 = calc_sinc_B1(cyc_rf2, pw_rf2, opflip);
	fprintf(stderr, "predownload(): maximum B1 for rf2 pulse: %f\n", rf2_b1);
	if (rf2_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = rf2_b1;

	blksat_b1 = 0.03867; /* nominal max b1 of sech_7360 pulse */
	fprintf(stderr, "predownload(): maximum B1 for blksat pulse: %f Gauss \n", blksat_b1);
	if (blksat_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = blksat_b1;
	
	bkgsup_b1 = blksat_b1; /* nominal max b1 of sech_7360 pulse */
	/*bkgsup_b1 = 0.03867;*/ /* nominal max b1 of sech_7360 pulse */
	fprintf(stderr, "predownload(): maximum B1 for bkgsup pulse: %f Gauss \n", bkgsup_b1);
	if (bkgsup_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = bkgsup_b1;
	
	fatsat_b1 = calc_sinc_B1(cyc_fatsatrho, pw_fatsatrho, 90.0);
	fprintf(stderr, "predownload(): maximum B1 for fatsat pulse: %f Gauss\n", fatsat_b1);
	if (fatsat_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = fatsat_b1;
	
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

	a_rf2 = rf2_b1 / maxB1Seq;
	ia_rf2 = a_rf2 * MAX_PG_WAMP;
	
	a_blksatrho = blksat_b1 / maxB1Seq;
	ia_blksatrho = a_blksatrho * MAX_PG_WAMP;
	
	a_bkgsuprho = bkgsup_b1 / maxB1Seq;
	ia_bkgsuprho = a_bkgsuprho * MAX_PG_WAMP;
	
	a_fatsatrho = fatsat_b1 / maxB1Seq;
	ia_fatsatrho = a_fatsatrho * MAX_PG_WAMP;
	
	a_prep1rholbl = prep1_b1 / maxB1Seq;
	ia_prep1rholbl = a_prep1rholbl * MAX_PG_WAMP;
	
	a_prep1rhoctl = prep1_b1 / maxB1Seq;
	ia_prep1rhoctl = a_prep1rhoctl * MAX_PG_WAMP;
	
	a_prep2rholbl = prep2_b1 / maxB1Seq;
	ia_prep2rholbl = a_prep2rholbl * MAX_PG_WAMP;
	
	a_prep2rhoctl = prep2_b1 / maxB1Seq;
	ia_prep2rhoctl = a_prep2rhoctl * MAX_PG_WAMP;

	/* Calculate the duration of blksatcore */
	dur_blksatcore = 0;
	dur_blksatcore += pgbuffertime;
	dur_blksatcore += pw_blksatrho;
	dur_blksatcore += pgbuffertime;
	dur_blksatcore += pw_gzblksatcrusha + pw_gzblksatcrush + pw_gzblksatcrushd;
	dur_blksatcore += pgbuffertime;

	/* Calcualte the duration of prep1core */
	dur_prep1core = 0;
	dur_prep1core += pgbuffertime;
	dur_prep1core += GRAD_UPDATE_TIME*prep1_len;
	dur_prep1core += pgbuffertime;

	/* Calcualte the duration of prep2core */
	dur_prep2core = 0;
	dur_prep2core += pgbuffertime;
	dur_prep2core += GRAD_UPDATE_TIME*prep2_len;
	dur_prep2core += pgbuffertime;

	/* Calculate the duration of bkgsupcore */
	dur_bkgsupcore = 0;
	dur_bkgsupcore += pgbuffertime;
	dur_bkgsupcore += pw_bkgsuprho;
	dur_bkgsupcore += pgbuffertime;	

	/* Calculate the duration of fatsatcore */
	dur_fatsatcore = 0;
	dur_fatsatcore += pgbuffertime;
	dur_fatsatcore += pw_fatsatrho;
	dur_fatsatcore += pgbuffertime;
	dur_fatsatcore += pw_gzfatsatcrusha + pw_gzfatsatcrush + pw_gzfatsatcrushd;
	dur_fatsatcore += pgbuffertime;
	
	/* Calcualte the duration of tipcore */
	dur_tipcore = 0;
	dur_tipcore += pgbuffertime;
	dur_tipcore += pw_gzrf1a + pw_gzrf1 + pw_gzrf1d;
	dur_tipcore += pgbuffertime;
	dur_tipcore += pw_gzrf1ra + pw_gzrf1r + pw_gzrf1rd;
	dur_tipcore += pgbuffertime;
	dur_tipcore += deadtime_tipcore;

	/* Calculate the duration of flipcore (GRE) this is also the refocuser is FSE */
	dur_flipcore = 0;
	dur_flipcore += pgbuffertime;
	dur_flipcore += pw_gzrf2crush1a + pw_gzrf2crush1 + pw_gzrf2crush1d;
	dur_flipcore += pgbuffertime;
	dur_flipcore += pw_gzrf2a + pw_gzrf2 + pw_gzrf2d;
	dur_flipcore += pgbuffertime;
	dur_flipcore += pw_gzrf2crush2a + pw_gzrf2crush2 + pw_gzrf2crush2d;
	dur_flipcore += pgbuffertime; 

	/* Calculate the duration of seqcore */
	dur_seqcore = 0;
	dur_seqcore += deadtime1_seqcore + pgbuffertime;	
	dur_seqcore += GRAD_UPDATE_TIME*grad_len;
	dur_seqcore += pgbuffertime + deadtime2_seqcore;
	
	/* Calculate the duration of the whole readout section (with TIMESSI's)*/
	dur_readout = 0;
	if (ro_mode == 1) /* FSE */
		dur_readout += dur_tipcore + TIMESSI;
	dur_readout += (opetl + ndisdaqechoes) * (dur_flipcore + TIMESSI + dur_seqcore + TIMESSI);

	/* Calculate minimum TR */
	absmintr = doblksat*dur_blksatcore + TIMESSI;
	absmintr += (prep1_id > 0)*(dur_prep1core + TIMESSI + prep1_pld + TIMESSI);
	absmintr += (prep2_id > 0)*(dur_prep2core + TIMESSI + prep2_pld + TIMESSI);
	absmintr += (dofatsat)*(dur_fatsatcore + TIMESSI);
	absmintr += dur_readout;
	if (exist(opautotr) == PSD_MINIMUMTR)
		optr = absmintr;	
	cvmin(optr, absmintr);
	
	/* 
	 * Calculate RF filter and update RBW:
	 *   &echo1_rtfilt: I: all the filter parameters.
	 *   exist(oprbw): I/O: desired and final allowable bw.
	 *   exist(opxres): I: output pts generated by filter.
	 *   OVERWRITE_OPRBW: oprbw will be updated.
	 */
	if( calcfilter( &echo1_rtfilt,
				exist(oprbw),
				grad_len,
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


	/* Read in tadjusttbl schedule */
	sprintf(tmpstr, "tadjusttbl");
	fprintf(stderr, "predownload(): reading in %s using readschedule(), schedule_id = %d\n", tmpstr, schedule_id);	
	switch (readschedule(schedule_id, tadjusttbl, tmpstr, nframes)) {
		case 0:   /* what to do if the schedule is not there  (readschedule returns 0)  */
			for (framen = 0; framen < nframes; framen++) {

				/* start with tadjust set to TR and then subtract all the other time blocks */
				tadjusttbl[framen] = optr;

				if (doblksat)
					tadjusttbl[framen] -= (dur_blksatcore + TIMESSI); /* bulk sat core */

				if (prep1_id > 0) { /* prep1 pulse/pld core */
					tadjusttbl[framen] -= (dur_prep1core + TIMESSI);
					tadjusttbl[framen] -= (prep1_pldtbl[framen] > 0) * (prep1_pldtbl[framen] + TIMESSI);
				}

				if (prep2_id > 0) { /* prep2 pulse/pld core */
					tadjusttbl[framen] -= (dur_prep2core + TIMESSI);
					tadjusttbl[framen] -= (prep2_pldtbl[framen] > 0) * (prep2_pldtbl[framen] + TIMESSI);
				}

				if (dofatsat) /* if fat sat is enabled */
					tadjusttbl[framen] -= (dur_fatsatcore + TIMESSI); /* add fat sat core */

				tadjusttbl[framen] -= dur_readout; /* add the readout cores */
			}
			break;
		case -1:
			return FAILURE;
	}

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
	entry_point_table[L_SCAN].epprexres = grad_len;

	(void)strcpy( entry_point_table[L_APS2].epname, "aps2" );
	entry_point_table[L_APS2].epfilter = (unsigned char)echo1_filt->fslot;
	entry_point_table[L_APS2].epprexres = grad_len;

	(void)strcpy( entry_point_table[L_MPS2].epname, "mps2" );
	entry_point_table[L_MPS2].epfilter = (unsigned char)echo1_filt->fslot;
	entry_point_table[L_MPS2].epprexres = grad_len;

	/* Loop through sequence and write out timing */
	pidmode = PSD_CLOCK_NORM;
	pitslice = optr;
	pitscan = 0; /* pitscan controls the clock time on the interface */	
	fseq = fopen("scansequence.txt","w");
	fprintf(fseq, "%-50s%22s%22s\n\n", "event label", "start time", "duration");
	for (ddan = 0; ddan < ndisdaqtrains; ddan++) {
		pitscan += optr - dur_readout;
		fprintf(fseq, "%-50s%20dus%20dus\n", "disdaq readout", (int)pitscan, dur_readout);
		pitscan += dur_readout;
	}
	for (framen = 0; framen < nframes; framen++) {
		for (shotn = 0; shotn < opnshots; shotn++) {
			if (doblksat) {
				if (doblksattbl[framen] == 1)
					fprintf(fseq, "%-50s%20dus%20dus\n", "bulk sat pulse", (int)pitscan, dur_blksatcore);
				pitscan += dur_blksatcore + TIMESSI; /* add bulk sat core */
			}

			/* add tadjust */
			pitscan += tadjusttbl[framen];

			if (prep1_id > 0) { /* add prep1 pulse/pld core */
				if (prep1_lbltbl[framen] == 0)
					fprintf(fseq, "%-50s%20dus%20dus\n", "prep1 ctl pulse", (int)pitscan, dur_prep1core);
				else if (prep1_lbltbl[framen] == 1)
					fprintf(fseq, "%-50s%20dus%20dus\n", "prep1 lbl pulse", (int)pitscan, dur_prep1core);
				pitscan += dur_prep1core + TIMESSI;
				pitscan += (prep1_pldtbl[framen] > 0) * (prep1_pldtbl[framen] + TIMESSI);
			}
			if (prep2_id > 0) { /* add prep2 pulse/pld core */
				if (prep2_lbltbl[framen] == 0)
					fprintf(fseq, "%-50s%20dus%20dus\n", "prep2 ctl pulse", (int)pitscan, dur_prep2core);
				else if (prep2_lbltbl[framen] == 1)
					fprintf(fseq, "%-50s%20dus%20dus\n", "prep2 lbl pulse", (int)pitscan, dur_prep2core);
				pitscan += dur_prep2core + TIMESSI;
				pitscan += (prep2_pldtbl[framen] > 0) * (prep2_pldtbl[framen] + TIMESSI);
			}

			if (dofatsat) { /* if fat sat is enabled */
				fprintf(fseq, "%-50s%20dus%20dus\n", "fat sat pulse", (int)pitscan, dur_fatsatcore);
				pitscan += dur_fatsatcore + TIMESSI; /* add fat sat core */
			}

			fprintf(fseq, "%-50s%20dus%20dus\n", "readout", (int)pitscan, dur_readout);
			pitscan += dur_readout; /* add the readout cores */
		}
	}
	fclose(fseq);
	
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
	setupslices(rf2_freq, rsp_info, opslquant, a_gzrf2, 1.0, opfov, TYPTRANSMIT);
	setupslices(receive_freq, rsp_info, opslquant, 0.0, 1.0, 2.0, TYPREC);

	/* Average together all slice frequencies */
	xmitfreq1 = 0;
	xmitfreq2 = 0;
	recfreq = 0;	
	for (slice = 0; slice < opslquant; slice++) {
		xmitfreq1 += (float)rf1_freq[slice] / (float)opslquant;
		xmitfreq2 += (float)rf2_freq[slice] / (float)opslquant;
		recfreq += (float)receive_freq[slice] / (float)opslquant;
	}

	if( orderslice( TYPNORMORDER, opslquant, opslquant, TRIG_INTERN ) == FAILURE )
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

	rhfrsize = grad_len;
	rhnframes = 2*ceil((float)(opetl * opnshots + 1) / 2.0);
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
	fprintf(finfo, "\t%-50s%20d\n", "dofatsat:", dofatsat);
	fprintf(finfo, "\t%-50s%20d\n", "ro_mode:", ro_mode);
	fprintf(finfo, "\t%-50s%20d\n", "pgbuffertime:", pgbuffertime);
	fprintf(finfo, "\t%-50s%20f\n", "phs_tip:", phs_tip);
	fprintf(finfo, "\t%-50s%20f\n", "phs_inv:", phs_inv);
	fprintf(finfo, "\t%-50s%20f\n", "phs_rx:", phs_rx);
	fprintf(finfo, "\t%-50s%20d\n", "phscyc_fse:", phscyc_fse);
	fprintf(finfo, "\t%-50s%20f\n", "spgr_phsinc:", spgr_phsinc);
	fprintf(finfo, "\t%-50s%20f\n", "varflipfac:", varflipfac);
	fprintf(finfo, "\t%-50s%20d\n", "kill_grads:", kill_grads);

	fprintf(finfo, "trajectory cvs:\n");
	fprintf(finfo, "\t%-50s%20d\n", "nnav:", nnav);
	fprintf(finfo, "\t%-50s%20f\n", "spvd0:", spvd0);
	fprintf(finfo, "\t%-50s%20f\n", "spvd1:", spvd1);
	fprintf(finfo, "\t%-50s%20d\n", "sptype2d:", sptype2d);
	fprintf(finfo, "\t%-50s%20d\n", "sptype3d:", sptype3d);
	
	fprintf(finfo, "ASL prep cvs:\n");
	fprintf(finfo, "\t%-50s%20d\n", "nm0frames:", nm0frames);
	fprintf(finfo, "\t%-50s%20d\n", "schedule_id:", schedule_id);
	fprintf(finfo, "\t%-50s%20d\n", "doblksat:", doblksat);
	fprintf(finfo, "\t%-50s%20d\n", "prep1_id:", prep1_id);
	fprintf(finfo, "\t%-50s%20d\n", "prep1_pld:", prep1_pld);
	fprintf(finfo, "\t%-50s%20d\n", "prep1_ncycles", prep1_ncycles);
	fprintf(finfo, "\t%-50s%20f\n", "prep1_rfmax:", prep1_rfmax);
	fprintf(finfo, "\t%-50s%20f\n", "prep1_gmax:", prep1_gmax);
	fprintf(finfo, "\t%-50s%20d\n", "prep1_mod", prep1_mod);
	fprintf(finfo, "\t%-50s%20d\n", "prep1_tbgs1", prep1_tbgs1);
	fprintf(finfo, "\t%-50s%20d\n", "prep1_tbgs2", prep1_tbgs2);
	fprintf(finfo, "\t%-50s%20d\n", "prep2_id:", prep2_id);
	fprintf(finfo, "\t%-50s%20d\n", "prep2_pld:", prep2_pld);
	fprintf(finfo, "\t%-50s%20d\n", "prep2_ncycles", prep2_ncycles);
	fprintf(finfo, "\t%-50s%20f\n", "prep2_rfmax:", prep2_rfmax);
	fprintf(finfo, "\t%-50s%20f\n", "prep2_gmax:", prep2_gmax);
	fprintf(finfo, "\t%-50s%20d\n", "prep2_mod", prep2_mod);
	fprintf(finfo, "\t%-50s%20d\n", "prep2_tbgs1", prep2_tbgs1);
	fprintf(finfo, "\t%-50s%20d\n", "prep2_tbgs2", prep2_tbgs2);

	fprintf(finfo, "\ntime cvs:\n");
	fprintf(finfo, "\t%-50s%20f\n", "pitscan:", pitscan);
	fprintf(finfo, "\t%-50s%20d\n", "dur_blksatcore:", dur_blksatcore);
	fprintf(finfo, "\t%-50s%20d\n", "dur_prep1core:", dur_prep1core);
	fprintf(finfo, "\t%-50s%20d\n", "dur_prep2core:", dur_prep2core);
	fprintf(finfo, "\t%-50s%20d\n", "dur_bkgsupcore:", dur_bkgsupcore);
	fprintf(finfo, "\t%-50s%20d\n", "dur_fatsatcore:", dur_fatsatcore);
	fprintf(finfo, "\t%-50s%20d\n", "dur_tipcore:", dur_tipcore);
	fprintf(finfo, "\t%-50s%20d\n", "dur_flipcore:", dur_flipcore);
	fprintf(finfo, "\t%-50s%20d\n", "dur_seqcore:", dur_seqcore);
	
	fclose(finfo);

@inline Prescan.e PSpredownload	

	return SUCCESS;
}   /* end predownload() */


@inline Prescan.e PShost


@pg
/*********************************************************************
 *                 ASL3DFLEX.E PULSEGEN SECTION                      *
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

	/*********************************/
	/* Generate bulk saturation core */
	/*********************************/	
	fprintf(stderr, "pulsegen(): beginning pulse generation of blksatcore (bulk saturation core)\n");
	tmploc = 0;	

	fprintf(stderr, "pulsegen(): generating blksatrho & blksattheta (bulk saturation rf)...\n");
	tmploc += pgbuffertime; /* start time for blksat rf pulse */
	EXTWAVE(RHO, blksatrho, tmploc, 5000, 1.0, 250, sech_7360.rho, , loggrd);
	EXTWAVE(THETA, blksattheta, tmploc, 5000, 1.0, 250, sech_7360.theta, , loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_blksatrho; /* end time for blksat rf pulse */
	fprintf(stderr, " end: %dus\n", tmploc);	

	fprintf(stderr, "pulsegen(): generating gzblksatcrush (bulk saturation crusher)...\n");
	tmploc += pgbuffertime; /* start time for blksat crusher */
	TRAPEZOID(ZGRAD, gzblksatcrush, tmploc + pw_gzblksatcrusha, 0, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzblksatcrusha + pw_gzblksatcrush + pw_gzblksatcrushd; /* end time for blksat crusher */
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime; /* add some buffer */
	
	fprintf(stderr, "pulsegen(): finalizing blksatcore...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_blksatcore, tmploc);
	SEQLENGTH(blksatcore, dur_blksatcore, blksatcore);
	fprintf(stderr, "\tDone.\n");


	/*****************************/
	/* Generate prep1 label core */
	/*****************************/	
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


	/*******************************/
	/* Generate prep1 control core */
	/*******************************/	
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
	

	/*****************************/
	/* Generate prep2 label core */
	/*****************************/	
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


	/*******************************/
	/* Generate prep2 control core */
	/*******************************/	
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
		
	
	/****************************************/
	/* Generate background suppression core */
	/****************************************/	
	fprintf(stderr, "pulsegen(): beginning pulse generation of bkgsupcore (background suppression core)\n");
	tmploc = 0;	

	fprintf(stderr, "pulsegen(): generating bkgsuprho & bkgsuptheta (background suppression rf)...\n");
	tmploc += pgbuffertime; /* start time for bkgsup rf */
	EXTWAVE(RHO, bkgsuprho, tmploc, 5000, 1.0, 500, sech_7360.rho, , loggrd);
	EXTWAVE(THETA, bkgsuptheta, tmploc, 5000, 1.0, 500, sech_7360.theta, , loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_bkgsuprho; /* end time for bkg sup rf */
	fprintf(stderr, " end: %dus\n", tmploc);	
	tmploc += pgbuffertime; /* add some buffer */

	fprintf(stderr, "pulsegen(): finalizing bkgsupcore...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_bkgsupcore, tmploc);
	SEQLENGTH(bkgsupcore, dur_bkgsupcore, bkgsupcore);
	fprintf(stderr, "\tDone.\n");


	/*********************************/
	/* Generate fat saturation pulse */
	/*********************************/
	fprintf(stderr, "pulsegen(): beginning pulse generation of fatsatcore (fat saturation core)\n");	
	tmploc = 0;
	
	fprintf(stderr, "pulsegen(): generating fatsatrho (fat saturation rf pulse)...\n");
	tmploc += pgbuffertime; /* start time for fatsatrho */
	SINC2(RHO, fatsatrho, tmploc + psd_rf_wait, 3200, 1.0, ,0.5, , , loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_fatsatrho; /* end time for fatsatrho */
	fprintf(stderr, " end: %dus\n", tmploc);	
 
	fprintf(stderr, "pulsegen(): generating gzfatsatcrush (fat saturation crusher gradients)...\n");
	tmploc += pgbuffertime; /* start time for gzfatsatcrush */
	TRAPEZOID(ZGRAD, gzfatsatcrush, tmploc + pw_gzfatsatcrusha, GRAD_UPDATE_TIME*1000, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzfatsatcrusha + pw_gzfatsatcrush + pw_gzfatsatcrushd; /* end time for gzfatsatcrush */
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime; /* add some buffer */

	fprintf(stderr, "pulsegen(): finalizing fatsatcore...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_fatsatcore, tmploc);
	SEQLENGTH(fatsatcore, dur_fatsatcore, fatsatcore);
	fprintf(stderr, "\tDone.\n");

	
	/***********************************/
	/* Generate spin echo tipdown core */
	/***********************************/	
	fprintf(stderr, "pulsegen(): beginning pulse generation of tipcore (rf tipdown core for FSE readout)\n");
	tmploc = 0;

	fprintf(stderr, "pulsegen(): generating rf1 (90deg tipdown pulse)...\n");
	tmploc += pgbuffertime; /* start time for rf1 pulse */
	SLICESELZ(rf1, tmploc + pw_gzrf1a, 3200, (opslthick + opslspace)*opslquant, 90.0, 2, 1, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzrf1a + pw_gzrf1 + pw_gzrf1d; /* end time for rf1 pulse */
	fprintf(stderr, " end: %dus\n", tmploc);

	fprintf(stderr, "pulsegen(): generating gzrf1r (90deg tipdown gradient refocuser)...\n");
	tmploc += pgbuffertime; /* start time for gzrf1r */
	TRAPEZOID(ZGRAD, gzrf1r, tmploc + pw_gzrf1ra, 3200, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzrf1ra + pw_gzrf1r + pw_gzrf1rd; /* end time for gzrf1r pulse */
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime; /* add some buffer */

	tmploc += deadtime_tipcore; /* add deadtime to account for TE */

	fprintf(stderr, "pulsegen(): finalizing spin echo tipdown core...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_tipcore, tmploc);
	SEQLENGTH(tipcore, dur_tipcore, tipcore);
	fprintf(stderr, "\tDone.\n");


	/*************************************/
	/* Generate spin echo refocuser core */
	/*************************************/
	fprintf(stderr, "pulsegen(): beginning pulse generation of (flipcore) FSE inversion/GRE tipdown core\n");
	tmploc = 0;

	fprintf(stderr, "pulsegen(): generating gzrf2crush1 (pre-rf2 crusher)...\n");
	tmploc += pgbuffertime; /* start time for gzrf2crush1 */
	TRAPEZOID(ZGRAD, gzrf2crush1, tmploc + pw_gzrf2crush1a, 3200, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzrf2crush1a + pw_gzrf2crush1 + pw_gzrf2crush1d; /* end time for gzrf2crush1 pulse */
	fprintf(stderr, " end: %dus\n", tmploc);

	fprintf(stderr, "pulsegen(): generating rf2 (FSE inversion/GRE tipdown)...\n");
	tmploc += pgbuffertime; /* start time for rf2 */
	SLICESELZ(rf2, tmploc + pw_gzrf2a, 3200, (opslthick + opslspace)*opslquant, 180.0, 2, 1, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzrf2a + pw_gzrf2 + pw_gzrf2d; /* end time for rf2 pulse */
	fprintf(stderr, " end: %dus\n", tmploc);

	fprintf(stderr, "pulsegen(): generating gzrf2crush2 (post-rf2 crusher for FSE/tipdown gradient rewinder for GRE)...\n");
	tmploc += pgbuffertime; /* start time for gzrf2crush2 */
	TRAPEZOID(ZGRAD, gzrf2crush2, tmploc + pw_gzrf2crush2a, 3200, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzrf2crush2a + pw_gzrf2crush2 + pw_gzrf2crush2d; /* end time for gzrf2crush2 pulse */
	fprintf(stderr, " end: %dus\n", tmploc);	
	tmploc += pgbuffertime;

	fprintf(stderr, "pulsegen(): finalizing flip core...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_flipcore, tmploc);
	SEQLENGTH(flipcore, dur_flipcore, flipcore);
	fprintf(stderr, "\tDone.\n");


	/*************************/
	/* Generate readout core */
	/*************************/
	fprintf(stderr, "pulsegen(): beginning pulse generation of readout core (seqcore)\n");
	tmploc = 0;

	fprintf(stderr, "pulsegen(): generating gxw, gyw, & gzw (readout gradients) and echo1 (data acquisition window)...\n");
	tmploc += deadtime1_seqcore + pgbuffertime; /* start time for readout */
	INTWAVE(XGRAD, gxw, tmploc, XGRAD_max, grad_len, GRAD_UPDATE_TIME*grad_len, Gx, 1, loggrd);
	INTWAVE(YGRAD, gyw, tmploc, YGRAD_max, grad_len, GRAD_UPDATE_TIME*grad_len, Gy, 1, loggrd);
	INTWAVE(ZGRAD, gzw, tmploc, ZGRAD_max, grad_len, GRAD_UPDATE_TIME*grad_len, Gz, 1, loggrd);
	ACQUIREDATA(echo1, tmploc + psd_grd_wait,,,);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += GRAD_UPDATE_TIME*grad_len; /* end time for readout */
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime; /* add some buffer */

	tmploc += deadtime2_seqcore; /* add post-readout deadtime */

	fprintf(stderr, "pulsegen(): finalizing spiral readout core...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_seqcore, tmploc);
	SEQLENGTH(seqcore, dur_seqcore, seqcore);
	fprintf(stderr, "\tDone.\n");


	/**********************************/
	/* Generate deadtime (empty) core */
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
 *                   ASL3DFLEX.E RSPVAR SECTION                      *
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
 *                   ASL3DFLEX.E RSP SECTION                         *
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
	setfrequency((int)xmitfreq1, &rf1, 0);
	setfrequency((int)xmitfreq2, &rf2, 0);
	setfrequency((int)recfreq, &echo1, 0);

	/* Set fat sat frequency */
	setfrequency( (int)(-520 / TARDIS_FREQ_RES), &fatsatrho, 0);
	
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
/* PLAY_BLKSAT() Function for playing bulk saturation pulse */
int play_blksat(int doblksatf) {

	if (doblksatf) {
		/* Play bulk saturation pulse */	
		fprintf(stderr, "\tplay_blksat(): playing bulk saturation pulse (%d us)...\n", dur_blksatcore + TIMESSI);

		boffset(off_blksatcore);
		startseq(0, MAY_PAUSE);
		settrigger(TRIG_INTERN, 0);
	}
	else {
		play_deadtime(dur_blksatcore + TIMESSI);	
	}

	fprintf(stderr, "\tplay_blksat(): Done.\n");

	return dur_blksatcore + TIMESSI;
}

/* PLAY_ASLPREP() Function for playing asl prep pulses & delays */
int play_aslprep(int type, s32* off_ctlcore, s32* off_lblcore, int dur, int tbgs1, int tbgs2, int pld) {
	int ttotal = 0;
	int ttmp;

	/* Play the VSASL pulse */	
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

	/* Play pld and background suppression */
	if (pld > 0) {

		/* Initialize pld before subtracting out tbgs timing */
		ttmp = pld;

		if (tbgs1 > 0) {
			/* Play first background suppression delay/pulse */
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
			/* Play second background suppression delay/pulse */
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

		/* Check that ttmp is non-negative */
		if (ttmp < 0) {
			fprintf(stderr, "\tplay_aslprep(): ERROR: invalid pld and background suppression time combination\n");
			rspexit();
		}

		/* Play remaining PLD deadtime */
		fprintf(stderr, "\tplay_aslprep(): playing post-label delay (%d us), total end delay = %d us...\n", pld, ttmp);
		setperiod(ttmp - TIMESSI, &emptycore, 0);
		boffset(off_emptycore);
		startseq(0, MAY_PAUSE);
		settrigger(TRIG_INTERN, 0);
		ttotal += ttmp;
	}

	return ttotal;
}

/* PLAY_FATSAT() Function for playing fat sat pulse */
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

/* PLAY_TIP() Function for playing FSE tipdown pulse and TE deadtime */
int play_tip() {
	int ttotal = 0;
	fprintf(stderr, "\tplay_tip(): playing FSE tipdown pulse and echo time delay (%d us)...\n", dur_tipcore + TIMESSI);

	/* Play the tipcore */
	setphase(phs_tip, &rf1, 0);
	boffset(off_tipcore);
	startseq(0, MAY_PAUSE);
	settrigger(TRIG_INTERN, 0);
	ttotal += dur_tipcore + TIMESSI;

	return ttotal;	
}

/* PLAY_FLIP() Function for playing GRE tipdown pulse or the FSE refocusing pulses */
int play_flip(int flipn) {
	int ttotal = 0;
	fprintf(stderr, "\tplay_flip(): playing flipcore (%d us)...\n", dur_flipcore);

	/* Set the flip angle & phase */
	setiamp((int)(flipfactbl[flipn]*ia_rf2), &rf2, 0);
	setphase(flipphstbl[flipn], &rf2, 0);
	if (ro_mode == 1) /* FSE, sweep the phs table */
		setphase(flipphstbl[flipn] + M_PI/2, &echo1, 0);
	else /* GRE, phs_flip = phs_tip */
		setphase(flipphstbl[flipn], &echo1, 0);

	/* Play the flip core */
	boffset(off_flipcore);
	startseq(0, MAY_PAUSE);
	settrigger(TRIG_INTERN, 0);
	ttotal += dur_flipcore + TIMESSI;

	return ttotal;	
}

/* PLAY_READOUT() Function for playing the readout section */
int play_readout(int grad_off) {
	int ttotal = 0;
	fprintf(stderr, "\tplay_readout(): playing seqcore (%d us)...\n", dur_seqcore);

	if (grad_off) {
		/* Kill the gradients */
		setiamp(0, &gxw, 0);
		setiamp(0, &gyw, 0);
		setiamp(0, &gzw, 0);
	}

	/* Play the seqcore */
	boffset(off_seqcore);
	startseq(0, MAY_PAUSE);
	settrigger(TRIG_INTERN, 0);
	ttotal += dur_seqcore + TIMESSI; 

	/* Restore the gradients */
	setiamp(ia_gxw, &gxw, 0);
	setiamp(ia_gyw, &gyw, 0);
	setiamp(ia_gzw, &gzw, 0);

	fprintf(stderr, "\tplay_readout(): Done.\n");
	return ttotal;
}

/* PLAY_ENDSCAN() Function for sending endpass packet at end of sequence */
STATUS play_endscan() {
	fprintf(stderr, "\tplay_endscan(): sending endpass packet...\n");
	
	/* Send SSP packet to end scan */
	boffset( off_pass );
	setwamp(SSPD + DABPASS + DABSCAN, &endpass, 2);
	settrigger(TRIG_INTERN, 0);
	startseq(0, MAY_PAUSE);  

	fprintf(stderr, "\tplay_endscan(): Done.\n");
	return SUCCESS;
}

/* PRESCANCORE() Function for playing prescan sequence */
STATUS prescanCore() {

	/* Initialize the rotation matrix */
	setrotate( tmtx0, 0 );
	
	for (view = 1 - rspdda; view < rspvus + 1; view++) {

		if (ro_mode == 1) {
			fprintf(stderr, "prescanCore(): Playing FSE tip pulse for prescan iteration %d...\n", view);
			play_tip();
		}

		fprintf(stderr, "prescanCore(): Playing flip pulse for prescan iteration %d...\n", view);
		play_flip(0);

		/* Load the DAB */	
		if (view < 1) {
			fprintf(stderr, "prescanCore(): loaddab(&echo1, 0, 0, 0, 0, DABOFF, PSD_LOAD_DAB_ALL)...\n");
			loaddab(&echo1, 0, 0, 0, 0, DABOFF, PSD_LOAD_DAB_ALL);
		}
		else {
			fprintf(stderr, "prescanCore(): loaddab(&echo1, 0, 0, 0, %d, DABON, PSD_LOAD_DAB_ALL)...\n", view);
			loaddab(&echo1, 0, 0, 0, view, DABON, PSD_LOAD_DAB_ALL);
		}

		fprintf(stderr, "prescanCore(): playing readout for prescan iteration %d...\n", view);
		play_readout(1); /* kill the gradients */

		fprintf(stderr, "prescanCore(): playing deadtime for prescan iteration %d...\n", view);
		play_deadtime(500000);

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
	fprintf(stderr, "scan(): Beginning scan (t = %d / %.0f us)...\n", ttotal, pitscan);	
	
	/* Play an empty acquisition to reset the DAB after prescan */
	if (disdaqn == 0) {
		/* David Frey (1/22/2024):
		When disdaqs play, they reset the buffer by calling DABOFF so that prescan data can be written.
		If no disdaqs are played (i.e. MRF), then the first acquisiton may be corrupted by prescan data.
		These lines run a single empty acqusition, but will introduce a slight delay in the sequence.
		This may cause issues when running fMRI stimulus that starts at the beginning of the sequence trigger.
		*/

		/* Turn the DABOFF */
		loaddab(&echo1, 0, 0, DABSTORE, 0, DABOFF, PSD_LOAD_DAB_ALL);
		play_readout(1); /* 1 = kill the gradients */
	}

	/* Play disdaqs */
	for (disdaqn = 0; disdaqn < ndisdaqtrains; disdaqn++) {
		/* Calculate and play deadtime */
		fprintf(stderr, "scan(): Playing TR deadtime for disdaq train %d (t = %d / %.0f us)...\n", disdaqn, ttotal, pitscan);
		ttotal += play_deadtime(optr - dur_readout);
		
		if (ro_mode == 1) {
			fprintf(stderr, "scan(): Playing FSE tip pulse for disdaq train %d (t = %d / %.0f us)...\n", disdaqn, ttotal, pitscan);
			ttotal += play_tip();	
		}

		/* Loop through echoes */
		for (echon = 0; echon < opetl; echon++) {
			fprintf(stderr, "scan(): Playing flip pulse for disdaq train %d (t = %d / %.0f us)...\n", disdaqn, ttotal, pitscan);
			ttotal += play_flip(0);

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
			ttotal += play_readout(1); /* 1 = kill the gradients */
		}
	}


	/* Loop through frames and shots */
	for (framen = 0; framen < nframes; framen++) {
		for (shotn = 0; shotn < opnshots; shotn++) {

			/* Play the Bulk saturation pulse for background suppression */
			if (doblksat) {
				fprintf(stderr, "scan(): Playing bulk saturation pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
				ttotal += play_blksat(doblksattbl[framen]);
			}
			/* delay 1 */
			fprintf(stderr, "scan(): Playing TR deadtime for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
			ttotal += play_deadtime(tadjusttbl[framen]);		

			if (prep1_id > 0 ) {
				fprintf(stderr, "scan(): Playing prep1 pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
				ttotal += play_aslprep(prep1_lbltbl[framen], off_prep1ctlcore, off_prep1lblcore, dur_prep1core, prep1_tbgs1tbl[framen], prep1_tbgs2tbl[framen], prep1_pldtbl[framen]);
			}
			
			if (prep2_id > 0) {
				fprintf(stderr, "scan(): Playing prep2 pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
				ttotal += play_aslprep(prep2_lbltbl[framen], off_prep2ctlcore, off_prep2lblcore, dur_prep2core, prep2_tbgs1tbl[framen], prep2_tbgs2tbl[framen], prep2_pldtbl[framen]);
			}

			/* fat sat pulse */
			if (dofatsat) {
				fprintf(stderr, "scan(): Playing fat sat pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
				ttotal += play_fatsat();
			}

			/* FSE:  tipdown */
			if (ro_mode == 1) {
				fprintf(stderr, "scan(): Playing FSE tip pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
				ttotal += play_tip();	
			}

			/* play disdaq echoes */
			for (echon = 0; echon < ndisdaqechoes; echon++) {
				fprintf(stderr, "scan(): Playing flip pulse for frame %d, shot %d, disdaq echo %d (t = %d / %.0f us)...\n", framen, shotn, echon, ttotal, pitscan);
				ttotal += play_flip(echon);
				
				fprintf(stderr, "scan(): playing deadtime in place of readout for frame %d, shot %d, disdaq echo %d (%d us)...\n", framen, shotn, echon, dur_seqcore);
				ttotal += play_deadtime(dur_seqcore);
			};

			for (echon = 0; echon < opetl; echon++) {
				fprintf(stderr, "scan(): Playing flip pulse for frame %d, shot %d, echo %d (t = %d / %.0f us)...\n", framen, shotn, echon, ttotal, pitscan);
				ttotal += play_flip(echon);
		
				/* Load the DAB */
				slice = framen+1;
				view = 	shotn*opetl + echon + 1;
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
				rotidx = framen*opnshots*opetl + shotn*opetl + echon;
				setrotate( tmtxtbl[rotidx], 0 );

				fprintf(stderr, "scan(): playing readout for frame %d, shot %d, echo %d (%d us)...\n", framen, shotn, echon, dur_seqcore);
				ttotal += play_readout(kill_grads);

				/* Reset the rotation matrix */
				setrotate( tmtx0, 0 );
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
int genspiral(FILE* fID_partitions) {

	/* 4 parts of the gradient waveform:
	 * 	- Z-encode (*_ze):		kZ encoding step
	 *	- Spiral (*_sp):		kXY-plane spiral trajectory (readout)
	 *	- Ramp-down (*_rd):		XY gradient ramp down
	 *	- kspace Rewinder (*_kr):	kspace rewinder (XYZ trapezoidal gradient)
	 */

	/* gradient waveforms will be in units of rad*G/ms */

	FILE* fID_ktraj = fopen("ktraj.txt", "w");
	FILE* fID_grad = fopen("grad.txt", "w");
	FILE* fID_graddac = fopen("graddac.txt", "w");

	/* declare variables */
	int n;
	float t;
	int np, np_ze, np_sp, np_rd, np_kr;
	int tmp_pwa, tmp_pw, tmp_pwd;
	float Tr_ze, Tp_ze, T_sp, T_rd, Tr_kr, Tp_kr;
	float t1, t2, t3;
	float gxn, gyn, gzn, kxn, kyn, kzn;
	float gx0, gy0, g0, kx0, ky0, kz0, k0;
	float h_ze, h_kr;
	float F[3];

	/* declare waveforms */
	float *gx_sp, *gy_sp;
	float gx[MAXWAVELEN], gy[MAXWAVELEN], gz[MAXWAVELEN];

	/* convert units */
	float dt = GRAD_UPDATE_TIME*1e-6; /* raster time (s) */
	float D = (float)opfov / 10.0; /* fov (cm) */
	float gm = GMAX; /* gradient amplitude limit (G/cm) */
	float sm = SLEWMAX; /* slew limit (G/cm/s) */
	float gam = 4258; /* gyromagnetic ratio (Hz/G) */
	float kxymax = opxres / D / 2.0; /* kspace xy sampling radius (cm^-1) */
	float kzmax = (sptype3d == 1) ? (kz_acc * opetl / D / 2.0) : (kxymax);

	/* generate the z encoding trapezoid gradient */
	amppwgrad(kzmax/gam*1e6, gm, 0, 0, ZGRAD_risetime, 0, &h_ze, &tmp_pwa, &tmp_pw, &tmp_pwd);
	Tr_ze = (tmp_pwa + tmp_pwd) / 2.0 * 1e-6;
	Tp_ze = tmp_pw * 1e-6;
	np_ze = round((2*Tr_ze + Tp_ze) / dt);

	/* generate the spiral trajectory */
	F[0] = F0;
	F[1] = F1;
	F[2] = F2;
	calc_vds(sm, gm, dt, dt, (sptype2d<3) ? (1) : (2), F, 2, kxymax, MAXWAVELEN, &gx_sp, &gy_sp, &np_sp);
	T_sp = dt * np_sp;

	/* calculate gradients at end of spiral */
	gx0 = gx_sp[np_sp - 1];
	gy0 = gy_sp[np_sp - 1];
	g0 = sqrt(pow(gx0,2) + pow(gy0,2));

	/* calculate gradient ramp down time and round up to nearest sampling interval */
	T_rd = g0 / sm;
	T_rd = dt * ceil(T_rd / dt);
	np_rd = round(T_rd/dt);

	/* calculate gradients at end of ramp down */
	kx0 = gam * dt * fsumarr(gx_sp, np_sp - 1) + gam * 1/2 * (T_rd + dt) * gx0;
	ky0 = gam * dt * fsumarr(gy_sp, np_sp - 1) + gam * 1/2 * (T_rd + dt) * gy0;
	kz0 = kzmax;
	k0 = sqrt(pow(kx0,2) + pow(ky0,2) + pow(kz0,2));

	/* generate the kspace rewinder */
	amppwgrad(k0/gam*1e6, gm, 0, 0, ZGRAD_risetime, 0, &h_kr, &tmp_pwa, &tmp_pw, &tmp_pwd);
	Tr_kr = (tmp_pwa + tmp_pwd) / 2.0 * 1e-6;
	Tp_kr = tmp_pw * 1e-6;
	np_kr = round((2*Tr_kr + Tp_kr) / dt);

	/* calculate time markers */
	t1 = 2*Tr_ze + Tp_ze;
	t2 = t1 + T_sp;
	t3 = t2 + T_rd;

	/* calculate total number of points */
	np = np_ze + np_sp + np_rd + np_kr + 1;

	/* loop through time points */
	memset(gx, 0, sizeof gx);
	memset(gy, 0, sizeof gx);
	memset(gz, 0, sizeof gx);
	for (n = 0; n < np; n++) {
		t = dt * n;

		if (t <= t1) { /* Z-encode gradient */
			gxn = 0;
			gyn = 0;
			gzn = h_ze * trap(t, 0, Tr_ze, Tp_ze);
		}

		else if (t <= t2) { /* Spiral trajectory */
			gxn = gx_sp[n - np_ze - 1];
			gyn = gy_sp[n - np_ze - 1];
			gzn = 0;
		}

		else if (t <= t3) { /* Gradient ramp-down */
			gxn = gx0 * (1 - (t - t2) / T_rd);
			gyn = gy0 * (1 - (t - t2) / T_rd);
			gzn = 0;
		}
		else { /* Kspace rewinder */
			gxn = -kx0/k0 * h_kr * trap(t, t3, Tr_kr, Tp_kr);
			gyn = -ky0/k0 * h_kr * trap(t, t3, Tr_kr, Tp_kr);
			gzn = -kz0/k0 * h_kr * trap(t, t3, Tr_kr, Tp_kr);
		}


		switch (sptype2d) {
			case 1: /* spiral out */
				gx[n + nnav] = gxn;
				gy[n + nnav] = gyn;
				gz[n + nnav] = gzn;
				break;
			case 2: /* spiral in */
				gx[np - 1 - n + nnav] = gxn;
				gy[np - 1 - n +  nnav] = gyn;
				gz[np - 1 - n + nnav] = gzn;
				break;
			case 3: /* spiral out-in */
				gx[n] = gxn;
				gy[n] = gyn;
				gz[n] = gzn;
				gx[2*np + nnav - 1 - n] = gxn;
				gy[2*np + nnav - 1 - n] = gyn;
				gz[2*np + nnav - 1 - n] = -gzn;
				break;
			case 4: /* spiral in-out */
				gx[np - 1 - n] = gxn;
				gy[np - 1 - n] = gyn;
				gz[np - 1 - n] = -gzn;
				gx[np + nnav + n] = gxn;
				gy[np + nnav + n] = gyn;
				gz[np + nnav + n] = gzn;
				break;
			default:
				return 0;
				break;
		}
	}

	/* calculate total number of points */
	if (sptype2d > 2)
		grad_len = 2*np + nnav;
	else
		grad_len = np + nnav;

	/* calculate kspace location, fs gradients, and write to file */
	kxn = 0.0;
	kyn = 0.0;
	kzn = 0.0;
	for (n = 0; n < grad_len; n++) {
		kxn += gam * gx[n] * dt;
		kyn += gam * gy[n] * dt;
		kzn += gam * gz[n] * dt;

		Gx[n] = 2*round(MAX_PG_WAMP/XGRAD_max * gx[n] / 2.0);
		Gy[n] = 2*round(MAX_PG_WAMP/YGRAD_max * gy[n] / 2.0);
		Gz[n] = 2*round(MAX_PG_WAMP/ZGRAD_max * gz[n] / 2.0);
		
		fprintf(fID_ktraj, "%f \t%f \t%f\n", kxn, kyn, kzn);
		fprintf(fID_grad, "%f \t%f \t%f\n", gx[n], gy[n], gz[n]);
		fprintf(fID_graddac, "%d \t%d \t%d\n", Gx[n], Gy[n], Gz[n]);
	}

	fclose(fID_ktraj);
	fclose(fID_grad);
	fclose(fID_graddac);

	return 1;
}

int genviews(FILE* fID_partitions) {

	/* Declare values and matrices */
	FILE* fID_kviews = fopen("kviews.txt","w");
	char buff[200];
	int framen, shotn, echon, n;
	float rz1, rx, ry, rz2, dz;
	float Rz1[9], Rx[9], Ry[9], Rz2[9], Tz[9];
	float T_0[9], T[9];

	/* Initialize z translation to identity matrix */
	eye(Tz, 3);

        /* Get original transformation matrix */
        for (n = 0; n < 9; n++) T_0[n] = (float)rsprot[0][n] / MAX_PG_WAMP;
        orthonormalize(T_0, 3, 3);

	/* Loop through all views */
	for (framen = 0; framen < nframes; framen++) {
		for (shotn = 0; shotn < opnshots; shotn++) {
			for (echon = 0; echon < opetl; echon++) {

				if (fID_partitions == 0) { /* generate partitions */

					/* Initialize rotation angles/kz shift */
					rz1 = 2.0*M_PI * (float)shotn / (float)opnshots;
					rx = 0.0;
					ry = 0.0;
					rz2 = 0.0;
					dz = 0.0;

					/* Determine type of transformation */
					switch (sptype3d) {
						case 0 : /* 2D - shot by shot rotations only */
							break;
						case 1 : /* Kz shifts */
							dz = 2.0/(float)(opetl) * pow(-1.0,echon)*floor((float)(echon + 1)/2.0);
							break;
						case 2 : /* Single axis rotation */
							rx = 2.0*M_PI * echon / PHI;
							break;
						case 3 : /* Double axis rotations */
							rx = acos(fmod(echon*phi1, 1.0)); /* polar angle */
							rz2 = 2.0*M_PI * fmod(echon*phi2, 1.0); /* azimuthal angle */
							break;		
						case 4: /* Debugging case */
							rx = M_PI/2.0*echon;
							ry = M_PI/4.0*shotn;	
							break;
						default:
							return 0;
					}

				}
				else {
					fprintf(stderr, "genviews(): reading in rotation angles and kz fraction from file for frame %d, shot %d, echo %d\n", framen, shotn, echon);

					/* Loop through points in theta file */
					fgets(buff, 200, fID_partitions);
					sscanf(buff, "%f %f %f %f %f", &rz1, &rx, &ry, &rz2, &dz);
				}			

				/* Calculate the transformation matrices */
				Tz[8] = dz;
				genrotmat('z', rz1, Rz1);
				genrotmat('x', rx, Rx);
				genrotmat('y', ry, Ry);
				genrotmat('z', rz2, Rz2);

				/* Multiply the transformation matrices */
				multmat(3,3,3,T_0,Tz,T);
				multmat(3,3,3,Rz1,T,T);
				multmat(3,3,3,Rx,T,T);
				multmat(3,3,3,Ry,T,T);
				multmat(3,3,3,Rz2,T,T);

				/* Save the matrix to the table of matrices */
				fprintf(fID_kviews, "%d \t%d \t%d \t%f \t%f \t%f \t%f \t%f \t", framen, shotn, echon, rz1, rx, ry, rz2, dz);
				for (n = 0; n < 9; n++) {
					fprintf(fID_kviews, "%f \t", T[n]);
					tmtxtbl[framen*opnshots*opetl + shotn*opetl + echon][n] = (long)round(MAX_PG_WAMP*T[n]);
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
		grad_ctl[i] = (int)ctlval*(!zero_CTL_grads);
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
/* This function adds a linear phase shift to the velocity selective pulses
 in order to shift the velocity selectivity profile to a different velocity */
int calc_prep_phs_from_velocity (
	int* vsi_pulse_mag, 
	int* vsi_pulse_phs, 
	int* vsi_pulse_grad, 
	float vel_target, 
	int vsi_train_len, 
	double vsi_Gmax)
{
	/* GAMMA_H1 26754 in (rad/s)/Gauss */
	double phase_val;
	double grad_val;
	double pos=0.0;
	double dt = 4e-6;
	double delta_phs = 0.0;
	int 	i;
	int	DACMAX = 32766;
	int	tmp;
	double  pulseMax = 0.0;


	/* find the segments with 180 degree pulses- assume the 180s have the highest B1 in the train */
	for (i=0; i<vsi_train_len; i++)
	{
		if (pulseMax < vsi_pulse_mag[i] ) 
			pulseMax = vsi_pulse_phs[i] ;

	}


	for (i=1; i<vsi_train_len; i++)
	{
		/* from DAC units to radians */
		phase_val = M_PI * (double)(vsi_pulse_phs[i]) /  (double)FS_PI  ; 		
		/* from DAC units to G/cm */
		grad_val = vsi_Gmax * (double)(vsi_pulse_grad[i]) / (double)DACMAX ;

		/* calc the phase gained by moving spins during THIS dt interval */
		pos += vel_target*dt; 
		delta_phs += GAMMA * grad_val * pos * dt ;

		/* change the phase of the pulse accordingly */
		phase_val -=  delta_phs;

		/* from radians to DAC ... unwrap first  , then make them even numbers only. */	
		phase_val = atan2( sin(phase_val), cos(phase_val));
		tmp = (int)(phase_val / M_PI * FS_PI);
		vsi_pulse_phs[i] = 2*(tmp/2);
	}

	for (i=0; i<vsi_train_len; i++)
	{
		if (vsi_pulse_mag[i] == 0) 
			vsi_pulse_phs[i] = 0;

	}

	return 1;
}	


int readschedule(int id, int* var, char* varname, int lines) {

	FILE* fID;
	char fname[200];
	int val;
	
	if (id == 0)
		return 0;

	/* Open the schedule file */
	sprintf(fname, "./aslprep/schedules/%05d/%s.txt", id, varname);
	fprintf(stderr, "readschedule(): opening %s...\n", fname);
	fID = fopen(fname, "r");
	if (fID == 0) {
		fprintf(stderr, "File not found.\n");
		return 0;
	}

	/* Read in the array */
	int i = 0;
	while (fscanf(fID, "%d\n", &val) != EOF) {
		var[i] = val;
		i++;
	}
	fclose(fID);

	/* If only 1 line is read in (scalar --> array) */
	if (i == 1) {
		for (i = 1; i < lines; i++)
			var[i] = var[0];
	}

	/* If number of lines is less than number of lines to read */
	if (i < lines - 1)
		return -1;

	return 1;
}

int readschedulef(int id, float* var, char* varname, int lines) {

	FILE* fID;
	char fname[200];
	float val;
	
	if (id == 0)
		return 0;

	/* Open the schedule file */
	sprintf(fname, "./aslprep/schedules/%05d/%s.txt", id, varname);
	fprintf(stderr, "readschedule(): opening %s...\n", fname);
	fID = fopen(fname, "r");
	if (fID == 0) {
		fprintf(stderr, "File not found.\n");
		return 0;
	}

	/* Read in the array */
	int i = 0;
	while (fscanf(fID, "%f\n", &val) != EOF) {
		var[i] = val;
		i++;
	}
	fclose(fID);

	/* If only 1 line is read in (scalar --> array) */
	if (i == 1) {
		for (i = 1; i < lines; i++)
			var[i] = var[0];
	}

	/* If number of lines is less than number of lines to read */
	if (i < lines - 1)
		return -1;

	return 1;
}

int genlbltbl(int mod, int* lbltbl)
{
	int framen;

	/* Loop through frames */
	for (framen = 0; framen < nframes; framen++) {

		/* Set labeling scheme */
		if (framen < nm0frames){
			lbltbl[framen] = -1;
		}
		else {
			switch (mod) {
				case 1: /* Label, control... */
					lbltbl[framen] = (framen - nm0frames + 1) % 2; /* 1, 0, 1, 0 */
					break;
				case 2: /* Control, label... */
					lbltbl[framen] = (framen - nm0frames) % 2; /* 0, 1, 0, 1 */
					break;
				case 3: /* Label */
					lbltbl[framen] = 1;
					break;
				case 4: /* Control */
					lbltbl[framen] = 0;
					break;
			}
		}
	}	

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


/************************ END OF ASL3DFLEX.E ******************************/
