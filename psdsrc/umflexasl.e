/*
 * File Name : umflexasl.e
 * Language  : EPIC/ANSI C
 * Date      : july 31, 2024 
 *
 * this is velocity spectrum imaging branch
 *
 * Based on sequence by 
 * David Frey
 * University of Michigan Medicine Department of Radiology
 * Functional MRI Laboratory, PI: Luis Hernandez-Garcia
 *
 * GE Medical Systems
 * Copyright (C) 1996-2003 The General Electric Company
 *
 *
 * a velocity selective ASL-prepped turboFLASH sequence (umvsasl),
 * built on grass.e
 * also rotating spirals and FSE readout functionality
 *  
 * LHG: aug 4, 2024 : adding PCASL pulses to the sequence and other functionality (umflexasl)
 * 
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

#include "umflexasl.h"

/* Define important values */
#define MAXWAVELEN 50000 /* Maximum wave length for gradients */
#define MAXNSHOTS 48 /* Maximum number of echo trains per frame */
#define MAXNECHOES 64 /* Maximum number of echoes per echo train */
#define MAXNFRAMES 500 /* Maximum number of temporal frames */
#define MAXITR 50 /* Maximum number of iterations for iterative processes */
#define GAMMA 26754 /* Gyromagnetic ratio (rad/s/G) */
#define TIMESSI 120 /* SSP instruction time */
#define SPOIL_SEED 21001 /* rf spoiler seed */
#define MAXPCASLSEGMENTS 5000 /* maximum n. of iterations that can fit in a PCASL label block */

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
long tmtxtbl[MAXNSHOTS*MAXNECHOES*MAXNFRAMES][9];
float Tex[MAXNSHOTS*MAXNECHOES*MAXNFRAMES][9];

/* rotation matrices for the VS pulses*/	
float R0[9];
float R[9]; 
float Rp[9];
long rotmat0[9];
long prep_rotmat[9];

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

/* allocate space for PCASL pulse train phases */
int pcasl_iphase_tbl[MAXPCASLSEGMENTS];

/* Allocate space and intialize table with values as if it were not MRF */
float mrf_deadtime[MAXNFRAMES];
float mrf_prep1_pld[MAXNFRAMES];
float mrf_prep2_pld[MAXNFRAMES];
float mrf_pcasl_pld[MAXNFRAMES];
float mrf_pcasl_duration[MAXNFRAMES];
int mrf_prep1_type[MAXNFRAMES];
int mrf_prep2_type[MAXNFRAMES];
int mrf_pcasl_type[MAXNFRAMES];

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

float SLEWMAX = 12000.0 with {1000, 25000.0, 12500.0, VIS, "maximum allowed slew rate (G/cm/s)",};
float GMAX = 4.0 with {0.5, 5.0, 4.0, VIS, "maximum allowed gradient (G/cm)",};

/* readout cvs */
int nframes = 2 with {1, , 2, VIS, "number of frames",};
int ndisdaqtrains = 2 with {0, , 2, VIS, "number of disdaq echo trains at beginning of scan loop",};
int ndisdaqechoes = 0 with {0, , 0, VIS, "number of disdaq echos at beginning of echo train",};

int varflip = 1 with {0,1,1, VIS, "do variable flip angles (FSE case)- make sure opflip=180 for this", };
float arf180, arf180ns; 

int ro_type = 2 with {1, 3, 2, VIS, "FSE (1), SPGR (2), or bSSFP (3)",};
float SE_factor = 1.5 with {0.01, 10.0 , 1.5, VIS, "Adjustment for the slice width of the refocuser",};
int	doNonSelRefocus = 1 with {0, 1, 0, VIS, "Use a RECT non-selective refocuser pulse",};

int fatsup_mode = 1 with {0, 3, 1, VIS, "none (0), CHESS (1), or SPIR (2)",};
int fatsup_off = -520 with { , , -520, VIS, "fat suppression pulse frequency offset (Hz)",};
int fatsup_bw = 440 with { , , 440, VIS, "fat suppression bandwidth (Hz)",};
float spir_fa = 110 with {0, 360, 1, VIS, "SPIR pulse flip angle (deg)",};
int spir_ti = 52ms with {0, 1000ms, 52ms, VIS, "SPIR inversion time (us)",};
int rfspoil_flag = 1 with {0, 1, 1, VIS, "option to do RF phase cycling (117deg increments to rf1 phase)",};
int flowcomp_flag = 0 with {0, 1, 0, VIS, "option to use flow-compensated slice select gradients",};
int rf1_b1calib = 0 with {0, 1, 0, VIS, "option to sweep B1 amplitudes across frames from 0 to nominal B1 for rf1 pulse",};

int pgbuffertime = 248 with {0, , 248, VIS, "gradient IPG buffer time (us)",}; /* used to be 248 */
int pcasl_buffertime = 0 with {0, , 248, VIS, "PCASL core - gradient IPG buffer time (us)",}; /* used to be 100 */
float crushfac = 2.0 with {0, 10, 0, VIS, "crusher amplitude factor (a.k.a. cycles of phase/vox; dk_crush = crushfac*kmax)",};
int kill_grads = 0 with {0, 1, 0, VIS, "option to turn off readout gradients",};

/* Trajectory cvs */
int nnav = 250 with {0, 1000, 250, VIS, "number of navigator points in spiral",};
int narms = 1 with {1, 1000, 1, VIS, "number of spiral arms - in SOS, this is interleaves/shots",};
int spi_mode = 2 with {0, 4, 0, VIS, "SOS (0), TGA (1), 3DTGA (2) rotmats from File (3) traj AND rotmats from file (4)",};
int grad_id = 13; /* file ID for arbitrary gradients and rotation matrices*/
float grad_scale = 1.0;  /* scaling factor for the arbitrary gradients - for safety and troubleshooting*/
float kz_acc = 1.0 with {1, 100.0, 1.0, VIS, "kz acceleration (SENSE) factor (for SOS only)",};
float vds_acc0 = 1.0 with {0.001, 50.0, 1.0, VIS, "spiral center oversampling factor",};
float vds_acc1 = 1.0 with {0.001, 50.0, 1.0, VIS, "spiral edge oversampling factor",};
float F0 = 0 with { , , 0, INVIS, "vds fov coefficient 0",};
float F1 = 0 with { , , 0, INVIS, "vds fov coefficient 1",};
float F2 = 0 with { , , 0, INVIS, "vds fov coefficient 2",};

/* ASL prep pulse cvs */
int presat_flag = 0 with {0, 1, 0, VIS, "option to play asl pre-saturation pulse at beginning of each tr",};
int presat_delay = 1000000 with {0, , 1000000, VIS, "ASL pre-saturation delay (us)",};
int nm0frames = 0 with {0, , 2, VIS, "Number of M0 frames (no prep pulses are played)",};

int zero_ctl_grads = 0 with {0, 1, 0, VIS, "option to zero out control gradients for asl prep pulses",};

int prep1_id = 0 with {0, , 0, VIS, "ASL prep pulse 1: ID number (0 = no pulse)",};
int prep1_pld = 0 with {0, , 0, VIS, "ASL prep pulse 1: post-labeling delay (us; includes background suppression)",};
float prep1_rfmax = 234 with {0, , 0, VIS, "ASL prep pulse 1: maximum RF amplitude",};
float prep1_gmax = 1.5 with {-4.0, 4.0 , 1.5, VIS, "ASL prep pulse 1: maximum gradient amplitude",};
int prep1_mod = 1 with {1, 4, 1, VIS, "ASL prep pulse 1: labeling modulation scheme (1 = label/control, 2 = control/label, 3 = always label, 4 = always control)",};
int prep1_tbgs1 = 0 with {0, , 0, VIS, "ASL prep pulse 1: 1st background suppression delay (0 = no pulse)",};
int prep1_tbgs2 = 0 with {0, , 0, VIS, "ASL prep pulse 1: 2nd background suppression delay (0 = no pulse)",};
int prep1_tbgs3 = 0 with {0, , 0, VIS, "ASL prep pulse 1: 3rd background suppression delay (0 = no pulse)",};
int prep1_b1calib = 0 with {0, 1, 0, VIS, "ASL prep pulse 1: option to sweep B1 amplitudes across frames from 0 to nominal B1",};

int prep2_id = 0 with {0, , 0, VIS, "ASL prep pulse 2: ID number (0 = no pulse)",};
int prep2_pld = 0 with {0, , 0, VIS, "ASL prep pulse 2: post-labeling delay (us; includes background suppression)",};
float prep2_rfmax = 234 with {0, , 0, VIS, "ASL prep pulse 2: maximum RF amplitude",};
float prep2_gmax = 1.5 with {0, , 1.5, VIS, "ASL prep pulse 2: maximum gradient amplitude",};
int prep2_mod = 1 with {1, 4, 1, VIS, "ASL prep pulse 2: labeling modulation scheme (1 = label/control, 2 = control/label, 3 = always label, 4 = always control)",};
int prep2_tbgs1 = 0 with {0, , 0, VIS, "ASL prep pulse 2: 1st background suppression delay (0 = no pulse)",};
int prep2_tbgs2 = 0 with {0, , 0, VIS, "ASL prep pulse 2: 2nd background suppression delay (0 = no pulse)",};
int prep2_tbgs3 = 0 with {0, , 0, VIS, "ASL prep pulse 2: 3rd background suppression delay (0 = no pulse)",};
int prep2_b1calib = 0 with {0, 1, 0, VIS, "ASL prep pulse 2: option to sweep B1 amplitudes across frames from 0 to nominal B1",};
int prep2_Npoints = 0 ;

/* Velocity Selective prep pulse axis*/
int prep_axis = 0;

/* cardiac trigger option */
int do_cardiac_gating = 0 with {0, 1, 0, VIS, "Flag to control cardiac gating. (1) Do Cardiac Gating. (0) Don't do Cardiac Gating"};

/* Declare PCASL variables (cv)*/
int pcasl_pld 	= 1500*1e3 with {0, , 0, VIS, "PCASL prep  : (ms) post-labeling delay ( includes background suppression)",};
int pcasl_duration = 1500*1e3 with {0, , 0, VIS, "PCASL prep  : (ms) Labeling Duration)",}; /* this is the bolus duration, NOT the duration of a single cycle */
int pcasl_mod 	= 1 with {1, 4, 1, VIS, "PCASL prep  : labeling modulation scheme (1 = label/control, 2 = control/label, 3 = always label, 4 = always control)",};
int pcasl_tbgs1 = 0 with {0, , 0, VIS, "PCASL prep  : (us) 1st background suppression delay (0 = no pulse)",};
int pcasl_tbgs2 = 0 with {0, , 0, VIS, "PCASL prep  : (us) 2nd background suppression delay (0 = no pulse)",};

int		pcasl_flag = 0; /* when this is turned to 1, prep1 pulse gets replaced with a PCASL pulse*/

int		pcasl_calib = 0 with {0, 1 , 0, VIS, "do PCASL phase calibration?",};
int		pcasl_calib_frames = 2 with {0, , 100, VIS, "N. frames per phase increment",};
int		pcasl_calib_cnt = 0;
float 	phs_cal_step = 0.0;

int		pcasl_period = 1200; /* (us) duration of one of the PCASL 'units' 
					(here it was 1500 originally))	
					Li Zhao paper used 1200 - I really want it to shrink it to 1000*/ 
int 	pcasl_Npulses = 1700;
int 	pcasl_RFamp_dac = 0;
float 	pcasl_RFamp = 20;   /*mGauss-  Li Zhao paper used 18 mGauss-is an ~8 deg flip for a 0.5ms hanning
								Jahanian used ~80 mG (~35 deg.) */
float 	pcasl_delta_phs = 0;
float 	pcasl_delta_phs_correction = 0;  /* this is about typical */
int		pcasl_delta_phs_dac = 0;
int 	pcasl_RFdur = 500us;
float 	pcasl_Gamp =  0.6;  /* slice select lobe for PCASL RF pulse G/cm .... Lizhao paper: 0.35
								Jahanian used 0.6*/
float	pcasl_Gave = 0.06;  /* average gradient for each pulse in the train.  LiZhao paper: 0.05 
								Jahanian used 0.039- eASL uses 0.7 / 0.07 */
float	pcasl_Gref_amp;     /* refocuser gradient */
int		pcasl_tramp =  120us; 
int		pcasl_tramp2 =  120us; 
float	pcasl_distance = 12.0; /*cm - distance from the iso-center */
float	pcasl_distance_adjust; /*cm - distance from the iso-center after table movement */
float	pcasl_RFfreq;

/* adding velocity selectivity shift to the VSI pulses (optional)*/
int		doVelSpectrum = 0 with {0,3,0, VIS, "Velocity Spectrum imaging: nothing(0) FTVS velocity target sweep (1) BIR8 velocity encoding gradient(2) PIR8, positive only (3)"};  /* sweep velocity target in FTVS pulses (change the phase)*/
float	vel_target = 0.0 /* use for FTVS case */;
float   vspectrum_grad = -4.0;  /* use this for BIR8 */
int		vspectrum_Navgs = 2;
float	vel_target_incr = 1.0;
float   prep2_delta_gmax = 0.1 ;
int	min_dur_pcaslcore = 0;
int	zero_CTL_grads = 0; /* option to use zero gradients for the control pulses */

/* Declare core duration variables */
int dur_presatcore = 0 with {0, , 0, INVIS, "duration of the ASL pre-saturation core (us)",};
int dur_pcaslcore = 0 with {0, , 0, INVIS, "Duration of the PCASL core (us)",};
int dur_prep1core = 0 with {0, , 0, INVIS, "duration of the ASL prep 1 cores (us)",};
int dur_prep2core = 0 with {0, , 0, INVIS, "duration of the ASL prep 2 cores (us)",};
int dur_bkgsupcore = 0 with {0, , 0, INVIS, "duration of the background suppression core (us)",};
int dur_fatsupcore = 0 with {0, , 0, INVIS, "duration of the fat suppression core (us)",};
int dur_rf0core = 0 with {0, , 0, INVIS, "duration of the slice selective rf0 core (us)",};
int dur_rf1core = 0 with {0, , 0, INVIS, "duration of the slice selective rf1 core (us)",};
int dur_rf1nscore = 0 with {0, , 0, INVIS, "duration of the NON-slice selective rf1 core (us)",};
int dur_seqcore = 0 with {0, , 0, INVIS, "duration of the spiral readout core (us)",};
int deadtime_pcaslcore = 0 with {0, , 0, INVIS, "deadtime at end of pcasl core (us)",};
int deadtime_fatsupcore = 0 with {0, , 0, INVIS, "deadtime at end of fatsup core (us)",};
int deadtime_rf0core = 0 with {0, , 0, INVIS, "post-tipdown deadtime for FSE (us)",};
int deadtime1_seqcore = 0 with {0, , 0, INVIS, "pre-readout deadtime within core (us)",};
int deadtime2_seqcore = 0 with {0, , 0, INVIS, "post-readout deadtime within core (us)",};
int tr_deadtime = 0 with {0, , 0, INVIS, "TR deadtime (us)",};

/* inhereted from grass.e, not sure if it's okay to delete: */
float xmtaddScan;
int obl_debug = 0 with {0, 1, 0, INVIS, "On(=1) to print messages for obloptimize",};
int obl_method = 0 with {0, 1, 0, INVIS, "On(=1) to optimize the targets based on actual rotation matrices",};
int debug = 0 with {0,1,0,INVIS,"1 if debug is on ",};
float echo1bw = 16 with {,,,INVIS,"Echo1 filter bw.in KHz",};

/*MRF mode features*/
int mrf_mode = 0 with {0, 2, 0, VIS, "MRF mode. (0)=none, (1)= update ASL timings + rotations every frame, (2)=updates rotations only",};
int mrf_sched_id = 1;
float prev_theta = 0.0;  /* rotation angles from last frame */
float prev_phi = 0.0;    /* rotation angles from last frame */



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
abstract("umflexasl sequence");
psdname("umflexasl");

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
float phi2D = (1.0 + sqrt(5.0)) / 2.0; /* 2d golden ratio */
float phi3D_1 = 0.4656; /* 3d golden ratio 1 */
float phi3D_2 = 0.6823; /* 3d golden ratio 2 */

/*---- This is where we put function prototype declarations for the HOST,  not the IPG ------*/

/* Declare trajectory generation function prototypes */
int genspiral();
int genviews();

/* Declare function prototypes from aslprep.h */
int readprep(int id, int *len,
		int *rho_lbl, int *theta_lbl, int *grad_lbl,
		int *rho_ctl, int *theta_ctl, int *grad_ctl); 
float calc_sinc_B1(float cyc_rf, int pw_rf, float flip_rf);
float calc_hard_B1(int pw_rf, float flip_rf);
int write_scan_info();

/* function to fetch radout gradients from file */
int readGrads(int id, float scale);

/* currently unusued */
int make_pcasl_schedule(
		int *pcasl_lbltbl, 
		int *pcasl_pldtbl, 
		int *pcasl_tbgs1tbl, 
		int *pcasl_tbgs2tbl);

int calc_pcasl_phases(
		int *iphase_tbl, 
		float  myphase_increment, 
		int nreps);

/* declare MRF related functions*/
int init_mrf(
	float 	*mrf_deadtime, 
	int 	*mrf_pcasl_type,
	float 	*mrf_pcasl_duration,
	float 	*mrf_pcasl_pld,
	int 	*mrf_prep1_type,
	float 	*mrf_prep1_pld, 
	int 	*mrf_prep2_type,
	float 	*mrf_prep2_pld);

int read_mrf_fromfile(
	int 	file_id, 
	float 	*mrf_deadtime, 
	int 	*mrf_pcasl_type,
	float 	*mrf_pcasl_duration,
	float 	*mrf_pcasl_pld,
	int 	*mrf_prep1_type,
	float 	*mrf_prep1_pld, 
	int 	*mrf_prep2_type_pld,
	float 	*mrf_prep2_pld);

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
	ZGRAD_risetime *= 1.2; /* slowing the gradient down a little to avoid PNS */
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
	piuset2 = 0;	
	/* Add opuser fields to the Adv. pulse sequence parameters interface */	
	/* weird EPIC note: in epic_ui_contrl.h, you can see that you can have use0-use30 
	and add them into piuset.  You can also have use31-use48, but then you add them to piuset2.
	Notice that the actual valuse of use0-use17 are the same as the ones in use31-use48 */
 
	piuset += use0;
	cvdesc(opuser0, "Readout type: (1) FSE, (2) SPGR, (3) bSSFP");
	cvdef(opuser0, ro_type);
	opuser0 = ro_type;
	cvmin(opuser0, 1);
	cvmax(opuser0, 3);	
	ro_type = opuser0;

	if (ro_type > 1) /* Not applicable for FSE */
		piuset += use1;
	cvdesc(opuser1, "ESP (short TR) (ms)");
	cvdef(opuser1, esp*1e-3);
	opuser1 = esp*1e-3;
	cvmin(opuser1, 0);
	cvmax(opuser1, 1000);	
	esp = 4*round(opuser1*1e3/4);
	

	piuset += use3;
	cvdesc(opuser3, "Number of frames");
	cvdef(opuser3, nframes);
	opuser3 = nframes;
	cvmin(opuser3, 1);
	cvmax(opuser3, MAXNFRAMES);
	nframes = opuser3;
	
	piuset += use4;
	cvdesc(opuser4, "Number of spiral arms");
	cvdef(opuser4, narms);
	opuser4 = narms;
	cvmin(opuser4, 1);
	cvmax(opuser4, 1000);
	narms = opuser4;
	
	piuset += use5;
	cvdesc(opuser5, "Number of disdaq echo trains");
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
	
	piuset += use9;
	cvdesc(opuser9, "FID mode (no spiral): (0) off, (1) on");
	cvdef(opuser9, 0);
	opuser9 = kill_grads;
	cvmin(opuser9, 0);
	cvmax(opuser9, 1);	
	kill_grads = opuser9;

	if (kill_grads == 0) /* only if spirals are on */
		piuset += use10;
	cvdesc(opuser10, "SPI mode: (0) SOS, (1) 2DTGA, (2) 3DTGA, (3)use myrotmats.txt file (4) use both myrotmats and grad files");
	cvdef(opuser10, 2);
	opuser10 = spi_mode;
	cvmin(opuser10, 0);
	cvmax(opuser10, 3);	
	spi_mode = opuser10;
	
	if (kill_grads == 0 && spi_mode == 0) /* only if spirals are on & SOS*/
		piuset += use11;
	cvdesc(opuser11, "kz acceleration (SENSE) factor");
	cvdef(opuser11, 0);
	opuser11 = kz_acc;
	cvmin(opuser11, 0);
	cvmax(opuser11, 10);	
	kz_acc = opuser11;
	
	if (kill_grads == 0) /* only if spirals are on*/
		piuset += use12;
	cvdesc(opuser12, "VDS center acceleration factor");
	cvdef(opuser12, 0);
	opuser12 = vds_acc0;
	cvmin(opuser12, 0);
	cvmax(opuser12, 100);	
	vds_acc0 = opuser12;
	
	if (kill_grads == 0) /* only if spirals are on*/
		piuset += use13;
	cvdesc(opuser13, "VDS edge acceleration factor");
	cvdef(opuser13, 0);
	opuser13 = vds_acc1;
	cvmin(opuser13, 0);
	cvmax(opuser13, 100);	
	vds_acc1 = opuser13;
	
	if (kill_grads == 0) /* only if spirals are on*/
		piuset += use14;
	cvdesc(opuser14, "Number of navigator points");
	cvdef(opuser14, 0);
	opuser14 = nnav;
	cvmin(opuser14, 0);
	cvmax(opuser14, 2000);	
	nnav = opuser14;
	
	piuset += use15;
	cvdesc(opuser15, "Fat supppresion mode: (0) off, (1) CHESS, (2) SPIR");
	cvdef(opuser15, 0);
	opuser15 = fatsup_mode;
	cvmin(opuser15, 0);
	cvmax(opuser15, 2);	
	fatsup_mode = opuser15;
	
	piuset += use16;
	cvdesc(opuser16, "Num. M0 frames (no label or BGS)");
	cvdef(opuser16, 0);
	opuser16= nm0frames;
	cvmin(opuser16, 0);
	cvmax(opuser16, 500);	
	nm0frames = opuser16;

	piuset += use17;
	cvdesc(opuser17, "Use presaturation pulse? (1) Yes, (0) No ");
	cvdef(opuser17, 0);
	opuser17= presat_flag; 
	cvmin(opuser17, 0);
	cvmax(opuser17, 1);
	presat_flag = opuser17;

	piuset += use18;
	cvdesc(opuser18, "Do Cardiac Gating ? (1) Yes, (0) No");
	cvdef(opuser18, 0);
	opuser18 = do_cardiac_gating; 
	cvmin(opuser18, 0); 
	cvmax(opuser18, 1);
	do_cardiac_gating = opuser18;
	
	/* GUI variables for  PCASL pulses */
	
	piuset += use19;
	cvdesc(opuser19, "use PCASL? (0=off)");
	cvdef(opuser19, 0);
	opuser19 = pcasl_flag;
	cvmin(opuser19, 0);
	cvmax(opuser19, 1);	
	pcasl_flag = opuser19;

	if (pcasl_flag > 0)
		piuset += use20;
	cvdesc(opuser20, "Labeling plane distance to center of Rx (mm)");
	cvdef(opuser20, 0);
	opuser20 = pcasl_distance*10;
	cvmin(opuser20, 0);
	cvmax(opuser20, 500);	
	pcasl_distance = opuser20/10.0;  /* cm */

	if (pcasl_flag > 0)
		piuset += use21;
	cvdesc(opuser21, "PCASL PLD (ms)");
	cvdef(opuser21, 0);
	opuser21 = pcasl_pld/1e3;
	cvmin(opuser21, 0);
	cvmax(opuser21, 99999);	
	pcasl_pld = 4*round(opuser21*1e3/4);
	
	if (pcasl_flag > 0)
		piuset += use22;
	cvdesc(opuser22, "PCASL labeling duration (ms)");
	cvdef(opuser22, 1);
	opuser22 = pcasl_duration/1e3;
	cvmin(opuser22, 0);
	cvmax(opuser22, 5000);	
	pcasl_duration = 4*round(opuser22*1e3/4);

	/* GUI variables for prep1 pulse*/

	piuset += use23;
	cvdesc(opuser23, "Prep 1 pulse id (0=off)");
	cvdef(opuser23, 0);
	opuser23 = prep1_id;
	cvmin(opuser23, 0);
	cvmax(opuser23, 99999);	
	prep1_id = opuser23;
		
	if (prep1_id > 0)
		piuset += use24;
	cvdesc(opuser24, "Prep 1 PLD (ms)");
	cvdef(opuser24, 0);
	opuser24 = prep1_pld*1e-3;
	cvmin(opuser24, 0);
	cvmax(opuser24, 99999);	
	prep1_pld = 4*round(opuser24*1e3/4);
	
	if (prep1_id > 0)
		piuset += use25;
	cvdesc(opuser25, "Prep 1 max B1 amp (mG)");
	cvdef(opuser25, 0);
	opuser25 = prep1_rfmax;
	cvmin(opuser25, 0);
	cvmax(opuser25, 500);	
	prep1_rfmax = opuser25;
	
	if (prep1_id > 0)
		piuset += use26;
	cvdesc(opuser26, "Prep 1 max G amp (G/cm)");
	cvdef(opuser26, 0);
	opuser26 = prep1_gmax;
	cvmin(opuser26, -GMAX);
	cvmax(opuser26, GMAX);	
	prep1_gmax = opuser26;
	
	/* GUI variables for prep pulse 2 */
	piuset += use27;
	cvdesc(opuser27, "Prep 2 pulse id (0=off)");
	cvdef(opuser27, 0);
	opuser27 = prep2_id;
	cvmin(opuser27, 0);
	cvmax(opuser27, 99999);	
	prep2_id = opuser27;
		
	if (prep2_id > 0)
		piuset += use28;
	cvdesc(opuser28, "Prep 2 PLD (ms)");
	cvdef(opuser28, 0);
	opuser28 = prep2_pld*1e-3;
	cvmin(opuser28, 0);
	cvmax(opuser28, 99999);	
	prep2_pld = 4*round(opuser28*1e3/4);
	
	if (prep2_id > 0)
		piuset += use29;
	cvdesc(opuser29, "Prep 2 max B1 amp (mG)");
	cvdef(opuser29, 0);
	opuser29 = prep2_rfmax;
	cvmin(opuser29, 0);
	cvmax(opuser29, 500);	
	prep2_rfmax = opuser29;
	
	if (prep2_id > 0)
		piuset += use30;
	cvdesc(opuser30, "Prep 2 max G amp (G/cm)");
	cvdef(opuser30, 0);
	opuser30 = prep2_gmax;
	cvmin(opuser30, 0);
	cvmax(opuser30, GMAX);	
	prep2_gmax = opuser30;
	
	/* Let's not use these ... GE reserved opuser CVs start at opuser36 */

	/*
	if (fatsup_mode == 2) // only for SPIR
		piuset += use16;
	cvdesc(opuser16, "SPIR flip angle (deg)");
	cvdef(opuser16, 0);
	opuser16 = spir_fa;
	cvmin(opuser16, 0);
	cvmax(opuser16, 360);	
	spir_fa = opuser16;
	
	if (fatsup_mode == 2) // only for SPIR 
		piuset += use17;
	cvdesc(opuser17, "SPIR inversion time (ms)");
	cvdef(opuser17, 0);
	opuser17 = spir_ti*1e-3;
	cvmin(opuser17, 0);
	cvmax(opuser17, 5000);	
	spir_ti = 4*round(opuser17*1e3/4);

	if (prep2_id > 0)
		piuset += use30;
	cvdesc(opuser30, "Prep 2 mod pattern: (1) LC, (2) CL, (3) L, (4), C");
	cvdef(opuser30, 1);
	opuser30 = prep2_mod;
	cvmin(opuser30, 1);
	cvmax(opuser30, 4);	
	prep2_mod = opuser30;
	
	if (prep2_id > 0)
		piuset2 += use31;
	cvdesc(opuser31, "Prep 2 BGS 1 delay (0=off) (ms)");
	cvdef(opuser31, 0);
	opuser31 = prep2_tbgs1*1e-3;
	cvmin(opuser31, 0);
	cvmax(opuser31, 20000);	
	prep2_tbgs1 = 4*round(opuser31*1e3/4);
	
	if (prep2_id > 0)
		piuset2 += use32;
	cvdesc(opuser32, "Prep 2 BGS 2 delay (0=off) (ms)");
	cvdef(opuser32, 0);
	opuser32 = prep2_tbgs2*1e-3;
	cvmin(opuser32, 0);
	cvmax(opuser32, 20000);	
	prep2_tbgs2 = 4*round(opuser32*1e3/4);
	
	if (prep2_id > 0)
		piuset2 += use33;
	cvdesc(opuser33, "Prep 2 BGS 3 delay (0=off) (ms)");
	cvdef(opuser33, 0);
	opuser33 = prep2_tbgs3*1e-3;
	cvmin(opuser33, 0);
	cvmax(opuser33, 20000);	
	prep2_tbgs3= 4*round(opuser33*1e3/4);

		
	
	if (pcasl_flag > 0)
		piuset2 += use37;
	cvdesc(opuser37, "PCASL BGS 1 delay (0=off) (ms)");
	cvdef(opuser37, 0);
	opuser37 = pcasl_tbgs1*1e-3;
	cvmin(opuser37, 0);
	cvmax(opuser37, 20000);	
	pcasl_tbgs1 = 4*round(opuser37*1e3/4);
	
	if (pcasl_flag > 0)
		piuset2 += use38;
	cvdesc(opuser38, "PCASL BGS 2 delay (0=off) (ms)");
	cvdef(opuser38, 0);
	opuser38 = pcasl_tbgs2*1e-3;
	cvmin(opuser38, 0);
	cvmax(opuser38, 20000);	
	pcasl_tbgs2 = 4*round(opuser38*1e3/4);
	
	if (pcasl_flag > 0)
		piuset2 += use39;
	cvdesc(opuser39, "Do PCASL phase calibration? (0=No)");
	cvdef(opuser39, 0);
	opuser39 = pcasl_calib;
	cvmin(opuser39, 0);
	cvmax(opuser39, 1);	
	pcasl_calib = opuser39;
	
	if (pcasl_flag > 0 && pcasl_calib >0)
		piuset2 += use40;
	cvdesc(opuser40, "Num. calibration reps per calibration phase increment");
	cvdef(opuser40, 0);
	opuser40 = pcasl_calib_frames;
	cvmin(opuser40, 0);
	cvmax(opuser40, 100);	
	pcasl_calib_frames = opuser40;
	

	if (ro_type == 2) // SPGR only 
		piuset += use2;
	cvdesc(opuser2, "RF spoiling: (0) off, (1) on");
	cvdef(opuser2, 0);
	opuser2 = rfspoil_flag;
	cvmin(opuser2, 0);
	cvmax(opuser2, 1);	
	rfspoil_flag = opuser2;

	piuset += use7;
	cvdesc(opuser7, "Crusher area factor (% kmax)");
	cvdef(opuser7, crushfac);
	opuser7 = crushfac;
	cvmin(opuser7, 0);
	cvmax(opuser7, 10);
	crushfac = opuser7;
	
	piuset += use8;
	cvdesc(opuser8, "Flow comp: (0) off, (1) on");
	cvdef(opuser8, 0);
	opuser8 = flowcomp_flag;
	cvmin(opuser8, 0);
	cvmax(opuser8, 1);	
	flowcomp_flag = opuser8;
	----------------*/




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
	int echo1_freq[opslquant], rf1_freq[opslquant];
	int slice;
	float kzmax;
	int minesp, minte, absmintr;	
	float rf0_b1, rf1_b1, rf1ns_b1;
	float rf180_b1, rf180ns_b1;
	float rfps1_b1, rfps2_b1, rfps3_b1, rfps4_b1;
	float rffs_b1, rfbs_b1;
	float prep1_b1, prep2_b1;
	int tmp_pwa, tmp_pw, tmp_pwd;
	float tmp_a, tmp_area;
	float mom1;
	int ctr = 0;
	int i;

	/*********************************************************************/
#include "predownload.in"	/* include 'canned' predownload code */
	/*********************************************************************/

	if (ro_type != 1){
		SE_factor=1.0;
		doNonSelRefocus = 0;
	}
	fprintf(stderr, "\npredownload(): SE_factor (SE refocuse sl. thick factor)  %f\n", SE_factor);

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
	prep2_Npoints = prep2_len;
	
	/* update presat pulse parameters */
	pw_rfps1 = 1ms; /* 1ms hard pulse */
	pw_rfps1a = 0; /* hard pulse - no ramp */
	pw_rfps1d = 0;
	pw_rfps2 = 1ms; /* 1ms hard pulse */
	pw_rfps2a = 0; /* hard pulse - no ramp */
	pw_rfps2d = 0;
	pw_rfps3 = 1ms; /* 1ms hard pulse */
	pw_rfps3a = 0; /* hard pulse - no ramp */
	pw_rfps3d = 0;
	pw_rfps4 = 1ms; /* 1ms hard pulse */
	pw_rfps4a = 0; /* hard pulse - no ramp */
	pw_rfps4d = 0;
	
	/* update sinc pulse parameters */
	pw_rf0 = 3200;
	pw_rf1 = 3200;
	pw_rf1ns = 1000;

	/* adjust fat sup pw s.t. desired bandwidth is achieved */
	pw_rffs = 3200; /* nominal SINC1 pulse width */
	pw_rffs *= (int)round(NOM_BW_SINC1_90 / (float)fatsup_bw); /* adjust bandwidth */

	/* Update the background suppression pulse parameters */
	res_rfbs_rho = 500;
	pw_rfbs_rho = 5000;
	a_rfbs_theta = 1.0;
	res_rfbs_theta = res_rfbs_rho;
	pw_rfbs_theta = pw_rfbs_rho;
		
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
	
	rf0_b1 = calc_sinc_B1(cyc_rf0, pw_rf0, 90.0);
	fprintf(stderr, "predownload(): maximum B1 for rf0 pulse: %f\n", rf0_b1);
	if (rf0_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = rf0_b1;

	rf180_b1 = calc_sinc_B1(cyc_rf1, pw_rf1, 180);
	if (ro_type==1){
		fprintf(stderr, "predownload(): maximum B1 for a 180 deg pulse: %f\n", rf180_b1);
		if (rf180_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = rf180_b1;
	}

	rf180ns_b1 = calc_hard_B1(pw_rf1ns, 180);
	if (ro_type==1){
		fprintf(stderr, "predownload(): maximum B1 for a 180 deg RECT pulse: %f\n", rf180ns_b1);
		if (rf180ns_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = rf180ns_b1;
	}

	rf1_b1 = calc_sinc_B1(cyc_rf1, pw_rf1, opflip);
	fprintf(stderr, "predownload(): maximum B1 for rf1 pulse: %f\n", rf1_b1);
	if (rf1_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = rf1_b1;

	rf1ns_b1 = calc_hard_B1(pw_rf1ns, opflip);
	fprintf(stderr, "predownload(): maximum B1 for rf1ns pulse: %f\n", rf1ns_b1);
	if (rf1ns_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = rf1ns_b1;

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
	
	if (fatsup_mode < 2)
		rffs_b1 = calc_sinc_B1(cyc_rffs, pw_rffs, 90.0);
	else
		rffs_b1 = calc_sinc_B1(cyc_rffs, pw_rffs, spir_fa);
	fprintf(stderr, "predownload(): maximum B1 for fatsup pulse: %f Gauss\n", rffs_b1);
	if (rffs_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = rffs_b1;
	
	prep1_b1 = (prep1_id > 0) ? (prep1_rfmax*1e-3) : (0);
	fprintf(stderr, "predownload(): maximum B1 for prep1 pulse: %f Gauss\n", prep1_b1);
	if (prep1_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = prep1_b1;

	prep2_b1 = (prep2_id > 0) ? (prep2_rfmax*1e-3) : (0);
	fprintf(stderr, "predownload(): maximum B1 for prep2 pulse: %f Gauss\n", prep2_b1);
	if (prep2_b1 > maxB1[L_SCAN]) maxB1[L_SCAN] = prep2_b1;
	
	/* Determine peak B1 across all entry points */
	maxB1Seq = 0.0;
	for (entry=0; entry < MAX_ENTRY_POINTS; entry++) {
		if (entry != L_SCAN) { /* since we aleady computed the peak B1 for L_SCAN entry point */
			if (peakB1(&maxB1[entry], entry, RF_FREE, rfpulse) == FAILURE) {
				epic_error(use_ermes,"peakB1 failed",EM_PSD_SUPPORT_FAILURE,1,STRING_ARG,"peakB1");
				return FAILURE;
			}
		}
		if (maxB1[entry] > maxB1Seq)
			maxB1Seq = maxB1[entry];
	}
	fprintf(stderr, "predownload(): maxB1Seq = %f Gauss\n", maxB1Seq);
	
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
	
	/* Update all the rf amplitudes : convert RF amplitude from gauss to fraction of b1max*/
	a_rf0 = rf0_b1 / maxB1Seq;
	ia_rf0 = a_rf0 * MAX_PG_WAMP;

	/* (this one is for the variable flip angle calculation) */
	arf180 = rf180_b1 / maxB1Seq;	
	arf180ns = rf180ns_b1 / maxB1Seq;	

	a_rf1 = rf1_b1 / maxB1Seq;
	ia_rf1 = a_rf1 * MAX_PG_WAMP;

	a_rf1ns = rf1ns_b1 / maxB1Seq;
	ia_rf1ns = a_rf1ns * MAX_PG_WAMP;


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
	
	if (doVelSpectrum==2){
		fprintf(stderr, "predownload(): doVelspectrum=2 -> setting the first prep1_gmax to -4.0\n");
		vspectrum_grad = -4.0;
	}
	if (doVelSpectrum==3){
		fprintf(stderr, "predownload(): doVelspectrum=3 -> setting the first prep1_gmax to 0.0\n");
		vspectrum_grad = 0.0;
	}
	
	/* Update the asl prep pulse gradients */
	a_prep1gradlbl = (prep1_id > 0) ? (prep1_gmax) : (0);
	ia_prep1gradlbl = (int)ceil(a_prep1gradlbl / ZGRAD_max * (float)MAX_PG_WAMP);
	a_prep1gradctl = (prep1_id > 0) ? (prep1_gmax) : (0); 
	ia_prep1gradctl = (int)ceil(a_prep1gradctl / ZGRAD_max * (float)MAX_PG_WAMP);
	a_prep2gradlbl = (prep2_id > 0) ? (prep2_gmax) : (0);
	ia_prep2gradlbl = (int)ceil(a_prep2gradlbl / ZGRAD_max * (float)MAX_PG_WAMP);
	a_prep2gradctl = (prep2_id > 0) ? (prep2_gmax) : (0); 
	ia_prep2gradctl = (int)ceil(a_prep2gradctl / ZGRAD_max * (float)MAX_PG_WAMP);

	/* ---------------------------------------------*/
	/* update the PCASL train specific calculations  */
	/* ---------------------------------------------*/
	fprintf(stderr, "\npredownload(): calculations for PCASL pulses ");
	fprintf(stderr,"\n---------------------------------------------\n");

	for(ctr=0; ctr<MAXPCASLSEGMENTS; ctr++) pcasl_iphase_tbl[ctr]=0;

	/* Making sure that everything fits in the PCASL train */
	pcasl_period = 4*round(pcasl_period/4); /* includes TIMESSI*/
	pcasl_Npulses = round(pcasl_duration/pcasl_period);
	pcasl_duration = pcasl_period * pcasl_Npulses;

	fprintf(stderr, "predownload(): pcasl_period : %d \n", pcasl_period);  
	fprintf(stderr, "predownload(): pcasl_Npulses : %d \n", pcasl_Npulses);  
	fprintf(stderr, "predownload(): pcasl_duration : %d \n", pcasl_duration);  
	fprintf(stderr, "predownload(): deadtime_pcaslcore: %d \n", deadtime_pcaslcore);  
	fprintf(stderr, "predownload(): dur_pcaslcore: %d \n", deadtime_pcaslcore);  


	/* Trapezoid durations */
	pw_gzpcasl = pcasl_RFdur;
	pw_gzpcasla = pcasl_tramp;
	pw_gzpcasld = pcasl_tramp;

	pw_gzpcaslref =  0;
	pw_gzpcaslrefa = pcasl_tramp2  ;
	pw_gzpcaslrefd = pcasl_tramp2 ;

	/* Check that things fit in the pcasl core */
	min_dur_pcaslcore =  
                 pw_gzpcasl + pw_gzpcasla + pw_gzpcasld 
                 + pw_gzpcaslref+ pw_gzpcaslrefa + pw_gzpcaslrefd
                 + 3*pcasl_buffertime 
				 + TIMESSI;

	deadtime_pcaslcore = pcasl_period  - min_dur_pcaslcore; 
	dur_pcaslcore = min_dur_pcaslcore + deadtime_pcaslcore;
	fprintf(stderr, "predownload(): min_dur_pcaslcore: %d \n", min_dur_pcaslcore);  
	fprintf(stderr, "predownload(): dur_pcaslcore: %d \n", dur_pcaslcore);  

	if (deadtime_pcaslcore<0){
		/* recalculate PCASL timings without any deadtime in the pcaslcore*/
		deadtime_pcaslcore = 0;
		dur_pcaslcore = min_dur_pcaslcore;
		pcasl_period = dur_pcaslcore;
		pcasl_Npulses = (int)floor(pcasl_duration/pcasl_period);
		pcasl_duration = pcasl_period * pcasl_Npulses;

		fprintf(stderr, "\npredownload(): _WARNING_: pcasl_period too short ... ");  
		fprintf(stderr, "\n changing pcasl_period : %d \n", pcasl_period);  
		fprintf(stderr, "\n changing pcasl_Npulses : %d \n", pcasl_Npulses);  
		fprintf(stderr, "\n changing pcasl_duration : %d \n", pcasl_duration);  
		fprintf(stderr, "\n changing  dur_pcaslcore: %d \n", dur_pcaslcore);  
		fprintf(stderr, "\n changing  deadtime_pcaslcore: %d \n", deadtime_pcaslcore);  
	}	
	 
	fprintf(stderr, "\npredownload(): PCASL SS gradient: %f ", pcasl_Gamp);
	fprintf(stderr, "\npredownload(): pcasl_distance: %f", pcasl_distance);
	/* account for the table position */
	pcasl_distance_adjust = pcasl_distance - (float)piscancenter/10.0;
	fprintf(stderr, "\npredownload(): piscancenter: %f", piscancenter);
	fprintf(stderr, "\npredownload(): pcasl_distance (adjusted) : %f", pcasl_distance_adjust);

	pcasl_RFfreq = -pcasl_distance_adjust * GAMMA/2/M_PI * pcasl_Gamp ; /* Hz */
	/* DEBUGGING:   is the piscancenter not right???? 
	if we use the pcasl_distance_adjust here, it should also be used for 
	the pcasl_delta_phs calculation!  */
	fprintf(stderr, "\npredownload(): PCASL RF frequency offset: %f", pcasl_RFfreq);

	/* The gradient moment from the first trapezoid */
	mom1 = pcasl_Gamp*(pcasl_RFdur + pcasl_tramp) ;
	/* Gradient moment of refocuser is calculated to maintain a specific gradient moment
	for the pcasl module.  This moment is expressed as an average gradient : pcasl_Gave 
	mom1 and mom2 are the gradient moments (areas) of each of the lobes in the pcasl module
	most papers specify G_ss and G_ave, so we have to calculate the amplitude of the G_ref
	(units of mom1 and mom2:  G/cm * us)
  
		pcasl_Gave =  (mom1 + mom2)/pcasl_period );
		mom2 = pcasl_Gref_amp * (pw_gzpcaslref + pw_gzpcaslrefa) 

	now solve for pcasl_Gref_amp ...       */
	pcasl_Gref_amp = (pcasl_Gave * pcasl_period - mom1) / (pw_gzpcaslref + pw_gzpcaslrefa);

	/* Change the values of the gradient amplitudes */
	a_gzpcasl = pcasl_Gamp;
	a_gzpcaslref = pcasl_Gref_amp;

	fprintf(stderr, "\npredownload(): PCASL refocuser gradient : %f ", pcasl_Gref_amp);
	fprintf(stderr, "\npredownload(): PCASL refocuser gradient ramps: %d ", pw_gzpcaslrefa);

	/* calculate linear phase increment for PCASL train:
	units: 
		GAMMA (rad/s/G) , mom (G/cm*us) , pcasl_distance_adjust (cm) , 
		pcasl_period (us), pcasl_Gave (G/cm)*/
	pcasl_delta_phs = -GAMMA* pcasl_Gave * pcasl_period * pcasl_distance_adjust * 1e-6;
		
	fprintf(stderr, "\npredownload(): PCASL linear phase increment: %f radians", pcasl_delta_phs);

	/* scale the  PCASL amplitude of RF pulses for the RHO channel... also in DAC units*/
	a_rfpcasl = pcasl_RFamp * 1e-3 / maxB1Seq;
	pcasl_RFamp_dac = (int)(a_rfpcasl* MAX_PG_WAMP);
	ia_rfpcasl = pcasl_RFamp_dac;

	fprintf(stderr, "\npredownload(): maxB1Seq : %f Gauss",maxB1Seq  );
	fprintf(stderr, "\npredownload(): PCASL RF amplitude : %f mGauss, a_rfpcasl: %f ",pcasl_RFamp, a_rfpcasl  );
	fprintf(stderr, "\npredownload(): PCASL RF amplitude in DAC untis: %d \n", pcasl_RFamp_dac);

	/* Make adjustments to pcasl before getting into the pre-scan loop */
	/* add a manual correction for off-resonance (or whatever)
	   and calculate phase table for the PCASL train */
	pcasl_delta_phs = pcasl_delta_phs + pcasl_delta_phs_correction;
	pcasl_delta_phs = atan2( sin(pcasl_delta_phs), cos(pcasl_delta_phs));
	fprintf(stderr, "\npredownload(): CORRECTED PCASL linear phase increment: %f (rads)", pcasl_delta_phs);
	fprintf(stderr, "\npredownload(): calling calc_pcasl_phases() to make the linear phase sweep \n");
	calc_pcasl_phases(pcasl_iphase_tbl, pcasl_delta_phs, MAXPCASLSEGMENTS);

	if (doVelSpectrum==2){
		/* velocity spectrum default venc gradient increments - so that it goes from -4.0 to +4.0 G/cm */
		prep2_delta_gmax = 2.0*4.0 / ((float)nframes/(float)vspectrum_Navgs -1)  ;
	}
	if (doVelSpectrum==3){
		/* velocity spectrum default venc gradient increments - so that it goes from 0.0 to +4.0 G/cm */
		prep2_delta_gmax = 4.0 / ((float)nframes/(float)vspectrum_Navgs -1)  ;
	}

	/* -------------------------*/

	/* read MRF schedule from file */
	if (mrf_mode==1 && doVelSpectrum<1) 
	{
		if (read_mrf_fromfile(mrf_sched_id, mrf_deadtime, mrf_pcasl_type, mrf_pcasl_duration, mrf_pcasl_pld, mrf_prep1_type, mrf_prep1_pld, mrf_prep2_type, mrf_prep2_pld) ==0 )
		{
			epic_error(use_ermes,"Failure to read MRF schedule from file",EM_PSD_SUPPORT_FAILURE,1,STRING_ARG,"read_mrf_fromfile");
			return FAILURE;
		}
		else fprintf(stderr, "\nSuccess reading schedule %05d", mrf_sched_id);
	}
	/* -------------------------*/
	
	/* Set the parameters for the spin echo rf1 kspace rewinder */
	tmp_area = a_gzrf1 * (pw_gzrf1 + (pw_gzrf1a + pw_gzrf1d)/2.0);
	amppwgrad(tmp_area, GMAX, 0, 0, ZGRAD_risetime, 0, &tmp_a, &tmp_pwa, &tmp_pw, &tmp_pwd); 
	tmp_a *= -0.5;
	pw_gzrf1trap2 = tmp_pw;
	pw_gzrf1trap2a = tmp_pwa;
	pw_gzrf1trap2d = tmp_pwd;
	a_gzrf1trap2 = tmp_a;

	/* Set the parameters for the crusher gradients */
	tmp_area = crushfac * 2*M_PI/GAMMA * opxres/(opfov/10.0) * 1e6; /* Area under crusher s.t. dk = crushfac*kmax (G/cm*us) */
	amppwgrad(tmp_area, GMAX, 0, 0, ZGRAD_risetime, 0, &tmp_a, &tmp_pwa, &tmp_pw, &tmp_pwd); 	
	
	pw_rfps1c = tmp_pw;
	pw_rfps1ca = tmp_pwa;
	pw_rfps1cd = tmp_pwd;
	a_rfps1c = tmp_a;
	pw_rfps2c = tmp_pw;
	pw_rfps2ca = tmp_pwa;
	pw_rfps2cd = tmp_pwd;
	a_rfps2c = tmp_a;
	pw_rfps3c = tmp_pw;
	pw_rfps3ca = tmp_pwa;
	pw_rfps3cd = tmp_pwd;
	a_rfps3c = tmp_a;
	pw_rfps4c = tmp_pw;
	pw_rfps4ca = tmp_pwa;
	pw_rfps4cd = tmp_pwd;
	a_rfps4c = tmp_a;

	/* fat sat spoiler*/
	pw_gzrffsspoil = tmp_pw + 2000; /*Making sure that the transverse magnetization is spoiled and we don't get stimulated echoes from the prep pulse*/
	pw_gzrffsspoila = tmp_pwa ;
	pw_gzrffsspoild = tmp_pwd;
	a_gzrffsspoil = tmp_a ;

	pw_gzrf1trap1 = tmp_pw;
	pw_gzrf1trap1a = tmp_pwa;
	pw_gzrf1trap1d = tmp_pwd;
	a_gzrf1trap1 = tmp_a;

	/* set trap2 as a crusher (for FSE case) */
	pw_gzrf1trap2 = tmp_pw;
	pw_gzrf1trap2a = tmp_pwa;
	pw_gzrf1trap2d = tmp_pwd;
	a_gzrf1trap2 = tmp_a;
	
	/* Now for the rect non sel refocuser*/
	pw_rf1trap1ns = tmp_pw;
	pw_rf1trap1nsa = tmp_pwa;
	pw_rf1trap1nsd = tmp_pwd;
	a_rf1trap1ns = tmp_a;

	/* set trap2 as a crusher (for FSE case) */
	pw_rf1trap2ns = tmp_pw;
	pw_rf1trap2nsa = tmp_pwa;
	pw_rf1trap2nsd = tmp_pwd;
	a_rf1trap2ns = tmp_a;
	
	/* calculate slice select refocuser gradient */
	tmp_area = a_gzrf1 * (pw_gzrf1 + (pw_gzrf1a + pw_gzrf1d)/2.0);
	amppwgrad(tmp_area, GMAX, 0, 0, ZGRAD_risetime, 0, &tmp_a, &tmp_pwa, &tmp_pw, &tmp_pwd); 	
	tmp_a *= -0.5;
	
	pw_gzrf0r = tmp_pw;
	pw_gzrf0ra = tmp_pwa;
	pw_gzrf0rd = tmp_pwd;
	a_gzrf0r = tmp_a;
	
	if (ro_type > 1) { /* GRE modes - make trap2 a slice select refocuser */
		pw_gzrf1trap2 = tmp_pw;
		pw_gzrf1trap2a = tmp_pwa;
		pw_gzrf1trap2d = tmp_pwd;
		a_gzrf1trap2 = tmp_a;
	}

	/* set parameters for flow compensated kz-encode (pre-scaled to kzmax) */
	kzmax = (float)(kz_acc * opetl * opnshots) / ((float)opfov/10.0) / 2.0;
	tmp_area = 2*M_PI/(GAMMA*1e-6) * kzmax * (1 + flowcomp_flag); /* multiply by 2 if flow compensated */
	amppwgrad(tmp_area, GMAX, 0, 0, ZGRAD_risetime, 0, &tmp_a, &tmp_pwa, &tmp_pw, &tmp_pwd); 	
	pw_gzw1 = tmp_pw;
	pw_gzw1a = tmp_pwa;
	pw_gzw1d = tmp_pwd;
	a_gzw1 = tmp_a;
	
	/* set parameters with the kz-rewinder */
	tmp_area = 2*M_PI/(GAMMA*1e-6) * kzmax;
	amppwgrad(tmp_area, GMAX, 0, 0, ZGRAD_risetime, 0, &tmp_a, &tmp_pwa, &tmp_pw, &tmp_pwd); 	
	pw_gzw2 = tmp_pw;
	pw_gzw2a = tmp_pwa;
	pw_gzw2d = tmp_pwd;
	a_gzw2 = -tmp_a;

	/* set parameters for flowcomp pre-phaser */
	tmp_area = 2*M_PI/(GAMMA*1e-6) * kzmax; /* multiply by 2 if flow compensated */
	amppwgrad(tmp_area, GMAX, 0, 0, ZGRAD_risetime, 0, &tmp_a, &tmp_pwa, &tmp_pw, &tmp_pwd); 
	pw_gzfc = tmp_pw;
	pw_gzfca = tmp_pwa;
	pw_gzfcd = tmp_pwd;
	a_gzfc = -tmp_a;
	
	/* generate initial spiral trajectory */
	if (spi_mode==4){
		fprintf(stderr, "predownload(): reading external spiral gradients...\n");
		if (readGrads(grad_id, grad_scale)==0){
			epic_error(use_ermes,"failure to read external spiral waveform", EM_PSD_SUPPORT_FAILURE, EE_ARGS(0));
			return FAILURE;	
		}
	}
	else{
		fprintf(stderr, "predownload(): calculating spiral gradients...\n");
		if (genspiral() == 0) {
			epic_error(use_ermes,"failure to generate spiral waveform", EM_PSD_SUPPORT_FAILURE, EE_ARGS(0));
			return FAILURE;
		}
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
	fprintf(stderr, "predownload(): calculating views (ie-transformation matrices)...\n");
	if (genviews() == 0) {
		epic_error(use_ermes,"failure to generate view transformation matrices", EM_PSD_SUPPORT_FAILURE, EE_ARGS(0));
		return FAILURE;
	}
	if (mrf_mode>0){

		scalerotmats(tmtxtbl, &loggrd, &phygrd, opetl*opnshots*narms*nframes, 0);
	}else{
		scalerotmats(tmtxtbl, &loggrd, &phygrd, opetl*opnshots*narms, 0);
	}

	/* uncomment this for debugging rotmats ...					
	fprintf(stderr,"predownload(): mrf mode: rotations after scalerotmats.. %d frames\n", nframes);
	for (i=0; i<opetl*opnshots*narms*nframes; i++ ){
		for (ctr=0;ctr<9;ctr++){
			fprintf(stderr,"\t%ld", tmtxtbl[i][ctr] );
		}
		fprintf(stderr,"\n");
	}
	*/

	/* Generate prep pulse axis transformation matrix*/
	fprintf(stderr, "predownload(): calculating VSASL axis rotation matrices...\n");
	
    for (i = 0; i < 9; i++) 
		R0[i] = (float)rsprot[0][i] / MAX_PG_WAMP;
    
	orthonormalize(R0, 3, 3);

	/* figure out the  right transformation matrix to alter the axis - this core only*/
	eye(R,3); 
	switch(prep_axis){
		case 0:
			eye(R,3);
			break;
		case 1:
 			/* PI/2 rotation of  <0,0,1> about the y axis should land you on the x-axis*/
			genrotmat('y', M_PI/2.0, R); 
			break;
		case 2:
			genrotmat('x', M_PI/2.0, R);
			break;
	}	

	/* multiply original matrix by rotation*/			
	multmat(3,3,3,R, R0, Rp); /*  Rp = R * R_0 */
	/* convert Rp to long and scale the new rotation matrix*/
	for (i=0; i<9; i++){
		prep_rotmat[i] = (long)round(MAX_PG_WAMP * Rp[i]);
		fprintf(stderr, "\t%0.2f ", Rp[i]);
	}
	scalerotmats(&prep_rotmat, &loggrd, &phygrd, 0, 0);
	fprintf(stderr, "done\n");


	/* calculate minimum echo time and esp, and corresponding deadtimes */
	minesp = 0;
	minte = 0;
	switch (ro_type) {
		case 1: /* FSE */
			
			/* calculate minimum esp (time from rf1 to next rf1) */
			if (doNonSelRefocus)
				minesp += pw_rf1ns/2;
			else
				minesp += pw_gzrf1/2 + pw_gzrf1d; /* 2nd half of rf1 pulse */
			minesp += pgbuffertime;
			minesp += pw_gzrf1trap2a + pw_gzrf1trap2 + pw_gzrf1trap2d; /* post-rf crusher */
			minesp += pgbuffertime;
			minesp += TIMESSI; /* inter-core time */
			minesp += (flowcomp_flag == 1 && spi_mode == 0)*(pw_gzfca + pw_gzfc + pw_gzfcd + pgbuffertime); /* flow comp pre-phaser */
			minesp += pgbuffertime;
			minesp += (spi_mode == 0) * (pw_gzw1a + pw_gzw1 + pw_gzw1d + pgbuffertime); /* z encode gradient */
			minesp += pw_gxw; /* spiral readout */
			minesp += pgbuffertime;
			minesp += (spi_mode == 0) * (pw_gzw2a + pw_gzw2 + pw_gzw2d + pgbuffertime); /* z rewind gradient */
			minesp += (flowcomp_flag == 1 && spi_mode == 0)*(pw_gzfca + pw_gzfc + pw_gzfcd + pgbuffertime); /* for symmetry - add length of fc pre-phaser */
			minesp += TIMESSI; /* inter-core time */
			minesp += pgbuffertime;
			minesp += pw_gzrf1trap1a + pw_gzrf1trap1 + pw_gzrf1trap1d; /* pre-rf crusher */
			minesp += pgbuffertime;
			if (doNonSelRefocus)
				minesp += pw_rf1ns/2;
			else
				minesp += pw_gzrf1a + pw_gzrf1/2; /* 1st half of rf1 pulse */

			/* calculate minimum TE (time from center of rf0 to center of readout pulse) */
			minte += pw_gzrf0/2 + pw_gzrf0d; /* 2nd half of rf0 pulse */
			minte += pgbuffertime;
			minte += pw_gzrf0ra + pw_gzrf0r + pw_gzrf0rd; /* rf0 slice select rewinder */
			minte += pgbuffertime;
			minte += TIMESSI; /* inter-core time */
			minte += pgbuffertime;	
			minte += pw_gzrf1trap1a + pw_gzrf1trap1 + pw_gzrf1trap1d; /* pre-rf crusher */
			minte += pgbuffertime;
			if (doNonSelRefocus)
				minte += pw_rf1ns; /* rf1 NS pulse */
			else
				minte += pw_gzrf1a + pw_gzrf1 + pw_gzrf1d; /* rf1 pulse */
			minte += pgbuffertime;	
			minte += pw_gzrf1trap2a + pw_gzrf1trap2 + pw_gzrf1trap2d; /* post-rf crusher */
			minte += pgbuffertime;
			minte += TIMESSI; /* inter-core time */
			minte += pgbuffertime;
			minte += (flowcomp_flag == 1 && spi_mode == 0)*(pw_gzfca + pw_gzfc + pw_gzfcd + pgbuffertime); /* flow comp pre-phaser */
			minte += (spi_mode == 0) * (pw_gzw1a + pw_gzw1 + pw_gzw1d + pgbuffertime); /* SOS z encode gradient */
			minte += pw_gxw/2; /* first half of spiral readout */

			/* calculate deadtimes */
			deadtime1_seqcore = (opte - minesp)/2;
			deadtime1_seqcore -= (flowcomp_flag == 1 && spi_mode == 0)*(pw_gzfca + pw_gzfc + pw_gzfcd + pgbuffertime); /* adjust for flowcomp symmetry */

			deadtime2_seqcore = (opte - minesp)/2; 
			deadtime2_seqcore += (flowcomp_flag == 1 && spi_mode == 0)*(pw_gzfca + pw_gzfc + pw_gzfcd + pgbuffertime);		

			minte += deadtime1_seqcore; 
			deadtime_rf0core = opte - minte;
/*			
			deadtime_rf0core = opte/2 - (pw_gzrf0/2 + pw_gzrf0d);
			deadtime_rf0core -= pgbuffertime;
			deadtime_rf0core -= (pw_gzrf0ra + pw_gzrf0r + pw_gzrf0rd) ;
			deadtime_rf0core -= pgbuffertime;
			deadtime_rf0core -= TIMESSI;
			deadtime_rf0core -= pgbuffertime;
			deadtime_rf0core -= (pw_gzrf1trap1a + pw_gzrf1trap1 + pw_gzrf1trap1d);
			deadtime_rf0core -= pgbuffertime;
			deadtime_rf0core -= (pw_gzrf1/2 + pw_gzrf0d);
*/
			minte = (int)fmax(minte, minesp);
			minesp = 0; /* no restriction on esp cv - let opte control the echo spacing */
			fprintf(stderr, "\n -- calculated minTE and minESP: %d and  %d\n", minte , minesp);
	
			break;

		case 2: /* SPGR */
			
			/* calculate minimum esp (time from rf1 to next rf1) */
			minesp += pw_gzrf1/2 + pw_gzrf1d; /* 2nd half of rf1 pulse */
			minesp += pgbuffertime;
			minesp += pw_gzrf1trap2a + pw_gzrf1trap2 + pw_gzrf1trap2d; /* rf1 slice select rewinder */
			minesp += pgbuffertime;
			minesp += TIMESSI; /* inter-core time */
			minesp += pgbuffertime;
			minesp += (flowcomp_flag == 1 && spi_mode == 0)*(pw_gzfca + pw_gzfc + pw_gzfcd + pgbuffertime); /* flow comp pre-phaser */
			minesp += (spi_mode == 0) * (pw_gzw1a + pw_gzw1 + pw_gzw1d + pgbuffertime); /* z rewind gradient */
			minesp += pw_gxw; /* spiral readout */
			minesp += pgbuffertime;
			minesp += (spi_mode > 0) * (pw_gzw2a + pw_gzw2 + pw_gzw2d + pgbuffertime); /* z rewind gradient */
			minesp += TIMESSI; /* inter-core time */
			minesp += pgbuffertime;
			minesp += pw_gzrf1trap1a + pw_gzrf1trap1 + pw_gzrf1trap1d; /* pre-rf crusher */
			minesp += pgbuffertime;
			minesp += pw_gzrf1a + pw_gzrf1; /* 1st half of rf1 pulse */

			/* calculate minimum TE (time from center of rf1 to beginning of readout pulse) */
			minte += pw_gzrf1/2 + pw_gzrf1d; /* 2nd half of rf1 pulse */
			minte += pgbuffertime;	
			minte += pw_gzrf1trap2a + pw_gzrf1trap2 + pw_gzrf1trap2d; /* post-rf crusher */
			minte += pgbuffertime;
			minte += TIMESSI; /* inter-core time */
			minte += pgbuffertime;
			minte += (flowcomp_flag == 1 && spi_mode == 0)*(pw_gzfca + pw_gzfc + pw_gzfcd + pgbuffertime); /* flow comp pre-phaser */
			minte += (spi_mode == 0) * (pw_gzw1a + pw_gzw1 + pw_gzw1d + pgbuffertime); /* z rewind gradient */

			/* calculate deadtimes */
			deadtime_rf0core = 1ms; /* no effect here */
			deadtime1_seqcore = opte - minte;
			minesp += deadtime1_seqcore; /* add deadtime1 to minesp calculation */
			deadtime2_seqcore = esp - minesp;
		
			break;

		case 3: /* bSSFP */

			/* calculate minimum esp (time from rf1 to next rf1) */
			minesp += pw_gzrf1/2 + pw_gzrf1d; /* 2nd half of rf1 pulse */
			minesp += pgbuffertime;
			minesp += pw_gzrf1trap2a + pw_gzrf1trap2 + pw_gzrf1trap2d; /* rf1 slice select rewinder */
			minesp += pgbuffertime;
			minesp += TIMESSI; /* inter-core time */
			minesp += pgbuffertime;
			minesp += (flowcomp_flag == 1 && spi_mode == 0)*(pw_gzfca + pw_gzfc + pw_gzfcd + pgbuffertime); /* flow comp pre-phaser */
			minesp += (spi_mode == 0) * (pw_gzw1a + pw_gzw1 + pw_gzw1d + pgbuffertime); /* z rewind gradient */
			minesp += pw_gxw; /* spiral readout */
			minesp += pgbuffertime;
			minesp += pw_gzw2a + pw_gzw2 + pw_gzw2d; /* z rewind gradient */
			minesp += TIMESSI; /* inter-core time */
			minesp += pgbuffertime;
			minesp += pw_gzrf1a + pw_gzrf1; /* 1st half of rf1 pulse */

			/* calculate minimum TE (time from center of rf1 to beginning of readout pulse) */
			minte += pw_gzrf1/2 + pw_gzrf1d; /* 2nd half of rf1 pulse */
			minte += pgbuffertime;	
			minte += pw_gzrf1trap2a + pw_gzrf1trap2 + pw_gzrf1trap2d; /* post-rf crusher */
			minte += pgbuffertime;
			minte += TIMESSI; /* inter-core time */
			minte += pgbuffertime;
			minte += (flowcomp_flag == 1 && spi_mode == 0)*(pw_gzfca + pw_gzfc + pw_gzfcd + pgbuffertime); /* flow comp pre-phaser */
			minte += (spi_mode == 0) * (pw_gzw1a + pw_gzw1 + pw_gzw1d + pgbuffertime); /* z rewind gradient */
			minte += pw_gxw/2; /* first half of spiral readout */
			
			/* calculate deadtimes */
			deadtime_rf0core = 1ms; /* no effect here */
			deadtime1_seqcore = opte - minte;
			minesp += deadtime1_seqcore; /* add deadtime1 to minesp calculation */
			deadtime2_seqcore = esp - minesp;
			
			break;
	}
	fprintf(stderr, "\ncalculated minTE and minESP: %d and  %d\n", minte , minesp);

	/* set minimums */	
	cvmin(esp, minesp);
	cvmin(opte, minte);

	/* set fatsup deadtime */
	if (fatsup_mode < 2) /* CHESS/none */
		deadtime_fatsupcore = 0;
	else { /* SPIR */
		deadtime_fatsupcore = spir_ti;
		deadtime_fatsupcore -= pw_rffs/2; /* 2nd half of rf pulse */
		deadtime_fatsupcore -= pgbuffertime;
		deadtime_fatsupcore -= (pw_gzrffsspoila + pw_gzrffsspoil + pw_gzrffsspoild); /* crusher */
		deadtime_fatsupcore -= pgbuffertime;
		deadtime_fatsupcore -= TIMESSI;
		deadtime_fatsupcore -= pgbuffertime;
		switch (ro_type) {
			case 1: /* FSE */
				deadtime_fatsupcore -= (pw_gzrf0a + pw_gzrf0/2); /* first half of tipdown */
				break;
			case 2: /* SPGR */
				deadtime_fatsupcore -= (pw_gzrf1trap1a + pw_gzrf1trap1 + pw_gzrf1trap2);
				deadtime_fatsupcore -= pgbuffertime;
				deadtime_fatsupcore -= (pw_gzrf1a + pw_gzrf1/2);
				break;
			case 3: /* bSSFP */
				deadtime_fatsupcore -= (pw_gzrf1a + pw_gzrf1/2);
				break;
		}
	}

	/* Calculate the duration of presatcore */
	dur_presatcore = 0;
	dur_presatcore += pgbuffertime;
	dur_presatcore += pw_rfps1;
	dur_presatcore += pgbuffertime;
	dur_presatcore += pw_rfps1ca + pw_rfps1c + pw_rfps1cd;
	dur_presatcore += 1000 + pgbuffertime;
	dur_presatcore += pw_rfps2;
	dur_presatcore += pgbuffertime;
	dur_presatcore += pw_rfps2ca + pw_rfps2c + pw_rfps2cd;
	dur_presatcore += 1000 + pgbuffertime;
	dur_presatcore += pw_rfps3;
	dur_presatcore += pgbuffertime;
	dur_presatcore += pw_rfps3ca + pw_rfps3c + pw_rfps3cd;
	dur_presatcore += 1000 + pgbuffertime;
	dur_presatcore += pw_rfps4;
	dur_presatcore += pgbuffertime;
	dur_presatcore += pw_rfps4ca + pw_rfps4c + pw_rfps4cd;
	dur_presatcore += pgbuffertime;

	/* calculate the duration of the PCASL core */
	dur_pcaslcore = 0;
	dur_pcaslcore += pcasl_buffertime;
	dur_pcaslcore += pw_gzpcasl + pw_gzpcasla + pw_gzpcasld;
	dur_pcaslcore += pcasl_buffertime;
	dur_pcaslcore += pw_gzpcaslref + pw_gzpcaslrefa + pw_gzpcaslrefd;
	dur_pcaslcore += pcasl_buffertime;
	dur_pcaslcore += deadtime_pcaslcore;

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

	/* Calculate the duration of fatsupcore */
	dur_fatsupcore = 0;
	dur_fatsupcore += pgbuffertime;
	dur_fatsupcore += pw_rffs;
	dur_fatsupcore += pgbuffertime;
	dur_fatsupcore += pw_gzrffsspoila + pw_gzrffsspoil + pw_gzrffsspoild;
	dur_fatsupcore += pgbuffertime;
	dur_fatsupcore += deadtime_fatsupcore;
	
	/* calculate duration of rf0core */
	dur_rf0core = 0;
	dur_rf0core += pgbuffertime;
	dur_rf0core += pw_gzrf0a + pw_gzrf0 + pw_gzrf0d;
	dur_rf0core += pgbuffertime;
	dur_rf0core += pw_gzrf0ra + pw_gzrf0r + pw_gzrf0rd;
	dur_rf0core += pgbuffertime; 
	dur_rf0core += deadtime_rf0core;
	
	/* calculate duration of rf1core */
	dur_rf1core = 0;
	dur_rf1core += pgbuffertime;
	dur_rf1core += pw_gzrf1trap1a + pw_gzrf1trap1 + pw_gzrf1trap1d;
	dur_rf1core += pgbuffertime;
	dur_rf1core += pw_gzrf1a + pw_gzrf1 + pw_gzrf1d;
	dur_rf1core += pgbuffertime;
	dur_rf1core += pw_gzrf1trap2a + pw_gzrf1trap2 + pw_gzrf1trap2d;
	dur_rf1core += pgbuffertime; 

	/* calculate duration of nonselective rect refocuser rf1nscore */
	dur_rf1nscore = 0;
	dur_rf1nscore += pgbuffertime;
	dur_rf1nscore += pw_rf1trap1nsa + pw_rf1trap1ns + pw_rf1trap1nsd;
	dur_rf1nscore += pgbuffertime;
	dur_rf1nscore += pw_rf1ns;
	dur_rf1nscore += pgbuffertime;
	dur_rf1nscore += pw_rf1trap2nsa + pw_rf1trap2ns + pw_rf1trap2nsd;
	dur_rf1nscore += pgbuffertime; 

	/* calculate duration of seqcore */
	dur_seqcore = 0;
	dur_seqcore += deadtime1_seqcore + pgbuffertime;
	dur_seqcore += (flowcomp_flag == 1 && spi_mode == 0)*(pw_gzfca + pw_gzfc + pw_gzfcd + pgbuffertime);
	dur_seqcore += (spi_mode == 0) * (pw_gzw1a + pw_gzw1 + pw_gzw1d + pgbuffertime); /* SOS z encode gradient */
	dur_seqcore += pw_gxw;
	dur_seqcore += (spi_mode == 0) * (pw_gzw2a + pw_gzw2 + pw_gzw2d + pgbuffertime);  /* SOS - z rewinder*/
	dur_seqcore += deadtime2_seqcore + pgbuffertime;

	/* calculate minimum TR */
	absmintr = presat_flag*(dur_presatcore + TIMESSI + presat_delay + TIMESSI);
	absmintr += (pcasl_flag) *  pcasl_Npulses*(dur_pcaslcore + TIMESSI);
	absmintr += (pcasl_flag) * (pcasl_pld + TIMESSI);
	absmintr += (prep1_id > 0)*(dur_prep1core + TIMESSI + prep1_pld + TIMESSI);
	absmintr += (prep2_id > 0)*(dur_prep2core + TIMESSI + prep2_pld + TIMESSI);
	absmintr += (fatsup_mode > 0)*(dur_fatsupcore + TIMESSI);

	if (ro_type == 1) /* FSE - add the rf0 pulse */
		absmintr += dur_rf0core + TIMESSI;

	if(doNonSelRefocus) /* happens only in FSE mode */
		absmintr += (opetl + ndisdaqechoes) * (dur_rf1nscore + TIMESSI + dur_seqcore + TIMESSI);
	else
		absmintr += (opetl + ndisdaqechoes) * (dur_rf1core + TIMESSI + dur_seqcore + TIMESSI);
	

	if (exist(opautotr) == PSD_MINIMUMTR)
		optr = absmintr;	
	cvmin(optr, absmintr);

	/* calculate TR deadtime */
	tr_deadtime = optr - absmintr;

	/* troubleshooting */
	if(doNonSelRefocus)
		fprintf(stderr, "\npredownload(): DEBUG: total deadtime for disdaqs: %d ",
		(optr - opetl * (dur_rf1core + TIMESSI + dur_seqcore + TIMESSI)));
	else
		fprintf(stderr, "\npredownload(): DEBUG: total deadtime for disdaqs: %d ",
		(optr - opetl * (dur_rf1nscore + TIMESSI + dur_seqcore + TIMESSI)));
	
	fprintf(stderr, "\n--optr %d " , optr);
	fprintf(stderr, "\n--opetl %d " , opetl);
	fprintf(stderr, "\n--dur_rf1core %d " , dur_rf1core);
	fprintf(stderr, "\n--dur_rf1nscore %d " , dur_rf1core);
	fprintf(stderr, "\n--dur_seqcore %d " , dur_seqcore);
	fprintf(stderr, "\n--TIMESSI %d " , TIMESSI);
		
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

	/* Set up the filter structures to be downloaded for realtime 
	   filter generation. Get the slot number of the filter in the filter rack 
	   and assign to the appropriate acquisition pulse for the right 
	   filter selection - LxMGD, RJF */
	setfilter( echo1_filt, SCAN );
	filter_echo1 = echo1_filt->fslot;
	entry_point_table[L_SCAN].epxmtadd = (short)rint( (double)xmtaddScan );

	/* LHG : doing the same filters for the prescan */
	setfilter( echo1_filt, PRESCAN );

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

	write_scan_info();

@inline Prescan.e PSpredownload	

	fprintf(stderr, "\n\nEnd of Predownload()\n\n");

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

/* waveform pointers for real time updates in the scan loop 
we use them for updating the phase of the prep pulse in velocity spectrum imaging

WF_HW_WAVEFORM_PTR 	phsbuffer1_wf;
WF_HW_WAVEFORM_PTR 	phsbuffer2_wf;

int* 	phsbuffer1_wf;
int* 	phsbuffer2_wf;
*/
/* function to reserve memory for a dynamically updated waveform */
void tp_wreserve(WF_PROCESSOR wfp, WF_HW_WAVEFORM_PTR *wave_addr, int n) 
{
    SeqData seqdata;
    getWaveSeqDataWavegen(&seqdata, wfp, 0, 0, 0, PULSE_CREATE_MODE);
    *wave_addr = wreserve(seqdata, n);
}

STATUS pulsegen( void )
{
	sspinit(psd_board_type);
	int tmploc;	
	/*********************************/
	/* Generate PCASL core */
	/*********************************/	
	fprintf(stderr, "\npulsegen(): beginning pulse generation of PCASL core\n");
	tmploc = 0;	
	tmploc += pcasl_buffertime;
	tmploc += pw_gzpcasla ;

	fprintf(stderr, "\npulsegen(): generating PCASL slice select gradient and RF...\n");
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	TRAPEZOID(ZGRAD, gzpcasl,  tmploc , GRAD_UPDATE_TIME*1000, 0, loggrd);

	fprintf(stderr, "\tRF start: %dus, ", tmploc);
	EXTWAVE(RHO, rfpcasl, tmploc , 500, 1.0, 250, myhanning.rho, , loggrd);
	/* old line: 
	
	EXTWAVE(RHO, rfpcasl, tmploc + psd_rf_wait , 500, 1.0, 250, myhanning.rho, , loggrd);

	If I include the psd_rf_wait , I can see the psd_rf_wait delay 
	between the gradient and the RF on the scope. This is not the case in other pulses.
	pcasl code for MR750 did have the psd_rf_wait lag here ... what's different? 
	Is EXTWAVE handled differently than TRAPEZOID now in MR30?

	*/
	
	tmploc += pw_gzpcasl + pw_gzpcasld;
	fprintf(stderr, "\tend: %dus, ", tmploc);

	tmploc += pcasl_buffertime;

	fprintf(stderr, "\npulsegen(): generating PCASL refocus gradient...\n");
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	TRAPEZOID(ZGRAD, gzpcaslref,  tmploc + pw_gzpcaslrefa, GRAD_UPDATE_TIME*1000, 0, loggrd);
	tmploc += pw_gzpcaslref + pw_gzpcaslrefa + pw_gzpcaslrefd;
	fprintf(stderr, "\tend: %dus, ", tmploc);

	tmploc += pcasl_buffertime;

	tmploc += deadtime_pcaslcore;
	fprintf(stderr, "\npulsegen(): finalizing pcaslcore...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %d us)\n", dur_pcaslcore, tmploc);
	SEQLENGTH(pcaslcore, dur_pcaslcore , pcaslcore);
	fprintf(stderr, "\tDone.\n");


	/*************************/
	/* generate readout core */
	/*************************/
	fprintf(stderr, "pulsegen(): beginning pulse generation of seqcore\n");
	tmploc = 0;
	tmploc += deadtime1_seqcore + pgbuffertime; /* add pre-readout deadtime + buffer */

	if (flowcomp_flag && spi_mode == 0) { /* SOS only */
		fprintf(stderr, "pulsegen(): generating gzfc... (flow comp dephaser gradient\n");
		TRAPEZOID(ZGRAD, gzfc, tmploc + pw_gzfca, 1ms, 0, loggrd);	
		fprintf(stderr, "\tstart: %dus, ", tmploc);
		tmploc += pw_gzfca + pw_gzfc + pw_gzfcd; /* end time for gzfc */
		fprintf(stderr, " end: %dus\n", tmploc);
		tmploc += pgbuffertime; /* add some buffer */
	}	

	if (spi_mode == 0) { /* SOS only */
		fprintf(stderr, "pulsegen(): generating gzw1... (z encode + flow comp rephase gradient\n");
		TRAPEZOID(ZGRAD, gzw1, tmploc + pw_gzw1a, 1ms, 0, loggrd);	
		fprintf(stderr, "\tstart: %dus, ", tmploc);
		tmploc += pw_gzw1a + pw_gzw1 + pw_gzw1d; /* end time for gzw1 */
		fprintf(stderr, " end: %dus\n", tmploc);
		tmploc += pgbuffertime; /* add some buffer */
	}

	fprintf(stderr, "pulsegen(): generating gxw, gyw (spiral readout gradients) and echo1 (data acquisition window)...\n");
	INTWAVE(XGRAD, gxw, tmploc, XGRAD_max, grad_len, GRAD_UPDATE_TIME*grad_len, Gx, 1, loggrd);
	INTWAVE(YGRAD, gyw, tmploc, YGRAD_max, grad_len, GRAD_UPDATE_TIME*grad_len, Gy, 1, loggrd);
	ACQUIREDATA(echo1, tmploc + psd_grd_wait + GRAD_UPDATE_TIME*acq_offset,,,);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gxw; /* end time for readout */
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime; /* add some buffer */

	if (spi_mode == 0) { /* SOS only */
		fprintf(stderr, "pulsegen(): generating gzw2... (z rewind)\n");
		TRAPEZOID(ZGRAD, gzw2, tmploc + pw_gzw2a, 1ms, 0, loggrd);	
		fprintf(stderr, "\tstart: %dus, ", tmploc);
		tmploc += pw_gzw2a + pw_gzw2 + pw_gzw2d; /* end time for gzw2 */
		fprintf(stderr, " end: %dus\n", tmploc);
		tmploc += pgbuffertime; /* add some buffer */
	}

	tmploc += deadtime2_seqcore; /* add post-readout deadtime */

	fprintf(stderr, "pulsegen(): finalizing spiral readout core...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_seqcore, tmploc);
	SEQLENGTH(seqcore, dur_seqcore, seqcore);
	fprintf(stderr, "\tDone.\n");


	/************************/
	/* generate presat core */
	/************************/	
	fprintf(stderr, "pulsegen(): beginning pulse generation of presatcore\n");
	tmploc = 0;
	
	fprintf(stderr, "pulsegen(): generating rfps1 (presat rf pulse 1)...\n");
	tmploc += pgbuffertime; /* start time for rfps1 pulse */
	TRAPEZOID(RHO, rfps1, tmploc + psd_rf_wait, 1ms, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rfps1;
	fprintf(stderr, " end: %dus\n", tmploc);
	
	fprintf(stderr, "pulsegen(): generating rfps1c (presat rf crusher 1)...\n");
	tmploc += pgbuffertime; /*start time for gradient */
	TRAPEZOID(ZGRAD, rfps1c, tmploc + pw_rfps1ca, 1ms, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rfps1ca + pw_rfps1c + pw_rfps1cd; /* end time for gradient */
	fprintf(stderr, " end: %dus\n", tmploc);

	fprintf(stderr, "pulsegen(): generating rfps2 (presat rf pulse 2)...\n");
	tmploc += 1000 + pgbuffertime; /* start time for rfps2 pulse */
	TRAPEZOID(RHO, rfps2, tmploc + psd_rf_wait, 1ms, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rfps2;
	fprintf(stderr, " end: %dus\n", tmploc);
	
	fprintf(stderr, "pulsegen(): generating rfps2c (presat rf crusher 2)...\n");
	tmploc += pgbuffertime; /*start time for gradient */
	TRAPEZOID(ZGRAD, rfps2c, tmploc + pw_rfps1ca, 1ms, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rfps2ca + pw_rfps1c + pw_rfps1cd; /* end time for gradient */
	fprintf(stderr, " end: %dus\n", tmploc);
	
	fprintf(stderr, "pulsegen(): generating rfps3 (presat rf pulse 3)...\n");
	tmploc += 1000 + pgbuffertime; /* start time for rfps3 pulse */
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
	tmploc += 1000 + pgbuffertime;
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
	fprintf(stderr, "pulsegen(): beginning pulse generation of prep1lblcore\n");
	tmploc = 0;
	
	fprintf(stderr, "pulsegen(): generating prep1rholbl, prep1thetalbl & prep1gradlbl (prep1 label rf & gradients)...\n");
	tmploc += pgbuffertime; /* start time for prep1 pulse */
	INTWAVE(RHO, prep1rholbl, tmploc + psd_rf_wait, 0.0, prep1_len, GRAD_UPDATE_TIME*prep1_len, prep1_rho_lbl, 1, loggrd); 
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	INTWAVE(THETA, prep1thetalbl, tmploc + psd_rf_wait, 1.0, prep1_len, GRAD_UPDATE_TIME*prep1_len, prep1_theta_lbl, 1, loggrd); 
	fprintf(stderr, "\tstart: %dus, ", tmploc);
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
	fprintf(stderr, "pulsegen(): beginning pulse generation of prep1ctlcore\n");
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
	fprintf(stderr, "pulsegen(): beginning pulse generation of prep2lblcore\n");
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
	fprintf(stderr, "pulsegen(): beginning pulse generation of prep2ctlcore\n");
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
	fprintf(stderr, "pulsegen(): beginning pulse generation of bkgsupcore\n");
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
	/* generate fatsup core */
	/************************/
	fprintf(stderr, "pulsegen(): beginning pulse generation of fatsupcore\n");	
	tmploc = 0;
	
	fprintf(stderr, "pulsegen(): generating rffs (fat suppresion rf pulse)...\n");
	tmploc += pgbuffertime; /* start time for rffs */
	SINC(RHO, rffs, tmploc + psd_rf_wait, 3200, 1.0, ,0.5, , , loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rffs; /* end time for rffs */
	fprintf(stderr, " end: %dus\n", tmploc);	
 
	fprintf(stderr, "pulsegen(): generating gzrffsspoil (fat suppression crusher gradients)...\n");
	tmploc += pgbuffertime; /* start time for gzrffsspoil */
	TRAPEZOID(ZGRAD, gzrffsspoil, tmploc + pw_gzrffsspoila, GRAD_UPDATE_TIME*1000, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzrffsspoila + pw_gzrffsspoil + pw_gzrffsspoild; /* end time for gzrffsspoil */
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime; /* add some buffer */

	fprintf(stderr, "pulsegen(): finalizing fatsupcore...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_fatsupcore, tmploc);
	SEQLENGTH(fatsupcore, dur_fatsupcore, fatsupcore);
	fprintf(stderr, "\tDone.\n");
	

	/*************************/
	/* generate rf0 core */
	/*************************/
	fprintf(stderr, "pulsegen():  beginning pulse generation of rf0 core\n");
	tmploc = 0;

	fprintf(stderr, "pulsegen(): generating rf0 (rf0 pulse)...\n");
	tmploc += pgbuffertime; /* start time for rf0 */
	SLICESELZ(rf0, tmploc + pw_gzrf0a, 3200, (opslthick + opslspace) * opslquant * SE_factor , 90.0, 2, 1, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzrf0a + pw_gzrf0 + pw_gzrf0d; /* end time for rf2 pulse */
	fprintf(stderr, " end: %dus\n", tmploc);
		
	fprintf(stderr, "pulsegen(): generating gzrf1trap2 (post-rf1 gradient trapezoid)...\n");
	tmploc += pgbuffertime; /* start time for gzrf0r */
	TRAPEZOID(ZGRAD, gzrf0r, tmploc + pw_gzrf0ra, 3200, 0, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzrf0ra + pw_gzrf0r + pw_gzrf0rd; /* end time for gzrf1trap2 pulse */
	fprintf(stderr, " end: %dus\n", tmploc);
	tmploc += pgbuffertime; /* buffer */

	tmploc += deadtime_rf0core;

	fprintf(stderr, "pulsegen(): finalizing rf0 core...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_rf0core, tmploc);
	SEQLENGTH(rf0core, dur_rf0core, rf0core);
	fprintf(stderr, "\tDone.\n");

	
	/*************************/
	/* generate rf1 core */
	/*************************/
	fprintf(stderr, "pulsegen(): beginning pulse generation of rf1 core\n");
	tmploc = 0;

	if (ro_type != 3) { /* bSSFP - do not use trap1 */
		fprintf(stderr, "pulsegen(): generating gzrf1trap1 (pre-rf1 gradient trapezoid)...\n");
		tmploc += pgbuffertime; /* start time for gzrf1trap1 */
		TRAPEZOID(ZGRAD, gzrf1trap1, tmploc + pw_gzrf1trap1a, 3200, 0, loggrd);
		fprintf(stderr, "\tstart: %dus, ", tmploc);
		tmploc += pw_gzrf1trap1a + pw_gzrf1trap1 + pw_gzrf1trap1d; /* end time for gzrf1trap1 */
		fprintf(stderr, " end: %dus\n", tmploc);
	}

	fprintf(stderr, "pulsegen(): generating rf1 (rf1 pulse)...\n");
	tmploc += pgbuffertime; /* start time for rf1 */
	SLICESELZ(rf1, tmploc + pw_gzrf1a, 3200, (opslthick + opslspace) * opslquant * SE_factor, opflip, 2, 1, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_gzrf1a + pw_gzrf1 + pw_gzrf1d; /* end time for rf2 pulse */
	fprintf(stderr, " end: %dus\n", tmploc);

	if (ro_type != 3) { /* bSSFP - do not use trap2 */
		fprintf(stderr, "pulsegen(): generating gzrf1trap2 (post-rf1 gradient trapezoid)...\n");
		tmploc += pgbuffertime; /* start time for gzrf1trap2 */
		TRAPEZOID(ZGRAD, gzrf1trap2, tmploc + pw_gzrf1trap2a, 3200, 0, loggrd);
		fprintf(stderr, "\tstart: %dus, ", tmploc);
		tmploc += pw_gzrf1trap2a + pw_gzrf1trap2 + pw_gzrf1trap2d; /* end time for gzrf1trap2 pulse */
		fprintf(stderr, " end: %dus\n", tmploc);
	}
	
	tmploc += pgbuffertime;

	fprintf(stderr, "pulsegen(): finalizing rf1 core...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_rf1core, tmploc);
	SEQLENGTH(rf1core, dur_rf1core, rf1core);
	fprintf(stderr, "\tDone.\n");

	
	/*************************/
	/* generate rf1ns core (non-selective rect RF pulse refocuser)*/
	/*************************/
	fprintf(stderr, "pulsegen(): beginning pulse generation of rf1ns core\n");
	tmploc = 0;

	if (ro_type != 3) { /* bSSFP - do not use trap1 */
		fprintf(stderr, "pulsegen(): generating rf1trap1ns (pre-rf1 gradient trapezoid)...\n");
		tmploc += pgbuffertime; /* start time for rf1trap1ns */
		TRAPEZOID(ZGRAD, rf1trap1ns, tmploc + pw_rf1trap1nsa, 3200, 0, loggrd);
		fprintf(stderr, "\tstart: %dus, ", tmploc);
		tmploc += pw_rf1trap1nsa + pw_rf1trap1ns + pw_rf1trap1nsd; /* end time for rf1trap1ns */
		fprintf(stderr, " end: %dus\n", tmploc);
	}

	fprintf(stderr, "pulsegen(): generating RECT rf1 (rf1ns pulse)...\n");
	tmploc += pgbuffertime; /* start time for rf1ns */
	CONST(RHO, rf1ns, tmploc + psd_rf_wait, 1000, opflip, loggrd);
	fprintf(stderr, "\tstart: %dus, ", tmploc);
	tmploc += pw_rf1ns ; /* end time for rf1ns pulse */
	fprintf(stderr, " end: %dus\n", tmploc);

	if (ro_type != 3) { /* bSSFP - do not use trap2 */
		fprintf(stderr, "pulsegen(): generating gzrf1trap2 (post-rf1 gradient trapezoid)...\n");
		tmploc += pgbuffertime; /* start time for rf1trap2ns */
		TRAPEZOID(ZGRAD, rf1trap2ns, tmploc + pw_rf1trap2nsa, 3200, 0, loggrd);
		fprintf(stderr, "\tstart: %dus, ", tmploc);
		tmploc += pw_rf1trap2nsa + pw_rf1trap2ns + pw_rf1trap2nsd; /* end time for gzrf1trap2 pulse */
		fprintf(stderr, " end: %dus\n", tmploc);
	}
	
	tmploc += pgbuffertime;

	fprintf(stderr, "pulsegen(): finalizing rf1ns core...\n");
	fprintf(stderr, "\ttotal time: %dus (tmploc = %dus)\n", dur_rf1nscore, tmploc);
	SEQLENGTH(rf1nscore, dur_rf1nscore, rf1nscore);
	fprintf(stderr, "\tDone.\n");

	/**********************************/
	/* generate deadtime (empty) core */
	/**********************************/
	fprintf(stderr, "pulsegen(): beginning pulse generation of emptycore\n");

	fprintf(stderr, "pulsegen(): finalizing empty core...\n");
	SEQLENGTH(emptycore, 1000, emptycore);
	fprintf(stderr, "\tDone.\n");

@inline Prescan.e PSpulsegen

	PASSPACK(endpass, 49ms);   /* tell Signa system we're done */
	SEQLENGTH(pass, 50ms, pass);

	buildinstr();              /* load the sequencer memory       */
	fprintf(stderr, "\tDone with pulsegen().\n");

	/* reserve memory for waveform buffers in waveform memory (IPG?)
	for the phase of the prep pulses.
	use these for dynamic updates of the phase waveforms in the scan loop
	*/	
	/*
	tp_wreserve(TYPTHETA, phsbuffer1_wf, res_prep1thetalbl);
	tp_wreserve(TYPTHETA, phsbuffer2_wf, res_prep1thetactl);
	*/		

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

/* declare the seqdata object for dynamic updates of waveform using movewaveimmrsp) */
SeqData	sdTheta1, sdTheta2;


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
	setfrequency((int)xmitfreq, &rfps1, 0);
	setfrequency((int)xmitfreq, &rfps2, 0);
	setfrequency((int)xmitfreq, &rfps3, 0);
	setfrequency((int)xmitfreq, &rfps4, 0);
	setfrequency((int)xmitfreq, &rf0, 0);
	setfrequency((int)xmitfreq, &rf1, 0);
	setfrequency((int)recfreq, &echo1, 0);

	/* Set fat sup frequency */
	setfrequency( (int)(fatsup_off / TARDIS_FREQ_RES), &rffs, 0);
	
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
int play_aslprep(int type, s32* off_ctlcore, s32* off_lblcore, int dur, int pld, int tbgs1, int tbgs2, int tbgs3) 
{
	int ttotal = 0;
	int ttmp;
	

	
	setrotate(prep_rotmat, 0);


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

	/* restore rotation matrix */
	setrotate(tmtx0, 0);

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


/* play the PCASL train and the post label delay including background suppression pulses*/
int play_pcasl(int type,  int tbgs1, int tbgs2, int pld) {

	int ttmp;
	int ttotal = 0;
	int i;
	float pcasl_toggle = 1.0; /* variable to switch from control to label... (+1 or -1)*/
	/* 
	float tmpPHI=0.0;
	int iphase=0;
	*/
	pcasl_Npulses = (int)(floor(pcasl_duration/pcasl_period));

	/* set the RF frequency for labeling pulses */
	setfrequency((int)(pcasl_RFfreq/TARDIS_FREQ_RES), &rfpcasl, 0);

	switch (type) {
		case 0: /* control PCASL: loop alternating the sign of the RF pulse. update the phase*/
			fprintf(stderr, "\tplay_pcasl(): playing PCASL control pulse (%d us)...\n", pcasl_period);
			pcasl_toggle = -1.0;
			break;
		case -1: /* no PCASL - keep the gradients but don't play rf */
			fprintf(stderr, "\tplay_pcasl(): playing NO PCASL pulse (%d us)...\n", pcasl_period );
			pcasl_toggle = 0.0;
			break;
		case 1: /* label PCASL : loop the PCASL core updating the phse of the RF pulses*/
			fprintf(stderr, "\tplay_pcasl(): playing PCASL label pulse (%d us)...\n", pcasl_period );
			pcasl_toggle = 1.0;
			break;

		default: /* invalid */
			fprintf(stderr, "\tplay_pcasl(): ERROR - invalid type (%d)\n", type);
			rspexit();
			return -1;
	}

	/* Execute the PCASL train loop here */
	for (i=0; i<pcasl_Npulses; i++){
		/* set the amplitude of the blips */
		setiamp((int)(pow(pcasl_toggle,i) * pcasl_RFamp_dac), &rfpcasl, 0);

		/* fprintf(stderr, "\tplay_pcasl(): pulse phase %f (rads)...\n", tmpPHI );*/
		/* set the phase of the blips incrementing phase each time - previously calculated*/
		
		setiphase(pcasl_iphase_tbl[i], &rfpcasl,0);
		/*setiphase(-pcasl_iphase_tbl[i], &rfpcasl,0);*/
		 
		/*
	    tmpPHI = atan2 (sin(tmpPHI), cos(tmpPHI));     
        iphase = (int)(tmpPHI/M_PI * (float)FS_PI);
		setiphase(iphase, &rfpcasl,0);
		tmpPHI += pcasl_delta_phs; 
		*/

		boffset(off_pcaslcore);
		startseq(0, MAY_PAUSE);
		settrigger(TRIG_INTERN, 0);
	}
	ttotal += pcasl_Npulses*(pcasl_period);

	/* Play pld and background suppression - same code as in the play_pcasl() */
	if (pld > 0) {

		/* Initialize pld before subtracting out tbgs timing */
		ttmp = pld;

		if (tbgs1 > 0) {
			/* Play first background suppression delay/pulse */
			fprintf(stderr, "\tplay_pcasl(): playing bkg suppression pulse 1 delay (%d us)...\n", tbgs1 + TIMESSI);		
			setperiod(tbgs1, &emptycore, 0);
			ttmp -= (tbgs1 + TIMESSI);
			boffset(off_emptycore);
			startseq(0, MAY_PAUSE);
			settrigger(TRIG_INTERN, 0);
			ttotal += tbgs1 + TIMESSI;

			fprintf(stderr, "\tplay_pcasl(): playing bkg suppression pulse 1 (%d us)...\n", dur_bkgsupcore + TIMESSI);
			ttmp -= (dur_bkgsupcore + TIMESSI);
			boffset(off_bkgsupcore);
			startseq(0, MAY_PAUSE);
			settrigger(TRIG_INTERN, 0);
			ttotal += dur_bkgsupcore + TIMESSI;
		}

		if (tbgs2 > 0) {
			/* Play second background suppression delay/pulse */
			fprintf(stderr, "\tplay_pcasl(): playing bkg suppression pulse 2 delay (%d us)...\n", tbgs2 + TIMESSI);		
			setperiod(tbgs2, &emptycore, 0);
			ttmp -= (tbgs2 + TIMESSI);
			boffset(off_emptycore);
			startseq(0, MAY_PAUSE);
			settrigger(TRIG_INTERN, 0);
			ttotal += tbgs2 + TIMESSI;

			fprintf(stderr, "\tplay_pcasl(): playing bkg suppression pulse 2 (%d us)...\n", dur_bkgsupcore + TIMESSI);
			ttmp -= (dur_bkgsupcore + TIMESSI);
			boffset(off_bkgsupcore);
			startseq(0, MAY_PAUSE);
			settrigger(TRIG_INTERN, 0);
			ttotal += dur_bkgsupcore + TIMESSI;
		}

		/* Check that ttmp is non-negative */
		if (ttmp < 0) {
			fprintf(stderr, "\tplay_pcasl(): ERROR: invalid pld and background suppression time combination\n");
			rspexit();
		}

		/* Play remaining PLD deadtime */
		fprintf(stderr, "\tplay_pcasl(): playing post-label delay (%d us), total end delay = %d us...\n", pld, ttmp);
		setperiod(ttmp - TIMESSI, &emptycore, 0);
		boffset(off_emptycore);
		startseq(0, MAY_PAUSE);
		settrigger(TRIG_INTERN, 0);
		ttotal += ttmp;
	}	
	return ttotal;

}

/* LHG 12/6/12 : compute the linear phase increment of the PCASL pulses - NOT USED currently*/
int update_pcasl_phases(int *iphase_tbl, float  myphase_increment, int nreps)
{
	int     n;
	double  rfphase; 

	fprintf(stderr,"\nUpdating PCASL phase table .... "); 
	rfphase = 0.0;

	for (n=0; n<nreps; n++)
	{
		rfphase += myphase_increment;
		/* wrap phase to (-pi,pi) range */
		rfphase = atan2 (sin(rfphase), cos(rfphase));      
		/* translate to DAC units */
		iphase_tbl[n] = (int)(rfphase/M_PI * (float)FS_PI);
	}
	fprintf(stderr,"Done .\n "); 
	return 1;
}               
        
/* function to figure out whether it's a lable or a control */
int determine_label_type(int mode, int framen)
{
	int type = -1;

	switch (mode) {
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
	return type;
}



/* function for playing fat sup pulse */
int play_fatsup() {
	int ttotal = 0;

	fprintf(stderr, "\tplay_fatsup(): playing fat sup pulse (%d us)...\n", dur_fatsupcore + TIMESSI);

	/* Play fatsup core */
	boffset(off_fatsupcore);
	startseq(0, MAY_PAUSE);
	settrigger(TRIG_INTERN, 0);
	ttotal += dur_fatsupcore + TIMESSI;

	fprintf(stderr, "\tplay_fatsup(): Done.\n");
	return ttotal;
}

/* function for playing rf0 pulse */
int play_rf0(float phs) {
	int ttotal = 0;

	/* set tx phase */
	setphase(phs, &rf0, 0);

	/* Play the rf1 */
	fprintf(stderr, "\tplay_rf0(): playing rf0core (%d us)...\n", dur_rf0core);
	boffset(off_rf0core);
	startseq(0, MAY_PAUSE);
	settrigger(TRIG_INTERN, 0);
	ttotal += dur_rf0core + TIMESSI;

	return ttotal;	
}

/* function for playing GRE rf1 pulse
in FSE mode, this serves as the refocuser (180) */
int play_rf1(float phs) {
	int ttotal = 0;

	/* set rx and tx phase */
	setphase(phs, &rf1, 0);

	/* Play the rf1 */
	fprintf(stderr, "\tplay_rf1(): playing rf1core (%d us)...\n", dur_rf1core);
	boffset(off_rf1core);
	startseq(0, MAY_PAUSE);
	settrigger(TRIG_INTERN, 0);
	ttotal += dur_rf1core + TIMESSI;

	return ttotal;	
}

/* function for playing FSE rf1 refocuser pulse
this refocuser is a non-selective rect function */
int play_rf1ns(float phs) {
	int ttotal = 0;

	/* set rx and tx phase */
	setphase(phs, &rf1ns, 0);

	/* Play the rf1 */
	fprintf(stderr, "\tplay_rf1ns(): playing rf1nscore (%d us)...\n", dur_rf1nscore);
	boffset(off_rf1nscore);
	startseq(0, MAY_PAUSE);
	settrigger(TRIG_INTERN, 0);
	ttotal += dur_rf1nscore + TIMESSI;

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
int calc_prep_phs (int* prep_pulse_phs, short** prep_pulse_phs_out, int* prep_pulse_grad,  int vsi_train_len, float vsi_Gmax)
{
/* This function adds a linear phase shift to the velocity selective pulses
in order to shift the velocity selectivity profile to a different velocity

Pre computes a table of phase waveforms for the FTVS pulses so that they 
ill target different velocities on different frames of the time series */

/* matlab code (this works fine in simulation)
	t = [0:length(B1)-1]*dt;
	phsvel = gambar*vel_target*cumsum(Gz(:).*t(:))*dt;
	B1sel = B1 .* exp(-1i*phsvel);
	*/
	
	/* GAMMA 26754 in (rad/s)/Gauss */
	float phase_val;		/* rad */
	float grad_val;			/* G/cm */
	float dt = 4e-6;  		/* sec */
	float delta_phs = 0.0; 	/* rad */
	int 	i, iv;
	int	DACMAX = 32766;
	float t = 0.0;
	int numVels = 0;

	numVels = nframes/vspectrum_Navgs;
		
	FILE *fID = fopen("vs_phasewaves.txt", "w");

	for (iv=0 ; iv<numVels; iv++){
		t=0.0;
		for (i=0; i<vsi_train_len; i++)
		{
			/* from DAC units to radians */
			phase_val = M_PI * (float)(prep_pulse_phs[i]) /  (float)FS_PI  ; 		
			/* from DAC units to G/cm */
			grad_val = vsi_Gmax * (float)(prep_pulse_grad[i]) / (float)DACMAX ;

			/* calc the phase gained by moving spins during the last dt interval */
			delta_phs += GAMMA * grad_val * vel_target * t * dt ;

			/* change the phase of the pulse accordingly */
			phase_val -=  delta_phs;

			/* phase unwrapping */
			phase_val = atan2( sin(phase_val), cos(phase_val));
			/* from radians to DAC  */	
			phase_val = phase_val * (float)(FS_PI) / M_PI;
			/* make sure they are even numbers and can be written as shorts later*/
			prep_pulse_phs_out[iv][i] = (short)(phase_val);
			prep_pulse_phs_out[iv][i] /= 2 ;
			prep_pulse_phs_out[iv][i] *= 2 ;

			fprintf(fID, "%d, ", prep_pulse_phs_out[iv][i]);
			t += dt;
		}
		fprintf(stderr, "\ncalc_prep_phs(): integration successful (%d points) for %f cm/s", vsi_train_len, vel_target);
		fprintf(fID, "\n");

		/* make sure the EOS bit is set at the end, by adding 1 to the last number*/ 
		
		prep_pulse_phs_out[iv][vsi_train_len - 1] += 1;
		fprintf(stderr, "\ncalc_prep_phs(): setting zeros and EOS bit successful");
		
		vel_target += vel_target_incr;
	}
	fclose(fID);
	return 1;
}	

/* function for playing prescan sequence : MOSTLY the same as the scan loop */
STATUS prescanCore() {

	int ttotal; 
	float arf1_var = 0.0;
	//float tmpmax= 1.0;

	/* initialize the rotation matrix */
	setrotate( tmtx0, 0 );
	
	/* prep 1 labeling/control type*/
	int ptype = 1;
	int pcasl_type = 0;  /* use the control case for the prescan */

	for (view = 1 - rspdda; view < rspvus + 1; view++) {
		ttotal = 0;

		/* play the ASL pre-saturation pulse to reset magnetization */
		if (presat_flag) {
			fprintf(stderr, "prescanCore(): playing asl pre-saturation pulse for frame %d, arm %d, shot %d (t = %d / %.0f us)...\n", framen, armn, shotn, ttotal, pitscan);
			ttotal += play_presat();
		}
		/* PCASL pulse after pre-sat pulse */
		if (pcasl_flag > 0  &&  framen >= nm0frames){
			fprintf(stderr, "prescanCore(): Playing PCASL pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
			ttotal += play_pcasl(pcasl_type, pcasl_tbgs1, pcasl_tbgs2, pcasl_pld);
		}
		/* prep1 (vsasl module) */
		if (prep1_id > 0 ) {
			fprintf(stderr, "prescanCore(): playing prep1 pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
			ttotal += play_aslprep(ptype, off_prep1ctlcore, off_prep1lblcore,  dur_prep1core, prep1_pld, prep1_tbgs1, prep1_tbgs2, prep1_tbgs3);
		}

		/* prep2 (vascular crusher module) */
		if (prep2_id > 0 ) {
			fprintf(stderr, "prescanCore(): playing prep2 pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
			ttotal += play_aslprep(ptype, off_prep2ctlcore, off_prep2lblcore, dur_prep2core, prep2_pld, prep2_tbgs1, prep2_tbgs2, prep2_tbgs3);
		}

		/* fat sup pulse */
		if (fatsup_mode > 0) {
			fprintf(stderr, "prescanCore(): playing fat sup pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
			ttotal += play_fatsup();
		}


		if (ro_type == 1) { /* FSE - play 90 deg. pulse with 0 phase */
			fprintf(stderr, "prescanCore(): playing 90deg FSE tipdown for prescan iteration %d...\n", view);
			ttotal += play_rf0(0);
		}	

		/* Load the DAB */	
		/*
		if (view < 1 || n < ndisdaqechoes) {
			fprintf(stderr, "prescanCore(): loaddab(&echo1, 0, 0, 0, 0, DABOFF, PSD_LOAD_DAB_ALL)...\n");
			loaddab(&echo1, 0, 0, 0, 0, DABOFF, PSD_LOAD_DAB_ALL);
		}
		else {
			fprintf(stderr, "prescanCore(): loaddab(&echo1, 0, 0, 0, %d, DABON, PSD_LOAD_DAB_ALL)...\n", view);
			loaddab(&echo1, 0, 0, 0, view, DABON, PSD_LOAD_DAB_ALL);
		}*/

		/* kill gradients */				
		/* setrotate( zmtx, 0 );*/

		for (echon=0; echon<opetl; echon++){
			
			if (echon<ndisdaqechoes)
			{ /* use only the data from the first echo*/
				fprintf(stderr, "prescanCore(): loaddab(&echo1, 0, 0, 0, %d, DABON, PSD_LOAD_DAB_ALL)...\n", view);
				loaddab(&echo1, 0, 0, DABSTORE, view, DABOFF, PSD_LOAD_DAB_ALL);
			}
			else
			{
				fprintf(stderr, "prescanCore(): loaddab(&echo1, 0, 0, 0, %d, DABOFF, PSD_LOAD_DAB_ALL)...\n", view);
				loaddab(&echo1, 0, 0, DABSTORE, view, DABON, PSD_LOAD_DAB_ALL);
			}
		
			if (ro_type == 1){ 
				/* FSE - CPMG */
				/* For FSE case, include variable flip angle refocusers
				The default is to use a constant refocuser flip angle*/
				
				arf1_var = a_rf1;
			
				if (echon==0){
					/*using a "stabilizer" for the first of the echoes in the train
						flip[0] = 90 + (vflip)/2         
					the rest of them are whatever the refocuser flip angle is: 
						flip[n] = vflip            
					eg1: opflip = 120
					-->	90x - 150y - 120y - 120y -120y ...
					eg2: opflip = 180
					-->	90x - 180y - 180y - 180y -180y ...  */

					if(doNonSelRefocus)
						arf1_var = (arf180ns + a_rf1ns)/2;
					else
						arf1_var = (arf180 + a_rf1)/2;
				}
				if(varflip) {
					/* variable flip angle refocuser pulses to get more signal 
					- linearly increasing schedule */
					/* arf1_var = a_rf1 + (float)echon*(arf180 - a_rf1)/(float)(opetl-1); */
					
					/* in VFA, the first refocusers are higher - trying to approximate that here*/
					/* if(echon==0) arf1_var = (arf180 + a_rf1)/2.0;  */
						    
					/* New approach: do a quadrative schedule with 
					the minimum of parabola occurring at one quarter of the way in the echo train 
					Note - the min value will be a_rf1ns (or opflip) */
	    			//arf1_var = ((float)(echon) - (float)(opetl)/4.0) * ((float)(echon) - (float)(opetl)/4.0);  /* shifted parabola */
					//tmpmax = ((float)(opetl) - (float)(opetl)/4.0) *  ((float)(opetl) - (float)(opetl)/4.0) ;    /* max value of the parabola */

					/* New: flip angle schedule is fourth order polynomial based on data by Zhao 1997 , DOI: 10.1002/mrm.27118
					flip angles in degrees: */
   					arf1_var = 0.0044*pow(echon+1,4)   - 0.2521*pow(echon+1,3) +   5.3544 *pow(echon+1,2) - 45.0296*(echon+1) + 158.0661;
					/* cap the RF amplitude at 150 degree pulse */
					if (arf1_var > 150) arf1_var = 150;;

					if(doNonSelRefocus)
					{
						// scale to relative scale of rho channel [0,1]
						arf1_var *= arf180ns / 180.0;
					
						//arf1_var *= (arf180ns - a_rf1ns) / tmpmax; /* scale to the range from min to 180 */
						//arf1_var += a_rf1ns;  /* shift up */

					}
					else
					{
						// scale to relative scale of rho channel [0,1]
						arf1_var *= arf180 / 180.0;
					
						//arf1_var *= (arf180 - a_rf1) / tmpmax; /* scale */
						//arf1_var += a_rf1;  /* shift up */

					}
					
					/* set the transmitter gain after the adjustments */
					if(doNonSelRefocus)
					{
						setiamp(arf1_var * MAX_PG_WAMP, &rf1ns,0);
						fprintf(stderr,"\nadjusting var flip ang: %f (arf180=%f)", arf1_var, arf180 ); 
					}
					else
					{
						setiamp(arf1_var * MAX_PG_WAMP, &rf1,0);
						fprintf(stderr,"\nadjusting var flip ang: %f (arf180=%f)", arf1_var, arf180 ); 
					}
				}
			}
		
			fprintf(stderr, "prescanCore(): Playing flip pulse for prescan iteration %d...\n", view);
			if (doNonSelRefocus)
				ttotal += play_rf1ns(90*(ro_type == 1));
			else
				ttotal += play_rf1(90*(ro_type == 1));

			/* set rotation matrix for each echo readout */
			setrotate( tmtxtbl[echon], 0 );

			fprintf(stderr, "prescanCore(): playing readout for prescan iteration %d , echo %d...\n", view, echon);
			ttotal += play_readout();

			/*restore gradients for exctiation pulse*/				
			setrotate( tmtx0, 0 );
		}


		fprintf(stderr, "prescanCore(): playing deadtime for prescan iteration %d...\n", view);
		play_deadtime(optr - ttotal);
			
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

/*------------------------------------------------------------------
		SCAN() function
-------------------------------------------------------------------*/
STATUS scan( void )
{ 
	if( psdinit() == FAILURE )
	{
		return rspexit();
	}

	int ttotal = 0;
	int rotidx;
	float calib_scale;
	int vspectrum_rep=0;
	int vel_target_num = 0;
	int i;
	/* variable phase waveform for velocity spectrum */
	short **s_phsbuffer; 

	int pcasl_type, prep1_type, prep2_type;

	float arf1_var = 0;
	//float tmpmax = 1.0;

	fprintf(stderr, "scan(): beginning scan (t = %d / %.0f us)...\n", ttotal, pitscan);

	/* VELOCITY SPECTRUM IMAGING:
	generate seqData objects to contain the phase waveform in HW memory 
	so we can be update the prep pulse phase waveforms inside the scan loop*/
	
	/*
	getWaveSeqDataWavegen(&sdTheta1, TYPTHETA, 0, 0, 0, PULSE_CREATE_MODE);	
	phsbuffer1_wf = (WF_HW_WAVEFORM_PTR)AllocNode(prep1_len*sizeof(int));
	*/
	
	/*-------------------------------------------------
	Calculations for velocity spectrum mode 1: 
	--------------------------------------*/
	if (doVelSpectrum==1 && prep1_id >0)
	{
		/* pre-calculate a table of phase waveforms for the theta channel */
		fprintf(stderr, "scan():  allocating space for table of phase waveforms \n");
		s_phsbuffer = 	(short **)AllocNode(nframes/2*sizeof(short *));
		for(i=0; i< nframes/2; i++){
			s_phsbuffer[i] = (short *)AllocNode(prep1_len*sizeof(short));
		}

		fprintf(stderr, "scan(): calling calc_prep_phs() to create phase table for prep 1 , prep1_len: %d, prep1_len: %d \n", prep1_len, prep1_len);
		calc_prep_phs(prep1_theta_lbl, s_phsbuffer, prep1_grad_lbl, prep1_len, prep1_gmax);
		fprintf(stderr, "scan(): ... done \n");
	}

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
	fprintf(stderr, "\nscan(): ndisdaqtrains = %d", ndisdaqtrains);
	for (disdaqn = 0; disdaqn < ndisdaqtrains; disdaqn++) {
		
		/* Calculate and play deadtime */
		fprintf(stderr, "scan(): playing TR deadtime for disdaq train %d (t = %d / %.0f us)...\n", disdaqn, ttotal, pitscan);
		if (doNonSelRefocus)
			ttotal += play_deadtime(optr - opetl * (dur_rf1nscore + TIMESSI + dur_seqcore + TIMESSI));
		else
			ttotal += play_deadtime(optr - opetl * (dur_rf1core + TIMESSI + dur_seqcore + TIMESSI));
		
		if (ro_type == 1) { /* FSE - play 90 deg. with 0 phase*/
			fprintf(stderr, "scan(): playing 90deg FSE tipdown for disdaq train %d (t = %d / %.0f us)...\n", disdaqn, ttotal, pitscan);
			play_rf0(0);
		}	
		
		/* Loop through echoes */
		for (echon = 0; echon < opetl+ndisdaqechoes; echon++) {
			fprintf(stderr, "scan(): playing flip pulse for disdaq train %d (t = %d / %.0f us)...\n", disdaqn, ttotal, pitscan);
			if (ro_type == 1) {/* FSE - CPMG */
				if (doNonSelRefocus)
					ttotal += play_rf1ns(90 );
				else
					ttotal += play_rf1(90 );
				}
			else
				//ttotal += play_rf1(0);
				ttotal += play_rf1(rfspoil_flag*117*(echon + ndisdaqechoes));

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


	/* loop through frames arms and shots 
	frames are the number images in the time series
	shots are the number of kzsteps in stack of spirals, and also the number of echoes in the echo train
	arms is the spiral z-rotations in SOS, or also the second axis rotation in SERIOS*/

	
	for (framen = 0; framen < nframes; framen++) {

		/* Velocity spectrum imaging
		----------------------------------------------------
		If we're doing velocity spectrum imaging, 
		we have to sweep through a number of target velocities
		or velocity encoding gradients
		incrementing every "vspectrum_Navgs" 
		*/	
		/* Method 1: FTVS velocity target using phase control */
		if (doVelSpectrum==1 && framen >= nm0frames){

			fprintf(stderr, "\nscan(): velocity spectrum Avgs. counter: %d", vspectrum_rep); 

			if (vspectrum_rep == vspectrum_Navgs){

				fprintf(stderr, "\n\nscan() updating phase waveform for prep1 pulse (movewaveimm) %d\n", vel_target_num);
				movewaveimm(s_phsbuffer[vel_target_num], &prep1thetalbl, (int)0, prep1_len, TOHARDWARE); 
				fprintf(stderr, "scan(): ... done \n");

				vspectrum_rep = 0;	
				vel_target_num++;
			}	

			/* do the same thing with the control pulse ? -- NO!! */

			vspectrum_rep++ ;
		}

		/* Method 2: If using BIR8 pulses - sweep through the venc gradients (options 2 and 3)*/
		if (doVelSpectrum>=2 && framen >= nm0frames){
			fprintf(stderr, "\nscan():Encoding velocity spectrum Avgs. counter: %d \n", vspectrum_rep); 
			if (vspectrum_rep == vspectrum_Navgs)
			{
				fprintf(stderr, "\n\nscan() velocity spectrum: updating for prep2 pulse for vel %f \n\n", vel_target);
				vspectrum_rep = 0;
				vspectrum_grad += prep2_delta_gmax;
			}	
			ia_prep2gradlbl = (int)ceil(vspectrum_grad / ZGRAD_max * (float)MAX_PG_WAMP);
			ia_prep2gradctl = (int)ceil(vspectrum_grad / ZGRAD_max * (float)MAX_PG_WAMP);
			setiamp(ia_prep2gradctl, &prep2gradctl, 0);
			setiamp(ia_prep2gradlbl, &prep2gradlbl, 0);
			vspectrum_rep++ ;

		}

		/* PCASL phase calibration
		-------------------------------------------------
		if we want to calibrate the phase correction to correct for off-resonance
		we increment the size of the phase steps between pulses.
		We will do this every 'pcasl_calib_frames' frames - must be an even number! 
		*/
		if (pcasl_flag	&& pcasl_calib) {
			nm0frames = 0;
			phs_cal_step = 2.0*M_PI / (float)(nframes) * (float)(pcasl_calib_frames);

			fprintf(stderr, "\nscan(): Phase calibration counter: %d \n", pcasl_calib_cnt); 
			
			if (pcasl_calib_cnt == pcasl_calib_frames ){
				fprintf(stderr, "\n\n scan() CALIBRATION: updating PCASL linear phase increment: %f (rads) and phase table\n\n", pcasl_delta_phs);
				pcasl_delta_phs += phs_cal_step;				
				update_pcasl_phases(pcasl_iphase_tbl, pcasl_delta_phs, MAXPCASLSEGMENTS);
				pcasl_calib_cnt = 0;
			}
			pcasl_calib_cnt++;
		}

		/* MRF ASL mode
		-------------------------------------------------
		MRF case:use the values in the mrf tables for the acquisition.  
		We update the values in every frame
		*/
		if (mrf_mode == 1 && doVelSpectrum<1){
			fprintf(stderr, "\n\n scan(): MRF MODE : updating schedule from mrf tables\n");
			
			presat_flag = 0;
			nm0frames = 0;

			tr_deadtime		= 4* (int)(mrf_deadtime[framen]* 1e6 / 4);
			pcasl_type		= mrf_pcasl_type[framen];
			pcasl_duration 		= 4*(int)(mrf_pcasl_duration[framen] * 1e6 / 4); 	/* seconds -> us, resolution of 4 us */
			pcasl_Npulses 		= 4*(int)(floor(pcasl_duration/pcasl_period / 4));
			pcasl_duration 		= pcasl_period * pcasl_Npulses; 	
			pcasl_pld		= 4*(int)(mrf_pcasl_pld[framen] * 1e6 / 4);

			pcasl_tbgs1 		= 0;
			pcasl_tbgs2 		= 0;
			
			prep1_type		= mrf_prep1_type[framen];
			prep1_pld		= 4*(int)(mrf_prep1_pld[framen]* 1e6 / 4); 

			prep1_tbgs1 		= 0;
			prep1_tbgs2 		= 0;
			prep1_tbgs3 		= 0;
			
			prep2_type		= mrf_prep2_type[framen];
			prep2_pld		= 4*(int)(mrf_prep2_pld[framen]* 1e6 /4);  

			prep2_tbgs1 		= 0;
			prep2_tbgs2 		= 0;
			prep2_tbgs3 		= 0;
		
		}
		else
		{
			/* if we're not doing MRF, 
			determine prep1 and prep2  type based on the switching  mode 
			*/		
			pcasl_type = determine_label_type(pcasl_mod, framen);
			prep1_type = determine_label_type(prep1_mod, framen);	
			prep2_type = determine_label_type(prep2_mod, framen);	

		}

		/* Quick Sanity Check 
		------------------------------------
		Whether we are doing an MRF schedule or the working on non-MRF mode: Are we playing the right thing at the right time? 
		*/
		fprintf(stderr,"\nUpdating %d frame of %d : \t deadtime: %d \tpcasl_type: %d \tpcasl_duration: %d \tpcasl_pld: %d \tprep1_type: %d \tprep1_pld: %d \tprep2_type: %d \tprep2_pld; %d\n", 
			framen, nframes, tr_deadtime, pcasl_type, pcasl_duration, pcasl_pld, prep1_type, prep1_pld,  prep2_type,  prep2_pld);
  
		for (armn = 0; armn < narms; armn++) {

			for (shotn = 0; shotn < opnshots; shotn++) {

				/* B1 calibration:
				-------------------
				set amplitudes for rf calibration modes */
				calib_scale = (float)framen / (float)(nframes - 1);
				if (rf1_b1calib) {
					fprintf(stderr, "scan():  rf1_b1calib: setting ia_rf1 = %d\n", 2*(int)ceil(calib_scale*(float)ia_rf1 / 2.0));
					setiamp(2*(int)ceil(calib_scale*(float)ia_rf1 / 2.0), &rf1, 0);
				}
				if (prep1_b1calib) {
					fprintf(stderr, "scan(): prep1_b1calib: setting ia_prep1rholbl/ctl = %d\n", 2*(int)ceil(calib_scale*(float)ia_prep1rholbl / 2.0));
					setiamp(2*(int)ceil(calib_scale*(float)ia_prep1rholbl / 2.0), &prep1rholbl, 0);
					setiamp(2*(int)ceil(calib_scale*(float)ia_prep1rhoctl / 2.0), &prep1rhoctl, 0);
				}
				if (prep2_b1calib) {
					fprintf(stderr, "scan(): prep2_b1calib: setting ia_prep2lbl/ctl = %d\n", 2*(int)ceil(calib_scale*(float)ia_prep2rholbl / 2.0));
					setiamp(2*(int)ceil(calib_scale*(float)ia_prep2rholbl / 2.0), &prep2rholbl, 0);
					setiamp(2*(int)ceil(calib_scale*(float)ia_prep2rhoctl / 2.0), &prep2rhoctl, 0);
				}

				/* play TR deadtime */
				ttotal += play_deadtime(tr_deadtime);

				fprintf(stderr, "scan(): ************* beginning loop for frame %d, shot %d, arm %d *************\n", framen, shotn, armn);

				/* Cardiac gating option - prior to presaturation pulse */
				if (do_cardiac_gating) {
                	settrigger(TRIG_ECG, 0);
                	fprintf(stderr, "scan(): Setting ECG Trigger\n");
				}

				/* play the ASL pre-saturation pulse to reset magnetization */
				if (presat_flag) {
					fprintf(stderr, "scan(): playing asl pre-saturation pulse for frame %d, arm %d, shot %d (t = %d / %.0f us)...\n", framen, armn, shotn, ttotal, pitscan);
					ttotal += play_presat();
				}
				/* PCASL pulse after pre-sat pulse */
				if (pcasl_flag > 0  &&  framen >= nm0frames){
					fprintf(stderr, "scan(): Playing PCASL pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
					/* ttotal += play_pcasl(pcasl_lbltbl[framen], pcasl_tbgs1tbl[framen], pcasl_tbgs2tbl[framen], pcasl_pldtbl[framen]); * will use this for arbitrary labeling schedules */
					ttotal += play_pcasl(pcasl_type, pcasl_tbgs1, pcasl_tbgs2, pcasl_pld);
				}
				/* prep1 (vsasl module) */
				if (prep1_id > 0 && framen >= nm0frames) {
					fprintf(stderr, "scan(): playing prep1 pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
					ttotal += play_aslprep(prep1_type, off_prep1ctlcore, off_prep1lblcore, dur_prep1core, prep1_pld, prep1_tbgs1, prep1_tbgs2, prep1_tbgs3);
				}

				/* prep2 (vascular crusher module - or T2 weigthing module) */
				if (prep2_id > 0 ) {
					fprintf(stderr, "scan(): playing prep2 pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
					ttotal += play_aslprep(prep2_type, off_prep2ctlcore, off_prep2lblcore, dur_prep2core, prep2_pld, prep2_tbgs1, prep2_tbgs2, prep2_tbgs3);
				}

				/* fat sup pulse */
				if (fatsup_mode > 0) {
					fprintf(stderr, "scan(): playing fat sup pulse for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
					ttotal += play_fatsup();
				}

				if (ro_type == 1) { /* FSE - play 90 */
					fprintf(stderr, "scan(): playing 90deg FSE tipdown for frame %d, shot %d (t = %d / %.0f us)...\n", framen, shotn, ttotal, pitscan);
					play_rf0(0);
				}	

				/* play disdaq echoes */
				for (echon = 0; echon < ndisdaqechoes; echon++) {
					fprintf(stderr, "scan(): playing flip pulse for frame %d, shot %d, disdaq echo %d (t = %d / %.0f us)...\n", framen, shotn, echon, ttotal, pitscan);
					if (ro_type == 1) {/* FSE - CPMG */
						if (doNonSelRefocus)
							ttotal += play_rf1ns(90 );
						else
							ttotal += play_rf1(90 );
					}
					else
						ttotal += play_rf1(rfspoil_flag*117*echon);

					fprintf(stderr, "scan(): playing deadtime in place of readout for frame %d, shot %d, disdaq echo %d (%d us)...\n", framen, shotn, echon, dur_seqcore);
					ttotal += play_deadtime(dur_seqcore);
				}

				/* play the actual echo train */
				for (echon = 0; echon < opetl; echon++) {
					
					fprintf(stderr, "scan(): playing flip pulse for frame %d, shot %d, echo %d (t = %d / %.0f us)...\n", framen, shotn, echon, ttotal, pitscan);
					if (ro_type == 1){ /* FSE - CPMG */
						
						/* The default is to use a constant refocuser flip angle*/
						arf1_var = a_rf1;

						if (echon==0){
							/*using a "stabilizer" for the first of the echoes in the train
								flip[0] = 90 + (vflip)/2         
							the rest of them are whatever the refocuser flip angle is: 
								flip[n] = vflip            
							eg1: 
								90x - 150y - 120y - 120y -120y ...
							eg2: 
								90x - 180y - 180y - 180y -180y ...  */

							arf1_var = (arf180 + a_rf1)/2;
							
							if(doNonSelRefocus)
								arf1_var = (arf180ns + a_rf1ns)/2;
						}

						if(varflip ) {
							/* variable flip angle refocuser pulses to get more signal 
							   - linearly increasing schedule */
							/* arf1_var = a_rf1 + (float)echon*(arf180 - a_rf1)/(float)(opetl-1); */

							/* in VFA, the first refocusers are higher - trying to approximate that here*/
							/* if(echon==0) arf1_var = (arf180 + a_rf1)/2.0;  */

							/* New approach: do a quadratic schedule with 
							   the minimum of parabola is opflip and will occur at one quarter of the way in the echo train  
							   y = (x-xmax/4)^2 */
							//arf1_var = ((float)(echon) - (float)(opetl)/4.0) * ((float)(echon) - (float)(opetl)/4.0);  /* shifted parabola */
							//tmpmax = ((float)(opetl) - (float)(opetl)/4.0) *  ((float)(opetl) - (float)(opetl)/4.0) ;    /* max value of the parabola */
							
							/* New: flip angle schedule is fourth order polynomial based on data by Zhao 1997 , DOI: 10.1002/mrm.27118
							flip angles in degrees: */
   							arf1_var = 0.0044*pow(echon+1,4)   - 0.2521*pow(echon+1,3) +   5.3544 *pow(echon+1,2) - 45.0296*(echon+1) + 158.0661;

							if(doNonSelRefocus)
							{
								// scale to relative scale of rho channel [0,1]
								arf1_var *= a_rf1ns / 180.0;

								//arf1_var *= (arf180ns - a_rf1ns) / tmpmax; /* scale */
								//arf1_var += a_rf1ns;  /* shift up */


								/* but we cap it at 150 degree pulse */
								if (arf1_var > arf180ns * 0.833 ) arf1_var = arf180ns * 0.833;;

								/* set the transmitter gain after the adjustments */
								setiamp(arf1_var * MAX_PG_WAMP, &rf1ns,0);
								fprintf(stderr,"\nadjusting var flip ang: %f (arf180=%f)", arf1_var, arf180 ); 
							}
							else
							{
								// scale to relative scale of rho channel [0,1]
								arf1_var *= a_rf1 / 180.0;

								//arf1_var *= (arf180 - a_rf1) / tmpmax; /* scale */
								//arf1_var += a_rf1;  /* shift up */

								/* but we cap it at 150 degree pulse */
								if (arf1_var > arf180 * 0.833) arf1_var = arf180 * 0.833;;

								/* set the transmitter gain after the adjustments */
								setiamp(arf1_var * MAX_PG_WAMP, &rf1,0);
								fprintf(stderr,"\nadjusting var flip ang: %f (arf180=%f)", arf1_var, arf180 ); 
							}

						}

						if (doNonSelRefocus)
							ttotal += play_rf1ns(90 * (ro_type == 1) );
						else
							ttotal += play_rf1(90 * (ro_type == 1));
					}
					else  /* SPGR and SSFP cases */
					{
						ttotal += play_rf1(rfspoil_flag*117*(echon + ndisdaqechoes));
						setphase(rfspoil_flag*117*(echon + ndisdaqechoes), &echo1, 0);
					}

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
					if (mrf_mode>0){
						rotidx += framen*narms*opnshots*opetl;
					}

					if (kill_grads)
						setrotate( zmtx, 0 );
					else
						setrotate( tmtxtbl[rotidx], 0 );

					fprintf(stderr, "scan(): playing readout for frame %d, shot %d, echo %d (%d us)...\n", framen, shotn, echon, dur_seqcore);
					fprintf(stderr, "scan(): rotation matrix index=  %d, mrf_mode= %d, spi_mode = %d\n", rotidx, mrf_mode, spi_mode);
					fprintf(stderr, "rotation tmtxtbl entries for this echo : \n");
					for (i=0; i<9; i++){
						fprintf(stderr, "\t%ld", tmtxtbl[rotidx][i]);
					}
					fprintf(stderr, "\n");

					ttotal += play_readout();

					/* Reset the rotation matrix */
					setrotate( tmtx0, 0 );
				}

			}
		}
	}

	/* clear memory for the velocity spectrum buffers - if needed*/
	FreeNode(s_phsbuffer);

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
	FILE *fID_ktraj_all = fopen("ktraj_all.txt", "w");

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
	float kzmax = (spi_mode == 0) * (float)(kz_acc * opetl * opnshots) / ((float)opfov/10.0) / 2.0; /* max kz radius (1/cm), 0 if SPI */

	/* declare temporary variables */
	float gx_area, gy_area;
	float tmp_area, tmp_a;
	int tmp_pwa, tmp_pw, tmp_pwd;
	float kxn, kyn;
	
	/* calculate FOV coefficients */
	/* these calculations are a bit suspect 
	F0 = 1.1*(1.0/vds_acc1 / (float)narms * (float)opfov / 10.0);
	F1 = 1.1*(2*pow((float)opfov/10.0,2)/opxres *(1.0/vds_acc1 - 1.0/vds_acc0)/(float)narms);
	F2 = 0;
	...*/
	/* LHG: the above code doesn't work quite right - looking at trajectories in WTools and plotting them
	they are not making sense to me.
	.... Let's try something simpler */
	F0 = vds_acc0 * (float)opfov/10.0 / (float)narms ;
	F1 =(vds_acc1 * (float)opfov/10.0 / (float)narms - F0) / kxymax  ; 
	F2 = 0;
	
	if (ro_type == 1) { /* FSE and bSSFP - spiral in-out */
		F0 /= 2;
		F1 /= 2;
		F2 /= 2;
	}
	F[0] = F0;
	F[1] = F1;
	F[2] = F2;
	
	fprintf(stderr, "genspiral(): FOV coefficients are %0.2f %0.2f\n", F0, F1);

	/* generate the vd-spiral out gradients */	
	calc_vds(SLEWMAX, GMAX, dt, dt, 1, F, 2, kxymax, MAXWAVELEN, &gx_vds, &gy_vds, &n_vds);

	/* calculate gradient ramp-down */
	n_rmp = ceil(fmax(fabs(gx_vds[n_vds - 1]), fabs(gy_vds[n_vds - 1])) / SLEWMAX / dt);
	
	/* the ramp down may be too fast - lots of gradient errors.  slowing it down here */
	n_rmp *=2;

	gx_rmp = (float *)malloc(n_rmp*sizeof(float));
	gy_rmp = (float *)malloc(n_rmp*sizeof(float));
	for (n = 0; n < n_rmp; n++) {
		gx_rmp[n] = gx_vds[n_vds - 1]*(1 - (float)n/(float)n_rmp);
		gy_rmp[n] = gy_vds[n_vds - 1]*(1 - (float)n/(float)n_rmp);
	}

	gx_area = 1e6 * dt * (fsumarr(gx_vds, n_vds) + fsumarr(gx_rmp, n_rmp));
	gy_area = 1e6 * dt * (fsumarr(gy_vds, n_vds) + fsumarr(gy_rmp, n_rmp));
	tmp_area = fmax(fabs(gx_area), fabs(gy_area)); /* get max abs area */

	/* calculate optimal trapezoid kspace rewinder */
	amppwgrad(tmp_area, GMAX, 0, 0, ZGRAD_risetime, 0, &tmp_a, &tmp_pwa, &tmp_pw, &tmp_pwd);
	n_rwd = ceil((float)(tmp_pwa + tmp_pw + tmp_pwd)/(float)GRAD_UPDATE_TIME);
	gx_rwd = (float *)malloc(n_rwd*sizeof(float));
	gy_rwd = (float *)malloc(n_rwd*sizeof(float));
	for (n = 0; n < n_rwd; n++) {
		gx_rwd[n] = -gx_area/tmp_area*tmp_a*trap(n*1e6*dt,0.0,tmp_pwa,tmp_pw);
		gy_rwd[n] = -gy_area/tmp_area*tmp_a*trap(n*1e6*dt,0.0,tmp_pwa,tmp_pw);
	}

	/* calculate total points in spiral + rewinder */
	n_sprl = n_vds + n_rmp + n_rwd;
	gx_sprlo = (float *)malloc(n_sprl*sizeof(float));
	gy_sprlo = (float *)malloc(n_sprl*sizeof(float));
	gx_sprli = (float *)malloc(n_sprl*sizeof(float));
	gy_sprli = (float *)malloc(n_sprl*sizeof(float));

	/* concatenate gradients to form spiral out */
	gx_tmp = (float *)malloc((n_vds + n_rmp)*sizeof(float));
	gy_tmp = (float *)malloc((n_vds + n_rmp)*sizeof(float));
	catArray(gx_vds, n_vds, gx_rmp, n_rmp, 0, gx_tmp);
	catArray(gy_vds, n_vds, gy_rmp, n_rmp, 0, gy_tmp);
	catArray(gx_tmp, n_vds + n_rmp, gx_rwd, n_rwd, 0, gx_sprlo);
	catArray(gy_tmp, n_vds + n_rmp, gy_rwd, n_rwd, 0, gy_sprlo);
	free(gx_tmp);
	free(gy_tmp);

	/* reverse the gradients to form spiral in */
	reverseArray(gx_sprlo, n_sprl, gx_sprli);
	reverseArray(gy_sprlo, n_sprl, gy_sprli);

	if (ro_type == 2) { /* SPGR - spiral out */
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
	else { /* FSE & bSSFP - spiral in-out */
		
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
	kxn = 0.0;
	kyn = 0.0;
	for (n = 0; n < grad_len; n++) {
		/* integrate gradients */
		kxn += gam * gx[n] * dt;
		kyn += gam * gy[n] * dt;
		if (n > acq_offset-1 && n < acq_offset + acq_len)
			fprintf(fID_ktraj, "%f \t%f \t%f\n", kxn, kyn, kzmax);
		fprintf(fID_ktraj_all, "%f \t%f \t%f\n", kxn, kyn, kzmax);
		
		/* convert gradients to integer units */
		Gx[n] = 2*round(MAX_PG_WAMP/XGRAD_max * gx[n] / 2.0);
		Gy[n] = 2*round(MAX_PG_WAMP/YGRAD_max * gy[n] / 2.0);
	}

	fclose(fID_ktraj);
	fclose(fID_ktraj_all);

	return SUCCESS;
}

/* Function to read external waveform file for the gradients.  use G/cm */
int readGrads(int id, float scale)
{
	/* Declare variables */
	char fname[80];
	FILE *fID;
	char buff[200];
	int n;
	double gxval, gyval;

	float gx[MAXWAVELEN];
	float gy[MAXWAVELEN];
		
	float dt = GRAD_UPDATE_TIME*1e-6; /* raster time (s) */
	float gam = 4258; /* gyromagnetic ratio (Hz/G) */

	if (id == 0) {
		/* Set all values to zero and return */
		for (n = 0; n < MAXWAVELEN; n++) {
			gx[n] = 0;
			gy[n] = 0;
		}
		fprintf(stderr, "readGrads(): file not read - files set to zero %s\n", fname);
		return 1;
	}

	/* Read in gradient waveformt from text file */
	sprintf(fname, "RO_grads/%05d/grads.txt", id);
	fprintf(stderr, "readGrads(): opening %s...\n", fname);
	fID = fopen(fname, "r");

	/* Check if file was opened successfully */
	if (fID == 0) {
		fprintf(stderr, "readGrads(): failure opening %s\n", fname);
		return 0;
	}

	/* Loop through points in rho file */
	n = 0;
	while (fgets(buff, 200, fID)) {
		
		sscanf(buff, "%lf %lf", &gxval, &gyval);
		gx[n] = gxval * scale;
		gy[n] = gyval * scale;

		/* convert gradients to integer units */
		Gx[n] = 2*round(MAX_PG_WAMP/XGRAD_max * gx[n] / 2.0);
		Gy[n] = 2*round(MAX_PG_WAMP/YGRAD_max * gy[n] / 2.0);

		n++;
	}
	fclose(fID);

	grad_len = n;

	
	/* Integrate gradients to calculate kspace and write it to a file */
	float kxn = 0.0;
	float kyn = 0.0;
	FILE *fID_ktraj = fopen("ktraj.txt", "w");
	FILE *fID_ktraj_all = fopen("ktraj_all.txt", "w");

	for (n = 0; n < grad_len; n++) {
		/* integrate gradients */
		kxn += gam * gx[n] * dt;
		kyn += gam * gy[n] * dt;
	
		fprintf(fID_ktraj, "%f \t%f \t0.0\n", kxn, kyn);
		fprintf(fID_ktraj_all, "%f \t%f \t0.0\n", kxn, kyn);
	
	}

	fclose(fID_ktraj);
	fclose(fID_ktraj_all);

	return 1;
}



int genviews() {

	/* Declare values and matrices */
	FILE* fID_kviews = fopen("kviews.txt","w");
	int rotidx, armn, shotn, echon, n;
	float rz, dz;
	float phi = 0.0;
	float theta = 0.0;
	float Rz[9], Rtheta[9], Rphi[9], Tz[9];
	float T_0[9], T[9];

	FILE* pRotFile;
	char fname[80];
	int matnum = 0;
	char textline[256];
	float element; 
	int mrf_nframes = 1;
	int nfr =0;

	fprintf(stderr, "genviews...():\n");

	if(mrf_mode > 0) mrf_nframes=nframes;

	if (spi_mode>=3){
		/* external file with rotations*/
		sprintf(fname, "RO_grads/%05d/myrotmats.txt", grad_id);
		pRotFile=fopen(fname,  "r");

		/* Check if rotation mats  file was read successfully */
		if (pRotFile == 0) {
			fprintf(stderr, "genviews(): failure opening myrotmats.txt\n");
			return 0;
		}

		/* Loop through the text lines in  file */
		fprintf(stderr, "genviews(): reading myrotmats.txt\n");
		while (fgets(textline, sizeof(textline), pRotFile)) {
			/* each row is a matrix*/
			for(n=0; n<9; n++) {
				sscanf(textline, "%f", &element);
				Tex[matnum][n] = element;
				/* fprintf(stderr, "\t%0.2f", element);*/
			}	
			matnum++;
		}
		fclose(pRotFile);
	}

	/* Initialize z translation to identity matrix */
	eye(Tz, 3);

	/* Get original transformation matrix */
	for (n = 0; n < 9; n++) T_0[n] = (float)rsprot[0][n] / MAX_PG_WAMP;
	orthonormalize(T_0, 3, 3);

	/* Loop through all views */
	for(nfr=0; nfr < mrf_nframes; nfr++){
		fprintf(stderr, "genviews(): rotation table tmtxtbl[][] entries for frame %d : \n", nfr);
		for (armn = 0; armn < narms; armn++) {
			for (shotn = 0; shotn < opnshots; shotn++) {
				for (echon = 0; echon < opetl; echon++) {

					/* calculate view index */
					rotidx = armn*opnshots*opetl + shotn*opetl + echon;
					if(mrf_mode >0){
						rotidx += narms*opnshots*opetl*nfr;
					}

					/* Set the z-axis rotation angles and kz step (as a fraction of kzmax) */ 
					rz = 2* M_PI * (float)armn / (float)narms;
					phi = 0.0;
					theta = 0.0;
					dz = 0.0;

					/* the spiral in-out case rotates by only 90 degreesq  -
					This happens in FSE and SSFP readouts*/
					if (ro_type != 2) 
						rz /= 2;
					
					switch (spi_mode) {
						case 0: /* SOS */
							phi = 0.0;
							theta = 0.0;
							dz = 2.0/(float)opetl * (center_out_idx(opetl,echon) - 1.0/(float)opnshots*center_out_idx(opnshots,shotn)) - 1.0;
							break;
						case 1: /* 2D TGA */
							phi = 0.0;
							theta = 2* M_PI /phi2D * (shotn*opetl + echon);
							
							/* test: use the arms, instead of the shots to advance the angle*/
							theta = 2* M_PI /phi2D * (armn*opetl + echon);
							
							dz = 0.0;
							break;
						case 2: /* 3D TGA */

							/* using the fiboancci sphere formulas */
							theta = (float)(shotn*opetl + echon)*2*M_PI / phi2D; /* polar angle */
							phi = acos(1 - 2*(float)(shotn*opetl + echon)/(float)(opnshots*opetl)); /* azimuthal angle */
							
							/* test: use the arms, instead of the shots to advance the angle*/
							theta = (float)(armn*opetl + echon)*2*M_PI / phi2D; /* polar angle */
							phi = acos(1 - 2*(float)(armn*opetl + echon)/(float)(narms*opetl)); /* azimuthal angle */
							
							if (mrf_mode>0){
								theta += prev_theta;
								phi += prev_phi;
							}
							/* theta = acos(fmod(echon*phi3D_1, 1.0));  */
							/* phi = 2.0*M_PI * fmod(echon*phi3D_2, 1.0); */
							dz = 0.0;
							break;
						
					}

					/* Calculate the transformation matrices */
					Tz[8] = dz;
					genrotmat('z', rz, Rz);
					genrotmat('x', theta, Rtheta);
					genrotmat('z', phi, Rphi);

					/* Multiply the transformation matrices */
					multmat(3,3,3,T_0,Tz,T); /* kz scale T = T_0 * Tz */
					multmat(3,3,3,Rz,T,T); /* z rotation (arm-to-arm) T = Rz * T */
					multmat(3,3,3,Rtheta,T,T); /* polar angle rotation T = Rtheta * T */
					multmat(3,3,3,Rphi,T,T); /* azimuthal angle rotation T = Rphi * T */

					/* if the rotations are external we re-calculate 
					the transformation using the values from the file instead*/
					if (spi_mode>=3){
						multmat(3,3,3,Tex[shotn*opetl + echon],T_0,T);
					}

					/* Save the matrix to file and assign to tmtxtbl[][] - the table of rotation  matrices */
					fprintf(fID_kviews, "%d \t%d \t%d \t%f \t%f \t", armn, shotn, echon, rz, dz);	
					for (n = 0; n < 9; n++) {
						fprintf(fID_kviews, "%f \t", T[n]);
						tmtxtbl[rotidx][n] = (long)round(MAX_PG_WAMP*T[n]);
					}
					fprintf(fID_kviews, "\n");
					
					/* debugging : 
					fprintf(stderr,"rotidx ; %d -", rotidx);
					for (n=0; n<9; n++){
						fprintf(stderr, "\t%ld",tmtxtbl[rotidx][n]);
					}
					fprintf(stderr, "\n");
					*/
				}

				
			}
		}
		/* in MRF mode -  we use different rotations in each frame.
		Increment the first of the rotations by the last rotation in the previous frame */
		if (mrf_mode >0) {
			/* prev_theta = (float)(opnshots*opetl*narms*(nfr + 1))*phi3D_1 *2*M_PI / phi2D;  */
			/* prev_phi = acos(1 - 2*(float)(opnshots*opetl*narms*(nfr + 1))/(float)(opnshots*opetl));  */
			prev_theta = theta;
			prev_phi = phi;
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

/* assign default values to the MRF table*/
int init_mrf(	
	float 	*mrf_deadtime, 
	int 	*mrf_pcasl_type,
	float 	*mrf_pcasl_duration,
	float 	*mrf_pcasl_pld,
	int		*mrf_prep1_type,
	float 	*mrf_prep1_pld, 
	int 	*mrf_prep2_type,
	float 	*mrf_prep2_pld
	)
{
	int n;
	for (n=0; n < nframes; n++){
		mrf_deadtime[n] = tr_deadtime;

		mrf_pcasl_type[n] = pow(-1,n);
		mrf_pcasl_pld[n] = pcasl_pld;
		mrf_pcasl_duration[n] = pcasl_duration;

		mrf_prep1_type[n] = pow(-1,n)  * !(prep1_id==0); /* puts in zeros if there is no pulse id specified*/
		mrf_prep1_pld[n] = prep1_pld ; /* puts in zeros if there is no pulse id specified*/

		mrf_prep2_type[n] = !(prep2_id==0);  				/*always 1 if there is a pulse id specified*/
		mrf_prep2_pld[n] = prep2_pld ;   /* puts in zeros if there is no pulse id specified*/
	}
	return 1;
}
/*read all mrf schedule of events from specified values in a single schedule file
the format is in 7 columns as follows 
deadtime	dopcasl	 pcasl_pld	doprep1	prep1_pld doprep2	prep2_pld */
int read_mrf_fromfile(
	int 	file_id, 
	float 	*mrf_deadtime, 
	int 	*mrf_pcasl_type,
	float 	*mrf_pcasl_duration,
	float 	*mrf_pcasl_pld,
	int 	*mrf_prep1_type,
	float 	*mrf_prep1_pld, 
	int 	*mrf_prep2_type,
	float 	*mrf_prep2_pld
	)
{
	int n;
	char fname[80];	
	char line[500];
	FILE *fID;

	float pc, p1, p2;
	float dt, pcd, pct, p1t, p2t;

	sprintf(fname, "mrfasl_schedules/%05d/mrf_schedule.txt", file_id);
	fprintf(stderr, "read_mrf_fromfile(): opening %s...\n", fname);
	fID = fopen(fname, "r");

	/* Check if rho file was opened successfully */
	if (fID == 0) {
		fprintf(stderr, "read_mrf_fromfile(): failure opening %s\n", fname);
		return 0;
	}

	/* Loop through the rows of number in the file */
	n = 0;
	while (fgets(line, 500, fID)) {
		
		sscanf(line, "%f %f %f %f %f %f %f %f", &dt, &pc , &pcd, &pct, &p1, &p1t, &p2, &p2t);
		mrf_deadtime[n] 	= dt;
		mrf_pcasl_type[n] 	= (int)pc;
		mrf_pcasl_duration[n] = pcd;
		mrf_pcasl_pld[n] 	= pct;
		mrf_prep1_type[n] 	= (int)p1;
		mrf_prep1_pld[n] 	= p1t;
		mrf_prep2_type[n] 	= (int)p2;
		mrf_prep2_pld[n] 	= p2t;

		fprintf(stderr,"\nreading MRF  \tframe: %d \t deadtime: %f \tpcasl_type: %d \tpcasl_duration: %f \tpcasl_pld: %f \tprep1_type: %d \tprep1_pld: %f \tprep2_type: %d \tprep2_pld; %f", 
			n, mrf_deadtime[n], mrf_pcasl_type[n], mrf_pcasl_duration[n], mrf_pcasl_pld[n], mrf_prep1_type[n], mrf_prep1_pld[n],  mrf_prep2_type[n],  mrf_prep2_pld[n]);

		n++;
	}
	fclose(fID);
 
 	nframes = n;

	return 1;
}

float calc_sinc_B1(float cyc_rf, int pw_rf, float flip_rf) {

	int M = 1001;
	int n;
	float w[M], x[M];
	float area = 0.0;

	/* Create an M-point symmetrical Hamming window */
	for (n = 0; n < M; n++) {
		w[n] = 0.54 - 0.46*cos( 2*M_PI*n / (M-1) );
	}	

	/* Create a sinc pulse */
	for (n = -(M-1)/2; n < (M-1)/2 + 1; n++) {
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

/* LHG 12/6/12 : compute the linear phase increment of the PCASL pulses */
int calc_pcasl_phases(int *iphase_tbl, float  myphase_increment, int nreps)
{
	int     n;
	double  rfphase; 

	fprintf(stderr,"\nUpdating PCASL phase table .... "); 
	rfphase = 0.0;

	for (n=0; n<nreps; n++)
	{
		rfphase += myphase_increment;
		/* wrap phase to (-pi,pi) range */
		rfphase = atan2 (sin(rfphase), cos(rfphase));
		/* translate to DAC units */
		iphase_tbl[n] = (int)(rfphase/M_PI * (float)FS_PI);
	}
	fprintf(stderr,"Done .\n "); 
	return 1;
}               
        

/*in the absence of MRF schedule files, configure the PCASL label and control schedules , including BGS pulses
these timings will be in us units ----- unused*/
int make_pcasl_schedule(int *pcasl_lbltbl, int *pcasl_pldtbl, int *pcasl_tbgs1tbl, int *pcasl_tbgs2tbl)
{
	int framen=0;
	int toggle = 1;

	sprintf(tmpstr, "pcasl_pldtbl");
	for (framen = 0; framen < nframes; framen++)
		pcasl_pldtbl[framen] = (framen >= nm0frames) * pcasl_pld ; /* conversion to us */

	sprintf(tmpstr, "pcasl_lbltbl");
	for (framen = 0; framen < nframes; framen++){
		pcasl_lbltbl[framen]= toggle;
		toggle = -toggle;
	}

	sprintf(tmpstr, "pcasl_tbgs1tbl");
	for (framen = 0; framen < nframes; framen++)
		pcasl_tbgs1tbl[framen] = (framen >= nm0frames) * pcasl_tbgs1; /* conversion to us */

	sprintf(tmpstr, "pcasl_tbgs2tbl");
	for (framen = 0; framen < nframes; framen++)
		pcasl_tbgs2tbl[framen] = (framen >= nm0frames) * pcasl_tbgs2; /* coversion to us */

	return 1;
}	



int write_scan_info() {

	FILE *finfo = fopen("scaninfo.txt","w");
	fprintf(finfo, "Rx parameters:\n");
	fprintf(finfo, "\t%-50s%20f %s\n", "X/Y FOV:", (float)opfov/10.0, "cm");
	fprintf(finfo, "\t%-50s%20d \n", "Matrix size:", opxres);
	fprintf(finfo, "\t%-50s%20f %s\n", "3D slab thickness:", (float)opslquant*opslthick/10.0, "cm"); 	

	fprintf(finfo, "Hardware limits:\n");
	fprintf(finfo, "\t%-50s%20f %s\n", "Max gradient amplitude:", GMAX, "G/cm");
	fprintf(finfo, "\t%-50s%20f %s\n", "Max slew rate:", SLEWMAX, "G/cm/s");

	fprintf(finfo, "Readout parameters:\n");
	switch (ro_type) {
		case 1: /* FSE */
			fprintf(finfo, "\t%-50s%20s\n", "Readout type:", "FSE");
			fprintf(finfo, "\t%-50s%20f %s\n", "Flip (inversion) angle:", opflip, "deg");
			fprintf(finfo, "\t%-50s%20f %s\n", "Echo time:", (float)opte*1e-3, "ms");
			fprintf(finfo, "\t%-50s%20d \n", "variable FA flag:", varflip );
			fprintf(finfo, "\t%-50s%20d \n", "Non-selective rect pulse refocuser ", doNonSelRefocus );		
			break;
		case 2: /* SPGR */
			fprintf(finfo, "\t%-50s%20s\n", "Readout type:", "SPGR");
			fprintf(finfo, "\t%-50s%20f %s\n", "Flip angle:", opflip, "deg");
			fprintf(finfo, "\t%-50s%20f %s\n", "Echo time:", (float)opte*1e-3, "ms");
			fprintf(finfo, "\t%-50s%20f %s\n", "ESP (short TR):", (float)esp*1e-3, "ms");
			fprintf(finfo, "\t%-50s%20s\n", "RF phase spoiling:", (rfspoil_flag) ? ("on") : ("off"));	
			break;
		case 3: /* bSSFP */
			fprintf(finfo, "\t%-50s%20s\n", "Readout type:", "bSSFP");
			fprintf(finfo, "\t%-50s%20f %s\n", "Flip angle:", opflip, "deg");
			fprintf(finfo, "\t%-50s%20f %s\n", "Echo time:", (float)opte*1e-3, "ms");
			fprintf(finfo, "\t%-50s%20f %s\n", "ESP (short TR):", (float)esp*1e-3, "ms");
			break;
	}
	fprintf(finfo, "\t%-50s%20f %s\n", "Shot interval (long TR):", (float)optr*1e-3, "ms");
	fprintf(finfo, "\t%-50s%20d\n", "ETL:", opetl);
	fprintf(finfo, "\t%-50s%20d\n", "Number of frames:", nframes);
	fprintf(finfo, "\t%-50s%20d\n", "Number of shots:", opnshots);
	fprintf(finfo, "\t%-50s%20d\n", "Number of spiral arms:", narms);
	fprintf(finfo, "\t%-50s%20d\n", "Number of disdaq echo trains:", ndisdaqtrains);
	fprintf(finfo, "\t%-50s%20d\n", "Number of disdaq echoes:", ndisdaqechoes);
	fprintf(finfo, "\t%-50s%20f %s\n", "Crusher area factor:", crushfac, "% kmax");
	fprintf(finfo, "\t%-50s%20s\n", "Flow compensation:", (flowcomp_flag) ? ("on") : ("off"));	
	if (kill_grads == 1)
			fprintf(finfo, "\t%-50s%20s\n", "Spiral readout:", "off (FID only)");
	else {
		switch (spi_mode) {
			case 0: /* SOS */
				fprintf(finfo, "\t%-50s%20s\n", "Projection mode:", "SOS");
				fprintf(finfo, "\t%-50s%20f\n", "kz acceleration (SENSE) factor:", kz_acc);
				break;
			case 1: /* 2DTGA */
				fprintf(finfo, "\t%-50s%20s\n", "Projection mode:", "2DTGA");
				break;
			case 2: /* 3DTGA */
				fprintf(finfo, "\t%-50s%20s\n", "Projection mode:", "3DTGA");
				break;
			case 3: /* arbitrary rotations from file */
				fprintf(finfo, "\t%-50s%20s\n", "Projection mode:", "rotations from file");
				break;
			case 4: /* arbitrary rotations and trajectory from file */
				fprintf(finfo, "\t%-50s%10s\n", "Projection mode:", "gradients + rotations from file");
				break;
		}
		fprintf(finfo, "\t%-50s%20f\n", "VDS center acceleration factor:", vds_acc0);
		fprintf(finfo, "\t%-50s%20f\n", "VDS edge acceleration factor:", vds_acc1);
		fprintf(finfo, "\t%-50s%20d\n", "Number of navigator points:", nnav);
	}
	fprintf(finfo, "\t%-50s%20f %s\n", "Acquisition window duration:", acq_len*GRAD_UPDATE_TIME*1e-3, "ms");
	fprintf(finfo, "Prep parameters:\n");
	switch (fatsup_mode) {
		case 0: /* Off */
			fprintf(finfo, "\t%-50s%20s\n", "Fat suppression:", "off");
			break;
		case 1: /* CHESS */
			fprintf(finfo, "\t%-50s%20s\n", "Fat suppression:", "CHESS");
			fprintf(finfo, "\t%-50s%20d %s\n", "Fat suppression frequency offset:", fatsup_off, "Hz");
			fprintf(finfo, "\t%-50s%20d %s\n", "Fat suppression bandwidth:", fatsup_bw, "Hz");
			break;
		case 2: /* SPIR */
			fprintf(finfo, "\t%-50s%20s\n", "Fat suppression:", "SPIR");
			fprintf(finfo, "\t%-50s%20d %s\n", "Fat suppression frequency offset:", fatsup_off, "Hz");
			fprintf(finfo, "\t%-50s%20d %s\n", "Fat suppression bandwidth:", fatsup_bw, "Hz");
			fprintf(finfo, "\t%-50s%20f %s\n", "SPIR flip angle:", spir_fa, "deg");
			fprintf(finfo, "\t%-50s%20f %s\n", "SPIR inversion time:", (float)spir_ti*1e-3, "ms");
			break;
	}

	if (mrf_mode > 0){
		fprintf(finfo, "\n\tMRF MODE : %d ktrajectory is rotated from frame to frame (see kviews.txt) \n", mrf_mode);
		if (mrf_mode==1)
			fprintf(finfo, "\t%-50s%20s%05d \n", "MRF Labeling timing file in:", "mrfasl_schedules/", mrf_sched_id );
	}

	if (prep1_id == 0)
		fprintf(finfo, "\t%-50s%20s\n", "Prep 1 pulse:", "off");
	else {
		fprintf(finfo, "\t%-50s%20d\n", "Prep 1 pulse id:", prep1_id);
		fprintf(finfo, "\t%-50s%20f %s\n", "Prep 1 post-labeling delay:", (float)prep1_pld*1e-3, "ms");
		fprintf(finfo, "\t%-50s%20f %s\n", "Prep 1 max B1 amplitude:", prep1_rfmax, "mG");
		fprintf(finfo, "\t%-50s%20f %s\n", "Prep 1 max gradient amplitude:", prep1_gmax, "G/cm");
		switch (prep1_mod) {
			case 1:
				fprintf(finfo, "\t%-50s%20s\n", "Prep 1 pulse modulation:", "1 (LCLC)");
				break;
			case 2:
				fprintf(finfo, "\t%-50s%20s\n", "Prep 1 pulse modulation:", "2 (CLCL)");
				break;
			case 3:
				fprintf(finfo, "\t%-50s%20s\n", "Prep 1 pulse modulation:", "3 (LLLL)");
				break;
			case 4:
				fprintf(finfo, "\t%-50s%20s\n", "Prep 1 pulse modulation:", "4 (CCCC)");
				break;
		}
		fprintf(finfo, "\t%-50s%20f %s\n", "Prep 1 BGS 1 delay:", (float)prep1_tbgs1*1e-3, "ms");
		fprintf(finfo, "\t%-50s%20f %s\n", "Prep 1 BGS 2 delay:", (float)prep1_tbgs2*1e-3, "ms");
		fprintf(finfo, "\t%-50s%20f %s\n", "Prep 1 BGS 3 delay:", (float)prep1_tbgs3*1e-3, "ms");
	}
	if (prep2_id == 0)
		fprintf(finfo, "\t%-50s%20s\n", "Prep 2 pulse:", "off");
	else {
		fprintf(finfo, "\t%-50s%20d\n", "Prep 2 pulse id:", prep2_id);
		fprintf(finfo, "\t%-50s%20f %s\n", "Prep 2 post-labeling delay:", (float)prep2_pld*1e-3, "ms");
		fprintf(finfo, "\t%-50s%20f %s\n", "Prep 2 max B1 amplitude:", prep2_rfmax, "mG");
		fprintf(finfo, "\t%-50s%20f %s\n", "Prep 2 max gradient amplitude:", prep2_gmax, "G/cm");
		switch (prep2_mod) {
			case 1:
				fprintf(finfo, "\t%-50s%20s\n", "Prep 2 pulse modulation:", "1 (LCLC)");
				break;
			case 2:
				fprintf(finfo, "\t%-50s%20s\n", "Prep 2 pulse modulation:", "2 (CLCL)");
				break;
			case 3:
				fprintf(finfo, "\t%-50s%20s\n", "Prep 2 pulse modulation:", "3 (LLLL)");
				break;
			case 4:
				fprintf(finfo, "\t%-50s%20s\n", "Prep 2 pulse modulation:", "4 (CCCC)");
				break;
		}
		fprintf(finfo, "\t%-50s%20f %s\n", "Prep 2 BGS 1 delay:", (float)prep2_tbgs1*1e-3, "ms");
		fprintf(finfo, "\t%-50s%20f %s\n", "Prep 2 BGS 2 delay:", (float)prep2_tbgs2*1e-3, "ms");
		fprintf(finfo, "\t%-50s%20f %s\n", "Prep 2 BGS 3 delay:", (float)prep2_tbgs3*1e-3, "ms");
	}
	if (presat_flag == 0)
		fprintf(finfo, "\t%-50s%20s\n", "Presaturation pulse:", "off");
	else {
		fprintf(finfo, "\t%-50s%20s\n", "Presaturation pulse:", "on");
		fprintf(finfo, "\t%-50s%20f %s\n", "Presaturation delay:", (float)presat_delay*1e-3, "ms");
	}

	if (pcasl_flag){
		fprintf(finfo, "PCASL prep cvs:\n");
		fprintf(finfo, "\t%-50s%20d\n", "pcasl_flag", pcasl_flag);
		fprintf(finfo, "\t%-50s%20f\n", "pcasl_distance", pcasl_distance);
		fprintf(finfo, "\t%-50s%20f\n", "pcasl_delta_phs", pcasl_delta_phs);
		fprintf(finfo, "\t%-50s%20f\n", "pcasl_delta_phs_correction", pcasl_delta_phs_correction);
		fprintf(finfo, "\t%-50s%20d\n", "pcasl_pld (us)", pcasl_pld);
		fprintf(finfo, "\t%-50s%20d\n", "pcasl_duration (us)", pcasl_duration);
		fprintf(finfo, "\t%-50s%20d\n", "pcasl_tbgs1 (us)", pcasl_tbgs1);
		fprintf(finfo, "\t%-50s%20d\n", "pcasl_tbgs2 (us)", pcasl_tbgs2);
		fprintf(finfo, "\t%-50s%20f\n", "pcasl_Gamp", pcasl_Gamp);
		fprintf(finfo, "\t%-50s%20f\n", "pcasl_Gave", pcasl_Gave);
		fprintf(finfo, "\t%-50s%20d\n", "pcasl_period (us)", pcasl_period);
		fprintf(finfo, "\t%-50s%20f\n", "pcasl_RFamp (mG)", pcasl_RFamp);

		if(pcasl_calib){
			fprintf(finfo, "\t%-50s%20d\n", "Phase calibration run", pcasl_calib);
			fprintf(finfo, "\t%-50s%20f\n", "Cal. phase increment", phs_cal_step);
			fprintf(finfo, "\t%-50s%20d\n", "N. frames per cal. increment", pcasl_calib_frames);
		}
	}

	if(doVelSpectrum){
		fprintf(finfo, "Collecting Velocity Spectrum:\n");
		fprintf(finfo, "\t%-50s%20d\n", "VS Type (FTVS phase(1) BIR8 venc(2))", doVelSpectrum); 
		fprintf(finfo, "\t%-50s%20d\n", "N. frames per encode/target_velocity", vspectrum_Navgs);
		fprintf(finfo, "\t%-50s%20f\n", "Velocity target increments (cm/s) if VS type 1", vel_target_incr); 
		fprintf(finfo, "\t%-50s%20f\n", "Velocity encoding Grad increments (G/cm) if VS type 2", prep2_delta_gmax); 
	}

	fclose(finfo);
	return 1;
}

/************************ END OF UMVSASL.E ******************************/

