(* ::Package:: *)

(************************************************************************)
(* This file was generated automatically by the Mathematica front end.  *)
(* It contains Initialization cells from a Notebook file, which         *)
(* typically will have the same name as this file except ending in      *)
(* ".nb" instead of ".m".                                               *)
(*                                                                      *)
(* This file is intended to be loaded into the Mathematica kernel using *)
(* the package loading commands Get or Needs.  Doing so is equivalent   *)
(* to using the Evaluate Initialization Cells menu command in the front *)
(* end.                                                                 *)
(*                                                                      *)
(* DO NOT EDIT THIS FILE.  This entire file is regenerated              *)
(* automatically each time the parent Notebook file is saved in the     *)
(* Mathematica front end.  Any changes you make to this file will be    *)
(* overwritten.                                                         *)
(************************************************************************)



Print["[PrimiCosmo]: Setting up options"]
timingStart = AbsoluteTime[];


SetDirectory[NotebookDirectory[]]


Print["[PrimiCosmo]: The current Directory is ", Directory[]]


(* ::Input::Initialization:: *)
(*$EvolutionType = "WIMP"*)
(*\[CapitalDelta]Neffective = 1.0;*)
(*TnuStart = 30.0;*)


$InterpolateAnalytics=True;


$HistoryLength = 10;


$PaperPlots=False;
$ResultsPlots=False;


(* ::Input::Initialization:: *)
$CompileNDSolve=True;


(* ::Input::Initialization:: *)
$BDFOrder=2;


(* ::Input::Initialization:: *)
PrecisionNDSolve=2;


(* ::Input::Initialization:: *)
AccuracyNDSolve:=15+PrecisionNDSolve;


(* ::Input::Initialization:: *)
NTemperaturePoints=1200;


(* ::Input::Initialization:: *)
InterpOrder=3;


(* ::Input::Initialization:: *)
$FastPENRatesIntegrals=True;


(* ::Input::Initialization:: *)
$PENRatesIntegralsPoints=300;


(* ::Input::Initialization:: *)
$RecomputeWeakRates=False;
$ParallelWeakRates=True;


(* ::Input::Initialization:: *)
$RadiativeCorrections=True;

$ResummedLogsRadiativeCorrections=True;
$RelativisticFermiFunction=True;


(* ::Input::Initialization:: *)
$RadiativeThermal=True;
$CorrectionBremsstrahlung=True;


(* ::Input::Initialization:: *)
$FiniteNucleonMass=True;


(* ::Input::Initialization:: *)
$CoupledFMandRC=True;


(* ::Input::Initialization:: *)
$QEDMassShift=False;


(* ::Input::Initialization:: *)
$QEDPlasmaCorrections=True;
$CompleteQEDPressure=True;


(* ::Input::Initialization:: *)
$IncompleteNeutrinoDecoupling=True;


(* ::Input::Initialization:: *)
$RecomputePlasmaCorrections=False;


(* ::Input::Initialization:: *)
$DegenerateNeutrinos=False;
\[Mu]OverT\[Nu]=0.0;


(* ::Input::Initialization:: *)
Kelvin=1;
Tstart=10^11 Kelvin;
TMiddle:=0.9999*10^10 Kelvin;
T18:=1.25 *10^9 Kelvin;
Tend=6.*10^7 Kelvin;


(* ::Input::Initialization:: *)
Ti=10^11 Kelvin;
Tf=6.*10^7 Kelvin;
LogTi=1.Log10[Ti];
LogTf=1.Log10[Tf]; 


(* ::Input::Initialization:: *)
ListLogT=Sort@DeleteDuplicates@Join[{10.},Table[i,{i,LogTf,LogTi,(LogTi-LogTf)/NTemperaturePoints}]];
ListT=1. 10^ListLogT;


(* ::Input::Initialization:: *)
ListTRange[T1_,T2_]:=Module[{len=Length@ListT,imindown,imaxup,Tmin=Min[T1,T2],Tmax=Max[T1,T2]},
imindown=Max[1,-1+Position[ListT,SelectFirst[ListT,#>Tmin&]][[1,1]]];
imaxup=Min[len,Position[ListT,SelectFirst[ListT,#>=Tmax&]][[1,1]]];
ListT[[imindown;;imaxup]]
]


(* ::Input::Initialization:: *)
second=1;
cm=1;
gram=1;


(* ::Input::Initialization:: *)
kg=10^3 gram;
meter = 10^2 cm;
km=10^3 meter;
Joule =kg meter^2/second^2; (* This gives 10^7 ergs *)
DensityUnit=gram/cm^3;
Hz=1/second;


(* ::Input::Initialization:: *)
Giga=10^9;
Mega=10^6;
Kilo=10^3;


(* ::Input::Initialization:: *)
kB =1.3806488 10^-23 Joule / Kelvin;(* Boltzmann constant in J/K *)
clight=2.99792458*10^8*meter/second; (* speed of light in cm/s *)
hbar= 6.62606957/(2\[Pi]) 10^-34(*1.054571596 10^-34*) Joule second;
Avogadro=6.0221415 10^23;


(* ::Input::Initialization:: *)
eV=1.60217653 10^-19 Joule;
keV = Kilo eV;
MeV=Mega eV;
GeV=Giga eV;


(* ::Input::Initialization:: *)
GN=6.67384 10^-11 meter^3/kg/second^2; (* Gravitation constant *)
GF=1.1663787*10^-5/(GeV)^2; (* Fermi Constant*)
gA=1.2723; 
(* Axial current constant of structure of the nucleons Particle data group : 1.2723(+-23) PDG2016 *)
(* However post 2002 data suggest 1.2755(11) as advised by William Marciano*)


(* ::Input::Initialization:: *)
fWM=3.7058/2(*1.853*); (* Weak magnetism see 1212.0332*)
radiusproton=0.841*10^-15 meter; (*(arXiv:1212.0332)*)


(* ::Input::Initialization:: *)
\[Alpha]FS=1/137.03599911;(* Fine structure constant =e^2/(4\[Pi]) *)


(* ::Input::Initialization:: *)
me=0.510998918 MeV; 
mn=939.565360 MeV;
mp=938.272029 MeV; 
Q=mn-mp; (* Mass difference between neutrons and protons *)
Subscript[m, Nucleon]=mn;

Subscript[m, W]=80.385 GeV; (* Mass of the W Boson. *) 
Subscript[m, Z]=91.1876 GeV;


(* ::Input::Initialization:: *)
pc=3.0856777807 10^16 meter; (* The parsec *)
Mpc=Mega pc;
H0=100 h km/second/Mpc; (* Hubble constant today *)
H100=100 km/second/Mpc;(*Fake Hubble rate given by 100 km/s/Mpc so that h = H0/H100 *)


(* ::Input::Initialization:: *)
Subscript[\[Rho], crit]=3./(8\[Pi] GN) (H0)^2 ;(* in g cm^-3 by construction *)
\[Rho]crit100=3./(8\[Pi] GN) (H100)^2; (* in g cm^-3 by construction *)


(* ::Input::Initialization:: *)
Mean\[Tau]neutron:=879.5;(*880.2second+-1.1s was previous value from PDG2017 *);
(* Now we use 1712.05663 Section 11 which includes recente 2017 measurements.*)
\[Sigma]\[Tau]neutron:=0.8 second;
\[Tau]neutron=Mean\[Tau]neutron;


(* ::Input::Initialization:: *)
NeutrinosGenerations:=3.;
\[Xi]\[Nu]:=If[$DegenerateNeutrinos,\[Mu]OverT\[Nu],0];


(* ::Input::Initialization:: *)
\[Rho]FD[c_]=1/(2\[Pi]^2) \!\(
\*SubsuperscriptBox[\(\[Integral]\), \(0\), \(Infinity\)]\(
\*FractionBox[
SuperscriptBox[\(y\), \(3\)], \((
\*SuperscriptBox[\(E\), \(y - c\)] + 1)\)] \[DifferentialD]y\)\);
nFD[c_]=1/(2\[Pi]^2) \!\(
\*SubsuperscriptBox[\(\[Integral]\), \(0\), \(Infinity\)]\(
\*FractionBox[
SuperscriptBox[\(y\), \(2\)], \((
\*SuperscriptBox[\(E\), \(y - c\)] + 1)\)] \[DifferentialD]y\)\);
\[Rho]FDNonDegenerate=\[Rho]FD[0];


(* ::Input::Initialization:: *)
Nneu:=NeutrinosGenerations*(\[Rho]FD[\[Xi]\[Nu]]+\[Rho]FD[-\[Xi]\[Nu]])/(2\[Rho]FDNonDegenerate);


(* ::Input::Initialization:: *)
TCMB0:=2.7255Kelvin;
\[Sigma]TCMB0:=0.0006 Kelvin;(* [Planck 2015 XIII] *)


(* ::Input::Initialization:: *)
FourOverElevenQED:=4/11 (1+(25 \[Alpha]FS)/(22\[Pi]));
FourOverElevenNoQED:=4/11;
FourOverEleven:=If[$QEDPlasmaCorrections,FourOverElevenQED,FourOverElevenNoQED];

T\[Nu]0=(FourOverEleven)^(1/3) TCMB0;


(* ::Input::Initialization:: *)
h:=0.6727; (*+-0.0066 *)(*[Planck 2015 XIII]*)


(* ::Input::Initialization:: *)
Meanh2\[CapitalOmega]c0=0.1198;(* [Planck 2015 XIII]*)
\[Sigma]h2\[CapitalOmega]c0=0.0015;
h2\[CapitalOmega]c0=Meanh2\[CapitalOmega]c0;


(* ::Input::Initialization:: *)
aBB=((\[Pi]^2)/(15 hbar^3 (clight)^5)) 


(* ::Input::Initialization:: *)
Subscript[\[Rho], CMB0]:=aBB (kB TCMB0)^4 ;(* in g cm^-3*)

Subscript[n, CMB0]:=(2 Zeta[3])/(\[Pi]^2 hbar^3 (clight)^3) (kB TCMB0)^3


(* ::Input::Initialization:: *)
Subscript[n, CMB0]


(* ::Input::Initialization:: *)
Subscript[\[CapitalOmega], \[Gamma]0]:=Subscript[\[Rho], CMB0]/Subscript[\[Rho], crit];


(* ::Input::Initialization:: *)
Subscript[\[CapitalOmega], \[Nu]0]:=Nneu*7/8*(FourOverEleven)^(1/3) Subscript[\[CapitalOmega], \[Gamma]0];


(* ::Input::Initialization:: *)
FD[EoverT_]=1/(Exp[EoverT]+1);(* Fermi Dirac Distribution *)
FD[Energy_,x_]=1/(Exp[x Energy]+1);
BE[EoverT_]=1/(Exp[EoverT]-1); (* Bose Einstein Distribution *)
BE[Energy_,x_]=1/(Exp[x Energy]-1);

(* For neutrinos with a chemical potential *)
FD\[Nu][Energy_,\[Phi]_,x_]=1/(Exp[x Energy-\[Phi]]+1);


(* ::Input::Initialization:: *)
FDp[Energy_,x_]=D[1/(Exp[x Energy]+1),Energy];


(* ::Input::Initialization:: *)
NP[number_]:=NumberForm[number,8]


(* ::Input::Initialization:: *)
MyGrid[Table_List]:=Grid[Table,Frame->All]


(* ::Input::Initialization:: *)
MyInterpolation[Tab_List]:=Interpolation[Tab,InterpolationOrder->InterpOrder];

(* Does not work to interpolate the log of rates because it fails when rates vanish !!!*)
MyInterpolationLog[Tab_List]:=Function[{x},Exp[Interpolation[{#[[1]],Log[#[[2]]]}&/@Tab,InterpolationOrder->InterpOrder][x]]];

$InterpolateLogRate=False;
MyInterpolationRate[Tab_List]:=If[$InterpolateLogRate,MyInterpolationLog[Tab],MyInterpolation[Tab]]


(* ::Input::Initialization:: *)
MyChop[el_?NumericQ]:=(Chop[el,$MinMachineNumber]);
SetAttributes[MyChop,Listable];


(* ::Input::Initialization:: *)
MySet[Hold[expr_],value_]:=(expr=value);
MySetDelayed[Hold[expr_],value_]:=(expr:=value);


(* ::Input::Initialization:: *)
TableSimpsonC=Compile[{{a,_Real},{b,_Real},{Np,_Integer}},With[{h=1.(b-a)/Np,n2=Np/2},With[{h3=h/3.},Join[{{a,h3}},Table[{a+2. j h,2 h3},{j,1,n2-1}],Table[{a+(2. j-1) h,4 h3},{j,1,n2}],{{b,h3}}]]],CompilationTarget->"C","RuntimeOptions"->"Speed"];


(* ::Input::Initialization:: *)
MyCompile[LV_List,Body_]:=Compile[LV,Evaluate[Body],"RuntimeOptions"->"Speed",CompilationTarget->"C",CompilationOptions->{"InlineExternalDefinitions"->True},RuntimeAttributes->{Listable}]


(* ::Input::Initialization:: *)
V1dotV2=Compile[{{V1,_Real,1},{V2,_Real,1}},V1.V2,CompilationTarget->"C","RuntimeOptions"->"Speed"];


(* ::Input::Initialization:: *)
IntegrateFunction[fun_,pemin_,pemax_,Np_]:=With[{interv=(pemax-pemin)/(Np),tab=TableSimpsonC[pemin,pemax,Np]},V1dotV2[tab[[All,2]],MyChop[fun[tab[[All,1]]]]]];


(* ::Input::Initialization:: *)
SafeImport[args__]:=Module[{out},out=Catch[Check[Import[args],Print["File ",{args}[[1]]," not found. Quiting Kernel."];Throw[$Failed];,Import::nffil]];If[out===$Failed,Quit[]];out]


(* ::Input::Initialization:: *)
MyFrameTicksLog={{Automatic,Automatic},{{{Log[10^8],"\!\(\*SuperscriptBox[\(10\), \(8\)]\)"},{Log[10^8.5],"\!\(\*SuperscriptBox[\(10\), \(8.5\)]\)"},{Log[10^9],"\!\(\*SuperscriptBox[\(10\), \(9\)]\)"},{Log[10^9.5],"\!\(\*SuperscriptBox[\(10\), \(9.5\)]\)"},{Log[10^10],"\!\(\*SuperscriptBox[\(10\), \(10\)]\)"},{Log[10^10.5],"\!\(\*SuperscriptBox[\(10\), \(10.5\)]\)"},{Log[10^11],"\!\(\*SuperscriptBox[\(10\), \(11\)]\)"},{Log[10^11.5],"\!\(\*SuperscriptBox[\(10\), \(11.5\)]\)"}},Automatic}};

MyFrameTicks={{Automatic,Automatic},{{{10^8,"\!\(\*SuperscriptBox[\(10\), \(8\)]\)"},{10^8.5,"\!\(\*SuperscriptBox[\(10\), \(8.5\)]\)"},{10^9,"\!\(\*SuperscriptBox[\(10\), \(9\)]\)"},{10^9.5,"\!\(\*SuperscriptBox[\(10\), \(9.5\)]\)"},{10^10,"\!\(\*SuperscriptBox[\(10\), \(10\)]\)"},{10^10.5,"\!\(\*SuperscriptBox[\(10\), \(10.5\)]\)"},{10^11,"\!\(\*SuperscriptBox[\(10\), \(11\)]\)"},{10^11.5,"\!\(\*SuperscriptBox[\(10\), \(11\)]\).5"}},Automatic}};


(* ::Input::Initialization:: *)
Clear[Imn]
Imn[sgn_][m_,n_][x_]:=NIntegrate[((pe^2+x^2)^((m-1)/2) pe^(n+1))/(Exp[Sqrt[pe^2+x^2]]+sgn),{pe,0,Infinity},Method->{Automatic,"SymbolicProcessing"->0}]
ImnT[sgn_][m_,n_][T_]:=Imn[sgn][m,n][me/(kB T)]

(* Interpolations *)
ImnI[sgn_][m_,n_]:=ImnI[sgn][m,n]=Interpolation@Table[{me/(kB Tv),Imn[sgn][m,n][me/(kB Tv)]},{Tv,ListT}]
ImnIT[sgn_][m_,n_][T_]:=ImnI[sgn][m,n][me/(kB T)]


(* ::Input::Initialization:: *)
dme2[T_]:=((kB T)/me)^2 ((2\[Pi] \[Alpha]FS)/3+ (4\[Alpha]FS)/\[Pi] ImnT[1][0,1][T])(* Only main part of mass shift *)
dm\[Gamma]2[T_]:= (8\[Alpha]FS)/\[Pi] ImnT[1][0,1][T]((kB T)/me)^2


(* ::Input::Initialization:: *)
dme2Tab=Check[Import["Interpolations/dme2.dat","TSV"],Print["Precomputed data not found. We recompute and store the data."];$Failed,Import::nffil];

dmg2Tab=Check[Import["Interpolations/dmg2.dat","TSV"],Print["Precomputed data not found. We recompute and store the data."];$Failed,Import::nffil];


(* ::Input::Initialization:: *)
Timing[If[dme2Tab==$Failed||dmg2Tab==$Failed||$RecomputePlasmaCorrections,

dme2Tab=Table[{T,dme2[T]},{T,ListT}];
dmg2Tab=Table[{T,dm\[Gamma]2[T]},{T,ListT}];

Export["Interpolations/dme2.dat",dme2Tab,"TSV"];
Export["Interpolations/dmg2.dat",dmg2Tab,"TSV"];
];]


(* ::Input::Initialization:: *)
dme2I=MyInterpolation@ToExpression@dme2Tab;
dm\[Gamma]2I=MyInterpolation@ToExpression@dmg2Tab;


(* ::Input::Initialization:: *)
dme2N[T_?NumericQ]:=Which[T<Tf,0,T<=Ti ,dme2I[T],T>Ti,dme2I[Ti]];
dm\[Gamma]2N[T_?NumericQ]:=Which[T<Tf,0,T<=Ti ,dm\[Gamma]2I[T],T>Ti,dme2I[Ti]];


(* ::Input::Initialization:: *)
dme2x[x_]:=dme2N[me/(kB x)];


(* ::Input::Initialization:: *)
dPa[T_]:=dPa[T]=\[Alpha]FS/\[Pi] (kB T)^4 (-(2/3)ImnT[1][0,1][T]-2/\[Pi]^2 (ImnT[1][0,1][T])^2);


(* ::Input::Initialization:: *)
Fdp1dp2=Compile[{{p1,_Real},{p2,_Real},{x,_Real}},Evaluate[With[
{e1=Sqrt[p1^2+x^2],e2=Sqrt[p2^2+x^2]},
\[Alpha]FS/\[Pi]^3 (x^2 p1^2 p2^2)/(p1 p2 e1 e2) Log[Abs[(p1+p2)/(p1-p2)]] 1/((Exp[e1]+1)(Exp[e2]+1))
]],"RuntimeOptions"->"Speed",CompilationTarget->"C"];


Fdp1dp2N[p1_?NumericQ,p2_?NumericQ,x_]:=Fdp1dp2[p1,p2,x];


Clear[dPb]
dPb[Tv_]:=dPb[Tv]=(kB Tv)^4 With[{x=me /(kB Tv)},
0.5NIntegrate[
Fdp1dp2N[(p1pp2+p1mp2)/2,(p1pp2-p1mp2)/2,x]
+Fdp1dp2N[(p1pp2-p1mp2)/2,(p1pp2+p1mp2)/2,x],
{p1mp2,0.0001,Max[20,20* x]},{p1pp2,0.0001+Abs[p1mp2],Max[20,20*x]+Abs[p1mp2]},PrecisionGoal->4]
];


(* ::Input::Initialization:: *)
dP[T_]:=dP[T]= dPa[T]+If[$CompleteQEDPressure,dPb[T],0]

dPI:=dPI=Interpolation@Table[{Tv,dP[Tv]},{Tv,ListT}]


(* ::Input::Initialization:: *)
Clear[d\[Rho]]
d\[Rho][T_]:=d\[Rho][T]=-dP[T]+T dPI'[T]


(* ::Input::Initialization:: *)
dgP[T_]:=dP[T] 90/(\[Pi]^2 (kB T)^4);
dg\[Rho][T_]:=d\[Rho][T] 30/(\[Pi]^2 (kB T)^4);


(* ::Input::Initialization:: *)
dg\[Rho]dgP=Check[Import["Interpolations/dg.dat","TSV"],Print["Precomputed data not found. We recompute and store the data."];$Failed,Import::nffil];

Timing[If[dg\[Rho]dgP==$Failed||$RecomputePlasmaCorrections,

dg\[Rho]Tab=Table[{T,dg\[Rho][T]},{T,ListT}];
dgPTab=Table[{T,dgP[T]},{T,ListT}];

dg\[Rho]dgP={dg\[Rho]Tab,dgPTab};
Export["Interpolations/dg.dat",dg\[Rho]dgP,"TSV"];
];]


(* ::Input::Initialization:: *)
dg\[Rho]I=MyInterpolation@ToExpression[dg\[Rho]dgP[[1]]];
dgPI=MyInterpolation@ToExpression[dg\[Rho]dgP[[2]]];


(* ::Input::Initialization:: *)
dg\[Rho]N[T_?NumericQ]:=Which[T<Tf,0,T<=Ti ,dg\[Rho]I[T],T>Ti,dg\[Rho]I[Ti]];
dgPN[T_?NumericQ]:=Which[T<Tf,0,T<=Ti ,dgPI[T],T>Ti,dgPI[Ti]];


(* ::Input::Initialization:: *)
dg\[Rho]x[x_]:=dg\[Rho]N[me/(kB x)];
dgPx[x_]:=dgPN[me/(kB x)];


(* ::Input::Initialization:: *)
DSTNoQED=MyInterpolation@Table[{T,With[{x=me/(kB T)},1+45/(2\[Pi]^4) (1/3 Imn[1][0,3][x]+Imn[1][2,1][x])]},{T,ListT}];
DSTQED[Tv_]:=(3dg\[Rho]N[Tv]+dgPN[Tv])/8+DSTNoQED[Tv];


DST[Tv_]:=If[$QEDPlasmaCorrections,DSTQED[Tv],DSTNoQED[Tv]]
DSTN[T_?NumericQ]=Which[T<Tf,1,T<=Ti ,DST[T],T>Ti,DST[Ti]];


(* ::Input::Initialization:: *)
D\[Rho]TNoQED=MyInterpolation@Table[{T,With[{x=me/(kB T)},30/\[Pi]^4 (Imn[1][2,1][x])]},{T,ListT}];D\[Rho]T[T_]:=If[$QEDPlasmaCorrections,dg\[Rho]N[T]/2,0]+D\[Rho]TNoQED[T];


Print["[PrimiCosmo]: Integrating the Cosmology"]


If[$EvolutionType == "SM",
<<sm.m;
]
If[$EvolutionType == "Neff",
<< smNeff.m;
]
If[$EvolutionType == "WIMP",
<< simple_wimp.m;
]


(* ::Input::Initialization:: *)
T\[Nu]overTTable = Table[{T\[Nu]overT[ListT[[i]]], ListT[[i]]},{i,1,Length[ListT]}];


Print["[PrimiCosmo]: Computing Weak rates"]


(* ::Input::Initialization:: *)
FDe2p0[en_,x_]=Simplify[FD[en,x]en^2];
FDe3p0[en_,x_]=Simplify[FD[en,x]en^3];
FDe2p2[en_,x_]=Simplify@D[D[FD[en,x]en^2,en],en];

FDe3p2[en_,x_]=Simplify@D[D[FD[en,x]en^3,en],en];
FDe4p2[en_,x_]=Simplify@D[D[FD[en,x]en^4,en],en];
FDe2p1[en_,x_]=Simplify@D[FD[en,x]en^2,en];
FDe3p1[en_,x_]=Simplify@D[FD[en,x]en^3,en];
FDe4p1[en_,x_]=Simplify@D[FD[en,x]en^4,en];


(* ::Input::Initialization:: *)
FD\[Nu]e2p0[en_,\[Phi]_,x_]=Simplify[FD\[Nu][en,\[Phi],x]en^2];
FD\[Nu]e3p0[en_,\[Phi]_,x_]=Simplify[FD\[Nu][en,\[Phi],x]en^3];
FD\[Nu]e2p2[en_,\[Phi]_,x_]=Simplify@D[D[FD\[Nu][en,\[Phi],x]en^2,en],en];

FD\[Nu]e3p2[en_,\[Phi]_,x_]=Simplify@D[D[FD\[Nu][en,\[Phi],x]en^3,en],en];
FD\[Nu]e4p2[en_,\[Phi]_,x_]=Simplify@D[D[FD\[Nu][en,\[Phi],x]en^4,en],en];
FD\[Nu]e2p1[en_,\[Phi]_,x_]=Simplify@D[FD\[Nu][en,\[Phi],x]en^2,en];
FD\[Nu]e3p1[en_,\[Phi]_,x_]=Simplify@D[FD\[Nu][en,\[Phi],x]en^3,en];
FD\[Nu]e4p1[en_,\[Phi]_,x_]=Simplify@D[FD\[Nu][en,\[Phi],x]en^4,en];


(* ::Input::Initialization:: *)
\[Lambda]BORN=With[{q=Q/me},NIntegrate[en (en-q)^2 Sqrt[en^2-1],{en,1,q}]];


(* ::Input::Initialization:: *)
AgCzarnecki=-0.34;
CCzarnecki=0.891;
mA=1.2 GeV;
ConstantSirlin =4Log[Subscript[m, Z]/mp]+Log[mp/mA]+2CCzarnecki+AgCzarnecki;


(* ::Input::Initialization:: *)
Rd[x_]:=ArcTanh[x]/x;


(* ::Input::Initialization:: *)
Lfun[x_]=Integrate[Log[1-t]/t,{t,0,x},Assumptions->x<1&&x>0];(* Lfun is called the Spence function *)


(* ::Input::Initialization:: *)
LfunSeries[b_]=Normal@Series[-1/4*(1+b)^6*4/b Lfun[(2b)/(1+b)],{b,0,12(*22*)}];


(* ::Input::Initialization:: *)
$SeriesSpenceFunction=False;

SirlinGFunction[b_,y_,en_]:=(3Log[mp/(me)]-3/4+4(Rd[b]-1)(y/(3 en)-3/2+Log[2y])+Rd[b](2(1+b^2)+y^2/(6 en^2)-4 b Rd[b])+If[$SeriesSpenceFunction,-4/(1+b)^6*LfunSeries[b],4/b Lfun[(2b)/(1+b)]]);
Cd[b_,y_,en_]:=(ConstantSirlin+SirlinGFunction[b,y,en]);


(* ::Input::Initialization:: *)
LFactor=1.02094;
SFactor=1.02248;
\[Delta]factor=-0.00043*2Pi/\[Alpha]FS;
NLL=-0.0001;


(* ::Input::Initialization:: *)
RadiativeCorrectionsResummed[b_,y_,en_]:=(1+\[Alpha]FS/(2\[Pi]) (SirlinGFunction[b,y,en]-3Log[mp/(2Q)]))*
(LFactor+\[Alpha]FS/\[Pi] CCzarnecki+\[Alpha]FS/(2\[Pi]) \[Delta]factor)*(SFactor+1/(134*2*Pi)*(Log[mp/mA]+AgCzarnecki)+NLL);


(* ::Input::Initialization:: *)
RadiativeCorrections[b_,y_,en_]:=If[$ResummedLogsRadiativeCorrections,RadiativeCorrectionsResummed[b,y,en],(1+\[Alpha]FS/(2\[Pi]) Cd[b,y,en])];


(* ::Input::Initialization:: *)
FermiRelat[b_]:=With[{\[Gamma]=Sqrt[1-\[Alpha]FS^2]-1,\[Lambda]Compton=1/(me/(hbar clight))},
(1+\[Gamma]/2)*4((2radiusproton b)/\[Lambda]Compton)^(2\[Gamma])*1/Gamma[3+2\[Gamma]]^2 Exp[(\[Pi] \[Alpha]FS)/b]*1/(1-b^2)^\[Gamma] Abs[Gamma[1+\[Gamma]+I \[Alpha]FS/b]]^2];

FermiNonRelat[b_]:=(2\[Pi] \[Alpha]FS/b)/(1-Exp[-2\[Pi] \[Alpha]FS/b]);


(* ::Input::Initialization:: *)
If[$RelativisticFermiFunction,

Fermi[b_]:=FermiRelat[b];
bFermi[b_]:=b Fermi[b];,

Fermi[b_]:=FermiNonRelat[b];
bFermi[b_]:=(2\[Pi] \[Alpha]FS)/(1-Exp[-2\[Pi] \[Alpha]FS/b]);]


(* ::Input::Initialization:: *)
\[Lambda]FermiOnly=With[{q=Q/me ,b=Sqrt[en^2-1]/en,y=Q/me -en},
NIntegrate[en (en-q)^2 en*bFermi[b],{en,1.0000001,q}]];


(* ::Input::Initialization:: *)
\[Lambda]Rad=With[{q=Q/me ,b=Sqrt[en^2-1]/en,y=Q/me -en},
NIntegrate[en (en-q)^2 en(RadiativeCorrections[b,y,en])*bFermi[b],{en,1.0000001,q}]];


(* ::Input::Initialization:: *)
IntegrateCorrectionNeutronDecay[fun_]:=
NIntegrate[fun[pe],{pe,0.0000001,Sqrt[(Q/me)^2-1]},WorkingPrecision->MachinePrecision];


(* ::Input::Initialization:: *)
\[Chi]FMNeutronDecay[en_,pe_]:=
With[{M=mp/me,enu=en-Q/me,f1=((1+gA)^2+4 fWM gA)/(1+3gA^2),f2=((1-gA)^2-4fWM gA)/(1+3gA^2),f3=(gA^2-1)/(1+3gA^2)},
 f1*enu^2 (pe^2/(M*en))
+f2*enu^3(-(1/M))
+ (f1+f2+f3) 1/(2M)*(4enu^3+2enu pe^2)
+f3*1/(3M) 3enu^2  (pe^2)/(en)
];


(* ::Input::Initialization:: *)
I\[Lambda]FM[pe_]:=With[{en=Sqrt[pe^2+1]},With[{b=pe/en},pe^2*
(\[Chi]FMNeutronDecay[en,pe]*If[$RadiativeCorrections&&$CoupledFMandRC,(RadiativeCorrections[b,Abs[en- Q/me ],en])Fermi[b],1])
]];


(* ::Input::Initialization:: *)
\[Lambda]FM=If[$FiniteNucleonMass,IntegrateCorrectionNeutronDecay[I\[Lambda]FM],0];


(* ::Input::Initialization:: *)
\[Lambda]Rad;


(* ::Input::Initialization:: *)
\[Lambda]FM;


(* ::Input::Initialization:: *)
\[Lambda]RadandFM=\[Lambda]Rad+\[Lambda]FM;


(* ::Input::Initialization:: *)
\[Lambda]Cooper=1.03887*1.6887;
\[Lambda]Czarnecki=1.0390*1.6887 ;(* = (1+RC)*f with f=1.6887 and RC = 0.0390(8) [Czarnecki et al. 2004]] *)


(* ::Input::Initialization:: *)
MixingCosAngle=0.97420;(* (+-16) Value taken from CKM particle data group 2017. More precisely from the review on Vud Vus of the PDG 2017.*)
MyK=MixingCosAngle^2 (GF)^2 (1+3(gA)^2)/(2\[Pi]^3)*(me )^5 /hbar;
1/MyK/\[Lambda]RadandFM;
1/MyK/\[Lambda]Czarnecki;
1/MyK/\[Lambda]Cooper;


(* ::Input::Initialization:: *)
pemin=0.00001;
pemiddle[x_]:=Sqrt[Max[pemin^2,(Q/me )^2-1 -If[$QEDMassShift,dme2x[x],0]]];
pemaxC[x_]:=Max[7,30/x];
pemax[x_]:=Max[7,30/x];


(* ::Input::Initialization:: *)
$TnuEqualT=False;


(* ::Input::Initialization:: *)
IntegratedpNpoints[fun_,sgnq_,Tv_,Npoints_]:=With[{x=me/(kB Tv) ,znu=me/(kB Tv T\[Nu]overT[Tv]) },
If[$FastPENRatesIntegrals,
IntegrateFunction[fun[#,x,If[$TnuEqualT,x,znu],sgnq]&,pemin,pemaxC[x],Npoints],
NIntegrate[fun[pe,x,If[$TnuEqualT,x,znu],sgnq],{pe,pemin,pemiddle[x],pemax[x]}]
]]

IntegrateRatedp[fun_,sgnq_,Tv_]:=IntegratedpNpoints[fun,sgnq,Tv,$PENRatesIntegralsPoints];



(* ::Input::Initialization:: *)
enOFpe[pe_,x_]:=Sqrt[pe^2+1 +If[$QEDMassShift,dme2x[x],0]];


(* ::Input::Initialization:: *)
IPENdpFrom\[Chi]NoCCR[en_,pe_,x_,znu_,sgnq_,\[Chi]function_]:=With[{q=Q/me },With[{b=pe/en},
pe^2*(\[Chi]function[en,pe,x,znu,sgnq]+\[Chi]function[-en,pe,x,znu,sgnq])
]];


(* ::Input::Initialization:: *)
Fermi[sgnq_,signE_,b_?NumericQ]:=If[sgnq signE >0,Fermi[b],1];
SetAttributes[Fermi,Listable];


(* ::Input::Initialization:: *)
IPENdpFrom\[Chi]CCR[en_,pe_,x_,znu_,sgnq_,\[Chi]function_]:=With[{q=Q/me ,b=pe/en},
pe^2*(\[Chi]function[en,pe,x,znu,sgnq](RadiativeCorrections[b,Abs[sgnq Q/me-en],en])Fermi[sgnq,1,b]+\[Chi]function[-en,pe,x,znu,sgnq](RadiativeCorrections[b, Abs[sgnq Q/me+en],en])Fermi[sgnq,-1,b])
];


(* ::Input::Initialization:: *)
\[Chi][en_,pe_,x_,znu_,sgnq_]:=With[{q=Q/me },FD\[Nu][en-sgnq q,sgnq \[Xi]\[Nu],znu]FD[-en,x](en-sgnq q)^2];


(* ::Input::Initialization:: *)
IPENdp[pe_,x_,znu_,sgnq_]:=IPENdpFrom\[Chi]NoCCR[enOFpe[pe,x],pe,x,If[$TnuEqualT,x,znu],sgnq,\[Chi]]


(* ::Input::Initialization:: *)
IPENdpCheatNeutrinoTemperature[pe_,x_,znu_,sgnq_]:=IPENdpFrom\[Chi]NoCCR[Sqrt[pe^2+1],pe,x,x,sgnq,\[Chi]]


(* ::Input::Initialization:: *)
\[Lambda]nTOpBORN[Tv_]:=IntegrateRatedp[IPENdp,1,Tv];
\[Lambda]pTOnBORN[Tv_]:=IntegrateRatedp[IPENdp,-1,Tv];


(* ::Input::Initialization:: *)
\[Lambda]nTOpBORNCheatNeutrino[Tv_]:=IntegrateRatedp[IPENdpCheatNeutrinoTemperature,1,Tv];
\[Lambda]pTOnBORNCheatNeutrino[Tv_]:=IntegrateRatedp[IPENdpCheatNeutrinoTemperature,-1,Tv];


(* ::Input::Initialization:: *)
\[Chi]FM[en_,pe_,x_,znu_,sgnq_]:=
With[{\[Phi]=sgnq \[Xi]\[Nu],q=Q/me ,M=(mp+mn -sgnq Q)/(2me ),Mp=mp/me ,Mn=mn /me ,enu=en-sgnq Q/me,
f1=((1+sgnq gA)^2+4fWM sgnq gA)/(1+3gA^2),
f2=((1-sgnq gA)^2-4fWM sgnq gA)/(1+3gA^2),f3=(gA^2-1)/(1+3gA^2)},
f1*FD\[Nu]e2p0[enu,\[Phi],znu]FD[-en,x](pe^2/(M*en))
+f2*FD\[Nu]e3p0[enu,\[Phi],znu]FD[-en,x](-(1/M))
+(f1+f2+f3) 1/(2x M)*(FD\[Nu]e4p2[enu,\[Phi],znu]FD[-en,x]+FD\[Nu]e2p2[enu,\[Phi],znu]FD[-en,x]pe^2)
+(f1+f2+f3) 1/(2M)*(FD\[Nu]e4p1[enu,\[Phi],znu]FD[-en,x]+FD\[Nu]e2p1[enu,\[Phi],znu]FD[-en,x]pe^2)
-(f1+f2) 1/(x M)*(FD\[Nu]e3p1[enu,\[Phi],znu]FD[-en,x]+FD\[Nu]e2p1[enu,\[Phi],znu]FD[-en,x]pe^2/(-en))
-f3*3/(x M) FD\[Nu]e2p0[enu,\[Phi],znu]FD[-en,x](* This term seems to give very small corrections *)
+f3*1/(3M) FD\[Nu]e3p1[enu,\[Phi],znu]FD[-en,x] pe^2/(en)
+f3*2/(2 x*3M) FD\[Nu]e3p2[enu,\[Phi],znu]FD[-en,x] pe^2/(en)
-(f1+f2+f3)*3/(2x)*(1-(Mn/Mp)^sgnq)*(FD\[Nu]e2p1[enu,\[Phi],znu]FD[-en,x])
];


(* ::Input::Initialization:: *)
IPENdpFMNoCCR[pe_,x_,znu_,sgnq_]:=IPENdpFrom\[Chi]NoCCR[enOFpe[pe,x],pe,x,If[$TnuEqualT,x,znu],sgnq,\[Chi]FM]
IPENdpFMCCR[pe_,x_,znu_,sgnq_]:=IPENdpFrom\[Chi]CCR[enOFpe[pe,x],pe,x,If[$TnuEqualT,x,znu],sgnq,\[Chi]FM]


(* ::Input::Initialization:: *)
IPENdpFMCheatNeutrinoTemperature[pe_,x_,znu_,sgnq_]:=IPENdpFrom\[Chi]NoCCR[enOFpe[pe,x],pe,x,x,sgnq,\[Chi]FM]


(* ::Input::Initialization:: *)
Clear[\[Lambda]nTOpFMCCR,\[Lambda]pTOnFMCCR,\[Lambda]nTOpFMNoCCR,\[Lambda]pTOnFMNoCCR,\[Lambda]nTOpCheatNeutrinoFM,\[Lambda]pTOnCheatNeutrinoFM]


(* ::Input::Initialization:: *)
\[Lambda]nTOpFMCCR[Tv_]:=IntegrateRatedp[IPENdpFMCCR,1,Tv];
\[Lambda]pTOnFMCCR[Tv_]:=IntegrateRatedp[IPENdpFMCCR,-1,Tv];


(* ::Input::Initialization:: *)
\[Lambda]nTOpFMNoCCR[Tv_]:=IntegrateRatedp[IPENdpFMNoCCR,1,Tv];
\[Lambda]pTOnFMNoCCR[Tv_]:=IntegrateRatedp[IPENdpFMNoCCR,-1,Tv];


(* ::Input::Initialization:: *)
\[Lambda]nTOpCheatNeutrinoFM[Tv_]:=IntegrateRatedp[IPENdpFMCheatNeutrinoTemperature,1,Tv];
\[Lambda]pTOnCheatNeutrinoFM[Tv_]:=IntegrateRatedp[IPENdpFMCheatNeutrinoTemperature,-1,Tv];


(* ::Input::Initialization:: *)
DetailedBalanceRatio0[T_]:=Exp[-(Q/(kB T))-\[Xi]\[Nu]];


(* ::Input::Initialization:: *)
DetailedBalance0[T_]:=(\[Lambda]nTOpBORN[T])/(\[Lambda]pTOnBORN[T])*DetailedBalanceRatio0[T];
DetailedBalanceCheatNeutrino0[T_]:=(\[Lambda]nTOpBORNCheatNeutrino[T])/(\[Lambda]pTOnBORNCheatNeutrino[T])*DetailedBalanceRatio0[T];


(* ::Input::Initialization:: *)
DetailedBalanceRatio[T_]:=Exp[-(Q/(kB T))-\[Xi]\[Nu]](1+(1+\[Alpha]) Q/mp)^(3/2);


(* ::Input::Initialization:: *)
IPENdpCCR[pe_,x_,znu_,sgnq_]:=IPENdpFrom\[Chi]CCR[enOFpe[pe,x],pe,x,If[$TnuEqualT,x,znu],sgnq,\[Chi]];


(* ::Input::Initialization:: *)
\[Lambda]nTOpCCR[Tv_]:=IntegrateRatedp[IPENdpCCR,1,Tv];
\[Lambda]pTOnCCR[Tv_]:=IntegrateRatedp[IPENdpCCR,-1,Tv];


(* ::Input::Initialization:: *)
BEQ[en_,sq_]:=sq BE[sq en];


(* ::Input::Initialization:: *)
\[Chi]tilde[en_,znu_,sgnq_]:=With[{q=Q/me },FD\[Nu][en-sgnq q,sgnq \[Xi]\[Nu],znu](en-sgnq q)^2]


(* ::Input::Initialization:: *)
IPENCCRT[en_,k_,x_,znu_,sgnq_]:=With[{p=Sqrt[en^2-1]},With[{b=p/en,A=(2 en^2+k^2)Log[(en+p)/(en-p)]-4 p en,B=2 en Log[(en+p)/(en-p)]-4p},
\[Alpha]FS/(2\[Pi])*(BE[x k]/k)*(A(FD[-en,x]Fermi[sgnq,1,b](\[Chi]tilde[en-k,znu,sgnq]+\[Chi]tilde[en+k,znu,sgnq]-2\[Chi]tilde[en,znu,sgnq])+FD[en,x]Fermi[sgnq,-1,b](\[Chi]tilde[-en+k,znu,sgnq]+\[Chi]tilde[-en-k,znu,sgnq]-2\[Chi]tilde[-en,znu,sgnq]))
-k B *(FD[-en,x]Fermi[sgnq,1,b](\[Chi]tilde[en-k,znu,sgnq]-\[Chi]tilde[en+k,znu,sgnq])+FD[en,x]Fermi[sgnq,-1,b](\[Chi]tilde[-en+k,znu,sgnq]-\[Chi]tilde[-en-k,znu,sgnq]))
)
]];

(* Compiled version to compute the integrals slightly faster *)
IPENCCRTC=MyCompile[{{en,_Real},{k,_Real},{x,_Real},{znu,_Real},{sgnq,_Integer}},Evaluate[IPENCCRT[en,k,x,znu,sgnq]]];IPENCCRTCN[en_?NumericQ,k_,x_,znu_,sgnq_]:=IPENCCRTC[en,k,x,znu,sgnq];


(* ::Input::Initialization:: *)
Clear[IPENCCRDiffBremsstrahlungCN,IPENCCRDiffBremsstrahlungC,IPENCCRDiffBremsstrahlung]
IPENCCRDiffBremsstrahlung[en_,k_,x_,znu_,sgnq_]:=With[{p=Sqrt[en^2-1],q=Q/me},With[{b=p/en,A=(2 en^2+k^2)Log[(en+p)/(en-p)]-4 p en,B=2 en Log[(en+p)/(en-p)]-4p},With[{Fp=A+k B,Fm=A-k B},
\[Alpha]FS/(2\[Pi] k) ((FD[-en,x]Fermi[sgnq,1,b](Fp \[Chi]tilde[en+k,znu,sgnq]-If[k<Abs[en-sgnq q],Fp FD[en-sgnq q,znu](Abs[en-sgnq q]-k)^2,0]))
+(FD[en,x]Fermi[sgnq,-1,b](Fm \[Chi]tilde[-en+k,znu,sgnq]-If[k<Abs[en+sgnq q],Fp FD[-en-sgnq q,znu](Abs[en+sgnq q]-k)^2,0]))
)
]]];

(* We compile for the integration *)
IPENCCRDiffBremsstrahlungC=MyCompile[{{en,_Real},{k,_Real},{x,_Real},{znu,_Real},{sgnq,_Integer}},Evaluate[IPENCCRDiffBremsstrahlung[en,k,x,znu,sgnq]]];IPENCCRDiffBremsstrahlungCN[en_?NumericQ,k_,x_,znu_,sgnq_]:=IPENCCRDiffBremsstrahlungC[en,k,x,znu,sgnq];


(* ::Input::Initialization:: *)
IPENFiveBodyT0[en_,k_,x_,znu_,sgnq_]:=With[{p=Sqrt[en^2-1]},With[{A=(2 en^2+k^2)Log[(en+p)/(en-p)]-4 p en,B=2 en Log[(en+p)/(en-p)]-4p},
\[Alpha]FS/(2\[Pi] k) (FD[en,x])\[Chi]tilde[-en+k,znu,sgnq](A-k B)]];


(* Compiled version *)
IPENFiveBodyT0C=Compile[{{en,_Real},{k,_Real},{x,_Real},{znu,_Real},{sgnq,_Integer}},Evaluate[With[{p=Sqrt[en^2-1]},With[{A=(2 en^2+k^2)Log[(en+p)/(en-p)]-4 p en,B=2 en Log[(en+p)/(en-p)]-4p},
\[Alpha]FS/(2\[Pi] k) (-BE[-k x])*(FD[en,x])\[Chi]tilde[-en+k,znu,sgnq](A-k B)]]],"RuntimeOptions"->"Speed",CompilationTarget->"C"];
IPENFiveBodyT0CN[en_?NumericQ,k_,x_,znu_,sgnq_]:=IPENFiveBodyT0C[en,k,x,znu,sgnq];


(* ::Input::Initialization:: *)
C1dE[en_,x_,znu_,sgnq_]:=With[{pe=Sqrt[en^2-1],q=Q/me },-((\[Alpha]FS en)/(2\[Pi] pe))*(2\[Pi]^2)/(3x^2) (\[Chi][en,pe,x,znu,sgnq]+\[Chi][-en,pe,x,znu,sgnq])];


(* ::Input::Initialization:: *)
C2dE1dE2[e1_,e2_,x_,znu_,sgnq_]:=With[{p1=Sqrt[e1^2-1],p2=Sqrt[e2^2-1],q=Q/me },With[{L=Log[(e1 e2 +p1 p2 +1)/(e1 e2 -p1 p2 +1)]},
\[Alpha]FS/(2\[Pi] ) (\[Chi][e1,p1,x,znu,sgnq]+\[Chi][-e1,p1,x,znu,sgnq])
*(-(1/4) Log[((p1+p2)/(p1-p2))^2]^2*(FDp[e2,x] p2/p1 e1^2/e2 (e1+e2)+FD[e2,x] e1^2/(p1 p2) (e2+e1/e2^2))
+Log[((p1+p2)/(p1-p2))^2](FDp[e2,x](p2^2 e1/e2 (1/p1^2+2)-e1^2 p2/p1 L)+FD[e2,x](e1/(p1^2 e2^2) (e2^2+2p1^2+1)-(e1^2+e2^2)/(e1+e2)-(e1^2 e2)/(p1 p2) L))
-FD[e2,x](4 e1 p2/p1+2 e2 L))
]];


(* Compiled version *)
C2dE1dE2C=Compile[{{e1,_Real},{e2,_Real},{x,_Real},{znu,_Real},{sgnq,_Integer}},Evaluate[With[{p1=Sqrt[e1^2-1],p2=Sqrt[e2^2-1],q=Q/me },With[{L=Log[(e1 e2 +p1 p2 +1)/(e1 e2 -p1 p2 +1)]},
\[Alpha]FS/(2\[Pi] ) (\[Chi][e1,p1,x,znu,sgnq]+\[Chi][-e1,p1,x,znu,sgnq])
*(-(1/4) Log[((p1+p2)/(p1-p2))^2]^2*(FDp[e2,x] p2/p1 e1^2/e2 (e1+e2)+FD[e2,x] e1^2/(p1 p2) (e2+e1/e2^2))
+Log[((p1+p2)/(p1-p2))^2](FDp[e2,x](p2^2 e1/e2 (1/p1^2+2)-e1^2 p2/p1 L)+FD[e2,x](e1/(p1^2 e2^2) (e2^2+2p1^2+1)-(e1^2+e2^2)/(e1+e2)-(e1^2 e2)/(p1 p2) L))
-FD[e2,x](4 e1 p2/p1+2 e2 L))
]]],"RuntimeOptions"->"Speed",CompilationTarget->"C"];
C2dE1dE2CN[e1_?NumericQ,e2_?NumericQ,x_,znu_,sgnq_]:=C2dE1dE2C[e1,e2,x,znu,sgnq];


(* ::Input::Initialization:: *)
\[Lambda]nTOpThermalTruePhoton[Tv_]:=(*\[Lambda]nTOpThermalTruePhoton[Tv]=*)With[{x=me/(kB Tv),znu=me/(kB Tv T\[Nu]overT[Tv]),q=Q/me },
NIntegrate[IPENCCRTCN[en,k,x,If[$TnuEqualT,x,znu],1],{k,0.001,Max[10,20/x]},{en,1.001,Max[10,20/x]},PrecisionGoal->4]
];

\[Lambda]nTOpThermalDiffBremsstrahlung[Tv_]:=(*\[Lambda]nTOpThermalDiffBremsstrahlung[Tv]=*)With[{x=me/(kB Tv),znu=me/(kB Tv T\[Nu]overT[Tv]),q=Q/me },
NIntegrate[IPENCCRDiffBremsstrahlungCN[en,k,x,If[$TnuEqualT,x,znu],1],{en,1.001,Max[10,20/x]},{k,0.001,Abs[en-q],Abs[en+q],Max[10,20/x]},PrecisionGoal->4]
];


(* ::Input::Initialization:: *)
\[Lambda]nTOpThermal[Tv_]:=(*\[Lambda]nTOpThermal[Tv]=*)With[{x=me/(kB Tv),znu=me/(kB Tv T\[Nu]overT[Tv]),q=Q/me },
NIntegrate[C1dE[en,x,If[$TnuEqualT,x,znu],1],{en,1,Max[25,150/x]}]
+NIntegrate[1/2C2dE1dE2CN[(e1pe2+e1me2)/2,(e1pe2-e1me2)/2,x,If[$TnuEqualT,x,znu],1],{e1me2,-Max[10,15/x],-0.001},{e1pe2,2.001+Abs[e1me2],2+Abs[e1me2]+Max[10,15/x]},PrecisionGoal->3,Exclusions->{0}]
+NIntegrate[1/2C2dE1dE2CN[(e1pe2+e1me2)/2,(e1pe2-e1me2)/2,x,If[$TnuEqualT,x,znu],1],{e1me2,0.001,Max[10,15/x]},{e1pe2,2.001+Abs[e1me2],2+Abs[e1me2]+Max[10,15/x]},PrecisionGoal->3]
];



(* ::Input::Initialization:: *)
\[Lambda]nTOp5bodies[Tv_]:=(*\[Lambda]nTOp5bodies[Tv]=*)With[{x=me/(kB Tv),znu=me/(kB Tv T\[Nu]overT[Tv]),q=Q/me },
NIntegrate[IPENFiveBodyT0CN[en,k,x,If[$TnuEqualT,x,znu],1],{en,1,Max[20,20/x]},{k,en+q,en+q+Max[20,20/x]},PrecisionGoal->4]
];


(* ::Input::Initialization:: *)
\[Lambda]pTOnThermalTruePhoton[Tv_]:=(*\[Lambda]pTOnThermalTruePhoton[Tv]=*)If[Tv<10^8.2 (* When the temperature is too low it is better to put 0 *),0,
With[{x=me/(kB Tv),znu=me/(kB Tv T\[Nu]overT[Tv]),q=Q/me },
NIntegrate[IPENCCRTCN[en,k,x,If[$TnuEqualT,x,znu],-1],{k,0.001,Max[10,20/x]},{en,1.001,Max[10,20/x]},PrecisionGoal->4]
]];

\[Lambda]pTOnThermalDiffBremsstrahlung[Tv_]:=(*\[Lambda]pTOnThermalDiffBremsstrahlung[Tv]=*)If[Tv<10^8.2 (* When the temperature is too low it is better to put 0 *),0,
With[{x=me/(kB Tv),znu=me/(kB Tv T\[Nu]overT[Tv]),q=Q/me },
NIntegrate[IPENCCRDiffBremsstrahlungCN[en,k,x,If[$TnuEqualT,x,znu],-1],{en,1.001,Max[10,20/x]},{k,0.001,Abs[en-q],Abs[en+q],Max[10,20/x]},PrecisionGoal->4]
]];


\[Lambda]pTOnThermal[Tv_]:=(*\[Lambda]pTOnThermal[Tv]=*)If[Tv<10^8.2 (* When the temperature is too low it is better to put 0 *),0,
With[{x=me/(kB Tv),znu=me/(kB Tv T\[Nu]overT[Tv]),q=Q/me },
NIntegrate[C1dE[en,x,If[$TnuEqualT,x,znu],-1],{en,1,Max[25,150/x]}]
+NIntegrate[1/2C2dE1dE2CN[(e1pe2+e1me2)/2,(e1pe2-e1me2)/2,x,If[$TnuEqualT,x,znu],-1],{e1me2,-Max[10,15/x],-0.001},{e1pe2,2.001+Abs[e1me2],2+Abs[e1me2]+Max[10,15/x]},PrecisionGoal->3]
+NIntegrate[1/2C2dE1dE2CN[(e1pe2+e1me2)/2,(e1pe2-e1me2)/2,x,If[$TnuEqualT,x,znu],-1],{e1me2,0.001,Max[10,15/x]},{e1pe2,2.001+Abs[e1me2],2+Abs[e1me2]+Max[10,15/x]},PrecisionGoal->3]
]];


(* ::Input::Initialization:: *)
\[Lambda]nTOpCCRTh[Tv_]:=(\[Lambda]nTOpThermal[Tv]+\[Lambda]nTOpThermalTruePhoton[Tv]+If[$CorrectionBremsstrahlung,\[Lambda]nTOpThermalDiffBremsstrahlung[Tv],\[Lambda]nTOp5bodies[Tv]]);

\[Lambda]pTOnCCRTh[Tv_]:=(\[Lambda]pTOnThermal[Tv]+\[Lambda]pTOnThermalTruePhoton[Tv]+If[$CorrectionBremsstrahlung,\[Lambda]pTOnThermalDiffBremsstrahlung[Tv],0]);


(* ::Input::Initialization:: *)
\[Lambda]0:=If[$RadiativeCorrections,\[Lambda]Rad,\[Lambda]BORN]+If[$FiniteNucleonMass,\[Lambda]FM,0];


(* ::Input::Initialization:: *)
Clear[\[Lambda]nTOp,\[Lambda]pTOn,\[Lambda]nTOpNormalized,\[Lambda]pTOnNormalized];
\[Lambda]nTOpNormalized[Tv_]:=(\[Lambda]0)^-1 (
If[$RadiativeCorrections,\[Lambda]nTOpCCR[Tv],\[Lambda]nTOpBORN[Tv]]
+If[$RadiativeThermal,\[Lambda]nTOpCCRTh[Tv],0]
+If[$FiniteNucleonMass,If[$CoupledFMandRC,\[Lambda]nTOpFMCCR[Tv],\[Lambda]nTOpFMNoCCR[Tv]],0]
);

\[Lambda]pTOnNormalized[Tv_]:=(\[Lambda]0)^-1 (
If[$RadiativeCorrections,\[Lambda]pTOnCCR[Tv],\[Lambda]pTOnBORN[Tv]]
+If[$RadiativeThermal,\[Lambda]pTOnCCRTh[Tv],0]
+If[$FiniteNucleonMass,If[$CoupledFMandRC,\[Lambda]pTOnFMCCR[Tv],\[Lambda]pTOnFMNoCCR[Tv]],0]);


(* ::Input::Initialization:: *)
Clear[\[Lambda]nTOp]
\[Lambda]nTOp[Tv_]:=1/\[Tau]neutron \[Lambda]nTOpNormalized[Tv];
\[Lambda]pTOn[Tv_]:=1/\[Tau]neutron \[Lambda]pTOnNormalized[Tv];


(* ::Input::Initialization:: *)
LetterFromBoolean[Bool_]:=If[Bool,"T","F"];
StringFromBoolean[BoolList_List]:=StringJoin[LetterFromBoolean/@BoolList];
BooleanSuffix=StringFromBoolean[{$RadiativeCorrections,$RadiativeThermal,$FiniteNucleonMass,$CoupledFMandRC,$QEDPlasmaCorrections,$IncompleteNeutrinoDecoupling}]


(* ::Input::Initialization:: *)
NamePENFilenp="Interpolations/NP_RATES";
NamePENFilepn="Interpolations/PN_RATES";


(* ::Input::Initialization:: *)
$BornBool=Not[$RadiativeThermal]&&Not[$RadiativeCorrections]&&Not[$FiniteNucleonMass];


(* ::Input::Initialization:: *)
MyTableWeakRate:=If[$ParallelWeakRates,ParallelEvaluate[Off[NIntegrate::slwcon];];ParallelTable,Table]

PreComputeWeakRates:=(
Off[NIntegrate::slwcon];
\[Lambda]nTOpTab=MyTableWeakRate[{T,\[Lambda]nTOpNormalized[T]},{T,ListT}];
\[Lambda]pTOnTab=MyTableWeakRate[{T,\[Lambda]pTOnNormalized[T]},{T,ListT}];
TabRatenp=\[Lambda]nTOpTab;
TabRatepn=\[Lambda]pTOnTab;
On[NIntegrate::slwcon];
\[Lambda]nTOpI=MyInterpolationRate[ToExpression[TabRatenp]];
\[Lambda]pTOnI=MyInterpolationRate[ToExpression[TabRatepn]];
);


(* ::Input::Initialization:: *)
TabRatenp=Check[Import[NamePENFilenp,"TSV"],Print["Precomputed n -> p rate not found. We recompute the rates and store them. This can take very long"];$Failed,Import::nffil];

TabRatepn=Check[Import[NamePENFilepn,"TSV"],Print["Precomputed p -> n rate not found. We recompute the rates and store them. This can take very long"];$Failed,Import::nffil];

Timing[If[TabRatenp===$Failed||TabRatepn===$Failed||$RecomputeWeakRates,
PreComputeWeakRates;,
\[Lambda]nTOpIPre=MyInterpolationRate[ToExpression[TabRatenp]]@@@T\[Nu]overTTable;
\[Lambda]pTOnIPre=MyInterpolationRate[ToExpression[TabRatepn]]@@@T\[Nu]overTTable;
\[Lambda]nTOpI=MyInterpolationRate[Table[{ListT[[i]], \[Lambda]nTOpIPre[[i]]},{i,1,Length[ListT]}]];
\[Lambda]pTOnI=MyInterpolationRate[Table[{ListT[[i]], \[Lambda]pTOnIPre[[i]]},{i,1,Length[ListT]}]];
];]


(* ::Input::Initialization:: *)
LnTOp[Tv_]:=1/\[Tau]neutron*\[Lambda]nTOpI[Tv];
LpTOn[Tv_]:=1/\[Tau]neutron*\[Lambda]pTOnI[Tv];
LbarnTOp[Tv_]:=LpTOn[Tv];


(* ::Input::Initialization:: *)
1/LnTOp[Tf]


timingEnd = AbsoluteTime[];
Print["[PrimiCosmo]: Completed. Total Time - ", Round[timingEnd - timingStart], " seconds"]
