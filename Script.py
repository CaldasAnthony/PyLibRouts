import sys
sys.path.append('/Users/caldas/Pytmosph3R/ParaPyRouts/')

from pytransfert import *
from pyfunction import *
from pymatrix import *
from pygcmtreat import *
from pydataread import *
from pyconvert import *
from pyremind import *

########################################################################################################################
########################################################################################################################

# Informations diverses sur l'etude

path = "/Users/caldas/Pytmosph3R/"
name_file = "Files"
name_source = "Source"
#name_exo = "GJ1214b"
name_exo = "GJ1214b"
opac_file, param_file, stitch_file = 'Opacity', 'Parameters', 'Stitch'
version = 6.3

########################################################################################################################
########################################################################################################################

# Donnees de base

data_base,diag_file = "/Users/caldas/Pytmosph3R/Simulations/GJ1214b/",'diagfi'
data_source = "/Users/caldas/Pytmosph3R/Simulations/GJ1214b/Sources/"
planet = planet()
if diag_file == '' :
    information = pickle.load(open(planet.pressure_profile_data))
    information = information['params']
    reso_long, reso_lat = planet.longitude, planet.latitude

t, t_selec, phi_rot, phi_obli, inclinaison = 0, 5, 0.00, 0.00, 0.00

Record = True

########################################################################################################################

# Proprietes de l'exoplanete

#Rp = 0.246384689105*R_J
#Mp = 0.0206006322445*M_J
if diag_file == '' :
    Rp = information[planet.planet_radius_key]
    Mp = information[planet.planet_mass_key]
    g0 = G*Mp/(Rp**2)

# Proprietes de l'etoile hote

#Rs = 0.206470165349*R_S
#Ts = 3000.
if diag_file == '' :
    Rs = information[planet.star_radius_key]
    Ts = information[planet.star_temperature_key]
else :
    Rs = 0.206470165349*R_S
    Ts = 3000.

d_al = 100.*9.461e+15
error = np.array([1.e-5])

# Proprietes en cas de lecture d'un diagfi

if data_base != '' :
    Rp, g0, reso_long, reso_lat, long_lat, Inverse = diag('%s%s'%(data_base,diag_file))
else :
    long_lat = np.zeros((2,int(np.amax(np.array([reso_long,reso_lat])))+1))
    degpi = np.pi/180.
    long_lat[0,0:reso_long+1] = np.linspace(-180.*degpi,180.*degpi,reso_long+1,dtype=np.float64)
    long_lat[1,0:reso_lat+1] = np.linspace(-90*degpi,90.*degpi,reso_lat+1,dtype=np.float64)
    Inverse = 'False'

alpha_step, delta_step = 2*np.pi/np.float(reso_long), np.pi/np.float(reso_lat)

# Proprietes de l'atmosphere

#n_species = np.array(['H2','He','H2O','CH4','N2','NH3','CO','CO2'])
#n_species_active = np.array(['H2O','CH4','NH3','CO','CO2'])
if data_base != '' :
    n_species = np.array(['H2','He','H2O','CH4','N2','NH3','CO','CO2'])
    n_species_active = np.array(['H2O','CH4','NH3','CO','CO2'])
else :
    n_species = np.array(['H2','He','H2O'])
    n_species_active = np.array([information[planet.active_species_key]])

# Proprietes de l'atmosphere isotherme

if data_base != '' :
    T_iso, P_surf = 0,0
    x_ratio_species_active = np.array([0,0])
    IsoComp = False
else :
    T_iso, P_surf = information[planet.planet_temperature_key], information[planet.extreme_pressure_key[0]]
    x_ratio_species_active = information[planet.planet_active_ratio_key]
    Isocomp = True

M_species, M, x_ratio_species = ratio(n_species,x_ratio_species_active,IsoComp)

# Proprietes des nuages

c_species = np.array(['gen_cond','gen_cond2'])
c_species_name = np.array(['KCl','ZnS'])
c_species_file = np.array(['KCl','ZnS'])
rho_p = np.array([1980.,4090.])
r_eff = np.array([0.5e-6,0.5e-6])

########################################################################################################################

# Crossection

n_species_cross = np.array(['H2O','CH4','NH3','CO','CO2'])
m_species = np.array([])
m_file = np.array([])
domain, domainn, source = "IR", "IR", "bin10"
dim_bande, dim_gauss = 3000, 16

# Selection des sections efficaces

ind_cross, ind_active = index_active (n_species,n_species_cross,n_species_active)

# Informations generale sur les donnees continuum

#cont_tot = np.array(['H2-He_2011.cia','H2-He_2011.cia','H2O_CONT_SELF.dat','H2O_CONT_FOREIGN.dat','H2-CH4_eq_2011.cia','N2-H2_2011.cia'])
#cont_species = np.array(['H2','He','H2Os','H2O','CH4','N2'])
#cont_associations = np.array(['h2h2','h2he','h2oh2o','h2ofor','h2ch4','h2n2'])
cont_tot = np.array(['H2-He_2011.cia','H2-He_2011.cia','H2O_CONT_SELF.dat','H2O_CONT_FOREIGN.dat'])
cont_species = np.array(['H2','He','H2Os','H2O'])
cont_associations = np.array(['h2h2','h2he','h2oh2o','h2ofor'])
#cont_tot = np.array(['H2O_CONT_SELF.dat','H2O_CONT_FOREIGN.dat'])
#cont_species = np.array(['H2Os','H2O'])
#cont_associations = np.array(['h2oh2o','h2ofor'])

########################################################################################################################

# Proprietes de maille

if data_base != '' :
    h, P_h, n_layers = 9.e+6, 1.e-4, 100
else :
    h, P_h, n_layers = 9.e+6, information[planet.extreme_pressure_key[1]], information[planet.number_layer_key]

delta_z, r_step, x_step, theta_number = 10e+4, 10e+4, 10e+4, 2*reso_lat
z_array = np.arange(h/np.float(delta_z)+1)*float(delta_z)
theta_step = 2*np.pi/np.float(theta_number)
Upper = "Isotherme"
number = 3 + n_species.size + m_species.size + c_species.size

# Choix dans la section de la maille

lim_alt, rupt_alt = 0.e+0, 0.e+0
lat, long = 0, 0
z_lim = int(lim_alt/delta_z)
z_reso = int(h/delta_z) + 1

# En cas de modulation de la densite

type = np.array(['',0.00])

########################################################################################################################
########################################################################################################################

# Importation des donnees GCM

if Record == True :

    class composition :
        def __init__(self):
            self.file = ''
            self.parameters = np.array(['T','p'])
            self.species = n_species
            self.ratio = ratio_HeH2
            self.renorm = np.array([])
    class aerosol :
        def __init__(self):
            self.number = c_species.size
            self.species = c_species
            self.nspecies = c_species_name
            self.file_name = c_species_file
            self.continuity = np.array([False,False])
    class continuum :
        def __init__(self) :
            self.number = cont_tot.size
            self.associations = cont_associations
            self.species = cont_species
            self.file_name = cont_tot
    class kcorr :
        def __init__(self) :
            self.resolution = ''
            self.resolution_n = np.array([dim_bande,dim_gauss])
            self.type = np.array(['IR'])
            self.parameters = np.array(['T','p','Q'])
            self.tracer = m_file
            self.exception = np.array([])
            self.jump = True
    class crossection :
        def __init__(self) :
            self.file = '/data1/caldas/Pytmosph3R/xsec/10wno/'
            self.type = source
            self.species = n_species_cross
            self.type_ref = np.array(['xsecarr','wno','p','t'])

    data_record(path,name_source,data_source,name_exo,aerosol(),continuum(),kcorr(),crossection(),composition(),Renorm=False)
else :
    class continuum :
        def __init__(self) :
            self.number = cont_tot.size
            self.associations = cont_associations
            self.species = cont_species
            self.file_name = cont_tot

########################################################################################################################
########################################################################################################################

# Les options choisies lors de l'etude

Tracer = False          ###### S'il y a des marqueurs
Cloudy = True          ###### S'il y a des nuages
Middle = True          ###### Construction de la maille sur le milieu des couches
NoH2 = False            ###### Une atmosphere sans H2 et He ou avec
TauREx = False          ###### Une atmosphere TauREx

########################################################################################################################

# Parameters

Parameters = True

Profil = True          ###### Reproduit la maille spherique en altitude

Box = True             ###### Transpose la maille spherique en altitude
Layers = True          ###### Decoupage en nombre de couche
Surf = True            ###### Si des donnees de surface sont accessibles
LogInterp = False       ###### Interpolation de la pression via le logarithme
N_fixe = True          ###### Si nous voulons imposer le nombre de couche atmospherique
TopPressure = 'Up'    ###### Si nous voulons fixer le toit de l'atmosphere par rapport a une pression minimale
MassAtm = False         ###### Si on tient compte de la masse atmospherique
compo_type = np.array(['composition'])

Cylindre = False        ###### Construit la maille cylindrique
Inclinaison = False     ###### En presence d'une inclinaison par rapport au plan ecliptique, l'angle d'inclinaison est
                             # defini positivement si le parametre d'impact est negatif et qu'elle passe en dessous de
                             # l'etoile
Obliquity = False       ###### Si l'exoplanete est inclinee

Corr = False            ###### Traite les parcours optiques
Gravity = False         ###### Pour travailler a gravite constante
Discret = True         ###### Calcul les distances discretes
Integral = True        ###### Effectue l'integration sur les chemins optiques
Ord = False             ###### Si Discreet == False, Ord permet de calculer les indices avec l'integration

Matrix = False          ###### Transposition de la maille spherique dans la maille cylindrique

Convert = False       ###### Lance la fonction convertator qui assure l'interpolation des sections efficaces
Kcorr = False           ###### Sections efficaces ou k-correles
Molecul = False       ###### Effectue les calculs pour l'absorption moleculaire
Cont = False            ###### Effectue les calculs pour l'absorption par les collisions
Scatt = True           ###### Effectue les calculs pour l'absorption par diffusion Rayleigh
Cl = False              ###### Effectue les calculs pour l'absorption par les nuages
Optimal = False         ###### Interpolation optimal (Waldmann et al.)
TimeSelec = True       ###### Si nous etudions un temps precis de la simulation

########################################################################################################################

# Cylindric transfert

Cylindric_transfert_3D = True

Molecular = True       ###### Ne tiens pas compte de l'absorption moleculaire
Continuum = True       ###### Tiens compte de l'absorption par les collisions
Scattering = True      ###### Tiens compte de l'absorption par la diffusion
Clouds = False          ###### Tiens compte de l'absoprtion par les nuages
Single = "no"           ###### Isole une espece de nuage
Rupt = False            ###### Si l'atmosphere est tronquee
Discreet = True        ###### Calcul discret
Integration = False     ###### Calcul integral
Module = False          ###### Si nous souhaitons moduler la densite de reference

D3Maille = False        ###### Si nous voulons resoudre le transfert dans la maille 3D
TimeSel = True         ###### Si nous etudions un temps precis de la simulation

########################################################################################################################

Script = True          ###### Si nous voulons avoir une version .dat du spectre
ErrOr = False           ###### Si calculons le bruit de photon pour un instrument donne
detection = JWST()
Noise = False           ###### Si nous voulons bruiter le signal a partir du bruit de photon calcule
resolution = 'low'
Push = np.array([False,False,False,False])
###### Si nous voulons forcer le code a faire les spcectres intermediaires meme s'ils existent

########################################################################################################################

# Plot

View = False

Radius = True          ###### Spectre rayon effective = f(longueur d'onde)
Flux = False            ###### Spectre flux = f(longueur d'onde)

########################################################################################################################
########################################################################################################################

# Sauvegardes

save_adress = "/Users/caldas/Pytmosph3R/I/"
special = ''
stud = stud_type(r_eff,Single,Continuum,Molecular,Scattering,Clouds)
save_name_1D = saving('1D',type,special,save_adress,version,name_exo,reso_long,reso_lat,t,h,dim_bande,dim_gauss,r_step,\
            inclinaison,phi_rot,phi_obli,r_eff,domain,stud,lim_alt,rupt_alt,long,lat,Discreet,Integration,Module,Optimal,Kcorr,False)
save_name_3D = saving('3D',type,special,save_adress,version,name_exo,reso_long,reso_lat,t,h,dim_bande,dim_gauss,r_step,\
            inclinaison,phi_rot,phi_obli,r_eff,domain,stud,lim_alt,rupt_alt,long,lat,Discreet,Integration,Module,Optimal,Kcorr,False)

########################################################################################################################
