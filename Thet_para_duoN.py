import os
import sys
sys.path.append('/data1/caldas/Pytmosph3R/PyLibRouts/')

from pytransfert import *
from pyfunction import *
from pymatrix import *
from pydataread import *
from pyremind import *
from pyparagcmtreat import *
from pyparaconvert import *

from mpi4py import MPI

sys.path.append('/data1/caldas/Pytmosph3R/Tools/')

from Tool_pressure_generator import *

########################################################################################################################
########################################################################################################################

# Initialisation de la parallelisation

comm = MPI.COMM_WORLD
rank = comm.rank
number_rank = comm.size

########################################################################################################################
########################################################################################################################

# Informations diverses sur l'etude

path = "/data1/caldas/Pytmosph3R/"
name_file = "Files/Para"
name_source = "Source"
#name_exo = "HD209458"
name_exo = "GJ1214b"
opac_file, param_file, stitch_file = 'Opacity', 'Parameters', 'Stitch'
version = 6.3

########################################################################################################################
########################################################################################################################

# Donnees de base

reso_long, reso_lat = 64, 48
t, t_selec, phi_rot, phi_obli, inclinaison = 0, 5, 0.00, 0.00, 0.00
data_source = "/data1/caldas/Pytmosph3R/Simulations/Trappist/Sources/"
lat_obs, long_obs = 0.00, 0.000

Record = False

########################################################################################################################

# Proprietes de l'exoplanete

#Rp = 15.*R_T
#Mp = 220.*M_T

Rp = 0.246384689105*R_J
Mp = 0.0206006322445*M_J
g0 = G*Mp/(Rp**2)

# Proprietes de l'etoile hote

#Rs = 1.155*R_S
#Ts = 6065.
#d_al = 154*9.461e+15

Rs = 0.206470165349*R_S
Ts = 3000.
d_al = 42.4*9.461e+15

error = np.array([1.e-5])

# Proprietes en cas de lecture d'un diagfi

long_step, lat_step = 2*np.pi/np.float(reso_long), np.pi/np.float(reso_lat)
long_lat = np.zeros((2,int(np.amax(np.array([reso_long,reso_lat])))+1))
degpi = np.pi/180.
long_lat[0,0:reso_long+1] = np.linspace(-180.*degpi,180.*degpi,reso_long+1,dtype=np.float64)
long_lat[1,0:reso_lat+1] = np.linspace(-90*degpi,90.*degpi,reso_lat+1,dtype=np.float64)
Inverse = 'False'

# Proprietes de l'atmosphere

#n_species = np.array(['H2','He','H2O','CH4','N2','NH3','CO','CO2'])
#n_species_active = np.array(['H2O','CH4','NH3','CO','CO2'])
n_species = np.array(['H2','He','H2O'])
n_species_active = np.array(['H2O'])

# Proprietes de l'atmosphere isotherme

#T_iso_array, P_surf, P_tau = np.array([1000.,2000.]), 1.e+6, 1.e+3
#x_ratio_species_active = np.array([0.01,0.01,0.01,0.01,0.01,0.01])
#x_ratio_species_inactive = np.array([0.01])
T_iso, P_surf, P_tau = 500, 1.e+6, 1.e+3
x_ratio_species_active_array = np.array([0.10,0.20])
x_ratio_species_inactive = np.array([])
M = np.zeros(2,dtype=np.float)
M_species, M[0], x_ratio_species = ratio(n_species,np.array([x_ratio_species_active_array[0]]),IsoComp=True)
M_species, M[1], x_ratio_species = ratio(n_species,np.array([x_ratio_species_active_array[1]]),IsoComp=True)

# Proprietes des nuages

c_species = np.array([])
c_species_name = np.array([])
c_species_file = np.array([])
rho_p = np.array([])
r_eff = np.array([])

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
#cont_tot = np.array(['H2-He_2011.cia','H2-He_2011.cia','H2O_CONT_SELF.dat','H2O_CONT_FOREIGN.dat'])
#cont_species = np.array(['H2','He','H2Os','H2O'])
#cont_associations = np.array(['h2h2','h2he','h2oh2o','h2ofor'])
cont_tot = np.array(['H2-H2_2011.cia','H2-He_2011.cia','H2O_CONT_SELF.dat','H2O_CONT_FOREIGN.dat'])
cont_species = np.array(['H2','He','H2Os','H2O'])
cont_associations = np.array(['h2h2','h2he','h2oh2o','h2ofor'])

########################################################################################################################

# Proprietes de maille

h, P_h, n_layers = 1.36e+7, 1.e-4, 100
delta_z, r_step, x_step, theta_number = 3.0e+4, 3.0e+4, 3.0e+4, 96
z_array = np.arange(h/np.float(delta_z)+1)*float(delta_z)
theta_step = 2*np.pi/np.float(theta_number)
Upper = "Isotherme"
number = 3 + n_species.size + m_species.size + c_species.size

# Choix dans la section de la maille

lim_alt, rupt_alt, beta = h, 0.e+0, np.linspace(0,theta_number/4,theta_number/4+1)*360./np.float(theta_number)
beta_rad_array = beta*2*np.pi/(360.)
lat, long = 24, 47
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
            self.resolution = '38x36'
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

    if rank == 0 :
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
Cloudy = False          ###### S'il y a des nuages
Middle = True          ###### Construction de la maille sur le milieu des couches
NoH2 = False            ###### Une atmosphere sans H2 et He ou avec
TauREx = False          ###### Une atmosphere TauREx

########################################################################################################################

# Parameters

Profil = True

Surf = True            ###### Si des donnees de surface sont accessibles
LogInterp = False       ###### Interpolation de la pression via le logarithme
TopPressure = True     ###### Si nous voulons fixer le toit de l'atmosphere par rapport a une pression minimale
Composition = False     ###### Se placer a l'equilibre thermodynamique
obs = np.array([lat_obs,long_obs,'NotModified',long_obs])

Parameters = True

Corr = True            ###### Traite les parcours optiques
Integral = False        ###### Effectue l'integration sur les chemins optiques
Cylindre = True        ###### Construit la maille cylindrique
Gravity = False         ###### Pour travailler a gravite constante

Matrix = True          ###### Transposition de la maille spherique dans la maille cylindrique

Convert = True         ###### Lance la fonction convertator qui assure l'interpolation des sections efficaces
Kcorr = False           ###### Sections efficaces ou k-correles
Molecul = True       ###### Effectue les calculs pour l'absorption moleculaire
Cont = True            ###### Effectue les calculs pour l'absorption par les collisions
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
LimTop = False         ###### Si on limite l'altitude max
Discreet = True        ###### Calcul discret
Integration = False     ###### Calcul integral
Module = False          ###### Si nous souhaitons moduler la densite de reference

D3Maille = False        ###### Si nous voulons resoudre le transfert dans la maille 3D
TimeSel = True         ###### Si nous etudions un temps precis de la simulation

########################################################################################################################

Script = True          ###### Si nous voulons avoir une version .dat du spectre
ErrOr = True           ###### Si calculons le bruit de photon pour un instrument donne
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

save_adress = "/data1/caldas/Pytmosph3R/I/"
special = ''
if rank == 0 :
    stud = stud_type(r_eff,Single,Continuum,Molecular,Scattering,Clouds)
    save_name_1D = saving('1D',type,special,save_adress,version,name_exo,reso_long,reso_lat,t,h,dim_bande,dim_gauss,r_step,\
            obs,r_eff,domain,stud,lim_alt,rupt_alt,long,lat,Discreet,Integration,Module,Optimal,Kcorr,False)
    save_name_3D = saving('3D',type,special,save_adress,version,name_exo,reso_long,reso_lat,t,h,dim_bande,dim_gauss,r_step,\
            obs,r_eff,domain,stud,lim_alt,rupt_alt,long,lat,Discreet,Integration,Module,Optimal,Kcorr,False)

########################################################################################################################

if rank == 0 :
    print 'Ratio He/H2 : ',ratio_HeH2

########################################################################################################################
########################################################################################################################

reso_alt = int(h/1000)
reso_long = int(reso_long)
reso_lat = int(reso_lat)

if rank == 0 :

    message_clouds = ''
    if Cloudy == True :
        for i in range(c_species.size) :
            message_clouds += '%s (%.2f microns/%.3f)  '%(c_species[i],r_eff[i]*10**6,rho_p[i]/1000.)
        print 'Clouds in the atmosphere (grain radius/density) : %s'%(message_clouds)
    else :
        print 'There is no clouds'
    print 'Mean radius of the exoplanet : %i m'%(Rp)
    print 'Mean surface gravity : %.2f m/s^2'%(g0)
    print 'Mean molar mass : %.5f kg/mol'%(M[0])
    print 'Mean molar mass : %.5f kg/mol'%(M[1])
    print 'Extrapolation type for the upper atmosphere : %s'%(Upper)
    number = 2 + m_species.size + c_species.size + n_species.size + 1
    print 'Resolution of the GCM simulation (latitude/longitude) : %i/%i'%(reso_lat,reso_long)
    print 'Position of the observer (long,lat) : (%.3f,%.3f)'%(long_obs,lat_obs)

########################################################################################################################





########################################################################################################################
########################################################################################################################
###########################################      PARAMETERS      #######################################################
########################################################################################################################
########################################################################################################################

# Telechargement des sections efficaces ou des k_correles

########################################################################################################################

X_min, X_max = np.amin(x_ratio_species_active_array), np.amax(x_ratio_species_active_array)
X_min_log, X_max_log = np.log10(X_min), np.log10(X_max)

for beta_rad in beta_rad_array :

    beta = beta_rad*360./(2*np.pi)
    stu_name = '_%.3f'%(beta)

    if Profil == True :

        if TopPressure == True :
            M_Ref = np.amin(M)
            alp = R_gp*T_iso/(g0*M_Ref)*np.log(P_h/P_surf)
            h_top = round(-alp/(1+alp/Rp),-5)
            delta_z, r_step, x_step = h_top/np.float(n_layers), h_top/np.float(n_layers), h_top/np.float(n_layers)
            if rank == 0 :
                print 'Width of layers : %i m'%(delta_z)
                print 'Top of the atmosphere : %i m'%(h_top)

        data_convert = np.zeros((number,1,n_layers+2,reso_lat+1,reso_long+1))
        d_lim = (Rp+h)*np.cos(np.pi/2.-beta_rad)
        alp_max = R_gp*T_iso/(g0*np.amax(M))*np.log(P_tau/P_surf)
        n_lim = -alp_max/(1+alp_max/Rp)

        if rank == 0 :
            bar = ProgressBar(reso_lat+1,'Data generation')

        for i_lat in range(reso_lat+1) :
            for i_long in range(reso_long+1) :
                z_maxi = 0
                phi_lat = -np.pi/2.+i_lat*np.pi/(np.float(reso_lat))
                phi_long = -np.pi + i_long*2*np.pi/(reso_long)
                x = np.abs((Rp+h)*np.cos(phi_lat)*np.cos(phi_long))

                if x <= d_lim and i_lat != 0  and i_lat != reso_lat and beta_rad >= theta_step :
                    if i_long >= 0. and i_long < reso_long/4. :
                        X = 10**(X_min_log + (d_lim - x)*(X_max_log-X_min_log)/(2*d_lim))
                    if i_long >= 3*reso_long/4. and i_long < reso_long :
                        X = 10**(X_min_log + (d_lim - x)*(X_max_log-X_min_log)/(2*d_lim))
                    if i_long >= reso_long/4. and i_long < reso_long/2. :
                        X = 10**(X_max_log - (d_lim - x)*(X_max_log-X_min_log)/(2*d_lim))
                    if i_long >= reso_long/2. and i_long < 3*reso_long/4. :
                        X = 10**(X_max_log - (d_lim - x)*(X_max_log-X_min_log)/(2*d_lim))
                else :
                    if i_long >= reso_long/4. and i_long < 3.*reso_long/4. :
                        X = X_max
                    else :
                        X = X_min
                if i_lat == 0 or i_lat == reso_lat :
                    if beta_rad >= theta_step :
                        X = 10**((X_max_log+X_min_log)/2.)
                    else :
                        if i_long >= reso_long/4. and i_long < 3.*reso_long/4. :
                            X = X_max
                        else :
                            X = X_min

                for i_n in range(n_layers+2) :
                    if i_n == 0 :
                        z = 0
                    else :
                        if i_n == n_layers+1 :
                            z = h_top
                        else :
                            z = (i_n-0.5)*delta_z

                    if z < n_lim :
                        data_convert[4,0,i_n,i_lat,i_long] = X_max
                    else :
                        if z_maxi == 0 :
                            z_maxi = z - delta_z
                            P_top = data_convert[0,0,i_n-1,i_lat,i_long]
                            data_convert[4,0,i_n,i_lat,i_long] = X
                    data_convert[1,0,i_n,i_lat,i_long] = T_iso

                    M_species, M_x, x_ratio_species = ratio(n_species,np.array([X]),IsoComp=True)

                    if z < n_lim :
                        data_convert[0,0,i_n,i_lat,i_long] = P_surf*np.exp(-g0*M_x/(R_gp*T_iso)*z/(1+z/Rp))
                    else :
                        data_convert[0,0,i_n,i_lat,i_long] = P_top*np.exp(-g0*(1/(1+z_maxi/Rp))**2*M_x/(R_gp*T_iso)*(z-z_maxi)/(1+(z-z_maxi)/(Rp+z_maxi)))
                    data_convert[2:2+n_species.size,0,i_n,i_lat,i_long] = x_ratio_species
                    data_convert[number-1,0,i_n,i_lat,i_long] = M_x

            if rank == 0 :
                bar.animate(i_lat+1)


        if TopPressure == True :
            h = h_top
            reso_alt = int(h/1000)
            z_array = np.arange(h/np.float(delta_z)+1)*float(delta_z)
            if LimTop == False :
                lim_alt = h
            save_adress = "/data1/caldas/Pytmosph3R/I/"
            save_name_3D = "%s%s_3D_duo_linear_real_%.3f_%.3f_%i_%.2f"%(save_adress,name_exo,X_min,X_max,beta,P_tau/(1.e+5))
            save_name_3D_dat = "%sdat/%s_3D_duo_linear_real_%.3f_%.3f_%i_%.2f"%(save_adress,name_exo,X_min,X_max,beta,P_tau/(1.e+5))
            if Noise == True :
                save_name_3D = '%s_n'%(save_name_3D)
                save_name_3D_dat = '%s_n'%(save_name_3D_dat)

        np.save("%s%s/%s/%s_data_convert_%ix%ix%i_lo%.2f%s.npy"%(path,name_file,param_file,name_exo,reso_alt,reso_long,reso_lat,long_obs,stu_name),\
                    data_convert)

########################################################################################################################

    if Parameters == True :

        if Corr == True and beta_rad == 0. :

                                        ###### Parallele encoding init ######

            n_level_rank = repartition(n_layers+1,number_rank,rank,False)

                                        ###### Parallele encoding end ######

            path_cyl = '%s%s/%s/'%(path,name_file,stitch_file)
            data = '%s%s/%s/%s_data_convert_%ix%ix%i_lo%.2f%s.npy'%(path,name_file,param_file,name_exo,reso_alt,reso_long,reso_lat,long_obs,stu_name)

            q_lat_grid_n, q_long_grid_n, q_z_grid_n, q_zh_grid_n, dx_grid_opt_n, pdx_grid_n, order_grid_n = \
                dx_correspondance(data,path_cyl,x_step,r_step,theta_number,Rp,g0,h,t,n_layers,reso_long,reso_lat,reso_alt,obs,n_level_rank,\
                              Middle,Cylindre,Integral,Gravity)

            obs = np.array([lat_obs,long_obs],dtype=np.float64)

                                        ###### Parallele encoding init ######

            for r_n in range(number_rank) :
                if r_n == 0 and rank == 0 :
                    length = np.zeros(number_rank,dtype=np.int)
                    length[0] = np.shape(dx_grid_opt_n)[2]
                elif r_n == 0 and rank != 0 :
                    sh_dx_n = np.array(np.shape(dx_grid_opt_n)[2],dtype=np.int)
                    comm.Send([sh_dx_n,MPI.INT],dest=0,tag=0)
                elif r_n != 0 and rank == 0 :
                    sh_dx = np.zeros(1,dtype=np.int)
                    comm.Recv([sh_dx,MPI.INT],source=r_n,tag=0)
                    length[r_n] = sh_dx[0]

            if rank == 0 :
                x_size = np.amax(length)

            comm.Barrier()

            if Cylindre == True :

                if rank == 0 :
                    sh_grid = np.shape(q_lat_grid_n)
                    q_lat_grid = np.ones((n_layers+1,theta_number,x_size),dtype=np.int)*(-1)
                    q_lat_grid[n_level_rank,:,:sh_grid[2]] = q_lat_grid_n

                comm.Barrier()

                for r_n in range(number_rank) :
                    if rank != 0 and r_n == rank :
                        sh_grid = np.array(np.shape(q_lat_grid_n),dtype=np.int)
                        q_lat_grid_n = np.array(q_lat_grid_n,dtype=np.int)
                        comm.Send([sh_grid,MPI.INT],dest=0,tag=3)
                        comm.Send([n_level_rank,MPI.INT],dest=0,tag=4)
                        comm.Send([q_lat_grid_n,MPI.INT],dest=0,tag=5)
                    elif rank == 0 and r_n != 0 :
                        sh_grid_ne = np.zeros(3,dtype=np.int)
                        comm.Recv([sh_grid_ne,MPI.INT],source=r_n,tag=3)
                        n_level_rank_ne = np.zeros(sh_grid_ne[0],dtype=np.int)
                        comm.Recv([n_level_rank_ne,MPI.INT],source=r_n,tag=4)
                        q_lat_grid_ne = np.zeros((sh_grid_ne),dtype=np.int)
                        comm.Recv([q_lat_grid_ne,MPI.INT],source=r_n,tag=5)
                        q_lat_grid[n_level_rank_ne,:,:sh_grid_ne[2]] = q_lat_grid_ne

                                            ###### Parallele encoding end ######

                if rank == 0 :
                    np.save("%s%s/%s/q_lat_grid_%i_%ix%ix%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                        reso_alt,r_step,obs[0],obs[1]),q_lat_grid)
                    print 'Reconstitution of the latitude grid finished with success'
                    del q_lat_grid, q_lat_grid_ne
                del q_lat_grid_n

                comm.Barrier()

                                            ###### Parallele encoding init ######

                for r_n in range(number_rank) :
                    if rank != 0 and r_n == 0 :
                        sh_grid = np.array(np.shape(q_long_grid_n),dtype=np.int)
                        q_long_grid_n = np.array(q_long_grid_n,dtype=np.int)
                        comm.Send([sh_grid,MPI.INT],dest=0,tag=20)
                        comm.Send([n_level_rank,MPI.INT],dest=0,tag=21)
                        comm.Send([q_long_grid_n,MPI.INT],dest=0,tag=22)
                    elif rank == 0 and r_n == 0 :
                        sh_grid = np.shape(q_long_grid_n)
                        q_long_grid = np.ones((n_layers+1,theta_number,x_size),dtype=np.int)*(-1)
                        q_long_grid[n_level_rank,:,:sh_grid[2]] = q_long_grid_n
                    elif rank == 0 and r_n !=0 :
                        sh_grid_ne = np.zeros(3,dtype=np.int)
                        comm.Recv([sh_grid_ne,MPI.INT],source=r_n,tag=20)
                        n_level_rank_ne = np.zeros(sh_grid_ne[0],dtype=np.int)
                        comm.Recv([n_level_rank_ne,MPI.INT],source=r_n,tag=21)
                        q_long_grid_ne = np.zeros((sh_grid_ne),dtype=np.int)
                        comm.Recv([q_long_grid_ne,MPI.INT],source=r_n,tag=22)
                        q_long_grid[n_level_rank_ne,:,:sh_grid_ne[2]] = q_long_grid_ne

                                            ###### Parallele encoding end ######

                if rank == 0 :
                    np.save("%s%s/%s/q_long_grid_%i_%ix%ix%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                            reso_alt,r_step,obs[0],obs[1]),q_long_grid)
                    print 'Reconstitution of the longitude grid finished with success'
                    del q_long_grid, q_long_grid_ne
                del q_long_grid_n

                comm.Barrier()

                                            ###### Parallele encoding init ######

                for r_n in range(number_rank) :
                    if rank != 0 and r_n == 0 :
                        sh_grid = np.array(np.shape(q_z_grid_n),dtype=np.int)
                        q_z_grid_n = np.array(q_z_grid_n,dtype=np.int)
                        comm.Send([sh_grid,MPI.INT],dest=0,tag=10)
                        comm.Send([n_level_rank,MPI.INT],dest=0,tag=11)
                        comm.Send([q_z_grid_n,MPI.INT],dest=0,tag=12)
                    elif rank == 0 and r_n == 0 :
                        sh_grid = np.shape(q_z_grid_n)
                        q_z_grid = np.ones((n_layers+1,theta_number,x_size),dtype=np.int)*(-1)
                        q_z_grid[n_level_rank,:,:sh_grid[2]] = q_z_grid_n
                    elif r_n != 0 and rank == 0 :
                        sh_grid_ne = np.zeros(3,dtype=np.int)
                        comm.Recv([sh_grid_ne,MPI.INT],source=r_n,tag=10)
                        n_level_rank_ne = np.zeros(sh_grid_ne[0],dtype=np.int)
                        comm.Recv([n_level_rank_ne,MPI.INT],source=r_n,tag=11)
                        q_z_grid_ne = np.zeros((sh_grid_ne),dtype=np.int)
                        comm.Recv([q_z_grid_ne,MPI.INT],source=r_n,tag=12)
                        q_z_grid[n_level_rank_ne,:,:sh_grid_ne[2]] = q_z_grid_ne

                                            ###### Parallele encoding end ######

                if rank == 0 :
                    np.save("%s%s/%s/q_z_grid_%i_%ix%ix%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                            reso_alt,r_step,obs[0],obs[1]),q_z_grid)
                    print 'Reconstitution of the altitude grid finished with success'
                    del q_z_grid, q_z_grid_ne
                del q_z_grid_n

                                            ###### Parallele encoding init ######

                for r_n in range(number_rank) :
                    if rank != 0 and r_n == 0 :
                        sh_grid = np.array(np.shape(q_zh_grid_n),dtype=np.int)
                        q_zh_grid_n = np.array(q_zh_grid_n,dtype=np.float64)
                        comm.Send([sh_grid,MPI.INT],dest=0,tag=30)
                        comm.Send([n_level_rank,MPI.INT],dest=0,tag=31)
                        comm.Send([q_zh_grid_n,MPI.DOUBLE],dest=0,tag=32)
                    elif rank == 0 and r_n == 0 :
                        sh_grid = np.shape(q_zh_grid_n)
                        q_zh_grid = np.ones((n_layers+1,theta_number,x_size),dtype=np.float64)
                        q_zh_grid[n_level_rank,:,:sh_grid[2]] = q_zh_grid_n
                    elif r_n != 0 and rank == 0 :
                        sh_grid_ne = np.zeros(3,dtype=np.int)
                        comm.Recv([sh_grid_ne,MPI.INT],source=r_n,tag=30)
                        n_level_rank_ne = np.zeros(sh_grid_ne[0],dtype=np.int)
                        comm.Recv([n_level_rank_ne,MPI.INT],source=r_n,tag=31)
                        q_zh_grid_ne = np.zeros((sh_grid_ne),dtype=np.int)
                        comm.Recv([q_zh_grid_ne,MPI.DOUBLE],source=r_n,tag=32)
                        q_zh_grid[n_level_rank_ne,:,:sh_grid_ne[2]] = q_zh_grid_ne

                                            ###### Parallele encoding end ######

                if rank == 0 :
                    np.save("%s%s/%s/q_zh_grid_%i_%ix%ix%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                            reso_alt,r_step,obs[0],obs[1]),q_zh_grid)
                    print 'Reconstitution of the integral altitude grid finished with success'
                    del q_zh_grid, q_zh_grid_ne
                del q_zh_grid_n

                                            ###### Parallele encoding init ######

                if rank == 0 :
                    x_size = np.amax(length)
                    dx_grid_opt = np.zeros((n_layers+1,theta_number,x_size),dtype=np.float64)
                    order_grid = np.zeros((3,n_layers+1,theta_number,x_size),dtype=np.int)
                    dx_grid_opt[n_level_rank,:,:length[0]] = dx_grid_opt_n
                    order_grid[:,n_level_rank,:,:length[0]] = order_grid_n

                for r_n in range(number_rank) :
                    if r_n == 0 and rank != 0 :
                        order_grid_n = np.array(order_grid_n,dtype=np.int)
                        dx_grid_opt_n = np.array(dx_grid_opt_n,dtype=np.float64)
                        comm.Send([dx_grid_opt_n,MPI.DOUBLE],dest=0,tag=rank+1)
                        comm.Send([order_grid_n,MPI.INT],dest=0,tag=rank+2)
                    elif r_n != 0 and rank == 0 :
                        n_level_rank_ne = repartition(n_layers+1,number_rank,r_n,False)
                        dx_grid_opt_ne = np.zeros((n_level_rank_ne.size,theta_number,length[r_n]),dtype=np.float64)
                        comm.Recv([dx_grid_opt_ne,MPI.DOUBLE],source=r_n,tag=r_n+1)
                        order_grid_ne = np.zeros((3,n_level_rank_ne.size,theta_number,length[r_n]),dtype=np.int)
                        comm.Recv([order_grid_ne,MPI.INT],source=r_n,tag=r_n+2)
                        dx_grid_opt[n_level_rank_ne,:,:length[r_n]] = dx_grid_opt_ne
                        order_grid[:,n_level_rank_ne,:,:length[r_n]] = order_grid_ne
                        if length[r_n] != x_size :
                            dx_grid_opt[n_level_rank_ne,:,length[r_n]:x_size] = np.ones((n_level_rank_ne.size,theta_number,x_size-length[r_n]))*(-1)
                            order_grid[:,n_level_rank_ne,:,length[r_n]:x_size] = np.ones((3,n_level_rank_ne.size,theta_number,x_size-length[r_n]))*(-1)

                                            ###### Parallele encoding end ######

                if rank == 0 :
                    np.save("%s%s/%s/dx_grid_opt_%i_%ix%ix%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                                reso_lat,reso_alt,r_step,obs[0],obs[1]),dx_grid_opt)
                    np.save("%s%s/%s/order_grid_%i_%ix%ix%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                                reso_lat,reso_alt,r_step,obs[0],obs[1]),order_grid)
                    print 'Reconstitution of the sub-path length grid finished with success'
                    print 'Reconstitution of the order grid finished with success'
                    del dx_grid_opt, dx_grid_opt_ne
                    del order_grid, order_grid_ne
                del dx_grid_opt_n
                del order_grid_n

            comm.Barrier()

            if rank == 0 :
                print 'Computation of the cylindrical stictch finished with success'

            if Integral == True :

                                            ###### Parallele encoding init ######

                if rank == 0 :
                    pdx_grid = np.zeros((n_layers+1,theta_number,x_size),dtype=np.float64)
                    pdx_grid[n_level_rank,:,:length[0]] = pdx_grid_n

                for r_n in range(number_rank) :
                    if r_n == rank and rank != 0 :
                        pdx_grid_n = np.array(pdx_grid_n, dtype=np.float64)
                        comm.Send([pdx_grid_n,MPI.DOUBLE],dest=0,tag=rank)
                    elif r_n != 0 and rank == 0 :
                        n_lay_rank_ne = repartition(n_layers+1,number_rank,r_n,False)
                        pdx_grid_ne = np.zeros((n_lay_rank_ne.size,theta_number,length[r_n]),dtype=np.float64)
                        comm.Recv([pdx_grid_ne,MPI.DOUBLE],source=r_n,tag=r_n)
                        pdx_grid[n_level_rank_ne,:,:length[r_n]] = pdx_grid_ne
                        if length[r_n] != x_size :
                            pdx_grid[n_level_rank_ne,:,length[r_n]:x_size] = np.ones((n_level_rank_ne.size,theta_number,x_size-length[r_n]))*(-1)

                                            ###### Parallele encoding end ######

                if rank == 0 :
                    np.save("%s%s/%s/pdx_grid_%i_%ix%ix%i_%i_%.2f_%.2f.npy"
                        %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,obs[0],obs[1]),pdx_grid)
                    print 'Reconstitution of the integrated density grid finished with success'
                    del pdx_grid, pdx_grid_ne
                del pdx_grid_n


        comm.Barrier()

        if rank == 0 :
            print 'Computation of optical pathes finished with success'


########################################################################################################################

        if Matrix == True :

                                        ###### Parallele encoding init ######

            n_lay_rank = repartition(n_layers+1,number_rank,rank,False)

            data_convert = np.load("%s%s/%s/%s_data_convert_%ix%ix%i_lo%.2f%s.npy"\
                                   %(path,name_file,param_file,name_exo,reso_alt,reso_long,reso_lat,long_obs,stu_name))

            order_grid = np.load("%s%s/%s/order_grid_%i_%ix%ix%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,\
                        reso_long,reso_lat,reso_alt,r_step,obs[0],obs[1]))

            order_grid = order_grid[:,n_lay_rank,:,:]

                                        ###### Parallele encoding end ######

            result_n = atmospheric_matrix_3D(order_grid,data_convert,t,Rp,c_species,rank,Tracer,Cloudy)

                                        ###### Parallele encoding init ######

            if Tracer == True :
                m_m = 1
            else :
                m_m = 0
            if Cloudy == True :
                c_c = 1
            else :
                c_c = 0

            if rank == 0 :
                sh_res = np.shape(result_n)
                result_P = np.zeros((n_layers+1,theta_number, np.shape(result_n[0])[2]), dtype=np.float64)
                result_T = np.zeros((n_layers+1,theta_number, np.shape(result_n[0])[2]), dtype=np.float64)
                result_Cn = np.zeros((n_layers+1,theta_number, np.shape(result_n[0])[2]), dtype=np.float64)
                result_comp = np.zeros((n_species.size + 1, n_layers+1,theta_number, np.shape(result_n[0])[2]), dtype=np.float64)
                result_P[n_lay_rank,:,:] = result_n[0]
                result_T[n_lay_rank,:,:] = result_n[1]
                result_Cn[n_lay_rank,:,:] = result_n[2]
                result_comp[:,n_lay_rank,:,:] = result_n[3+m_m+c_c]
                if Tracer == True :
                    result_Q = np.zeros((n_layers+1,theta_number, np.shape(result_n[0])[2]), dtype=np.float64)
                    result_Q[n_lay_rank,:,:] = result_n[3]
                if Cloudy == True :
                    result_gen = np.zeros((c_species.size,n_layers+1,theta_number, np.shape(result_n[0])[2]), dtype=np.float64)
                    result_gen[:,n_lay_rank,:,:] = result_n[3+m_m]

            length = np.shape(order_grid)[3]

            comm.Barrier()

            for r_n in range(number_rank) :
                if r_n == rank and rank != 0 :
                    comm.Send([result_n[0],MPI.DOUBLE],dest=0,tag=1)
                    comm.Send([result_n[1],MPI.DOUBLE],dest=0,tag=2)
                    comm.Send([result_n[2],MPI.DOUBLE],dest=0,tag=3)
                    if Tracer == True :
                        comm.Send([result_n[3],MPI.DOUBLE],dest=0,tag=4)
                    if Cloudy == True :
                        comm.Send([result_n[3+m_m],MPI.DOUBLE],dest=0,tag=5)
                    comm.Send([result_n[3+m_m+c_c],MPI.DOUBLE],dest=0,tag=6)
                elif r_n != 0 and rank == 0 :
                    n_lay_rank_ne = repartition(n_layers+1,number_rank,r_n,False)
                    result_n_P = np.zeros((n_lay_rank_ne.size,theta_number,length),dtype=np.float64)
                    comm.Recv([result_n_P,MPI.DOUBLE],source=r_n,tag=1)
                    result_P[n_lay_rank_ne,:,:] = result_n_P
                    result_n_T = np.zeros((n_lay_rank_ne.size,theta_number,length),dtype=np.float64)
                    comm.Recv([result_n_T,MPI.DOUBLE],source=r_n,tag=2)
                    result_T[n_lay_rank_ne,:,:] = result_n_T
                    result_n_Cn = np.zeros((n_lay_rank_ne.size,theta_number,length),dtype=np.float64)
                    comm.Recv([result_n_Cn,MPI.DOUBLE],source=r_n,tag=3)
                    result_Cn[n_lay_rank_ne,:,:] = result_n_Cn
                    if Tracer == True :
                        result_n_Q = np.zeros((n_lay_rank_ne.size,theta_number,length),dtype=np.float64)
                        comm.Recv([result_n_Q,MPI.DOUBLE],source=r_n,tag=4)
                        result_Q[n_lay_rank_ne,:,:] = result_n_Q
                    if Cloudy == True :
                        result_n_gen = np.zeros((c_species.size,n_lay_rank_ne.size,theta_number,length),dtype=np.float64)
                        comm.Recv([result_n_gen,MPI.DOUBLE],source=r_n,tag=5)
                        result_gen[:,n_lay_rank_ne,:,:] = result_n_gen
                    result_n_comp = np.zeros((n_species.size+1,n_lay_rank_ne.size,theta_number,length),dtype=np.float64)
                    comm.Recv([result_n_comp,MPI.DOUBLE],source=r_n,tag=6)
                    result_comp[:,n_lay_rank_ne,:,:] = result_n_comp

                                        ###### Parallele encoding end ######

            if rank == 0 :

                np.save("%s%s/%s/%s_P_%ix%ix%i_%i_%i_%.2f_%.2f.npy"%(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,\
                        t_selec,r_step,obs[0],obs[1]),result_P)
                del result_P,result_n_P
                np.save("%s%s/%s/%s_T_%ix%ix%i_%i_%i_%.2f_%.2f.npy"%(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,\
                        t_selec,r_step,obs[0],obs[1]),result_T)
                del result_T,result_n_T
                np.save("%s%s/%s/%s_Q_%ix%ix%i_%i_%i_%.2f_%.2f.npy"%(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,\
                    t_selec,r_step,obs[0],obs[1]),result_Cn)
                del result_Cn,result_n_Cn
                np.save("%s%s/%s/%s_compo_%ix%ix%i_%i_%i_%.2f_%.2f.npy"%(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,\
                    t_selec,r_step,obs[0],obs[1]),result_comp)
                del result_comp,result_n_comp

                if Tracer == True :
                    np.save("%s%s/%s/%s_Cn_%ix%ix%i_%i_%i_%.2f_%.2f.npy"%\
                            (path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,obs[0],obs[1]),\
                            result_Q)
                    del result_Q,result_n_Q
                if Cloudy == True :
                    np.save("%s%s/%s/%s_gen_%ix%ix%i_%i_%i_%.2f_%.2f.npy"%\
                            (path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,obs[0],obs[1]),\
                            result_gen)
                    del result_gen,result_n_gen

            del result_n,order_grid

        comm.Barrier()

########################################################################################################################

        if Convert == True :

            P = np.load("%s%s/%s/%s_P_%ix%ix%i_%i_%i_%.2f_%.2f.npy"%(path,name_file,param_file,name_exo,reso_long,reso_lat,\
                reso_alt,t_selec,r_step,obs[0],obs[1]))
            T = np.load("%s%s/%s/%s_T_%ix%ix%i_%i_%i_%.2f_%.2f.npy"%(path,name_file,param_file,name_exo,reso_long,reso_lat,\
                reso_alt,t_selec,r_step,obs[0],obs[1]))
            if Tracer == True :
                Q = np.load("%s%s/%s/%s_Q_%ix%ix%i_%i_%i_%.2f_%.2f.npy"\
                %(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,obs[0],obs[1]))
            else :
                Q = np.array([])
            if Cloudy == True :
                gen = np.load("%s%s/%s/%s_gen_%ix%ix%i_%i_%i_%.2f_%.2f.npy"\
                %(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,obs[0],obs[1]))
            else :
                gen = np.array([])
            comp = np.load("%s%s/%s/%s_compo_%ix%ix%i_%i_%i_%.2f_%.2f.npy"\
                %(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,obs[0],obs[1]))


########################################################################################################################

        dom_rank = repartition(theta_number,number_rank,rank,True)

        if Convert == True :

            comm.Barrier()

                                        ###### Parallele encoding end ######

            direc = "%s/%s/"%(name_file,opac_file)

            P = P[:,dom_rank,:]
            T = T[:,dom_rank,:]
            comp = comp[:,:,dom_rank,:]
            if Tracer == True :
                Q = Q[:,dom_rank,:]
            if Cloudy == True :
                gen = gen[:,:,dom_rank,:]
            P_rmd, T_rmd, Q_rmd, gen_cond_rmd, composit_rmd, wher, indices, liste = sort_set_param(P,T,Q,gen,comp,rank,Tracer,Cloudy)
            p = np.log10(P_rmd)
            p_min = int(np.amin(p)-1)
            p_max = int(np.amax(p)+1)
            rmind = np.zeros((2,p_max - p_min+1),dtype=np.float64)
            rmind[0,0] = 0

            for i_r in xrange(p_max - p_min) :

                wh, = np.where((p >= p_min + i_r)*(p <= p_min + (i_r+1)))

                if wh.size != 0 :
                    rmind[0,i_r+1] = wh[wh.size-1]
                    rmind[1,i_r] = p_min + i_r
                else :
                    rmind[0,i_r+1] = 0
                    rmind[1,i_r] = p_min + i_r

            rmind[1,i_r+1] = p_max

                                        ###### Parallele encoding end ######

            convertator_save(P_rmd,T_rmd,rmind,Q_rmd,gen_cond_rmd,composit_rmd,path,direc,reso_long,reso_lat,name_exo,t,\
                            x_step,obs[0],obs[1],domain,dim_bande,dim_gauss,rank,Kcorr,Tracer,Cloudy,True)

            del P,T,Q,gen,comp,P_rmd,T_rmd,Q_rmd,gen_cond_rmd,composit_rmd,rmind

            if Kcorr == True :
                rmind = np.load("%s%s/%s/Temp/rmind_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],\
                          domain,rank))
            else :
                rmind = np.load("%s%s/%s/Temp/rmind_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))

                comm.Barrier()

########################################################################################################################

            if Kcorr == True :

                rmind = np.array(rmind,dtype=np.int)
                T_rmd = np.load("%s%s/%s/Temp/T_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],\
                            domain,rank))
                P_rmd = np.load("%s%s/%s/Temp/P_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],\
                          domain,rank))
                composit_rmd = np.load("%s%s/%s/Temp/compo_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],\
                          domain,rank))
                if Cl == True :
                    gen_rmd = np.load("%s%s/%s/Temp/gen_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],\
                          domain,rank))
                else :
                    gen_rmd = np.array([])
                if Tracer == True :
                    Q_rmd = np.load("%s%s/%s/Temp/Q_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],\
                          domain,rank))
                else :
                    Q_rmd = np.array([])

            else :

                rmind = np.array(rmind,dtype=np.int)
                T_rmd = np.load("%s%s/%s/Temp/T_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))
                P_rmd = np.load("%s%s/%s/Temp/P_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))
                composit_rmd = np.load("%s%s/%s/Temp/compo_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))
                if Cl :
                    gen_rmd = np.load("%s%s/%s/Temp/gen_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))
                else :
                    gen_rmd = np.array([])
                if Tracer == True :
                    Q_rmd = np.load("%s%s/%s/Temp/Q_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))
                else :
                    Q_rmd = np.array([])

            data_convert = np.load("%s%s/%s/%s_data_convert_%ix%ix%i_lo%.2f%s.npy"\
                        %(path,name_file,param_file,name_exo,reso_alt,reso_long,reso_lat,long_obs,stu_name))

########################################################################################################################

            if Kcorr == True :

                gauss = np.arange(0,dim_gauss,1)
                gauss_val = np.load("%s%s/gauss_sample.npy"%(path,name_source))
                P_sample = np.load("%s%s/P_sample.npy"%(path,name_source))
                T_sample = np.load("%s%s/T_sample.npy"%(path,name_source))
                if Tracer == True :
                    Q_sample = np.load("%s%s/Q_sample.npy"%(path,name_source))
                else :
                    Q_sample = np.array([])
                bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,domain))

                k_corr_data_grid = "%s%s/k_corr_%s_%s.npy"%(path,name_source,name_exo,domain)

            else :

                gauss = np.array([])
                gauss_val = np.array([])
                P_sample = np.load("%s%s/P_sample_%s.npy"%(path,name_source,source))
                T_sample = np.load("%s%s/T_sample_%s.npy"%(path,name_source,source))
                Q_sample = np.array([])
                bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,source))

                k_corr_data_grid = "%s%s/crossection_%s.npy"%(path,name_source,source)

            # Telechargement des donnees CIA

            if Cont == True :
                K_cont = continuum()
            else :
                K_cont = np.array([])

            # Telechargement des donnees nuages

            if Cl == True :
                bande_cloud = np.load("%s%s/bande_cloud_%s.npy"%(path,name_source,name_exo))
                r_cloud = np.load("%s%s/radius_cloud_%s.npy"%(path,name_source,name_exo))
                cl_name = ''
                for i in range(c_species_name.size) :
                    cl_name += '%s_'%(c_species_name[i])
                Q_cloud = "%s%s/Q_%s%s.npy"%(path,name_source,cl_name,name_exo)
                message_clouds = ''
                for i in range(c_species.size) :
                    message_clouds += '%s (%.2f microns/%.3f)  '%(c_species[i],r_eff[i]*10**6,rho_p[i]/1000.)
            else :
                bande_cloud = np.array([])
                r_cloud = np.array([])
                Q_cloud = np.array([])


########################################################################################################################

            convertator (P_rmd,T_rmd,gen_rmd,c_species,Q_rmd,composit_rmd,ind_active,ind_cross,k_corr_data_grid,K_cont,\
                        Q_cloud,P_sample,T_sample,Q_sample,bande_sample,bande_cloud,x_step,r_eff,r_cloud,rho_p,direc,\
                        t,obs[0],obs[1],n_species,domain,ratio,path,name_exo,reso_long,reso_lat,rank,0,number_rank,name_source,\
                        Tracer,Molecul,Cont,Cl,Scatt,Kcorr,Optimal,True)

########################################################################################################################

    else :

        obs = np.array([lat_obs,long_obs])

    comm.Barrier()


########################################################################################################################
########################################################################################################################
##########################################      TRANSFERT 3D      ######################################################
########################################################################################################################
########################################################################################################################

    if Cylindric_transfert_3D == True :

        if rank == 0 :
            print('Download of stiches array')

        order_grid = np.load("%s%s/%s/order_grid_%i_%ix%ix%i_%i_%.2f_%.2f.npy"\
                    %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,obs[0],obs[1]))
        order_grid = order_grid[:,:,dom_rank,:]
        if Module == True :
            z_grid = np.load("%s%s/%s/z_grid_%i_%ix%ix%i_%i_%.2f_%.2f.npy"\
                    %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,obs[0],obs[1]))
            z_grid = z_grid[:,dom_rank,:]
        else :
            z_grid = np.array([])

        if Discreet == True :
            dx_grid = np.load("%s%s/%s/dx_grid_opt_%i_%ix%ix%i_%i_%.2f_%.2f.npy"\
                    %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,obs[0],obs[1]))
            dx_grid = dx_grid[:,dom_rank,:]
            pdx_grid = np.array([])

        else :

            pdx_grid = np.load("%s%s/%s/pdx_grid_%i_%ix%ix%i_%i_%.2f_%.2f.npy"\
                           %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,obs[0],obs[1]))
            pdx_grid = pdx_grid[:,dom_rank,:]
            dx_grid = np.load("%s%s/%s/dx_grid_opt_%i_%ix%ix%i_%i_%.2f_%.2f.npy"\
                          %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,obs[0],obs[1]))
            dx_grid = dx_grid[:,dom_rank,:]

        data_convert = np.load("%s%s/%s/%s_data_convert_%ix%ix%i_lo%.2f%s.npy"%(path,name_file,param_file,name_exo,reso_alt,reso_long,\
                    reso_lat,long_obs,stu_name))

########################################################################################################################

        if rank == 0 :
            print('Download of couples array')

        if Kcorr == True :
            T_rmd = np.load("%s%s/%s/Temp/T_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                    %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],\
                      domain,rank))
            P_rmd = np.load("%s%s/%s/Temp/P_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                    %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],\
                      domain,rank))
            if Clouds == True :
                gen_rmd = np.load("%s%s/%s/Temp/gen_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                    %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],\
                      domain,rank))
            else :
                gen_rmd = np.array([])
            if Tracer == True :
                Q_rmd = np.load("%s%s/%s/Temp/Q_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                    %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],\
                      domain,rank))
            else :
                Q_rmd = np.array([])
            rmind = np.load("%s%s/%s/Temp/rmind_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                    %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],\
                      domain,rank))
        else :
            T_rmd = np.load("%s%s/%s/Temp/T_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                    %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))
            P_rmd = np.load("%s%s/%s/Temp/P_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                    %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))
            if Clouds == True :
                gen_rmd = np.load("%s%s/%s/Temp/gen_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                    %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))
            else :
                gen_rmd = np.array([])
            if Tracer == True :
                Q_rmd = np.load("%s%s/%s/Temp/Q_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                    %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))
            else :
                Q_rmd = np.array([])
            rmind = np.load("%s%s/%s/Temp/rmind_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                    %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))

########################################################################################################################
    
        if rank == 0 :
            print 'Download of opacities data'

        rank_ref = 0

        Molecular = True       ###### Ne tiens pas compte de l'absorption moleculaire
        Continuum = True       ###### Tiens compte de l'absorption par les collisions
        Scattering = True      ###### Tiens compte de l'absorption par la diffusion
        Clouds = False          ###### Tiens compte de l'absoprtion par les nuages

        # Le simulateur de spectre va decouper en bande de facon a limiter au maximum la saturation en memoire
        # Une option permet un decoupage supplementaire en the ta ou exclusivement en theta si les tableaux de donnees ne
        # sont pas trop importants.

        cases = np.zeros(4,dtype=np.int)
        cases_names = ['molecular','continuum','scattering','clouds']
        if Molecular == True :
            cases[0] = 1
        if Continuum == True :
            cases[1] = 1
        if Scattering == True :
            cases[2] = 1
        if Clouds == True :
            cases[3] = 1

        wh_ca, = np.where(cases == 1)

        for i_ca in range(wh_ca.size) :

            proc = np.array([False,False,False,False])
            proc[wh_ca[i_ca]] = True
            Molecular, Continuum, Scattering, Clouds = proc[0],proc[1],proc[2],proc[3]

            stud = stud_type(r_eff,Single,Continuum,Molecular,Scattering,Clouds)
            save_name_3D_step = "%s%s_3D_duo_linear_real_%.3f_%.3f_%i_%.2f_%s"%(save_adress,name_exo,X_min,X_max,beta,P_tau/(1.e+5),stud)
            save_name_3D_step_dat = "%s/%s_3D_duo_linear_real_%.3f_%.3f_%i_%.2f_%s"%(save_adress,name_exo,X_min,X_max,beta,P_tau/(1.e+5),stud)
            if Noise == True :
                save_name_3D_step = '%s_n'%(save_name_3D_step)
                save_name_3D_step_dat = '%s_n'%(save_name_3D_step_dat)

            if os.path.isfile('%s.npy'%(save_name_3D_step)) != True or Push[i_ca] == True :

                if Molecular == True :
                    if Kcorr == True :
                        k_rmd = np.load("%s%s/%s/Temp/k_corr_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],domain,rank))
                        gauss_val = np.load("%s%s/gauss_sample.npy"%(path,name_source))
                    else :
                        if Optimal == True :
                            k_rmd = np.load("%s%s/%s/Temp/k_cross_opt_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                            %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))
                        else :
                            k_rmd = np.load("%s%s/%s/Temp/k_cross_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                            %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))
                        gauss_val = np.array([])
                else :
                    if Kcorr == True :
                        k_rmd = np.load("%s%s/%s/Temp/k_corr_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],domain,rank))
                        k_rmd = np.shape(k_rmd)
                    else :
                        k_rmd = np.load("%s%s/%s/Temp/k_cross_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                            %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))
                        k_rmd = np.shape(k_rmd)
                    gauss_val = np.array([])
                    if rank == 0 :
                        print 'No molecular'

                if Continuum == True :
                    if Kcorr == True :
                        k_cont_rmd = np.load("%s%s/%s/Temp/k_cont_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],domain,rank))
                    else :
                        k_cont_rmd = np.load("%s%s/%s/Temp/k_cont_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))
                else :
                    k_cont_rmd = np.array([])
                    if rank == 0 :
                        print 'No continuum'

                if Scattering == True :
                    if Kcorr == True :
                        k_sca_rmd = np.load("%s%s/%s/Temp/k_sca_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],domain,rank))
                    else :
                        k_sca_rmd = np.load("%s%s/%s/Temp/k_sca_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],domain,rank))
                else :
                    k_sca_rmd = np.array([])
                    if rank == 0 :
                        print 'No scattering'

                if Clouds == True :
                    r_enn = ''
                    for i_r in range(r_eff.size) :
                        if i_r != r_eff.size-1 :
                            r_enn += '%.2f_'%(r_eff[i_r]*10**6)
                        else :
                            r_enn += '%.2f'%(r_eff[i_r]*10**6)
                    if Kcorr == True :
                        k_cloud_rmd = np.load("%s%s/%s/Temp/k_cloud_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%s_%i.npy" \
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,obs[0],obs[1],\
                        r_enn,domain,rank))
                    else :
                        k_cloud_rmd = np.load("%s%s/%s/Temp/k_cloud_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%s_%i.npy" \
                        %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,obs[0],obs[1],r_enn,domain,rank))
                else :
                    k_cloud_rmd = np.array([])
                    if rank == 0 :
                        print 'No clouds'

########################################################################################################################
    
                if rank == 0 :
                    print 'Pytmosph3R will begin to compute the %s contribution'%(cases_names[wh_ca[i_ca]])
                    print 'Save directory : %s'%(save_name_3D_step)

                I_n = trans2fert3D (k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,Rp,h,g0,r_step,theta_step,gauss_val,dim_bande,data_convert,\
                          P_rmd,T_rmd,Q_rmd,dx_grid,order_grid,pdx_grid,z_grid,t,\
                          name_file,n_species,Single,rmind,lim_alt,rupt_alt,rank,rank_ref,\
                          Tracer,Continuum,Molecular,Scattering,Clouds,Kcorr,Rupt,Module,Integration,TimeSel)

                if rank == 0 :
                    sh_I = np.shape(I_n)
                    r_size, theta_size = sh_I[1], sh_I[2]
                    Itot = np.zeros((dim_bande,r_size,theta_number),dtype=np.float64)
                    Itot[:,:,dom_rank] = I_n
                else :
                    I_n = np.array(I_n,dtype=np.float64)
                    comm.Send([I_n,MPI.DOUBLE],dest=0,tag=0)

                if rank == 0 :
                    bar = ProgressBar(number_rank,'Reconstitution of transmitivity for the %s contribution'%(cases_names[wh_ca[i_ca]]))
                    for r_n in range(1,number_rank) :
                        new_dom_rank = repartition(theta_number,number_rank,r_n,True)
                        I_rn = np.zeros((dim_bande,r_size,new_dom_rank.size),dtype=np.float64)
                        comm.Recv([I_rn,MPI.DOUBLE],source=r_n,tag=0)
                        Itot[:,:,new_dom_rank] = I_rn
                        bar.animate(r_n+1)

                if rank == 0 :
                    np.save('%s.npy'%(save_name_3D_step),Itot)

                    if Script == True :

                        Itot = np.load('%s.npy'%(save_name_3D_step))
                        if Noise == True :
                            save_ad = '%s_n'%(save_name_3D_step_dat)
                        else :
                            save_ad = "%s"%(save_name_3D_step_dat)
                        class star :
                            def __init__(self):
                                self.radius = Rs
                                self.temperature = Ts
                                self.distance = d_al
                        if ErrOr == True :
                            bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,source))
                            bande_sample = np.delete(bande_sample,[0])
                            int_lambda = np.zeros((2,bande_sample.size))
                            bande_sample = np.sort(bande_sample)

                            if resolution == '' :
                                int_lambda = np.zeros((2,bande_sample.size))
                                for i_bande in range(bande_sample.size) :
                                    if i_bande == 0 :
                                        int_lambda[0,i_bande] = bande_sample[0]
                                        int_lambda[1,i_bande] = (bande_sample[i_bande+1]+bande_sample[i_bande])/2.
                                    elif i_bande == bande_sample.size - 1 :
                                        int_lambda[0,i_bande] = (bande_sample[i_bande-1]+bande_sample[i_bande])/2.
                                        int_lambda[1,i_bande] = bande_sample[bande_sample.size-1]
                                    else :
                                        int_lambda[0,i_bande] = (bande_sample[i_bande-1]+bande_sample[i_bande])/2.
                                        int_lambda[1,i_bande] = (bande_sample[i_bande+1]+bande_sample[i_bande])/2.
                                int_lambda = np.sort(10000./int_lambda[::-1])
                            else :
                                int_lambda = np.sort(10000./bande_sample[::-1])

                            noise = stellar_noise(star(),detection,int_lambda,resolution)
                            noise = noise[::-1]
                        else :
                            noise = error
                        if Kcorr == True :
                            flux_script(path,name_source,domain,save_ad,Itot,noise,Rs,Rp,r_step,Kcorr,Middle,Noise)
                        else :
                            flux_script(path,name_source,source,save_ad,Itot,noise,Rs,Rp,r_step,Kcorr,Middle,Noise)

                    del Itot
                del I_n

            else :

                if rank == 0 :
                    print 'The %s contribution was already computed'%(cases_names[wh_ca[i_ca]])
                    print 'Corresponding save directory : %s'%(save_name_3D_step)
                    print 'Please check that this is the expected file'

                    if Script == True :

                        Itot = np.load('%s.npy'%(save_name_3D_step))
                        if Noise == True :
                            save_ad = '%s_n'%(save_name_3D_step_dat)
                        else :
                            save_ad = "%s"%(save_name_3D_step_dat)
                        class star :
                            def __init__(self):
                                self.radius = Rs
                                self.temperature = Ts
                                self.distance = d_al
                        if ErrOr == True :
                            bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,source))
                            bande_sample = np.delete(bande_sample,[0])
                            int_lambda = np.zeros((2,bande_sample.size))
                            bande_sample = np.sort(bande_sample)

                            if resolution == '' :
                                int_lambda = np.zeros((2,bande_sample.size))
                                for i_bande in range(bande_sample.size) :
                                    if i_bande == 0 :
                                        int_lambda[0,i_bande] = bande_sample[0]
                                        int_lambda[1,i_bande] = (bande_sample[i_bande+1]+bande_sample[i_bande])/2.
                                    elif i_bande == bande_sample.size - 1 :
                                        int_lambda[0,i_bande] = (bande_sample[i_bande-1]+bande_sample[i_bande])/2.
                                        int_lambda[1,i_bande] = bande_sample[bande_sample.size-1]
                                    else :
                                        int_lambda[0,i_bande] = (bande_sample[i_bande-1]+bande_sample[i_bande])/2.
                                        int_lambda[1,i_bande] = (bande_sample[i_bande+1]+bande_sample[i_bande])/2.
                                int_lambda = np.sort(10000./int_lambda[::-1])
                            else :
                                int_lambda = np.sort(10000./bande_sample[::-1])

                            noise = stellar_noise(star(),detection,int_lambda,resolution)
                            noise = noise[::-1]
                        else :
                            noise = error
                        if Kcorr == True :
                            flux_script(path,name_source,domain,save_ad,Itot,noise,Rs,Rp,r_step,Kcorr,Middle,Noise)
                        else :
                            flux_script(path,name_source,source,save_ad,Itot,noise,Rs,Rp,r_step,Kcorr,Middle,Noise)

        if rank == 0 :
            for i_ca in range(wh_ca.size) :
                proc = np.array([False,False,False,False])
                proc[wh_ca[i_ca]] = True
                Molecular, Continuum, Scattering, Clouds = proc[0],proc[1],proc[2],proc[3]
                stud = stud_type(r_eff,Single,Continuum,Molecular,Scattering,Clouds)
                save_name_3D_step = "%s%s_3D_duo_linear_real_%.3f_%.3f_%i_%.2f_%s"%(save_adress,name_exo,X_min,X_max,beta,P_tau/(1.e+5),stud)
                save_name_3D_step_dat = "%s/%s_3D_duo_linear_real_%.3f_%.3f_%i_%.2f_%s"%(save_adress,name_exo,X_min,X_max,beta,P_tau/(1.e+5),stud)
                if Noise == True :
                    save_name_3D_step = '%s_n'%(save_name_3D_step)
                    save_name_3D_step_dat = '%s_n'%(save_name_3D_step_dat)
                I_step = np.load('%s.npy'%(save_name_3D_step))
                if i_ca == 0 :
                    Itot = I_step
                else :
                    Itot *= I_step
            np.save('%s.npy'%(save_name_3D),Itot)

            if Script == True :

                Itot = np.load('%s.npy'%(save_name_3D))
                save_ad = "%s"%(save_name_3D_dat)
                if Noise == True :
                    save_ad += '_n'
                if ErrOr == True :
                    class star :
                        def __init__(self):
                            self.radius = Rs
                            self.temperature = Ts
                            self.distance = d_al
                    bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,source))
                    bande_sample = np.delete(bande_sample,[0])
                    int_lambda = np.zeros((2,bande_sample.size))
                    bande_sample = np.sort(bande_sample)

                    if resolution == '' :
                        int_lambda = np.zeros((2,bande_sample.size))
                        for i_bande in range(bande_sample.size) :
                            if i_bande == 0 :
                                int_lambda[0,i_bande] = bande_sample[0]
                                int_lambda[1,i_bande] = (bande_sample[i_bande+1]+bande_sample[i_bande])/2.
                            elif i_bande == bande_sample.size - 1 :
                                int_lambda[0,i_bande] = (bande_sample[i_bande-1]+bande_sample[i_bande])/2.
                                int_lambda[1,i_bande] = bande_sample[bande_sample.size-1]
                            else :
                                int_lambda[0,i_bande] = (bande_sample[i_bande-1]+bande_sample[i_bande])/2.
                                int_lambda[1,i_bande] = (bande_sample[i_bande+1]+bande_sample[i_bande])/2.
                        int_lambda = np.sort(10000./int_lambda[::-1])
                    else :
                        int_lambda = np.sort(10000./bande_sample[::-1])

                    noise = stellar_noise(star(),detection,int_lambda,resolution)
                    noise = noise[::-1]
                else :
                    noise = error
                if Kcorr == True :
                    flux_script(path,name_source,domain,save_ad,Itot,noise,Rs,Rp,r_step,Kcorr,Middle,Noise)
                else :
                    flux_script(path,name_source,source,save_ad,Itot,noise,Rs,Rp,r_step,Kcorr,Middle,Noise)

            print 'Final save directory : %s'%(save_name_3D)


########################################################################################################################


    if rank == 0 :
        print 'Pytmosph3R process finished with success'


########################################################################################################################
