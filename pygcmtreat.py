from pyfunction import *
from pyconstant import *
from netCDF4 import Dataset
import math as math
import pickle
import scipy.integrate as integrate
import os,sys
import time

########################################################################################################################
########################################################################################################################

"""
    PYGCMTREAT

    Cette bibliotheque intervient a deux niveaux : premierement lors de la generation des matrices de conversion qui
    permettent de transposer les donnes a symetrie spherique du GCM dans la maille cylindrique utilisee par pytransfert.
    Deuxiemement dans la recuperation des trajets de rayons dans la dite maille et la recuperation des donnees pre-
    calculees.

    La fonction Boxes effectue le meme travail que zrecast du GCM, nous pouvons neanmoins aisement modifier l'echelle en
    altitude et extrapoler cette derniere pour la haute atmosphere, notamment lorsque le toit du modele est trop bas par
    rapport aux proprietes absorbantes en transmission de son atmosphere. Les fonctions ci-dessous tiennent compte de la
    presence de marqueurs, de nuages, du nombre de molecule d'interet, et de leurs influences sur le profil atmospherique.

    Nous pouvons ici tenir compte de la rotation de l'exoplanete ou d'une eventuelle obliquite.

    Version : 6.3

    Recentes mises a jour :

    >> Modifications de altitude_line_array1D_cyl_optimized_correspondance suite a une mauvaise estimation du l qui
    calculait l'epaisseur de correction sur les bords de l'atmosphere
    >> Suppression du retour de N
    >> Suppression de l'entree T
    >> Memes modifications altitude_line_array2D_cyl_optimized_correspondance
    >> cylindric_assymatrix_parameter va desormais pouvoir effectuer les chemins optiques au centre des couches de boxes
    a la limite inferieure ou a la limite superieure. Le nombre de rayon sortant est donc different et l'association des
    parametres necessite une nouvelle interpolation (modification necessaire de convertator puisque la diversite de P, T
    Q va necessairement varier de data_convert
    >> Nouvelle modification de Boxes, desormais la fonction peut calculer preferentiellement les proprietes de l'
    atmosphere au milieu des couches regulieres de la grille en altitude qui est souhaite. Ainsi, data_convert[:,:,0,:,:]
    correspond aux parametres de surface, et data_convert[:,:,1:,:,:] aux parametres au milieu des couches
    >> cylindric_assymatrix_parameter est donc a nouveau modifiee de maniere a tenir compte du fait que les data soient
    deja determines au milieu des couches, ainsi si l'option Middle est choisie, l'ordre de z_level devient l'ordre de
    la couche, et on conserve bien les proprietes, nous n'avons plus de demi-couche pour la surface puisque les
    parametres de surface n'interviennent plus (nous gardons cet ordre 0 de surface pour garder les informations sur
    le diagfi de la simu GCM).
    Note : le z_level doit rester un tableau allant de 0 a h avec un pas de delta_z, il donne la position des inter-
    couches
    >> Modification de dx_correspondance qui desormais calcule exactement les distances parcourues par un rayon dans
    les differentes cellules de l'atmosphere a une couche et un point de lattitude donnee. La fonction peut faire l'
    objet d'une amelioration en tenant compte des symetrie sur ces distances. On notera que les cas particuliers ou non
    seulement la strate en altitude et le point de latiude ou de longitude changent, les calculs favorisent le saut de
    couche sur les sauts de latitude ou longitude
    >> Correction d'un bug sur le calcul des l sur les chemins optiques, cette grandeur peut etre positive (il manque
    une partie de cellule) ou negative (la derniere cellule depasse le toit de l'atmosphere)

    Date de derniere modification : 14.09.2016

    >> Modification de dx_correspondance, les distances calculees sont desormais plus precises que jamais et decoupent
    reellement le chemin optique en deux pour s'abstenir des problemes aux poles
    >> Cette meme fonction peut desormais integrer sur le chemin optique des rayons la profondeur optique (divisee de
    la section efficace toutefois), nous ne tenons pas compte d'une eventuelle dependance de la fraction molaire ou de
    la section efficace avec l'altitude. Pour que cette hypothese reste valable, le pas en altitude doit etre bien
    inferieur a la hauteur d'echelle
    >> Desormais, chaque changement d'altitude, de longitude ou de latitude est traite de maniere independante, on notera
    que l'integrale diverse pour les variations d'altitude tres tres faibles (ce qui arrive typiquement lorsque les
    rayons traversent des coins de cellule spherique ou couramment pour les terminateurs aux poles)
    >> Cette ecriture est bien adaptee au cas Middle, il l'est moins si cette option est False
    >> Bien que conservees, les fonctions altitude_array ne sont plus appelees dans le transfert radiatif
    >> Modification dans la construction du profil P-T cylindrique, la loi hydrostatique a ete reecrite
    >> Une verification serait cependant avisee pour etre certain que cette reecriture ne s'eloigne pas de celle attendue
    (par exemple, cas isotherme)

    Date de derniere modification : 12.12.2016

    >> Modification complete de boxes, les fonctions d'interpolation sont predefinies desormais dans pyfunction,
    >> Correction d'un bug qui ne permettait pas de retenir convenablement les indices d'interpolations sur la
    temperature dans du recast. La pression et les compositions ne correspondaient alors plus du tout. (le first ne
    permettait le precalcul enregistre que de coeff1 et coeff5 mais pas de i_Tu ou i_Qu).
    >> Optimisation de egalement lors des calculs de composition, nous nous assurons bien que la somme des fractions
    molaires soit toujours egale a 1 sans etre biaisee par les etapes d'interpolation.

    Date de derniere modification : 17.05.2017

    >> Plusieurs options ont ete rajoutees, la possibilite notamment de raisonner en termes de nombre de couche et non
    plus en epaisseur de couche (on defini un nombre de couche fixe, et une fois que le toit de l'atmosphere est calcule
    on en deduit l'epaisseur de chaque couche), la multiplicite des options de toit atmospherique (desormais nous pouvons
    proposer une estimation du toit atmospherique defini en pression comme une altitude limite, et ce a partir de la
    connaissance des proprietes moyennes du toit du modele ou de ses proprietes extremes : 'Mean' on calcule une hauteur
    d'echelle moyenne du toit du modele et dans l'hypothese que l'extrapolation est bien isotherme, on en deduit une
    altitude correspondant a la pression P_h d'entree. 'Down' on fait la meme chose mais on cherche le point pour lequel
    la hauteur d'echelle est maximale, donc P_h va correspondre non pas au toit mais la pression maximale en toit que l
    on souhaite avoir, 'Up' on utilise le point ou la hauteur d'echelle est la plus faible et P_h correspond donc a la
    pression minimale que l'on souhaite avoir), et correction d'un bug dans le calcul du toit en altitude.
    >> L'epaisseur atmospherique est arrondie a la centaine afin d'eviter un bug dans la fonction dx_correspondance. Elle
    ne reconnaissait pas r_step/2. si r_step etait trop complique. Il est a noter qu'on evitera si possible les nombres
    de couche trop petits ou trop complexes : le mimimum etant la resolution maximale de la simulation, si c'est une
    48x64x50, 100 couches est une resolution raisonnable, mais on evitera 101, 97 couches par exemple.
    >> L'introduction d'une option n_layers a donne lieu a l'ecriture de la version NBoxes.

    Date de derniere modification : 29.10.2017

    >> Refonte de la partie extrapolation de la haute atmopshere pour rendre cette partie plus lisible et facile a
    modifier le cas echeant. Les calculs de fond restent globalement les memes bien que quelques bugs faisaient que
    pour certaines colonnes l'interpolation etait mal realisee.
    >> Ajout de la possibilite de travailler sans H2 et He, ce qui jusque la etait fastidieux car nous n'avions pas
    toujours la conservation de la matiere.

    Date de derniere modification : 12.12.2017

    >> Ajout de l'option TauREx dans Boxes et NBoxes afin de pouvoir exploiter des profils generes par create_spectrum.
    Correction de la fonction de calcul des altitudes lorsque les temperatures sont egales.

    Date de derniere modification : 09.03.2018

    >> Ajout de la possibilite de calculer les compositions sans les composition.in et a partir de la fraction massique
    du traceur

    Date de derniere modification : 03.04.2018

    >> Reecriture de la fonction cylindric_assymetrix, cette nouvelle version plus epuree permet de calculer plus
    facilement les correspondances en indice au sein de la maille cylindrique.

    Date de derniere modification : 24.04.2018

"""

########################################################################################################################
########################################################################################################################


def Boxes(data,delta_z,Rp,h,P_h,t,g0,M_atm,number,T_comp,P_comp,Q_comp,species,x_species,M_species,c_species,m_species,ratio,Upper,composition,obs,\
          TopPressure,Inverse,Surf=True,Tracer=False,Clouds=False,Middle=False,LogInterp=False,TimeSelec=False,MassAtm=False,NoH2=False,TauREx=True,Rotate=False) :

    if data != '' :
        file = Dataset("%s.nc"%(data))
        variables = file.variables
    else :
        from pyfunction import planet
        planet = planet()
    c_number = c_species.size
    if Tracer == True :
        m_number = 1
        m_species = m_species[0]
    else :
        m_number = 0

    # Si nous avons l'information sur les parametres de surface

    if data != '' :

        if Surf == True :

            # Si nous avons l'information sur la pression de surface, il nous faut donc rallonger les tableaux de parametres
            # de 1

            if TimeSelec == False :
                T_file = variables["temp"][:]
                n_t,n_l,n_lat,n_long = np.shape(T_file)
                T_surf = variables["tsurf"][:]
                P_file = variables["p"][:]
                P_surf = variables["ps"][:]
                P = np.zeros((n_t,n_l+1,n_lat,n_long),dtype=np.float64)
                T = np.zeros((n_t,n_l+1,n_lat,n_long),dtype=np.float64)
            else :
                T_prefile = variables["temp"][:]
                n_t,n_l,n_lat,n_long = np.shape(T_prefile)
                T_file = np.zeros((1,n_l,n_lat,n_long),dtype=np.float64)
                T_surf = np.zeros((1,n_lat,n_long),dtype=np.float64)
                P_file = np.zeros((1,n_l,n_lat,n_long),dtype=np.float64)
                P_surf = np.zeros((1,n_lat,n_long),dtype=np.float64)
                T_file[0] = variables["temp"][t,:,:,:]
                T_surf[0] = variables["tsurf"][t,:,:]
                P_file[0] = variables["p"][t,:,:,:]
                P_surf[0] = variables["ps"][t,:,:]
                P = np.zeros((1,n_l+1,n_lat,n_long),dtype=np.float64)
                T = np.zeros((1,n_l+1,n_lat,n_long),dtype=np.float64)

            P[:,0,:,:] = P_surf
            P[:,1:n_l+1,:,:] = P_file
            T[:,0,:,:] = T_surf
            T[:,1:n_l+1,:,:] = T_file

            if Tracer == True :
                if TimeSelec == False :
                    Q_vap = variables["%s_vap"%(m_species)][:]
                    Q_vap_surf = variables["%s_vap_surf"%(m_species)][:]
                    Q = np.zeros((n_t,n_l+1,n_lat,n_long),dtype=np.float64)
                else :
                    Q_vap = np.zeros((1,n_l,n_lat,n_long))
                    Q_vap_surf = np.zeros((1,n_lat,n_long))
                    Q_vap[0] = variables["%s_vap"%(m_species)][t,:,:,:]
                    Q_vap_surf[0] = variables["%s_vap_surf"%(m_species)][t,:,:]
                    Q = np.zeros((1,n_l+1,n_lat,n_long),dtype=np.float64)

                Q[:,0,:,:] = Q_vap_surf
                Q[:,1:n_l+1,:,:] = Q_vap
            else :
                Q = np.array([])

            if Clouds == True :
                if TimeSelec == False :
                    gen_cond = np.zeros((c_number,n_t,n_l,n_lat,n_long),dtype=np.float64)
                    gen_cond_surf = np.zeros((c_number,n_t,n_lat,n_long),dtype=np.float64)
                    for c_num in range(c_number) :
                        gen_cond_surf[c_num,:,:,:] = variables["%s_surf"%(c_species[c_num])][:]
                        gen_cond[c_num,:,:,:,:] = variables["%s"%(c_species[c_num])][:]
                    gen = np.zeros((c_species.size,n_t,n_l+1,n_lat,n_long))
                else :
                    gen_cond = np.zeros((c_number,1,n_l,n_lat,n_long),dtype=np.float64)
                    gen_cond_surf = np.zeros((c_number,1,n_lat,n_long),dtype=np.float64)
                    for c_num in range(c_number) :
                        gen_cond_surf[c_num,:,:,:] = variables["%s_surf"%(c_species[c_num])][t,:,:]
                        gen_cond[c_num,:,:,:,:] = variables["%s"%(c_species[c_num])][t,:,:,:]
                    gen = np.zeros((c_species.size,1,n_l+1,n_lat,n_long),dtype=np.float64)

                gen[:,:,0,:,:] = gen_cond_surf
                gen[:,:,1:n_l+1,:,:] = gen_cond
            else :
                gen = np.array([])

            if TimeSelec == True :
                n_t = 1
            T_mean = np.nansum(T_file[:,n_l-1,:,:])/(n_t*n_lat*n_long)
            T_max = np.amax(T_file[:,n_l-1,:,:])
            T_min = np.amin(T_file[:,n_l-1,:,:])
            print('Mean temperature : %i K, Maximal temperature : %i K, Minimal temperature : %i K'%(T_mean,T_max,T_min))

            P_mean = np.exp(np.nansum(np.log(P[:,n_l,:,:]))/(n_t*n_lat*n_long))
            print('Mean roof pressure : %f Pa'%(P_mean))

            n_l = n_l + 1

        # Si nous n'avons pas l'information sur les parametres de surface

        else :

            if TimeSelec == False :
                T = variables["temp"][:]
                n_t,n_l,n_lat,n_long = np.shape(T)
                P = variables["p"][:]
            else :
                T_prefile = variables["temp"][:]
                n_t,n_l,n_lat,n_long = np.shape(T_prefile)
                T = np.zeros((1,n_l,n_lat,n_long),dtype=np.float64)
                P = np.zeros((1,n_l,n_lat,n_long),dtype=np.float64)
                T[0] = variables["temp"][t,:,:,:]
                P[0] = variables["p"][t,:,:,:]

            if Tracer == True :
                if TimeSelec == False :
                    Q = variables["%s_vap"%(m_species)][:]
                else :
                    Q = np.zeros((1,n_l,n_lat,n_long))
                    Q[0] = variables["%s_vap"%(m_species)][t,:,:,:]
            else :
                Q = np.array([])

            if Clouds == True :
                if TimeSelec == False :
                    gen = np.zeros((c_number,n_t,n_l,n_lat,n_long))
                    for c_num in range(c_number) :
                        gen[c_num,:,:,:,:] = variables["%s"%(c_species[c_num])][:]
                else :
                    gen = np.zeros((c_number,1,n_l,n_lat,n_long))
                    for c_num in range(c_number) :
                        gen[c_num,:,:,:,:] = variables["%s"%(c_species[c_num])][t,:,:,:]
            else :
                gen = np.array([])

            if TimeSelec == True :
                n_t = 1
            T_mean = np.nansum(T[:,n_l-1,:,:]/(n_t*n_lat*n_long))
            T_max = np.amax(T[:,n_l-1,:,:])
            T_min = np.amin(T[:,n_l-1,:,:])
            print('Mean temperature : %i K, Maximal temperature : %i K, Minimal temperature of the high atmosphere : %i K'\
                  %(T_mean,T_max,T_min))

            P_mean = np.exp(np.nansum(np.log(P[:,n_l-1,:,:]))/(n_t*n_lat*n_long))
            print('Mean roof pressure : %f Pa'%(P_mean))

    else :

        data = pickle.load(open(planet.pressure_profile_data))
        param = data['params']
        T_file = data['data'][planet.pressure_profile_key][:,1]
        n_t,n_l,n_lat,n_long = 1, param[planet.number_layer_key],int(planet.latitude)+1,int(planet.longitude)+1
        T_surf = param[planet.planet_temperature_key]
        P_file = np.linspace(np.log10(param[planet.extreme_pressure_key[0]]),np.log10(param[planet.extreme_pressure_key[1]]),param[planet.number_layer_key]+1)
        P_file = 10**P_file
        T = np.zeros((n_t,n_l+1,n_lat,n_long),dtype=np.float64)
        P = np.zeros((n_t,n_l+1,n_lat,n_long),dtype=np.float64)

        T[:,0,:,:] = np.ones((n_t,n_lat,n_long),dtype=np.float64)*T_surf
        for i_n_t in range(n_t) :
            for i_n_lat in range(n_lat) :
                for i_n_long in range(n_long) :
                    T[i_n_t,1:n_l+1,i_n_lat,i_n_long] = T_file
                    P[i_n_t,:,i_n_lat,i_n_long] = P_file

        Q = np.array([])

        if Clouds == True :
            gen_cond = np.zeros((c_number,1,n_l,n_lat,n_long),dtype=np.float64)
            gen_cond_surf = np.zeros((c_number,1,n_lat,n_long),dtype=np.float64)
            for c_num in range(c_number) :
                gen_cond_surf[c_num,:,:,:] = data['data'][planet.extreme_pressure_key][:,c_num]
                gen_cond[c_num,:,:,:,:] = data['data'][planet.extreme_pressure_key][:,c_num]
            gen = np.zeros((c_species.size,1,n_l+1,n_lat,n_long),dtype=np.float64)

            gen[:,:,0,:,:] = gen_cond_surf
            gen[:,:,1:n_l+1,:,:] = gen_cond
        else :
            gen = np.array([])

        T_mean = np.mean(T_file[n_l-1])
        T_max = np.amax(T_file[n_l-1])
        T_min = np.amin(T_file[n_l-1])
        print('Mean temperature : %i K, Maximal temperature : %i K, Minimal temperature : %i K'%(T_mean,T_max,T_min))

        P_mean = np.exp(np.nansum(np.log(P[:,n_l,:,:]))/(n_t*n_lat*n_long))
        print('Mean roof pressure : %f Pa'%(P_mean))

    z = np.zeros((n_t,n_l,n_lat,n_long),dtype=np.float64)
    M = np.zeros((n_t,n_l,n_lat,n_long),dtype=np.float64)
    H = np.zeros((n_t,n_l,n_lat,n_long),dtype=np.float64)
    g = np.zeros((n_t,n_l,n_lat,n_long),dtype=np.float64)

    bar = ProgressBar(n_t*n_l,'Data convertion from pressure levels')

    if Tracer == False :

        size = species.size
        compo = np.zeros((size,n_t,n_l,n_lat,n_long),dtype=np.float64)

        if LogInterp == True :

            P_comp = np.log10(P_comp)

        for i in range(n_t) :
            for j in range(n_l) :
                for k in range(n_lat) :
                    if composition[0] == 'composition' :
                        if LogInterp == True :
                            res, c_grid, i_grid = interp2olation_multi(np.log10(P[i,j,k,:]),T[i,j,k,:],P_comp,T_comp,x_species)
                        else :
                            res, c_grid, i_grid = interp2olation_multi(P[i,j,k,:],T[i,j,k,:],P_comp,T_comp,x_species)

                        compo[2:size,i,j,k,:] = res[2:]
                        for l in range(n_long) :
                            if NoH2 == False :
                                compo[0,i,j,k,l] = (1. - np.nansum(compo[2:size,i,j,k,l]))/(ratio + 1.)
                            else :
                                compo[0,i,j,k,l] = 0.
                                compo[2:size,i,j,k,l] = compo[2:size,i,j,k,l]/(np.nansum(compo[2:size,i,j,k,l]))
                        if NoH2 == False :
                            compo[1,i,j,k,:] = compo[0,i,j,k,:]*ratio
                        else :
                            compo[1,i,j,k,l] = 0.
                        M[i,j,k,:] = np.dot(M_species,compo[:,i,j,k,:])
                bar.animate(i*n_l+j+1)

    else :

        size = species.size
        compo = np.zeros((size,n_t,n_l,n_lat,n_long),dtype=np.float64)

        if LogInterp == True :

            P_comp = np.log10(P_comp)

        for i in range(n_t) :
            for j in range(n_l) :
                for k in range(n_lat) :
                    if composition[0] == 'composition' :
                        if LogInterp == True :
                            res, c_grid, i_grid = interp3olation_multi(np.log10(P[i,j,k,:]),T[i,j,k,:],Q[i,j,k,:],P_comp,T_comp,Q_comp,x_species)
                        else :
                            res, c_grid, i_grid = interp3olation_multi(P[i,j,k,:],T[i,j,k,:],Q[i,j,k,:],P_comp,T_comp,Q_comp,x_species)

                        compo[2:size,i,j,k,:] = res[2:]
                        for l in range(n_long) :
                            if NoH2 == False :
                                compo[0,i,j,k,l] = (1. - np.nansum(compo[2:size,i,j,k,l]))/(ratio + 1.)
                            else :
                                compo[0,i,j,k,l] = 0.
                                compo[2:size,i,j,k,l] = compo[2:size,i,j,k,l]/(np.nansum(compo[2:size,i,j,k,l]))
                        if NoH2 == False :
                            compo[1,i,j,k,:] = compo[0,i,j,k,:]*ratio
                        else :
                            compo[1,i,j,k,l] = 0.
                        M[i,j,k,:] = np.dot(M_species,compo[:,i,j,k,:])

                    if composition[0] == 'tracer_other' :
                        ind, = np.where(species[2:] == m_species)
                        if NoH2 == False :
                            M_f = 1./(ratio + 1.)*M_species[0] + ratio/(ratio+1.)*M_species[1]
                            ratio_mf = M_f/M_species[ind+2]*Q[i,j,k,:]/(1.-Q[i,j,k,:])
                            compo[ind+2,i,j,k,:] = ratio_mf/(1.+ratio_mf)
                            compo[0,i,j,k,:] = (1. - compo[ind+2,i,j,k,:])/(ratio + 1.)
                        else :
                            indf, = np.where(species[2:] != m_species)
                            ratio_mf = M_species[indf+2]/M_species[ind+2]*Q[i,j,k,:]/(1.-Q[i,j,k,:])
                            compo[ind+2,i,j,k,:] = ratio_mf/(1.+ratio_mf)
                            compo[indf+2,i,j,k,:] = 1. - compo[ind+2,i,j,k,:]
                        if NoH2 == False :
                            compo[1,i,j,k,:] = compo[0,i,j,k,:]*ratio
                        else :
                            compo[1,i,j,k,:] = np.zeros(n_long,dtype=np.float64)
                        M[i,j,k,:] = np.dot(M_species,compo[:,i,j,k,:])

                bar.animate(i*n_l+j+1)

    # Une fois la composition dans chaque cellule des donnees GCM calculee, nous avons l'information manquante sur le
    # poids moleculaire moyen et donc sur la hauteur d'echelle locale. Nous pouvons alors transformer l'echelle de
    # pression en echelle d'altitude

    for pres in range(n_l) :

        if pres == 0 :

            z[:,0,:,:] = 0.
            Mass = np.zeros((n_t,n_lat,n_long),dtype=np.float64)
            g[:,0,:,:] = np.ones((n_t,n_lat,n_long),dtype=np.float64)*g0
            H[:,0,:,:] = R_gp*T[:,0,:,:]/(M[:,0,:,:]*g[:,0,:,:])

        else :

            # Premiere estmiation de l'altitude avec l'acceleration de la pesanteur de la couche precedente

            if TauREx == False :

                for i_n_t in range(n_t) :
                    for i_n_lat in range(n_lat) :
                        for i_n_long in range(n_long) :
                            g_z = g[i_n_t,pres-1,i_n_lat,i_n_long]
                            Rp_c = Rp + z[i_n_t,pres-1,i_n_lat,i_n_long]
                            if T[i_n_t,pres,i_n_lat,i_n_long] != T[i_n_t,pres-1,i_n_lat,i_n_long] :
                                a_z = -(1+z[i_n_t,pres-1,i_n_lat,i_n_long]/Rp_c)*R_gp*(T[i_n_t,pres,i_n_lat,i_n_long]-T[i_n_t,pres-1,i_n_lat,i_n_long])\
                                      /((M[i_n_t,pres,i_n_lat,i_n_long]+M[i_n_t,pres-1,i_n_lat,i_n_long])/2.*g_z*\
                                np.log(T[i_n_t,pres,i_n_lat,i_n_long]/T[i_n_t,pres-1,i_n_lat,i_n_long]))*np.log(P[i_n_t,pres,i_n_lat,i_n_long]/P[i_n_t,pres-1,i_n_lat,i_n_long])
                            else :
                                a_z = -(1+z[i_n_t,pres-1,i_n_lat,i_n_long]/Rp_c)*R_gp*T[i_n_t,pres-1,i_n_lat,i_n_long]/((M[i_n_t,pres,i_n_lat,i_n_long]+M[i_n_t,pres-1,i_n_lat,i_n_long])/2.*g_z)\
                                *np.log(P[i_n_t,pres,i_n_lat,i_n_long]/P[i_n_t,pres-1,i_n_lat,i_n_long])
                            dz = a_z*(1+z[i_n_t,pres-1,i_n_lat,i_n_long]/Rp_c)/(1-a_z/Rp_c)

                            z[i_n_t,pres,i_n_lat,i_n_long] = z[i_n_t,pres-1,i_n_lat,i_n_long] + dz

                if MassAtm == True :
                    g[:,pres,:,:] = g0 + Mass*G/(Rp + z[:,pres,:,:])**2
                else :
                    g[:,pres,:,:] = g0*1./(1.+z[i_n_t,pres-1,i_n_lat,i_n_long]/Rp)**2 + np.zeros((n_t,n_lat,n_long),dtype=np.float64)
                H[:,pres,:,:] = R_gp*T[:,pres,:,:]/(M[:,pres,:,:]*g[:,pres,:,:])

            else :
                for i_n_t in range(n_t) :
                    for i_n_lat in range(n_lat) :
                        for i_n_long in range(n_long) :
                            dz = H[i_n_t,pres-1,i_n_lat,i_n_long]*np.log(P[i_n_t,pres-1,i_n_lat,i_n_long]/P[i_n_t,pres,i_n_lat,i_n_long])
                            z[i_n_t,pres,i_n_lat,i_n_long] = z[i_n_t,pres-1,i_n_lat,i_n_long] + dz
                g[:,pres,:,:] = g0*1/(1+z[:,pres,:,:]/Rp)**2
                H[:,pres,:,:] = R_gp*T[:,pres,:,:]/(M[:,pres,:,:]*g[:,pres,:,:])

            # On incremente petit a petit la masse atmospherique

            if MassAtm == True :
                Mass += P[:,pres,:,:]/(R_gp*T[:,pres,:,:])*M[:,pres,:,:]*4/3.*np.pi*((Rp + z[:,pres,:,:])**3 - (Rp + z[:,pres-1,:,:])**3)

    print z

    if h < np.amax(z) :

        h = np.amax(z)
        hmax = h

    else :

        hmax = np.amax(z)

    dim = int(h/delta_z)+2
    np.save('/Users/caldas/Desktop/Pytmosph3R/ParaCompare/z.npy',z)

    if TopPressure == 'Mean' or TopPressure == 'No' :
        M_mean = np.nansum(M[:,n_l-1,:,:])/(n_t*n_lat*n_long)
        z_t = np.mean(z[:,n_l-1,:,:])
        g_roof = g0*1/(1+z_t/Rp)**2
        H_mean = R_gp*T_mean/(M_mean*g_roof)
    if TopPressure == 'Up' :
        wh_up = np.where(z[:,n_l-1,:,:] == np.amax(z))
        z_t = np.amax(z)
        g_roof = g0*1/(1.+z_t/Rp)**2
        H_mean = R_gp*T[wh_up[0],n_l-1,wh_up[1],wh_up[2]]/(M[wh_up[0],n_l-1,wh_up[1],wh_up[2]]*g_roof)
    if TopPressure == 'Down' :
        wh_dn = np.where(z[:,n_l-1,:,:] == np.amin(z[:,n_l-1,:,:]))
        z_t = z[wh_dn[0],n_l-1,wh_dn[1],wh_dn[2]]
        g_roof = g0*1/(1.+z_t/Rp)**2
        H_mean = R_gp*T[wh_dn[0],n_l-1,wh_dn[1],wh_dn[2]]/(M[wh_dn[0],n_l-1,wh_dn[1],wh_dn[2]]*g_roof)

    print("The thickness of the simulation is %i m"%(np.amax(z)))
    print("The thickness of the atmosphere is %i m"%((dim-2)*delta_z))
    print("The scale height at the roof is %f m"%(H_mean))

    if TopPressure != 'No' :
        alp_h = H_mean*np.log(P_mean/P_h)
        z_h = z_t + alp_h/(1.+alp_h/(Rp+z_t))
        dim = int(z_h/delta_z)+2
        z_h = (dim-2)*delta_z
        h = z_h

    print("The final thickness of the atmosphere is %i m"%((dim-2)*delta_z))

    data_convert = np.zeros((number,n_t,dim,n_lat,n_long),dtype=np.float64)

    Mass = np.zeros((n_t,n_lat,n_long),dtype=np.float64)
    Reformate = False

    bar = ProgressBar(dim,'Computation of the atmospheric dataset')

    for i_z in range(dim) :

        # Si la fonction Middle est selectionnee, le code va formater la grille cylindrique de maniere a ce que le
        # premier point corresponde aux donnees de surface tandis que les autres points correspondront aux donnees
        # des milieux de couche.

        if Middle == False :
            z_ref = i_z*delta_z
        else :
            if i_z == 0 :
                z_ref = 0.
            else :
                if i_z == dim-1 :
                    z_ref = (i_z-1)*delta_z
                else :
                    z_ref = (i_z-0.5)*delta_z

        if z_ref >= hmax :
            Reformate = True

        for t in range(n_t) :

            for lat in range(n_lat) :

                for long in range(n_long) :

                    # Nous cherchons l'intervalle dans lequel se situe le point d'altitude considere

                    wh, = np.where(z[t,:,lat,long] >= z_ref)

                    # Si le point en question n'est pas au dessus du toit du modele a cette lattitude et a cette longitude

                    if wh.size != 0 and i_z != 0 :

                        res, c_grid, i_grid = interpolation(z_ref,z[t,:,lat,long],np.log(P[t,:,lat,long]))

                        data_convert[0,t,i_z,lat,long] = np.exp(res)
                        data_convert[1,t,i_z,lat,long] = c_grid[1]*T[t,i_grid[0],lat,long] + c_grid[0]*T[t,i_grid[1],lat,long]
                        if composition[0] == 'composition' :
                            if Tracer == True :
                                data_convert[2,t,i_z,lat,long] = c_grid[1]*Q[t,i_grid[0],lat,long] + c_grid[0]*Q[t,i_grid[1],lat,long]

                                if Clouds == True :
                                    for c_num in range(c_number) :
                                        data_convert[3+c_num,t,i_z,lat,long] = c_grid[1]*gen[c_num,t,i_grid[0],lat,long] + c_grid[0]*gen[c_num,t,i_grid[1],lat,long]

                                if LogInterp == True :
                                    com, c_gr, i_gr = interp3olation_uni_multi(np.log10(data_convert[0,t,i_z,lat,long]),data_convert[1,t,i_z,lat,long],\
                                                                                data_convert[2,t,i_z,lat,long],np.log10(P_comp),T_comp,Q_comp,x_species)
                                else :
                                    com, c_gr, i_gr = interp3olation_uni_multi(data_convert[0,t,i_z,lat,long],data_convert[1,t,i_z,lat,long],\
                                                                                data_convert[2,t,i_z,lat,long],P_comp,T_comp,Q_comp,x_species)
                            else :
                                if Clouds == True :
                                    for c_num in range(c_number) :
                                        data_convert[2+c_num,t,i_z,lat,long] = c_grid[1]*gen[c_num,t,i_grid[0],lat,long] + c_grid[0]*gen[c_num,t,i_grid[1],lat,long]

                                if LogInterp == True :
                                    com, c_gr, i_gr = interp2olation_uni_multi(np.log10(data_convert[0,t,i_z,lat,long]),data_convert[1,t,i_z,lat,long],\
                                                                                np.log10(P_comp),T_comp,x_species)
                                else :
                                    com, c_gr, i_gr = interp2olation_uni_multi(data_convert[0,t,i_z,lat,long],data_convert[1,t,i_z,lat,long],\
                                                                                P_comp,T_comp,x_species)

                        if composition[0] == 'tracer_other' :
                            if Tracer == True :
                                data_convert[2,t,i_z,lat,long] = c_grid[1]*Q[t,i_grid[0],lat,long] + c_grid[0]*Q[t,i_grid[1],lat,long]

                                if Clouds == True :
                                    for c_num in range(c_number) :
                                        data_convert[3+c_num,t,i_z,lat,long] = c_grid[1]*gen[c_num,t,i_grid[0],lat,long] + c_grid[0]*gen[c_num,t,i_grid[1],lat,long]

                                ind, = np.where(species[2:] == m_species)
                                com = np.zeros(species.size,dtype=np.float64)
                                if NoH2 == False :
                                    M_f = 1./(ratio + 1.)*M_species[0] + ratio/(ratio+1.)*M_species[1]
                                    ratio_mf = M_f/M_species[ind+2]*data_convert[2,t,i_z,lat,long]/(1.-data_convert[2,t,i_z,lat,long])
                                    com[ind+2] = ratio_mf/(1.+ratio_mf)
                                else :
                                    indf, = np.where(species[2:] != m_species)
                                    ratio_mf = M_species[indf+2]/M_species[ind+2]*data_convert[2,t,i_z,lat,long]/(1.-data_convert[2,t,i_z,lat,long])
                                    com[ind+2] = ratio_mf/(1.+ratio_mf)
                                    com[indf+2] = 1. - com[ind+2]

                        if NoH2 == False :
                            data_convert[2+m_number+c_number,t,i_z,lat,long] = (1. - np.nansum(com[2:]))/(1. + ratio)
                            data_convert[2+m_number+c_number+1,t,i_z,lat,long] = data_convert[2+m_number+c_number,t,i_z,lat,long]*ratio
                            data_convert[2+m_number+c_number+2:number-1,t,i_z,lat,long] = com[2:]
                        else :
                            data_convert[2+m_number+c_number,t,i_z,lat,long] = 0.
                            data_convert[2+m_number+c_number+1,t,i_z,lat,long] = 0.
                            data_convert[2+m_number+c_number+2:number-1,t,i_z,lat,long] = com[2:]/(np.nansum(com[2:]))
                        data_convert[2+m_number+c_number+size,t,i_z,lat,long] = np.nansum(data_convert[2+m_number+c_number:2+m_number+c_number+size,t,i_z,lat,long]*M_species)

                        Mass[t,lat,long] += data_convert[0,t,i_z,lat,long]/(R_gp*data_convert[1,t,i_z,lat,long])*\
                                    data_convert[number-1,t,i_z,lat,long]*4/3.*np.pi*((Rp + i_z*delta_z)**3 - (Rp + (i_z - 1)*delta_z)**3)

                    # Si le point d'altitude est plus eleve que le toit du modele a cette lattitude et cette longitude
                    # il nous faut extrapoler

                    if i_z == 0 :

                        data_convert[0,t,i_z,lat,long] = P[t,0,lat,long]
                        data_convert[1,t,i_z,lat,long] = T[t,0,lat,long]
                        if Tracer == True :
                            data_convert[2,t,i_z,lat,long] = Q[t,0,lat,long]
                        if Clouds == True :
                            for c_num in range(c_number) :
                                data_convert[2+m_number+c_num,t,i_z,lat,long] = gen[c_num,t,0,lat,long]
                        data_convert[2+m_number+c_number:number-1,t,i_z,lat,long] = compo[:,t,0,lat,long]
                        data_convert[2+m_number+c_number+size,t,i_z,lat,long] = M[t,0,lat,long]

                    if wh.size == 0 :

                        # Nous avons besoin d'une temperature de reference pour trouver la composition sur le dernier point
                        # en altitude, suivant le type d'extrapolation, nous ne pouvons pas l'identifier a celle deja calculee
                        # et nous preferons l'executer a partir des donnees d'equilibre que sur des resultats d'interpolation

                        if Reformate == False :

                            data_convert[1,t,i_z,lat,long] = T[t,n_l-1,lat,long]

                        else :

                            if Upper == "Isotherme" :
                                data_convert[1,t,i_z,lat,long] = T[t,n_l-1,lat,long]
                            if Upper ==  "Isotherme_moyen" :
                                data_convert[1,t,i_z,lat,long] = T_mean
                            if Upper == "Maximum_isotherme" :
                                data_convert[1,t,i_z,lat,long] = T_max
                            if Upper == "Minimum_isotherme" :
                                data_convert[1,t,i_z,lat,long] = T_min

                        # On estime la pression au dela du toit a partir de la temperature choisie

                        if MassAtm == True :
                            g = g0 + Mass[t,lat,long]*G/(Rp + i_z*delta_z)**2
                        else :
                            g = g0

                        if i_z != dim-1 :
                            data_convert[0,t,i_z,lat,long] = data_convert[0,t,i_z-1,lat,long]*np.exp(-data_convert[number-1,t,i_z-1,lat,long]*g*\
                                delta_z/(R_gp*data_convert[1,t,i_z,lat,long])*1./((1+z_ref/Rp)*(1+(z_ref-delta_z)/Rp)))
                        else :
                            data_convert[0,t,i_z,lat,long] = data_convert[0,t,i_z-1,lat,long]*np.exp(-data_convert[number-1,t,i_z-1,lat,long]*g*\
                                delta_z/(2.*R_gp*data_convert[1,t,i_z,lat,long])*1./((1+z_ref/Rp)*(1+(z_ref-delta_z/2.)/Rp)))

                        T_ref = data_convert[1,t,i_z,lat,long]

                        # On incremente toujours la masse atmospherique pour la latitude et la longitude donnee, les
                        # ce point est a modifier

                        if MassAtm == True :
                            Mass[t,lat,long] += data_convert[0,t,i_z-1,lat,long]/(R_gp*data_convert[1,t,i_z-1,lat,long])*\
                                data_convert[number-1,t,i_z-1,lat,long]*4/3.*np.pi*((Rp + i_z*delta_z)**3 - (Rp + (i_z - 1)*delta_z)**3)

                        P_ref = data_convert[0,t,i_z,lat,long]

                        if composition[0] == 'composition' :
                            if Tracer == True :
                                data_convert[2,t,i_z,lat,long] = Q[t,n_l-1,lat,long]
                                Q_ref = data_convert[2,t,i_z,lat,long]

                                if LogInterp == True :
                                    compos, c_grid, i_grid = interp3olation_uni_multi(np.log10(P_ref),T_ref,Q_ref,np.log10(P_comp),T_comp,Q_comp,x_species)
                                else :
                                    compos, c_grid, i_grid = interp3olation_uni_multi(P_ref,T_ref,Q_ref,P_comp,T_comp,Q_comp,x_species)

                                if Clouds == True :
                                    data_convert[3:3+c_number,t,i_z,lat,long] = gen[:,t,n_l-1,lat,long]

                            else :
                                if LogInterp == True :
                                    compos, c_grid, i_grid = interp2olation_uni_multi(np.log10(P_ref),T_ref,np.log10(P_comp),T_comp,x_species)
                                else :
                                    compos, c_grid, i_grid = interp2olation_uni_multi(P_ref,T_ref,P_comp,T_comp,x_species)

                                if Clouds == True :
                                    data_convert[2:2+c_number,t,i_z,lat,long] = gen[:,t,n_l-1,lat,long]

                            if NoH2 == False :
                                compoH2 = (1 - np.nansum(compos[2:]))/(ratio + 1.)
                                compoHe = compoH2*ratio
                                data_convert[2+m_number+c_number,t,i_z,lat,long] = compoH2
                                data_convert[3+m_number+c_number,t,i_z,lat,long] = compoHe
                                data_convert[4+m_number+c_number:number-1,t,i_z,lat,long] = compos[2:]
                            else :
                                data_convert[2+m_number+c_number,t,i_z,lat,long] = 0.
                                data_convert[3+m_number+c_number,t,i_z,lat,long] = 0.
                                data_convert[4+m_number+c_number:number-1,t,i_z,lat,long] = compos[2:]/(np.nansum(compos[2:]))
                            data_convert[number-1,t,i_z,lat,long] = np.nansum(data_convert[2+m_number+c_number:number-1,t,i_z,lat,long]*\
                                        M_species)

                        if composition[0] == 'tracer_other' :
                            data_convert[2:,t,i_z,lat,long] = data_convert[2:,t,i_z-1,lat,long]

        bar.animate(i_z + 1)

    print 'Shape of the dataset :',np.shape(data_convert)

    list = np.array([])

    for i in range(number) :

        wh = np.where(data_convert[i] < 0)

        if len(wh[0]) != 0 :

            list = np.append(list,i)

    if list.size != 0 :

        mess = 'Dataset error, negative value encontered for axis : '

        for i in range(list.size) :

            mess += '%i, '%(list[i])

        mess += 'a correction is necessary, or Boxes failed'

        print mess

    if Inverse[0] == 'True' :
        data_convert = reverse_dim(data_convert,4,np.float64)
        print 'Data needs to be reverse on longitude.'
    if Inverse[1] == 'True' :
        data_convert = reverse_dim(data_convert,3,np.float64)
        print 'Data needs to be reverse on latitude.'
    if Rotate == True :
        data_convert_r = np.zeros(np.shape(data_convert),dtype=np.float64)
        long_rot = np.int(np.float(obs[1])/(2*np.pi)*n_long)
        for i_l in range(n_long) :
            i_l_r = (i_l + long_rot)%(n_long)
            data_convert_r[:,:,:,:,i_l] = data_convert[:,:,:,:,i_l_r]
        data_convert = data_convert_r

    return data_convert, h


########################################################################################################################


def NBoxes(data,n_layers,Rp,h,P_h,t,g0,M_atm,number,T_comp,P_comp,Q_comp,species,x_species,M_species,c_species,m_species,ratio,Upper,composition,obs,\
          TopPressure,Inverse,Surf=True,Tracer=False,Clouds=False,Middle=False,LogInterp=False,TimeSelec=False,MassAtm=False,NoH2=False,TauREx=True,Rotate=False) :

    print TauREx
    if data != '' :
        file = Dataset("%s.nc"%(data))
        variables = file.variables
    else :
        from pyfunction import planet
        planet = planet()
    c_number = c_species.size
    if Tracer == True :
        m_number = 1
        m_species = m_species[0]
    else :
        m_number = 0

    # Si nous avons l'information sur les parametres de surface

    if data != '' :

        if Surf == True :

            # Si nous avons l'information sur la pression de surface, il nous faut donc rallonger les tableaux de parametres
            # de 1

            if TimeSelec == False :
                T_file = variables["temp"][:]
                n_t,n_l,n_lat,n_long = np.shape(T_file)
                T_surf = variables["tsurf"][:]
                P_file = variables["p"][:]
                P_surf = variables["ps"][:]
                P = np.zeros((n_t,n_l+1,n_lat,n_long),dtype=np.float64)
                T = np.zeros((n_t,n_l+1,n_lat,n_long),dtype=np.float64)
            else :
                T_prefile = variables["temp"][:]
                n_t,n_l,n_lat,n_long = np.shape(T_prefile)
                T_file = np.zeros((1,n_l,n_lat,n_long),dtype=np.float64)
                T_surf = np.zeros((1,n_lat,n_long),dtype=np.float64)
                P_file = np.zeros((1,n_l,n_lat,n_long),dtype=np.float64)
                P_surf = np.zeros((1,n_lat,n_long),dtype=np.float64)
                T_file[0] = variables["temp"][t,:,:,:]
                T_surf[0] = variables["tsurf"][t,:,:]
                P_file[0] = variables["p"][t,:,:,:]
                P_surf[0] = variables["ps"][t,:,:]
                P = np.zeros((1,n_l+1,n_lat,n_long),dtype=np.float64)
                T = np.zeros((1,n_l+1,n_lat,n_long),dtype=np.float64)

            P[:,0,:,:] = P_surf
            P[:,1:n_l+1,:,:] = P_file
            T[:,0,:,:] = T_surf
            T[:,1:n_l+1,:,:] = T_file

            if Tracer == True :
                if TimeSelec == False :
                    Q_vap = variables["%s_vap"%(m_species)][:]
                    Q_vap_surf = variables["%s_vap_surf"%(m_species)][:]
                    Q = np.zeros((n_t,n_l+1,n_lat,n_long),dtype=np.float64)

                else :
                    Q_vap = np.zeros((1,n_l,n_lat,n_long))
                    Q_vap_surf = np.zeros((1,n_lat,n_long))
                    Q_vap[0] = variables["%s_vap"%(m_species)][t,:,:,:]
                    Q_vap_surf[0] = variables["%s_vap_surf"%(m_species)][t,:,:]
                    Q = np.zeros((1,n_l+1,n_lat,n_long),dtype=np.float64)

                Q[:,0,:,:] = Q_vap_surf
                Q[:,1:n_l+1,:,:] = Q_vap
            else :
                Q = np.array([])

            if Clouds == True :
                if TimeSelec == False :
                    gen_cond = np.zeros((c_number,n_t,n_l,n_lat,n_long),dtype=np.float64)
                    gen_cond_surf = np.zeros((c_number,n_t,n_lat,n_long),dtype=np.float64)
                    for c_num in range(c_number) :
                        gen_cond_surf[c_num,:,:,:] = variables["%s_surf"%(c_species[c_num])][:]
                        gen_cond[c_num,:,:,:,:] = variables["%s"%(c_species[c_num])][:]
                    gen = np.zeros((c_species.size,n_t,n_l+1,n_lat,n_long))
                else :
                    gen_cond = np.zeros((c_number,1,n_l,n_lat,n_long),dtype=np.float64)
                    gen_cond_surf = np.zeros((c_number,1,n_lat,n_long),dtype=np.float64)
                    for c_num in range(c_number) :
                        gen_cond_surf[c_num,:,:,:] = variables["%s_surf"%(c_species[c_num])][t,:,:]
                        gen_cond[c_num,:,:,:,:] = variables["%s"%(c_species[c_num])][t,:,:,:]
                    gen = np.zeros((c_species.size,1,n_l+1,n_lat,n_long),dtype=np.float64)

                gen[:,:,0,:,:] = gen_cond_surf
                gen[:,:,1:n_l+1,:,:] = gen_cond
            else :
                gen = np.array([])

            if TimeSelec == True :
                n_t = 1
            T_mean = np.nansum(T_file[:,n_l-1,:,:])/np.float(n_t*n_lat*n_long)
            T_max = np.amax(T_file[:,n_l-1,:,:])
            T_min = np.amin(T_file[:,n_l-1,:,:])
            print('Mean temperature : %i K, Maximal temperature : %i K, Minimal temperature : %i K'%(T_mean,T_max,T_min))

            P_mean = np.exp(np.nansum(np.log(P[:,n_l,:,:]))/(n_t*n_lat*n_long))
            print('Mean roof pressure : %f Pa'%(P_mean))

            n_l = n_l + 1
            z = np.zeros((n_t,n_l,n_lat,n_long),dtype=np.float64)

        # Si nous n'avons pas l'information sur les parametres de surface

        else :

            if TimeSelec == False :
                T = variables["temp"][:]
                n_t,n_l,n_lat,n_long = np.shape(T)
                P = variables["p"][:]
            else :
                T_prefile = variables["temp"][:]
                n_t,n_l,n_lat,n_long = np.shape(T_prefile)
                T = np.zeros((1,n_l,n_lat,n_long),dtype=np.float64)
                P = np.zeros((1,n_l,n_lat,n_long),dtype=np.float64)
                T[0] = variables["temp"][t,:,:,:]
                P[0] = variables["p"][t,:,:,:]

            if Tracer == True :
                if TimeSelec == False :
                    Q = variables["%s_vap"%(m_species)][:]
                else :
                    Q = np.zeros((1,n_l,n_lat,n_long))
                    Q[0] = variables["%s_vap"%(m_species)][t,:,:,:]
            else :
                Q = np.array([])

            if Clouds == True :
                if TimeSelec == False :
                    gen = np.zeros((c_number,n_t,n_l,n_lat,n_long))
                    for c_num in range(c_number) :
                        gen[c_num,:,:,:,:] = variables["%s"%(c_species[c_num])][:]
                else :
                    gen = np.zeros((c_number,1,n_l,n_lat,n_long))
                    for c_num in range(c_number) :
                        gen[c_num,:,:,:,:] = variables["%s"%(c_species[c_num])][t,:,:,:]
            else :
                gen = np.array([])

            if TimeSelec == True :
                n_t = 1
            T_mean = np.nansum(T[:,n_l-1,:,:])/np.float(n_t*n_lat*n_long)
            T_max = np.amax(T[:,n_l-1,:,:])
            T_min = np.amin(T[:,n_l-1,:,:])
            print('Mean temperature : %i K, Maximal temperature : %i K, Minimal temperature of the high atmosphere : %i K'\
                  %(T_mean,T_max,T_min))

            P_mean = np.exp(np.nansum(np.log(P[:,n_l-1,:,:]))/(n_t*n_lat*n_long))
            print('Mean roof pressure : %f Pa'%(P_mean))

    else :

        data = pickle.load(open(planet.pressure_profile_data))
        param = data['params']
        T_file = data['data'][planet.pressure_profile_key][:,1]
        n_t,n_l,n_lat,n_long = 1, param[planet.number_layer_key],int(planet.latitude)+1,int(planet.longitude)+1
        T_surf = param[planet.planet_temperature_key]
        P_file = np.linspace(np.log10(param[planet.extreme_pressure_key[0]]),np.log10(param[planet.extreme_pressure_key[1]]),param[planet.number_layer_key]+1)
        P_file = 10**P_file
        T = np.zeros((n_t,n_l+1,n_lat,n_long),dtype=np.float64)
        P = np.zeros((n_t,n_l+1,n_lat,n_long),dtype=np.float64)

        T[:,0,:,:] = np.ones((n_t,n_lat,n_long),dtype=np.float64)*T_surf
        for i_n_t in range(n_t) :
            for i_n_lat in range(n_lat) :
                for i_n_long in range(n_long) :
                    T[i_n_t,1:n_l+1,i_n_lat,i_n_long] = T_file
                    P[i_n_t,:,i_n_lat,i_n_long] = P_file

        Q = np.array([])

        if Clouds == True :
            gen_cond = np.zeros((c_number,1,n_l,n_lat,n_long),dtype=np.float64)
            gen_cond_surf = np.zeros((c_number,1,n_lat,n_long),dtype=np.float64)
            for c_num in range(c_number) :
                gen_cond_surf[c_num,:,:,:] = data['data'][planet.extreme_pressure_key][:,c_num]
                gen_cond[c_num,:,:,:,:] = data['data'][planet.extreme_pressure_key][:,c_num]
            gen = np.zeros((c_species.size,1,n_l+1,n_lat,n_long),dtype=np.float64)

            gen[:,:,0,:,:] = gen_cond_surf
            gen[:,:,1:n_l+1,:,:] = gen_cond
        else :
            gen = np.array([])

        T_mean = np.mean(T_file[n_l-1])
        T_max = np.amax(T_file[n_l-1])
        T_min = np.amin(T_file[n_l-1])
        print('Mean temperature : %i K, Maximal temperature : %i K, Minimal temperature : %i K'%(T_mean,T_max,T_min))

        P_mean = np.exp(np.nansum(np.log(P[:,n_l,:,:]))/(n_t*n_lat*n_long))
        print('Mean roof pressure : %f Pa'%(P_mean))

    z = np.zeros((n_t,n_l,n_lat,n_long),dtype=np.float64)
    M = np.zeros((n_t,n_l,n_lat,n_long),dtype=np.float64)
    H = np.zeros((n_t,n_l,n_lat,n_long),dtype=np.float64)
    g = np.zeros((n_t,n_l,n_lat,n_long),dtype=np.float64)

    bar = ProgressBar(n_t*n_l,'Data convertion from pressure levels')

    if Tracer == False :

        size = species.size
        compo = np.zeros((size,n_t,n_l,n_lat,n_long),dtype=np.float64)

        if LogInterp == True :

            P_comp = np.log10(P_comp)

        for i in range(n_t) :
            for j in range(n_l) :
                for k in range(n_lat) :

                    if composition[0] == 'composition' :
                        if LogInterp == True :
                            res, c_grid, i_grid = interp2olation_multi(np.log10(P[i,j,k,:]),T[i,j,k,:],P_comp,T_comp,x_species)
                        else :
                            res, c_grid, i_grid = interp2olation_multi(P[i,j,k,:],T[i,j,k,:],P_comp,T_comp,x_species)

                        compo[2:size,i,j,k,:] = res[2:]
                        for l in range(n_long) :
                            if NoH2 == False :
                                compo[0,i,j,k,l] = (1. - np.nansum(compo[2:size,i,j,k,l]))/(ratio + 1.)
                            else :
                                compo[0,i,j,k,l] = 0.
                                compo[2:size,i,j,k,l] = compo[2:size,i,j,k,l]/(np.nansum(compo[2:size,i,j,k,l]))
                        if NoH2 == False :
                            compo[1,i,j,k,:] = compo[0,i,j,k,:]*ratio
                        else :
                            compo[1,i,j,k,l] = 0.
                        M[i,j,k,:] = np.dot(M_species,compo[:,i,j,k,:])
                bar.animate(i*n_l+j+1)

    else :

        size = species.size
        compo = np.zeros((size,n_t,n_l,n_lat,n_long),dtype=np.float64)

        if LogInterp == True :

            P_comp = np.log10(P_comp)

        for i in range(n_t) :
            for j in range(n_l) :
                for k in range(n_lat) :
                    if composition[0] == 'composition' :
                        if LogInterp == True :
                            res, c_grid, i_grid = interp3olation_multi(np.log10(P[i,j,k,:]),T[i,j,k,:],Q[i,j,k,:],P_comp,T_comp,Q_comp,x_species)
                        else :
                            res, c_grid, i_grid = interp3olation_multi(P[i,j,k,:],T[i,j,k,:],Q[i,j,k,:],P_comp,T_comp,Q_comp,x_species)

                        compo[2:size,i,j,k,:] = res[2:]
                        for l in range(n_long) :
                            if NoH2 == False :
                                compo[0,i,j,k,l] = (1. - np.nansum(compo[2:size,i,j,k,l]))/(ratio + 1.)
                            else :
                                compo[0,i,j,k,l] = 0.
                                compo[2:size,i,j,k,l] = compo[2:size,i,j,k,l]/(np.nansum(compo[2:size,i,j,k,l]))
                        if NoH2 == False :
                            compo[1,i,j,k,:] = compo[0,i,j,k,:]*ratio
                        else :
                            compo[1,i,j,k,:] = np.zeros(n_long,dtype=np.float64)
                        M[i,j,k,:] = np.dot(M_species,compo[:,i,j,k,:])

                    if composition[0] == 'tracer_other' :
                        ind, = np.where(species[2:] == m_species)
                        if NoH2 == False :
                            M_f = 1./(ratio + 1.)*M_species[0] + ratio/(ratio+1.)*M_species[1]
                            ratio_mf = M_f/M_species[ind+2]*Q[i,j,k,:]/(1.-Q[i,j,k,:])
                            compo[ind+2,i,j,k,:] = ratio_mf/(1.+ratio_mf)
                            compo[0,i,j,k,:] = (1. - compo[ind+2,i,j,k,:])/(ratio + 1.)
                        else :
                            indf, = np.where(species[2:] != m_species)
                            ratio_mf = M_species[indf+2]/M_species[ind+2]*Q[i,j,k,:]/(1.-Q[i,j,k,:])
                            compo[ind+2,i,j,k,:] = ratio_mf/(1.+ratio_mf)
                            compo[indf+2,i,j,k,:] = 1. - compo[ind+2,i,j,k,:]
                        if NoH2 == False :
                            compo[1,i,j,k,:] = compo[0,i,j,k,:]*ratio
                        else :
                            compo[1,i,j,k,:] = np.zeros(n_long,dtype=np.float64)
                        M[i,j,k,:] = np.dot(M_species,compo[:,i,j,k,:])

                bar.animate(i*n_l+j+1)

    # Une fois la composition dans chaque cellule des donnees GCM calculee, nous avons l'information manquante sur le
    # poids moleculaire moyen et donc sur la hauteur d'echelle locale. Nous pouvons alors transformer l'echelle de
    # pression en echelle d'altitude

    for pres in range(n_l) :

        if pres == 0 :

            z[:,0,:,:] = 0.
            Mass = np.zeros((n_t,n_lat,n_long),dtype=np.float64)
            g[:,0,:,:] = np.ones((n_t,n_lat,n_long),dtype=np.float64)*g0
            H[:,0,:,:] = R_gp*T[:,0,:,:]/(M[:,0,:,:]*g[:,0,:,:])

        else :

            # Premiere estmiation de l'altitude avec l'acceleration de la pesanteur de la couche precedente

            if TauREx == False :

                for i_n_t in range(n_t) :
                    for i_n_lat in range(n_lat) :
                        for i_n_long in range(n_long) :
                            g_z = g[i_n_t,pres-1,i_n_lat,i_n_long]
                            Rp_c = Rp + z[i_n_t,pres-1,i_n_lat,i_n_long]
                            if T[i_n_t,pres,i_n_lat,i_n_long] != T[i_n_t,pres-1,i_n_lat,i_n_long] :
                                a_z = -(1+z[i_n_t,pres-1,i_n_lat,i_n_long]/Rp_c)*R_gp*(T[i_n_t,pres,i_n_lat,i_n_long]-T[i_n_t,pres-1,i_n_lat,i_n_long])\
                                      /((M[i_n_t,pres,i_n_lat,i_n_long]+M[i_n_t,pres-1,i_n_lat,i_n_long])/2.*g_z*\
                                np.log(T[i_n_t,pres,i_n_lat,i_n_long]/T[i_n_t,pres-1,i_n_lat,i_n_long]))*np.log(P[i_n_t,pres,i_n_lat,i_n_long]/P[i_n_t,pres-1,i_n_lat,i_n_long])
                            else :
                                a_z = -(1+z[i_n_t,pres-1,i_n_lat,i_n_long]/Rp_c)*R_gp*T[i_n_t,pres-1,i_n_lat,i_n_long]/((M[i_n_t,pres,i_n_lat,i_n_long]+M[i_n_t,pres-1,i_n_lat,i_n_long])/2.*g_z)\
                                *np.log(P[i_n_t,pres,i_n_lat,i_n_long]/P[i_n_t,pres-1,i_n_lat,i_n_long])
                            dz = a_z*(1+z[i_n_t,pres-1,i_n_lat,i_n_long]/Rp_c)/(1-a_z/Rp_c)

                            z[i_n_t,pres,i_n_lat,i_n_long] = z[i_n_t,pres-1,i_n_lat,i_n_long] + dz

                if MassAtm == True :
                    g[:,pres,:,:] = g0 + Mass*G/(Rp + z[:,pres,:,:])**2
                else :
                    g[:,pres,:,:] = g0*1./(1.+z[i_n_t,pres-1,i_n_lat,i_n_long]/Rp)**2 + np.zeros((n_t,n_lat,n_long),dtype=np.float64)
                H[:,pres,:,:] = R_gp*T[:,pres,:,:]/(M[:,pres,:,:]*g[:,pres,:,:])

            else :
                for i_n_t in range(n_t) :
                    for i_n_lat in range(n_lat) :
                        for i_n_long in range(n_long) :
                            dz = H[i_n_t,pres-1,i_n_lat,i_n_long]*np.log(P[i_n_t,pres-1,i_n_lat,i_n_long]/P[i_n_t,pres,i_n_lat,i_n_long])
                            z[i_n_t,pres,i_n_lat,i_n_long] = z[i_n_t,pres-1,i_n_lat,i_n_long] + dz
                g[:,pres,:,:] = g0*1/(1+z[:,pres,:,:]/Rp)**2
                H[:,pres,:,:] = R_gp*T[:,pres,:,:]/(M[:,pres,:,:]*g[:,pres,:,:])

            # On incremente petit a petit la masse atmospherique

            if MassAtm == True :
                Mass += P[:,pres,:,:]/(R_gp*T[:,pres,:,:])*M[:,pres,:,:]*4/3.*np.pi*((Rp + z[:,pres,:,:])**3 - (Rp + z[:,pres-1,:,:])**3)

    if h < np.amax(z) :
        h = np.amax(z)
        hmax = h
    else :
        hmax = np.amax(z)

    delta_z = h/np.float(n_layers)
    dim = n_layers+2

    if TopPressure == 'Mean' or TopPressure == 'No' :
        M_mean = np.nansum(M[:,n_l-1,:,:])/(n_t*n_lat*n_long)
        z_t = np.mean(z[:,n_l-1,:,:])
        g_roof = g0*1/(1+z_t/Rp)**2
        H_mean = R_gp*T_mean/(M_mean*g_roof)
    if TopPressure == 'Up' :
        wh_up = np.where(z[:,n_l-1,:,:] == np.amax(z))
        z_t = np.amax(z)
        g_roof = g0*1/(1.+z_t/Rp)**2
        H_mean = R_gp*T[wh_up[0],n_l-1,wh_up[1],wh_up[2]][0]/(M[wh_up[0],n_l-1,wh_up[1],wh_up[2]][0]*g_roof)
    if TopPressure == 'Down' :
        wh_dn = np.where(z[:,n_l-1,:,:] == np.amin(z[:,n_l-1,:,:]))
        z_t = z[wh_dn[0],n_l-1,wh_dn[1],wh_dn[2]][0]
        g_roof = g0*1/(1.+z_t/Rp)**2
        H_mean = R_gp*T[wh_dn[0],n_l-1,wh_dn[1],wh_dn[2]][0]/(M[wh_dn[0],n_l-1,wh_dn[1],wh_dn[2]][0]*g_roof)

    print("The scale height at the surface is %f m"%(R_gp*np.mean(T[:,0,:,:])/(np.mean(M[:,0,:,:])*g0)))
    print("The thickness of the simulation is %i m"%(np.amax(z)))
    print("The thickness of the atmosphere is %i m"%((dim-2)*delta_z))
    print("The scale height at the roof is %f m"%(H_mean))

    if TopPressure != 'No' :
        alp_h = H_mean*np.log(P_mean/P_h)
        z_h = z_t + alp_h/(1.+alp_h/(Rp+z_t))
        h = z_h
        delta_z =np.float(np.int(h/np.float(n_layers)))
        h = delta_z*n_layers

    print("The final thickness of the atmosphere is %i m"%((dim-2)*delta_z))

    data_convert = np.zeros((number,n_t,dim,n_lat,n_long),dtype=np.float64)

    Mass = np.zeros((n_t,n_lat,n_long),dtype=np.float64)
    Reformate = False

    bar = ProgressBar(dim,'Computation of the atmospheric dataset')

    for i_z in range(dim) :

        # Si la fonction Middle est selectionnee, le code va formater la grille cylindrique de maniere a ce que le
        # premier point corresponde aux donnees de surface tandis que les autres points correspondront aux donnees
        # des milieux de couche.

        if Middle == False :
            z_ref = i_z*delta_z
        else :
            if i_z == 0 :
                z_ref = 0.
            else :
                if i_z == dim-1 :
                    z_ref = (i_z-1)*delta_z
                else :
                    z_ref = (i_z-0.5)*delta_z

        if z_ref >= hmax :
            Reformate = True

        for t in range(n_t) :

            for lat in range(n_lat) :

                for long in range(n_long) :

                    # Nous cherchons l'intervalle dans lequel se situe le point d'altitude considere

                    wh, = np.where(z[t,:,lat,long] >= z_ref)

                    # Si le point en question n'est pas au dessus du toit du modele a cette lattitude et a cette longitude

                    if wh.size != 0 and i_z != 0 :

                        res, c_grid, i_grid = interpolation(z_ref,z[t,:,lat,long],np.log(P[t,:,lat,long]))

                        data_convert[0,t,i_z,lat,long] = np.exp(res)
                        data_convert[1,t,i_z,lat,long] = c_grid[1]*T[t,i_grid[0],lat,long] + c_grid[0]*T[t,i_grid[1],lat,long]
                        if composition[0] == 'composition' :
                            if Tracer == True :
                                data_convert[2,t,i_z,lat,long] = c_grid[1]*Q[t,i_grid[0],lat,long] + c_grid[0]*Q[t,i_grid[1],lat,long]

                                if Clouds == True :
                                    for c_num in range(c_number) :
                                        data_convert[3+c_num,t,i_z,lat,long] = c_grid[1]*gen[c_num,t,i_grid[0],lat,long] + c_grid[0]*gen[c_num,t,i_grid[1],lat,long]

                                if LogInterp == True :
                                    com, c_gr, i_gr = interp3olation_uni_multi(np.log10(data_convert[0,t,i_z,lat,long]),data_convert[1,t,i_z,lat,long],\
                                                                                data_convert[2,t,i_z,lat,long],np.log10(P_comp),T_comp,Q_comp,x_species)
                                else :
                                    com, c_gr, i_gr = interp3olation_uni_multi(data_convert[0,t,i_z,lat,long],data_convert[1,t,i_z,lat,long],\
                                                                                data_convert[2,t,i_z,lat,long],P_comp,T_comp,Q_comp,x_species)
                            else :
                                if Clouds == True :
                                    for c_num in range(c_number) :
                                        data_convert[2+c_num,t,i_z,lat,long] = c_grid[1]*gen[c_num,t,i_grid[0],lat,long] + c_grid[0]*gen[c_num,t,i_grid[1],lat,long]

                                if LogInterp == True :
                                    com, c_gr, i_gr = interp2olation_uni_multi(np.log10(data_convert[0,t,i_z,lat,long]),data_convert[1,t,i_z,lat,long],\
                                                                                np.log10(P_comp),T_comp,x_species)
                                else :
                                    com, c_gr, i_gr = interp2olation_uni_multi(data_convert[0,t,i_z,lat,long],data_convert[1,t,i_z,lat,long],\
                                                                                P_comp,T_comp,x_species)

                        if composition[0] == 'tracer_other' :
                            if Tracer == True :
                                data_convert[2,t,i_z,lat,long] = c_grid[1]*Q[t,i_grid[0],lat,long] + c_grid[0]*Q[t,i_grid[1],lat,long]

                                if Clouds == True :
                                    for c_num in range(c_number) :
                                        data_convert[3+c_num,t,i_z,lat,long] = c_grid[1]*gen[c_num,t,i_grid[0],lat,long] + c_grid[0]*gen[c_num,t,i_grid[1],lat,long]

                                ind, = np.where(species[2:] == m_species)
                                com = np.zeros(species.size,dtype=np.float64)
                                if NoH2 == False :
                                    M_f = 1./(ratio + 1.)*M_species[0] + ratio/(ratio+1.)*M_species[1]
                                    ratio_mf = M_f/M_species[ind+2]*data_convert[2,t,i_z,lat,long]/(1.-data_convert[2,t,i_z,lat,long])
                                    com[ind+2] = ratio_mf/(1.+ratio_mf)
                                else :
                                    indf, = np.where(species[2:] != m_species)
                                    ratio_mf = M_species[indf+2]/M_species[ind+2]*data_convert[2,t,i_z,lat,long]/(1.-data_convert[2,t,i_z,lat,long])
                                    com[ind+2] = ratio_mf/(1.+ratio_mf)
                                    com[indf+2] = 1. - com[ind+2]

                        if NoH2 == False :
                            data_convert[2+m_number+c_number,t,i_z,lat,long] = (1. - np.nansum(com[2:]))/(1. + ratio)
                            data_convert[2+m_number+c_number+1,t,i_z,lat,long] = data_convert[2+m_number+c_number,t,i_z,lat,long]*ratio
                            data_convert[2+m_number+c_number+2:number-1,t,i_z,lat,long] = com[2:]
                        else :
                            data_convert[2+m_number+c_number,t,i_z,lat,long] = 0.
                            data_convert[2+m_number+c_number+1,t,i_z,lat,long] = 0.
                            data_convert[2+m_number+c_number+2:number-1,t,i_z,lat,long] = com[2:]/(np.nansum(com[2:]))
                        data_convert[2+m_number+c_number+size,t,i_z,lat,long] = np.nansum(data_convert[2+m_number+c_number:2+m_number+c_number+size,t,i_z,lat,long]*M_species)

                        Mass[t,lat,long] += data_convert[0,t,i_z,lat,long]/(R_gp*data_convert[1,t,i_z,lat,long])*\
                                    data_convert[number-1,t,i_z,lat,long]*4/3.*np.pi*((Rp + i_z*delta_z)**3 - (Rp + (i_z - 1)*delta_z)**3)

                    # Si le point d'altitude est plus eleve que le toit du modele a cette lattitude et cette longitude
                    # il nous faut extrapoler

                    if i_z == 0 :

                        data_convert[0,t,i_z,lat,long] = P[t,0,lat,long]
                        data_convert[1,t,i_z,lat,long] = T[t,0,lat,long]
                        if Tracer == True :
                            data_convert[2,t,i_z,lat,long] = Q[t,0,lat,long]
                        if Clouds == True :
                            for c_num in range(c_number) :
                                data_convert[2+m_number+c_num,t,i_z,lat,long] = gen[c_num,t,0,lat,long]
                        data_convert[2+m_number+c_number:number-1,t,i_z,lat,long] = compo[:,t,0,lat,long]
                        data_convert[2+m_number+c_number+size,t,i_z,lat,long] = M[t,0,lat,long]


                    if wh.size == 0 :

                        if Reformate == False :

                            data_convert[1,t,i_z,lat,long] = T[t,n_l-1,lat,long]

                        else :

                            if Upper == "Isotherme" :
                                data_convert[1,t,i_z,lat,long] = T[t,n_l-1,lat,long]
                            if Upper ==  "Isotherme_moyen" :
                                data_convert[1,t,i_z,lat,long] = T_mean
                            if Upper == "Maximum_isotherme" :
                                data_convert[1,t,i_z,lat,long] = T_max
                            if Upper == "Minimum_isotherme" :
                                data_convert[1,t,i_z,lat,long] = T_min

                        # On estime la pression au dela du toit a partir de la temperature choisie

                        if MassAtm == True :
                            g = g0 + Mass[t,lat,long]*G/(Rp + i_z*delta_z)**2
                        else :
                            g = g0

                        if i_z != dim-1 :
                            data_convert[0,t,i_z,lat,long] = data_convert[0,t,i_z-1,lat,long]*np.exp(-data_convert[number-1,t,i_z-1,lat,long]*g*\
                                delta_z/(R_gp*data_convert[1,t,i_z,lat,long])*1./((1+z_ref/Rp)*(1+(z_ref-delta_z)/Rp)))
                        else :
                            data_convert[0,t,i_z,lat,long] = data_convert[0,t,i_z-1,lat,long]*np.exp(-data_convert[number-1,t,i_z-1,lat,long]*g*\
                                delta_z/(2.*R_gp*data_convert[1,t,i_z,lat,long])*1./((1+z_ref/Rp)*(1+(z_ref-delta_z/2.)/Rp)))

                        T_ref = data_convert[1,t,i_z,lat,long]

                        # On incremente toujours la masse atmospherique pour la latitude et la longitude donnee, les
                        # ce point est a modifier

                        if MassAtm == True :
                            Mass[t,lat,long] += data_convert[0,t,i_z-1,lat,long]/(R_gp*data_convert[1,t,i_z-1,lat,long])*\
                                data_convert[number-1,t,i_z-1,lat,long]*4/3.*np.pi*((Rp + i_z*delta_z)**3 - (Rp + (i_z - 1)*delta_z)**3)

                        P_ref = data_convert[0,t,i_z,lat,long]

                        if composition[0] == 'composition' :
                            if Tracer == True :
                                data_convert[2,t,i_z,lat,long] = Q[t,n_l-1,lat,long]
                                Q_ref = data_convert[2,t,i_z,lat,long]

                                if LogInterp == True :
                                    compos, c_grid, i_grid = interp3olation_uni_multi(np.log10(P_ref),T_ref,Q_ref,np.log10(P_comp),T_comp,Q_comp,x_species)
                                else :
                                    compos, c_grid, i_grid = interp3olation_uni_multi(P_ref,T_ref,Q_ref,P_comp,T_comp,Q_comp,x_species)

                                if Clouds == True :
                                    data_convert[3:3+c_number,t,i_z,lat,long] = gen[:,t,n_l-1,lat,long]

                            else :
                                if LogInterp == True :
                                    compos, c_grid, i_grid = interp2olation_uni_multi(np.log10(P_ref),T_ref,np.log10(P_comp),T_comp,x_species)
                                else :
                                    compos, c_grid, i_grid = interp2olation_uni_multi(P_ref,T_ref,P_comp,T_comp,x_species)

                                if Clouds == True :
                                    data_convert[2:2+c_number,t,i_z,lat,long] = gen[:,t,n_l-1,lat,long]

                            if NoH2 == False :
                                compoH2 = (1 - np.nansum(compos[2:]))/(ratio + 1.)
                                compoHe = compoH2*ratio
                                data_convert[2+m_number+c_number,t,i_z,lat,long] = compoH2
                                data_convert[3+m_number+c_number,t,i_z,lat,long] = compoHe
                                data_convert[4+m_number+c_number:number-1,t,i_z,lat,long] = compos[2:]
                            else :
                                data_convert[2+m_number+c_number,t,i_z,lat,long] = 0.
                                data_convert[3+m_number+c_number,t,i_z,lat,long] = 0.
                                data_convert[4+m_number+c_number:number-1,t,i_z,lat,long] = compos[2:]/(np.nansum(compos[2:]))
                            data_convert[number-1,t,i_z,lat,long] = np.nansum(data_convert[2+m_number+c_number:number-1,t,i_z,lat,long]*\
                                        M_species)

                        if composition[0] == 'tracer_other' :
                            data_convert[2:,t,i_z,lat,long] = data_convert[2:,t,i_z-1,lat,long]

        bar.animate(i_z + 1)

    print 'Shape of the dataset :',np.shape(data_convert)

    list = np.array([])

    for i in range(number) :

        wh = np.where(data_convert[i] < 0)

        if len(wh[0]) != 0 :

            list = np.append(list,i)

    if list.size != 0 :

        mess = 'Dataset error, negative value encontered for axis : '

        for i in range(list.size) :

            mess += '%i, '%(list[i])

        mess += 'a correction is necessary, or Boxes failed'

        print mess

    if Inverse[0] == 'True' :
        data_convert = reverse_dim(data_convert,4,np.float64)
        print 'Data needs to be reverse on longitude.'
    if Inverse[1] == 'True' :
        data_convert = reverse_dim(data_convert,3,np.float64)
        print 'Data needs to be reverse on latitude.'
    if Rotate == True :
        data_convert_r = np.zeros(np.shape(data_convert),dtype=np.float64)
        long_rot = np.int(np.float(obs[1])/(2*np.pi)*n_long)
        for i_l in range(n_long) :
            i_l_r = (i_l + long_rot)%(n_long)
            data_convert_r[:,:,:,:,i_l] = data_convert[:,:,:,:,i_l_r]
        data_convert = data_convert_r

    return data_convert, h



########################################################################################################################
########################################################################################################################

"""
    CYLINDRIC_MATRIX_PARAMETER

    Produit la matrice cylindrique de reference a partir de laquelle nous allons construire les tableaux de temperature
    de pression, de fraction molaire, de fraction massique, de concentration molaire et de concentration massique. Cette
    matrice tient desormais compte de la rotation de l'exoplanete, de son inclinaison ou de son obliquite. Seules les 
    valeurs positives de l'obliquite ont ete testees pour l'instant, dans le cas d'une obliquite negative, il suffit d'
    inevrser la matrice sur le chemin optique. 

"""

########################################################################################################################
########################################################################################################################


def cylindric_assymatrix_parameter(Rp,h,long_step,lat_step,r_step,theta_step,theta_number,x_step,z_level,phi_rot,\
                                   inc,phi_obli,reso_long,reso_lat,long_lat,Inclinaison=False,Obliquity=False,Middle=True,Layers=False) :

    # On definit un r maximal qui est la somme du rayon planetaire et du toit de l'atmosphere, on en deduit une valeur
    # entiere et qui est un multiple du pas en r

    if h/np.float(r_step)%r_step != 0 :
        r_reso = int(h/r_step) + 1
    else :
        r_reso = int(h/r_step) + 1 + 1
    lat_ref = np.linspace(-np.pi/2.-lat_step/2.,np.pi/2.+lat_step/2.,reso_lat+2)
    long_ref = np.linspace(-np.pi-long_step/2.,np.pi+long_step/2.,reso_long+2)

    # On calcule la distance maximale que peut parcourir un rayon lumineux rasant comme un entier et un multiple du pas
    # en x

    if Middle == True :
        L_max = 2*np.sqrt((Rp+h)**2 - (Rp+r_step/2.)**2)
    else :
        L_max = 2*np.sqrt((Rp+h)**2 - (Rp)**2)
    if L_max/2.%r_step >= r_step/2. :
        x_reso = 2*int(L_max/(2.*x_step)) + 1 + 1*2 + 2
    else :
        x_reso = 2*int(L_max/(2.*x_step)) + 1 + 1*2

    # q_lat pour la latitude, q_long pour la longitude, z pour l'altitude

    q_lat_grid = np.ones((r_reso ,theta_number , x_reso),dtype='int')*(-1)
    q_long_grid = np.ones((r_reso ,theta_number , x_reso),dtype='int')*(-1)
    z_grid = np.ones((r_reso ,theta_number , x_reso),dtype='int')*(-1)

    bar = ProgressBar(r_reso,'Transposition on cylindric stitch')

    for r_range in range(r_reso) :

        # Si les points de la maille spherique correspondent aux proprietes en milieu de couche, alors il faut tenir
        # compte de la demi-epaisseur de couche dans les calculs de correspondance

        r_layer = r_range*r_step
        if Middle == True :
            r = Rp + r_layer + r_step/2.
        else :
            r = Rp + r_layer

        # r_range est l'indice dans la maille cylindrique sur r

        if Inclinaison == False :
            theta_all = int(theta_number/2.)+1
        else :
            theta_all = theta_number

        for theta_range in range(theta_all) :
            theta = theta_range*theta_step

            for repeat in range(1,3) :

                for x_pos in range(0,int((x_reso-1)/2)) :

                    # x est la distance au centre et peut donc etre negatif comme positif, le 0 etant au terminateur

                    if Inclinaison == False :
                        x = x_pos*x_step*(-1)**(repeat)
                        x_range = int((x_reso-1)/2.) + x_pos*(-1)**(repeat)
                    else :
                        if repeat == 2 :
                            x = x_pos*x_step
                        if repeat == 1 :
                            x = (x_pos)*x_step - int((x_reso-1)/2 -1)*x_step
                        x_range = int((x_reso-1)/2.) + x_pos

                    # rho est la distance au centre de l'exoplanete du point de maille considere
                    rho = np.sqrt(r**2 + x**2)

                    if rho <= Rp + h :

                        # On verifie que le point considere est dans l'atmosphere
                        # alpha est la longitude correspondante

                        long = math.atan2(r*np.cos(theta),x)

                        # Les points de longitude allant de 0 a reso_long, le dernier point etant le meme que le premier, tandis qu'en
                        # angle ils correspondent a -pi a pi (pour le dernier), nous devons renormaliser la longitude

                        if r*np.cos(theta) >= 0 :
                            long = long - np.pi
                        else :
                            long = long + np.pi

                        # lat est la latitude correspondante
                        lat = np.arcsin((r*np.sin(theta))/(rho))

                        if Inclinaison == True :

                            long = long + phi_rot
                            if long > np.pi :
                                long = -np.pi + long%(np.pi)
                            if long < -np.pi :
                                long = long%(np.pi)

                            lat_o = np.arcsin(np.sin(lat)*np.cos(inc)+np.cos(lat)*np.sin(inc)*np.sin(long))
                            long_o = np.arctan2(np.sin(long)*np.cos(lat),np.cos(lat)*np.cos(inc)*np.cos(long)-np.sin(lat)*np.sin(inc))
                            lat = lat_o
                            long = long_o

                        else :
                            long = long + phi_rot
                            if long > np.pi :
                                long = -np.pi + long%(np.pi)
                            if long < -np.pi :
                                long = long%(np.pi)

                        lat_wh, = np.where(lat >= lat_ref)
                        q_lat = lat_wh[lat_wh.size-1]

                        long_wh, = np.where(long >= long_ref)
                        q_long = long_wh[long_wh.size-1]

                        if theta_range == 0 :
                            z_wh = np.where(np.round(z_level,7) == np.round((rho-Rp) - (rho-Rp)%(r_step),7))
                            z = z_wh[0] + 1

                        #print lat, q_lat, x

                        if Inclinaison == False :

                            q_lat_grid[r_range,theta_range,x_range] = q_lat
                            if theta_range != theta_number/4 or theta_range != 3*theta_number/4 :
                                q_long_grid[r_range,theta_range,x_range] = q_long
                            else :
                                q_long_grid[r_range,theta_range,x_range] = q_long_grid[r_range,theta_range,x_range-1]

                            # Conditions de symetrie
                            if theta_range == 0 :
                                z_grid[r_range,theta_range,x_range] = z
                            else :
                                z_grid[r_range,theta_range,x_range] = z_grid[r_range,0,x_range]
                                if theta_range != theta_number/4 or theta_range != 3*theta_number/4 :
                                    q_long_grid[r_range,theta_number - theta_range,x_range] = q_long
                                else :
                                    q_long_grid[r_range,theta_number - theta_range,x_range] = q_long_grid[r_range,theta_range,x_range-1]
                                q_lat_grid[r_range,theta_number - theta_range,x_range] = reso_lat - q_lat
                                z_grid[r_range,theta_number - theta_range,x_range] = z_grid[r_range,0,x_range]

                        else :

                            q_lat_grid[r_range,theta_range,x_range] = q_lat
                            q_long_grid[r_range,theta_range,x_range] = q_long

                            # Conditions de symetrie
                            if theta_range == 0 :
                                z_grid[r_range,theta_range,x_range] = z
                            else :
                                z_grid[r_range,theta_range,x_range] = z_grid[r_range,0,x_range]
                                z_grid[r_range,theta_number - theta_range,x_range] = z_grid[r_range,0,x_range]

        bar.animate(r_range+1)

    return q_lat_grid,q_long_grid,z_grid

########################################################################################################################
########################################################################################################################

"""
    DX_CORRESPONDANCE

    Cette fonction calcul prealablement les distances dx et permet par la suite aux fonctions de transfert de rayonnement
    de retrouver plus rapidement les parametres atmospheriques (P,T,X_mol). On suppose qu'au moins la pression varie
    entre deux altitudes donnees.

"""
########################################################################################################################
########################################################################################################################


def dx_correspondance(data,path,x_step,delta_r,theta_number,Rp,g0,h,t,n_layers,reso_long,reso_lat,reso_alt,obs,Middle=False,Cylindric=True,Integral=True,Gravity=False) :

    if np.str(obs[2]) != 'Modified' :
        lat_obs,long_obs = np.float(obs[0]), np.float(obs[1])
    else :
        lat_obs,long_obs = np.float(obs[0]), np.float(obs[3])

    Z = np.zeros(theta_number)
    Y = np.zeros(theta_number)
    X = np.zeros(theta_number)

    q_lat_grid = np.ones((n_layers+1,theta_number,reso_long+2*reso_lat+2*n_layers+3),dtype=np.int)*(-1)
    q_long_grid = np.ones((n_layers+1,theta_number,reso_long+2*reso_lat+2*n_layers+3),dtype=np.int)*(-1)
    q_z_grid = np.ones((n_layers+1,theta_number,reso_long+2*reso_lat+2*n_layers+3),dtype=np.int)*(-1)
    q_zh_grid = np.ones((n_layers+1,theta_number,reso_long+2*reso_lat+2*n_layers+3),dtype=np.int)*(-1)
    dx_grid_opt = np.ones((n_layers+1,theta_number,reso_long+2*reso_lat+2*n_layers+3),dtype=np.float64)*(-1)

    bar = ProgressBar(n_layers,'Transposition on the cylindric stitch : ')
    size_max = 0

    if Cylindric == True :
        for i_r in range(n_layers) :

            if Middle == True :
                r = Rp + (i_r+0.5)*delta_r
            else :
                r = Rp + (i_r)*delta_r
            L = np.sqrt((Rp+h)**2 - r**2)

            for i_theta in range(theta_number) :

                theta = i_theta*np.pi/np.float(reso_lat)

                Z[i_theta] = r*np.cos(theta)*np.cos(lat_obs)

                A = 1.
                B = 2*r*np.cos(theta)*np.sin(lat_obs)*np.sin(long_obs)
                C = r**2*(np.cos(theta)**2*(np.sin(lat_obs)**2 + np.cos(lat_obs)**2*np.cos(long_obs)**2) - np.cos(long_obs)**2)

                delta = B**2. - 4.*A*C
                if delta < 0. :
                    delta = 0

                if long_obs > -np.pi/2. and long_obs < np.pi/2. :
                    if theta >= 0. and theta <= np.pi :
                        Y[i_theta] = (-B - np.sqrt(delta))/(2.*A)
                    else :
                        Y[i_theta] = (-B + np.sqrt(delta))/(2.*A)
                else :
                    if theta >= 0. and theta <= np.pi :
                        Y[i_theta] = (-B + np.sqrt(delta))/(2.*A)
                    else :
                        Y[i_theta] = (-B - np.sqrt(delta))/(2.*A)

                X[i_theta] = -(r*np.sin(lat_obs)*np.cos(theta)/(np.cos(long_obs)) + Y[i_theta]*np.tan(long_obs))

                # Calcul des points de reference de l'entree dans l'atmosphere
                rho_ref = Rp + h

                lat_ref_init = np.arcsin((Z[i_theta] - L*np.sin(lat_obs))/(rho_ref))
                lat_ref_end = np.arcsin((Z[i_theta] + L*np.sin(lat_obs))/(rho_ref))
                lat_lim_init = np.pi - (Z[i_theta]/(np.cos(lat_obs)))/(rho_ref)
                lat_lim_end = (Z[i_theta]/(np.cos(lat_obs)))/(rho_ref)
                q_lat_ref_init = np.int(np.round((lat_ref_init)/(np.pi/np.float(reso_lat)))) + reso_lat/2
                q_lat_ref_end = np.int(np.round((lat_ref_end)/(np.pi/np.float(reso_lat)))) + reso_lat/2

                long_ref_end = np.arctan2(Y[i_theta] + L*np.cos(lat_obs)*np.sin(long_obs),X[i_theta] + L*np.cos(lat_obs)*np.cos(long_obs))
                long_ref_init = np.arctan2(Y[i_theta] - L*np.cos(lat_obs)*np.sin(long_obs),X[i_theta] - L*np.cos(lat_obs)*np.cos(long_obs))
                q_long_ref_init = np.int(np.round((long_ref_init)/(2*np.pi/np.float(reso_long))))
                q_long_ref_end = np.int(np.round((long_ref_end)/(2*np.pi/np.float(reso_long))))
                if q_long_ref_init < 0 :
                    q_long_ref_init += reso_long
                if q_long_ref_end < 0 :
                    q_long_ref_end += reso_long

                if long_ref_end >= np.pi/2. or long_ref_end <= -np.pi/2. :
                    q_lat_ref_end = 2*reso_lat - q_lat_ref_end - 1
                if long_ref_init >= np.pi/2. or long_ref_init <= -np.pi/2. :
                    q_lat_ref_init = 2*reso_lat - q_lat_ref_init - 1

                ############################### Resolution des positions des niveaux ###########################################

                d_z = np.zeros(n_layers+1)
                n_l = np.ones(n_layers+1,dtype=np.int)*(-1)
                for i_l in range(i_r+1,n_layers+1) :
                    d_z[i_l] = np.sqrt((Rp + i_l*delta_r)**2 - r**2)
                    n_l[i_l] = i_l

                wh, = np.where(d_z != 0)
                d_z = d_z[wh]
                n_l = n_l[wh]

                ############################### Resolution des positions de la latitude ########################################

                d_lat = np.ones(reso_lat*2)*(-1)
                n_lat = np.ones(reso_lat*2,dtype=np.int)*(-1)
                mod = 0

                if np.sin(lat_ref_init)*np.sin(lat_ref_end) >= 0. :
                    if np.abs(q_lat_ref_end-q_lat_ref_init) < reso_lat :
                        q_init = np.amin(np.array([q_lat_ref_init,q_lat_ref_end])) - 1
                        q_end = np.amax(np.array([q_lat_ref_init,q_lat_ref_end])) + 1
                        q_range = np.arange(q_init,q_end+1,1,dtype=np.int)
                    if np.abs(q_lat_ref_end-q_lat_ref_init) > reso_lat :
                        q_init = np.amin(np.array([q_lat_ref_init,q_lat_ref_end])) + 1
                        q_end = np.amax(np.array([q_lat_ref_init,q_lat_ref_end])) - 1
                        q_range = np.append(np.arange(q_end,2*reso_lat,1,dtype=np.int), np.arange(0,q_init+1,1,dtype=np.int))
                else :
                    if q_lat_ref_init == reso_lat/2 or q_lat_ref_end == reso_lat/2 :
                        if np.abs(q_lat_ref_end-q_lat_ref_init) < reso_lat :
                            q_init = np.amin(np.array([q_lat_ref_init,q_lat_ref_end])) - 1
                            q_end = np.amax(np.array([q_lat_ref_init,q_lat_ref_end])) + 1
                            q_range = np.arange(q_init,q_end+1,1,dtype=np.int)
                        if np.abs(q_lat_ref_end-q_lat_ref_init) > reso_lat :
                            q_init = np.amin(np.array([q_lat_ref_init,q_lat_ref_end])) + 1
                            q_end = np.amax(np.array([q_lat_ref_init,q_lat_ref_end])) - 1
                            q_range = np.append(np.arange(q_end,2*reso_lat,1,dtype=np.int), np.arange(0,q_init+1,1,dtype=np.int))
                    else :
                        q_init = np.amin(np.array([q_lat_ref_init,q_lat_ref_end]))
                        q_init_mir = 2*reso_lat - q_init - 1
                        q_end = np.amax(np.array([q_lat_ref_init,q_lat_ref_end]))
                        q_end_mir = 2*reso_lat - q_end - 1
                        q_range = np.append(np.arange(np.amin(np.array([q_end_mir,q_init])),np.amax(np.array([q_end_mir,q_init]))+1,1,dtype=np.int),\
                                            np.arange(np.amin(np.array([q_init_mir,q_end])),np.amax(np.array([q_init_mir,q_end]))+1,1,dtype=np.int))
                        mod = 1

                for i_la in q_range :
                    lat_o = -np.pi/2. + (i_la + 0.5)*np.pi/np.float(reso_lat)
                    if theta == np.pi/2. or theta == 3*np.pi/2. :
                        if lat_obs == 0 :
                            if i_la == 0 :
                                d_lat = np.array([])
                                n_lat = np.array([])
                        else:
                            A_phi = np.sin(lat_o)**2 - np.sin(lat_obs)**2
                            B_phi = -2*Z[i_theta]*np.sin(lat_obs)
                            C_phi = r**2*np.sin(lat_o)**2 - Z[i_theta]**2
                            Delta = B_phi**2 - 4*A_phi*C_phi
                            if lat_o > np.pi/2. :
                                d_lat[i_la] = np.amin(np.array([(-B_phi - np.sqrt(Delta))/(2.*A_phi),(-B_phi + np.sqrt(Delta))/(2.*A_phi)]))
                                if mod == 1 :
                                    if lat_o < np.pi :
                                        d_lat[i_la] = np.amax(np.array([(-B_phi - np.sqrt(Delta))/(2.*A_phi),(-B_phi + np.sqrt(Delta))/(2.*A_phi)]))
                                    if lat_o < lat_lim_init :
                                        d_lat[i_la] = 'nan'
                            else :
                                d_lat[i_la] = np.amax(np.array([(-B_phi - np.sqrt(Delta))/(2.*A_phi),(-B_phi + np.sqrt(Delta))/(2.*A_phi)]))
                                if mod == 1 :
                                    if lat_o < 0. :
                                        d_lat[i_la] = np.amin(np.array([(-B_phi - np.sqrt(Delta))/(2.*A_phi),(-B_phi + np.sqrt(Delta))/(2.*A_phi)]))
                                    if lat_o < lat_lim_end :
                                        d_lat[i_la] = 'nan'

                            #print reso_lat - lat_obs*reso_lat/np.pi

                    else :
                        A_phi = np.sin(lat_o)**2 - np.sin(lat_obs)**2
                        B_phi = -2*Z[i_theta]*np.sin(lat_obs)
                        C_phi = r**2*np.sin(lat_o)**2 - Z[i_theta]**2
                        Delta = B_phi**2 - 4*A_phi*C_phi
                        if lat_o > np.pi/2. :
                            d_lat[i_la] = np.amin(np.array([(-B_phi - np.sqrt(Delta))/(2.*A_phi),(-B_phi + np.sqrt(Delta))/(2.*A_phi)]))
                            if mod == 1 :
                                if lat_o < np.pi :
                                    d_lat[i_la] = np.amax(np.array([(-B_phi - np.sqrt(Delta))/(2.*A_phi),(-B_phi + np.sqrt(Delta))/(2.*A_phi)]))
                                if lat_o < lat_lim_init :
                                    d_lat[i_la] = 'nan'
                        else :
                            d_lat[i_la] = np.amax(np.array([(-B_phi - np.sqrt(Delta))/(2.*A_phi),(-B_phi + np.sqrt(Delta))/(2.*A_phi)]))
                            if mod == 1 :
                                if lat_o < 0. :
                                    d_lat[i_la] = np.amin(np.array([(-B_phi - np.sqrt(Delta))/(2.*A_phi),(-B_phi + np.sqrt(Delta))/(2.*A_phi)]))
                                if lat_o < lat_lim_end :
                                    d_lat[i_la] = 'nan'

                        #print reso_lat - lat_obs*reso_lat/np.pi
                    if d_lat[i_la] > L or d_lat[i_la] < -L :
                        d_lat[i_la] = 'nan'
                    if np.str(d_lat[i_la]) == 'nan' :
                        n_lat[i_la] = -1
                    else :
                        n_lat[i_la] = i_la

                wh, = np.where(n_lat != -1)
                n_lat = n_lat[wh]
                d_lat = d_lat[wh]

                qq = np.argsort(d_lat)
                d_lat = np.sort(d_lat)
                n_lat = n_lat[qq]

                wh_la, = np.where(n_lat >= reso_lat)
                n_lat[wh_la] = 2*reso_lat - 1 - n_lat[wh_la]

                ############################### Resolution des positions de la latitude ########################################

                d_long = np.ones(reso_long)*(-1)
                n_long = np.ones(reso_long,dtype=np.int)*(-1)
                q_init = np.amin(np.array([q_long_ref_init,q_long_ref_end]))
                q_end = np.amax(np.array([q_long_ref_init,q_long_ref_end]))
                for i_lo in range(q_init,q_end+1) :
                    long_o = (i_lo + 0.5)*2*np.pi/np.float(reso_long)

                    if theta == 0. or theta == np.pi :
                        if i_lo == 0 :
                            d_long[i_lo] = -L
                            d_long[i_lo+1] = 0.
                            d_long[i_lo+2] = L

                            n_long[i_lo] = q_long_ref_init
                            n_long[i_lo+1] = q_long_ref_end
                            n_long[i_lo+2] = q_long_ref_end
                    else :
                        d_long[i_lo] = (Y[i_theta] - X[i_theta]*np.tan(long_o))/(np.cos(lat_obs)*(np.cos(long_obs)*np.tan(long_o)-np.sin(long_obs)))

                        if d_long[i_lo] > L or d_long[i_lo] < -L :
                            d_long[i_lo] = 'nan'
                        if np.str(d_long[i_lo]) == 'nan' :
                            n_long[i_lo] = -1
                        else :
                            n_long[i_lo] = i_lo

                wh, = np.where(n_long != -1)

                if wh.size != 0 :
                    d_long = d_long[wh]
                    n_long = n_long[wh]
                else :
                    d_long = np.array([])
                    n_long = np.array([])

                ####################################### Assignation des coordonnees ############################################

                d = np.append(-d_z,np.append(d_z,np.append(d_long,d_lat)))
                n = np.append(n_l,np.append(n_l+1,np.append(10000+n_long,1000+n_lat)))

                q_ind = np.argsort(d)
                d = np.sort(d)
                n = n[q_ind]

                if q_lat_ref_init >= reso_lat :
                    q_lat_ref_init = 2*reso_lat - 1 - q_lat_ref_init
                if q_lat_ref_end >= reso_lat :
                    q_lat_ref_end = 2*reso_lat - 1 - q_lat_ref_end

                q_z = np.ones(d.size,dtype=np.int)*(-1)
                q_zh = np.ones(d.size,dtype=np.int)*(-1)
                dx_opt = np.ones(d.size,dtype=np.float64)*(-1)
                q_lat = np.ones(d.size,dtype=np.int)*(-1)
                q_long = np.ones(d.size,dtype=np.int)*(-1)

                q_z[0] = n_layers
                q_long[0] = q_long_ref_init
                q_lat[0] = q_lat_ref_init
                q_zh[0] = n_layers*delta_r

                for i_d in range(1,d.size) :
                    if n[i_d] < 1000:
                        if d[i_d] != L:
                            q_z[i_d] = n[i_d]
                            q_zh[i_d] = n[i_d] * delta_r
                        else:
                            q_z[i_d] = n_layers
                            q_zh[i_d] = n_layers * delta_r

                        lat_step = np.arcsin((Z[i_theta]+(d[i_d]-10)*np.sin(lat_obs))/(np.sqrt(r**2+(d[i_d]-10)**2)))
                        q_lat[i_d] = np.int(np.round((lat_step+np.pi/2.)/(np.pi)*reso_lat))

                        long_step = np.arctan2(Y[i_theta]+(d[i_d]-10)*np.cos(lat_obs)*np.sin(long_obs),X[i_theta]+(d[i_d]-10)*np.cos(lat_obs)*np.cos(long_obs))
                        q_long[i_d] = np.int(np.round((long_step)/(2*np.pi)*reso_long))
                        if q_long[i_d] < 0 :
                            q_long[i_d] += reso_long

                    else :
                        lat_step = np.arcsin((Z[i_theta]+(d[i_d]-10)*np.sin(lat_obs))/(np.sqrt(r**2+(d[i_d]-10)**2)))
                        q_lat[i_d] = np.int(np.round((lat_step+np.pi/2.)/(np.pi)*reso_lat))

                        long_step = np.arctan2(Y[i_theta]+(d[i_d]-10)*np.cos(lat_obs)*np.sin(long_obs),X[i_theta]+(d[i_d]-10)*np.cos(lat_obs)*np.cos(long_obs))
                        q_long[i_d] = np.int(np.round((long_step)/(2*np.pi)*reso_long))
                        if q_long[i_d] < 0 :
                            q_long[i_d] += reso_long

                        q_zh[i_d] = (Z[i_theta]+d[i_d]*np.sin(lat_obs))/(np.sin(lat_step)) - Rp

                        q_z[i_d] = q_z[i_d - 1]

                    if q_long[i_d] == reso_long :
                        q_long[i_d] = 0

                    dx_opt[i_d] = np.abs(d[i_d]-d[i_d - 1])


                wh, = np.where((dx_opt < 1.e-6)*(dx_opt != -1.))

                if wh.size != 0 :
                    q_lat = np.delete(q_lat,wh)
                    q_z = np.delete(q_z,wh)
                    q_zh = np.delete(q_zh,wh)
                    q_long = np.delete(q_long,wh)
                    dx_opt = np.delete(dx_opt,wh)

                wh = np.array([])
                for i_d in range(q_lat.size-1) :
                    if q_lat[i_d] == q_lat[i_d+1] and q_long[i_d] == q_long[i_d+1] and q_z[i_d] == q_z[i_d+1] :
                        wh = np.append(wh,np.array([i_d+1]))
                        dx_opt[i_d] = dx_opt[i_d] + dx_opt[i_d+1]

                if wh.size != 0 :
                    q_lat = np.delete(q_lat,wh)
                    q_z = np.delete(q_z,wh)
                    q_zh = np.delete(q_zh,wh)
                    q_long = np.delete(q_long,wh)
                    dx_opt = np.delete(dx_opt,wh)

                size = dx_opt.size

                q_lat_grid[i_r,i_theta,0:size] = q_lat
                q_z_grid[i_r,i_theta,0:size] = q_z
                q_zh_grid[i_r,i_theta,0:size] = q_zh
                dx_grid_opt[i_r,i_theta,0:size] = dx_opt
                q_long_grid[i_r,i_theta,0:size] = q_long

                if size_max < size :
                    size_max = size

            bar.animate(i_r)

        q_lat_grid = q_lat_grid[:,:,:size_max]
        q_long_grid = q_long_grid[:,:,:size_max]
        q_z_grid = q_z_grid[:,:,:size_max]
        q_zh_grid = q_zh_grid[:,:,:size_max]
        dx_grid_opt = dx_grid_opt[:,:,:size_max]

        order_grid = np.ones((3,n_layers+1,theta_number,size_max),dtype=np.int)*(-1)

        order_grid[0] = q_z_grid
        order_grid[1] = q_lat_grid
        order_grid[2] = q_long_grid

        np.save("%sq_lat_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,theta_number,reso_long,reso_lat,\
                    reso_alt,delta_r,np.float(obs[0]),np.float(obs[1])),q_lat_grid)
        np.save("%sq_long_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,theta_number,reso_long,reso_lat,\
                    reso_alt,delta_r,np.float(obs[0]),np.float(obs[1])),q_long_grid)
        np.save("%sq_z_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,theta_number,reso_long,reso_lat,\
                    reso_alt,delta_r,np.float(obs[0]),np.float(obs[1])),q_z_grid)
        np.save("%sq_zh_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,theta_number,reso_long,reso_lat,\
                    reso_alt,delta_r,np.float(obs[0]),np.float(obs[1])),q_z_grid)

        del q_lat_grid, q_long_grid, q_z_grid

    else :

        q_zh_grid = np.load('%sq_zh_grid_%i_%i%i%i_%i_%.2f_%.2f.npy'%(path,theta_number,reso_long,reso_lat,\
                reso_alt,delta_r,np.float(obs[0]),np.float(obs[1])))
        order_grid = np.load('%sorder_grid_%i_%i%i%i_%i_%.2f_%.2f.npy'%(path,theta_number,reso_long,reso_lat,\
                reso_alt,delta_r,np.float(obs[0]),np.float(obs[1])))

    ###################################### Calcul integral des sous-parcours ###########################################

    pdx_grid = np.ones((n_layers+1,theta_number,size_max),dtype=np.float64)*(-1)
    size = order_grid[0,0,0,:].size
    if Integral == True :
        data = np.load(data)
        sh = np.shape(data)
        number = sh[0]

        bar = ProgressBar(n_layers*theta_number,'Integration of pathes progression : ')

        for i_r in range(n_layers) :
            if Middle == True :
                r = (i_r+0.5)*delta_r
            else :
                r = (i_r)*delta_r

            for i_theta in range(theta_number) :
                for i_d in range(size) :

                    z_1 = q_zh_grid[i_r,i_theta,i_d]
                    if i_d != size-1 :
                        z_2 = q_zh_grid[i_r,i_theta,i_d+1]
                    else :
                        z_2 = h
                    M_1 = data[number-1,t,order_grid[0,i_r,i_theta,i_d],order_grid[1,i_r,i_theta,i_d],order_grid[2,i_r,i_theta,i_d]]
                    T_1 = data[1,t,order_grid[0,i_r,i_theta,i_d],order_grid[1,i_r,i_theta,i_d],order_grid[2,i_r,i_theta,i_d]]

                    if Gravity == False :
                        g_1 = g0/(1+z_1/Rp)**2
                        g_0 = g0/((1+(order_grid[0,i_r,i_theta,i_d]-0.5)*delta_r/Rp)*(1+z_1/Rp))
                        P_1 = data[0,t,order_grid[0,i_r,i_theta,i_d],order_grid[1,i_r,i_theta,i_d],order_grid[2,i_r,i_theta,i_d]]\
                              *np.exp(-M_1*g_0/(R_gp*T_1)*(z_1-(order_grid[0,i_r,i_theta,i_d]-0.5)*delta_r))

                        integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*\
                              (Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)
                    else :
                        g_1 = g0
                        P_1 = data[0,t,order_grid[0,i_r,i_theta,i_d],order_grid[1,i_r,i_theta,i_d],order_grid[2,i_r,i_theta,i_d]]\
                              *np.exp(-M_1*g_1/(R_gp*T_1)*(z_1-(order_grid[0,i_r,i_theta,i_d]-0.5)*delta_r))

                        integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*\
                              (Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)

                    if np.str(integ[0]) == 'inf' :
                        pdx_grid[i_r,i_theta,i_d] = P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                        print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i_r,i_theta,i_d,pdx_grid[i_r,i_theta,i_d])), 'initial result', integ[0]
                    else :
                        pdx_grid[i_r,i_theta,i_d] = integ[0]

                bar.animate(i_r*theta_number+i_theta+1)
    else :
        pdx_grid = np.array([])

    return dx_grid_opt, pdx_grid, order_grid


