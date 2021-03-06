from pyfunction import *
import pickle

########################################################################################################################
########################################################################################################################

"""
    PYDATAREAD

    Les routines suivantes assurent la conversion des fichiers de donnees de format texte en tableaux numpy afin de
    pouvoir exploiter pleinement le potentiel de traitement par cette bibliotheque python. Les fichiers de tableaux
    ainsi convertis sont enregistres dans un repertoire qui prendra le nom de l'exoplanete etudiee precede de l'
    attribut Source (exemple : Pytmosph3R/Source_Earth/).

    Version : 6.3

    Date de derniere modification : 28.06.2016

    Date de derniere modification : 12.12.2016

    >> Amelioration de l'ensemble des fonctions de lecture des donnees gcm, hitran et hitemp ainsi que creation d'un
    module complet d'enregistrement de ces donnees en tableaux numpy. A present, l'execution du module prepare
    le fichier Source consciencieusement de maniere a ce qu'il ne manque aucun fichier de base lors de l'execution
    de Pytmosph3R. Si la fonction est plus intelligente, certaines lectures necessitent tout de meme de preciser "a la
    main" le format adopte par l'auteur.

    Date de derniere modification : 29.10.2017

    >> Correction d'un bug d'enregistrement pour les P_sample ... ils etaient enregistre en Pa au lieu d'en puissance de
    10 et en mBar.

    Date de derniere modification : 01.01.2018

    >> Possibilite desormais de verifier l'existence des fichiers avant de lancer la recuperation des sonnees sources.
    >> Correction de tous les bugs de recuperation : aerosol qui ne recuperait pas le double tableau de longueur d'onde
    pour les differents domaines spectraux, les longueurs d'onde pour H2O qui n'etaient pas non plus collectes, les
    k-distributions lue a l'envers.

    Date de derniere modification : 11.04.2018

"""


########################################################################################################################
########################################################################################################################


def data_record(path,name_source,data_base,name_exo,aerosol,continuum,kcorr,crossection,composition,Renorm=False) :

    directory = '%s%s/'%(path,name_source)
    i_dec = 0
    if aerosol.number != 0 :
        for i_c in range(aerosol.number) :
            if aerosol.continuity[i_c] == True :
                if os.path.isfile('%sQsingle_%s_%s.npy'%(directory,aerosol.nspecies[i_c],name_exo)) == False :
                    for i_continuity in range(2) :
                        aerosol_file = '%saerosol_properties/optprop_%s.dat'%(data_base,aerosol.file_name[i_c+i_continuity+i_dec])
                        aerosol_data_read(aerosol_file,0,13,aerosol.file_name[i_c+i_continuity+i_dec],name_exo,directory,Save=True)
                        if i_continuity == 0 :
                            bande_cloud = np.load('%sbande_cloud_%s.npy'%(directory,name_exo))
                        else :
                            bande_continuity = np.load('%sbande_cloud_%s.npy'%(directory,name_exo))
                            bande_cloud = np.append(bande_cloud,bande_continuity)
                    Q_1 = np.load('%sQ_%s_%s.npy'%(directory,aerosol.file_name[i_c+i_dec],name_exo))
                    Q_2 = np.load('%sQ_%s_%s.npy'%(directory,aerosol.file_name[i_c+i_dec+1],name_exo))
                    sh_1 = np.shape(Q_1)
                    sh_2 = np.shape(Q_2)
                    Q_fin = np.zeros((sh_1[0],sh_1[1]+sh_2[1]))
                    Q_fin[:,0:sh_1[1]] = Q_1
                    Q_fin[:,sh_1[1]:sh_1[1]+sh_2[1]] = Q_2
                    np.save('%sQsingle_%s_%s.npy'%(directory,aerosol.nspecies[i_c],name_exo),Q_fin)
                    np.save('%sbande_cloud_%s.npy'%(directory,name_exo),bande_cloud)
                else :
                    print 'Aerosol data already recorded : %s'%(aerosol.nspecies[i_c])
                i_dec += 1
            else :
                if os.path.isfile('%sQsingle_%s_%s.npy'%(directory,aerosol.nspecies[i_c],name_exo)) == False :
                    aerosol_file = '%saerosol_properties/optprop_%s.dat'%(data_base,aerosol.file_name[i_dec+i_c])
                    aerosol_data_read(aerosol_file,0,13,aerosol.nspecies[i_c],name_exo,directory,Save=True)
                else :
                    print 'Aerosol data already recorded : %s'%(aerosol.nspecies[i_c])
        all_species = ''
        for i_c in range(aerosol.number):
            all_species += '%s_'%(aerosol.nspecies[i_c])
        Q_f = np.load('%sQsingle_%s_%s.npy'%(directory,aerosol.nspecies[0],name_exo))
        sh = np.shape(Q_f)
        Q_ext = np.zeros((aerosol.number,sh[0],sh[1]))
        for i_c in range(aerosol.number):
            print np.shape(np.load('%sQsingle_%s_%s.npy'%(directory,aerosol.nspecies[i_c],name_exo)))
            Q_ext[i_c] = np.load('%sQsingle_%s_%s.npy'%(directory,aerosol.nspecies[i_c],name_exo))
        np.save('%sQ_%s%s.npy'%(directory,all_species,name_exo),Q_ext)

    k_cont_number = continuum.number
    if k_cont_number != 0 :
        for i_n in range(k_cont_number) :
            k_cont_file = '%scontinuum_data/%s'%(data_base,continuum.file_name[i_n])
            species = continuum.species[i_n]
            if os.path.isfile('%sk_cont_%s.npy'%(directory,continuum.associations[i_n])) == False :
                if species != 'H2O' and species != 'H2Os' :
                    k_cont_data_read(k_cont_file,continuum.associations[i_n],directory)
                else :
                    k_cont_h2o_read(k_cont_file,continuum.associations[i_n],directory)
                    data = open('%scontinuum_data/H2O_CONT_NU.dat'%(data_base))
                    nu = data.readlines()
                    dim_b = np.shape(nu)[0]
                    k_cont_nu = np.zeros(dim_b)
                    for i_ban in range(dim_b) :
                        k_cont_nu[i_ban] = np.float(nu[i_ban])
                    np.save('%s%s/k_cont_nu_h2oh2o.npy'%(path,name_source),k_cont_nu)
                    np.save('%s%s/k_cont_nu_h2ofor.npy'%(path,name_source),k_cont_nu)
            else :
                print 'Continuum data already recorded : association %s'%(continuum.associations[i_n])

    if composition.file != '' :
        if os.path.isfile('%sT_comp_%s.npy'%(directory,name_exo)) == False :
            composition_data_read(composition,directory,name_exo,Renorm)
        else :
            'Composition data already recorded.'

    if crossection.file != '' :
        if os.path.isfile("%scrossection_%s.npy"%(directory,crossection.type)) == False :
            cross_data_read(crossection.file,crossection.type_ref,crossection.species,directory,crossection.type)
        else :
           print 'Cross-section already recorded'


########################################################################################################################
########################################################################################################################


def k_corr_data_read(kcorr,path,name_exo,parameters,domain,dim_bande,dim_gauss,exception,directory,All=True,Jump=False,Para=False) :

    data = open('%scorrk_data/%s.dat'%(path,parameters[0]),'r')
    T_read = data.readlines()
    T_dim = line_search(T_read[0])
    data = open('%scorrk_data/%s.dat'%(path,parameters[1]),'r')
    p_read = data.readlines()
    P_dim = line_search(p_read[0])
    if kcorr.parameters.size > 2 :
        data = open('%scorrk_data/%s.dat'%(path,parameters[2]),'r')
        Q_read = data.readlines()
        dec = line_search(Q_read[0])
        Q_dim = line_search(Q_read[int(dec[0])+1])
        Q_dim = int(Q_dim[0])
        Q_sample = np.zeros(Q_dim)
        for i_Q in range(Q_dim) :
            Q_sample[i_Q] = np.float(line_search(Q_read[int(dec[0])+2+i_Q])[0])
        np.save('%sQ_comp_%s.npy'%(directory,name_exo),Q_sample)
        np.save('%sQ_sample.npy'%(directory),Q_sample)
    else :
        Q_dim = 1
    P_dim, T_dim = np.int(P_dim[0]), np.int(T_dim[0])
    T_sample,P_sample = np.zeros(T_dim), np.zeros(P_dim)
    for i_T in range(T_dim) :
        T_sample[i_T] = np.float(line_search(T_read[1+i_T])[0])
    for i_P in range(P_dim) :
        P_sample[i_P] = np.float(line_search(p_read[1+i_P])[0])
    np.save('%sT_comp_%s.npy'%(directory,name_exo),T_sample)
    np.save('%sP_comp_%s.npy'%(directory,name_exo),P_sample)
    np.save('%sT_sample.npy'%(directory),T_sample)
    np.save('%sP_sample.npy'%(directory),P_sample)
    data = open('%scorrk_data/g.dat'%(path),'r')
    g_read = data.readlines()
    g_dim = np.int(line_search(g_read[0])[0])-1
    g_sample = np.zeros(g_dim)
    for i_g in range(g_dim) :
        g_sample[i_g] = np.float(line_search(g_read[1+i_g])[0])
    np.save('%sgauss_sample.npy'%(directory),g_sample)

    data = open('%scorrk_data/%s/narrowbands_%s.in'%(path,kcorr.resolution,domain),'r')
    bande_read = data.readlines()
    bande_dim = np.int(line_search(bande_read[0])[0])
    bande_sample = np.zeros(bande_dim+1)
    for i_ba in range(bande_dim) :
        bande_sample[i_ba] = np.float(line_search(bande_read[1+i_g])[0])
        if i_ba == bande_dim -1 :
            bande_sample[i_ba+1] = np.float(line_search(bande_read[1+i_g])[1])
    np.save('%sbande_sample_%s.npy'%(directory,domain),bande_sample)

    data = open('%scorrk_data/%s/corrk_gcm_%s.dat'%(path,kcorr.resolution,domain),'r')
    k_corr_data = data.readlines()

    ex_bande = np.array([],dtype='int')
    ex_gauss = np.array([],dtype='int')

    message = "We didn't take into account"

    if All == False :
        duo,ex_dim = np.shape(exception)
        for ex in range(ex_dim) :
            if exception[0,ex] == 'bande' :
                ex_bande = np.append(ex_bande,int(exception[1,ex]))
                message += " bande n%i,"%(int(exception[1,ex]))
            else :
                ex_gauss = np.append(ex_gauss ,int(exception[1,ex]))
                message += " gauss point n%i,"%(int(exception[1,ex]))
        message += " for this %s %ix%i resolution."%(domain,dim_bande,dim_gauss)
        if ex_dim != 0 :
            print(message)
    else :
        print('There is no exception')

    k_corr_plan = np.zeros((T_dim,P_dim,Q_dim,dim_bande,dim_gauss))
    if Jump == False :
        if Para == True :
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.rank
            number_rank = comm.size

            k_corr_data = k_corr_data[0]

            size = len(k_corr_data)
            i_dd = np.zeros(number_rank+1,dtype=np.int)
            for r_n in range(0,number_rank) :
                i_d = r_n*size/number_rank
                if k_corr_data[i_d] != ' ' :
                    supp = 1
                    while k_corr_data[i_d+supp] != ' ' :
                        supp += 1
                    i_d += supp
                i_dd[r_n] = i_d

            i_dd[i_dd.size-1] = size-1

            k_corr_nojump_n = line_search(k_corr_data[i_dd[rank]:i_dd[rank+1]+1])

            for i_k in range(k_corr_nojump_n.size) :
                if rank != number_rank -1 and i_k != k_corr_nojump_n.size-1 :
                    k_corr_nojump_n[i_k] = np.float(k_corr_nojump_n[i_k])
                else :
                    k_corr_nojump_n[i_k] = 0.0

            for r_n in range(number_rank) :
                if r_n != 0  and r_n == rank :
                    k_corr_nojump_n = np.array([k_corr_nojump_n],dtype=np.float64)
                    length = np.array([k_corr_nojump_n.size],dtype=np.int)
                    comm.Send([length,MPI.INT],dest=0,tag=1)
                    comm.Send([k_corr_nojump_n,MPI.DOUBLE],dest=0,tag=2)

                elif r_n == 0 and rank == 0 :
                    k_corr_nojump = k_corr_nojump_n
                elif r_n != 0 and rank == 0 :
                    length_n = np.zeros(1,dtype=np.int)
                    comm.Recv([length_n,MPI.INT],source=r_n,tag=1)
                    k_corr_nojump_ne = np.zeros(length_n,dtype=np.float64)
                    comm.Recv([k_corr_nojump_ne,MPI.DOUBLE],source=r_n,tag=2)
                    k_corr_nojump = np.append(k_corr_nojump,k_corr_nojump_ne)
        else :
            k_corr_nojump = line_search(k_corr_data[0])

    if Para == True :
        if rank == 0 :
            bar = ProgressBar(dim_gauss*dim_bande,'Kcorr record')

            for m in range(dim_gauss) :
                whg = ex_gauss[ex_gauss == m]
                m_coeff = m*T_dim*P_dim*Q_dim*dim_bande
                for l in range(dim_bande) :
                    wh = ex_bande[ex_bande == l]
                    l_coeff = l*T_dim*P_dim*Q_dim
                    if whg.size == 0 and wh.size == 0 :
                        for k in range(Q_dim) :
                            k_coeff = k*T_dim*P_dim
                            for j in range(P_dim) :
                                j_coeff = j*T_dim
                                for i in range(T_dim) :
                                    if Jump == False :
                                        i_data = i + j_coeff + k_coeff + l_coeff + m_coeff
                                        k_corr_plan[i,j,k,l,m] = np.float(k_corr_nojump[i_data])
                                    else :
                                        i_data = i + j_coeff + k_coeff + l_coeff + m_coeff
                                        i_line = i_data/3
                                        i_col = i_data%3
                                        k_corr_line = line_search(k_corr_data[i_line])
                                        k_corr_plan[i,j,k,l,m] = np.float(k_corr_line[i_col])

                    bar.animate(m*dim_bande+l+1)
    else :
        bar = ProgressBar(dim_gauss*dim_bande,'Kcorr record')

        for m in range(dim_gauss) :
            whg = ex_gauss[ex_gauss == m]
            m_coeff = m*T_dim*P_dim*Q_dim*dim_bande
            for l in range(dim_bande) :
                wh = ex_bande[ex_bande == l]
                l_coeff = l*T_dim*P_dim*Q_dim
                if whg.size == 0 and wh.size == 0 :
                    for k in range(Q_dim) :
                        k_coeff = k*T_dim*P_dim
                        for j in range(P_dim) :
                            j_coeff = j*T_dim
                            for i in range(T_dim) :
                                if Jump == False :
                                    i_data = i + j_coeff + k_coeff + l_coeff + m_coeff
                                    k_corr_plan[i,j,k,l,m] = np.float(k_corr_nojump[i_data])
                                else :
                                    i_data = i + j_coeff + k_coeff + l_coeff + m_coeff
                                    i_line = i_data/3
                                    i_col = i_data%3
                                    k_corr_line = line_search(k_corr_data[i_line])
                                    k_corr_plan[i,j,k,l,m] = np.float(k_corr_line[i_col])

                bar.animate(m*dim_bande+l+1)

    np.save("%sk_corr_%s_%s.npy"%(directory,name_exo,domain),k_corr_plan)


########################################################################################################################


def k_corr_data_write(k_corr_data,k_correlated_file,dec_first,dec_final,P_dim,T_dim,Q_dim,dim_bande,dim_gauss) :

    data = open(k_correlated_file,'w')
    first = ''
    for df in range(dec_first) :
        first += ' '
    final = ''
    for dfl in range(dec_final) :
        final += ' '

    bar = ProgressBar(dim_gauss*dim_bande,'Kcorr record')

    for m in range(dim_gauss) :
        for l in range(dim_bande) :
            for k in range(Q_dim) :
                for j in range(P_dim) :
                    for i in range(T_dim) :

                        if i == 0 and j == 0 and k == 0 and l == 0 and m == 0 :
                            data.write(first)

                        data.write('%.16E'%(k_corr_data[i,j,k,l,m]))
                        data.write(final)

            bar.animate(m*dim_bande+l+1)


########################################################################################################################


def cross_data_read(file_path,type_ref,n_species,path,source) :

    data = pickle.load(open('%s%s.db'%(file_path,n_species[0])))

    np.save("%sbande_sample_%s.npy"%(path,source), data["%s"%(type_ref[1])])
    np.save("%sP_sample_%s.npy"%(path,source), np.log10(data["%s"%(type_ref[2])]))
    np.save("%sT_sample_%s.npy"%(path,source), data["%s"%(type_ref[3])])

    dim_bande, dim_P, dim_T = data["%s"%(type_ref[1])].size, data["%s"%(type_ref[2])].size, data["%s"%(type_ref[3])].size

    section = np.zeros((n_species.size,dim_P,dim_T,dim_bande))

    for i in range(n_species.size) :

        data = pickle.load(open('%s%s.db'%(file_path,n_species[i])))

        section[i,:,:,:] = data[type_ref[0]]

    np.save("%scrossection_%s.npy"%(path,source), section)


########################################################################################################################


def k_cont_data_read(k_cont_file,associations,directory) :

    data = open(k_cont_file,'r')
    k_cont_data = data.readlines()

    res = line_search(k_cont_data[0])
    dim_bande = int(res[3])

    line = np.shape(k_cont_data)[0]

    dim_T = int(line/(dim_bande+1))

    bande_cont = np.zeros(dim_bande)
    T_cont = np.zeros(dim_T)
    i_c = 1

    while i_c < len(k_cont_data[1]) :
        if k_cont_data[1][i_c] == ' ' and k_cont_data[1][i_c-1] != ' ' :
            lim_wl = i_c
            i_c = len(k_cont_data[1])-1
        i_c += 1
    lim_k = len(k_cont_data[1])

    for i_T in range(dim_T) :

        res = line_search(k_cont_data[i_T*(dim_bande+1)])
        T_cont[i_T] = res[4]

    for i_bande in range(dim_bande) :

        bande_cont[i_bande] = np.float(k_cont_data[i_bande+1][0:lim_wl])

    np.save('%sk_cont_nu_%s.npy'%(directory,associations),bande_cont)
    print 'Bande_cont %s : '%(associations)
    print bande_cont
    np.save('%sT_cont_%s.npy'%(directory,associations),T_cont)
    print 'T_cont %s : '%(associations)
    print T_cont

    i_col = lim_wl + 2
    k_cont_plan = np.zeros((dim_T,dim_bande))

    bar = ProgressBar(dim_T*dim_bande,'K_cont_%s record'%(associations))

    for i_T in range(dim_T) :
        T_coeff = i_T*(dim_bande+1)
        for i_bande in range(dim_bande) :

            i_line = T_coeff + i_bande + 1

            k_cont_plan[i_T,i_bande] = np.float(k_cont_data[i_line][i_col:lim_k])

            bar.animate(i_T*dim_bande+i_bande+1)

    np.save('%sk_cont_%s.npy'%(directory,associations),k_cont_plan)


########################################################################################################################


def k_cont_h2o_read(k_cont_file,associations,directory) :

    data = open(k_cont_file,'r')
    k_cont_data = data.readlines()
    res = line_search(k_cont_data[0])

    dim_T = np.array([res]).size
    dim_bande = np.shape(k_cont_data)[0]
    i_c = 1
    T_cont = np.zeros(dim_T)

    while i_c < len(k_cont_data[1]) :
        if k_cont_data[1][i_c] == ' ' and k_cont_data[1][i_c-1] != ' ' :
            space_k = i_c
            i_c = len(k_cont_data[1]) - 1
        i_c += 1

    k_cont_plan = np.zeros((dim_T,dim_bande))

    bar = ProgressBar(dim_T*dim_bande,'K_cont_h2o record')

    for i_T in range(dim_T) :
        T_cont[i_T] = 200+i_T*50
        for i_bande in range(dim_bande) :
            k_cont_plan[i_T,i_bande] = np.float(k_cont_data[i_bande][i_T*space_k:(i_T+1)*space_k])
            bar.animate(i_T*dim_bande+i_bande+1)

    np.save('%sk_cont_%s.npy'%(directory,associations),k_cont_plan)
    np.save('%sT_cont_%s.npy'%(directory,associations),T_cont)
    print 'T_cont_H2O : '
    print T_cont


########################################################################################################################


def composition_data_read(composition,directory,name_exo,Renorm=False) :

    path = composition.file
    n_species, ratio = composition.species, ratio_HeH2
    renorm = composition.renorm
    data = open('%scomposition.in'%(path),'r')
    comp = data.readlines()
    data = open('%sT.dat'%(path),'r')
    T_read = data.readlines()
    T_dim = line_search(T_read[0])
    data =  open('%sp.dat'%(path),'r')
    p_read = data.readlines()
    P_dim = line_search(p_read[0])
    if composition.parameters.size > 2 :
        data =  open('%sQ.dat'%(path),'r')
        Q_read = data.readlines()
        dec = line_search(Q_read[0])
        Q_dim = line_search(Q_read[int(dec[0])+1])
        Q_dim = int(Q_dim[0])
        Q_sample = np.zeros(Q_dim)
        for i_Q in range(Q_dim) :
            Q_sample[i_Q] = np.float(line_search(Q_read[int(dec[0])+2+i_Q])[0])
        np.save('%sQ_comp_%s.npy'%(directory,name_exo),Q_sample)
    else :
        Q_dim = 1
    P_dim, T_dim = int(P_dim[0]), int(T_dim[0])
    T_sample,P_sample = np.zeros(T_dim), np.zeros(P_dim)
    for i_T in range(T_dim) :
        T_sample[i_T] = np.float(line_search(T_read[1+i_T])[0])
    for i_P in range(P_dim) :
        P_sample[i_P] = np.float(line_search(p_read[1+i_P])[0])
    np.save('%sT_comp_%s.npy'%(directory,name_exo),T_sample)
    np.save('%sP_comp_%s.npy'%(directory,name_exo),10**P_sample)

    x_species = np.zeros((n_species.size,P_dim,T_dim,Q_dim))

    bar = ProgressBar(P_dim*T_dim,'Composition record')

    for i_T in range(T_dim) :
        for i_P in range(P_dim) :
            for i_Q in range(Q_dim) :
                i_data = i_Q*T_dim*P_dim + i_P*T_dim + i_T + 5
                for i_spe in range(2,n_species.size) :
                    x_species[i_spe,i_P,i_T,i_Q] = np.float(comp[i_data][53+(i_spe-2)*17:53+(i_spe-1)*17])

                if Renorm == True :
                    for i_r in range(renorm[0].size) :
                        x_species[i_r,i_P,i_T,i_Q] = x_species[i_r,i_P,i_T,i_Q]/renorm[1,i_r]

                x_species[0,i_P,i_T,i_Q] = (1 - np.nansum(x_species[:,i_P,i_T,i_Q]))/(1+ratio)
                x_species[1,i_P,i_T,i_Q] = x_species[0,i_P,i_T,i_Q]*ratio

        bar.animate(i_T*P_dim+i_P+1)

    np.save('%sx_species_comp_%s.npy'%(directory,name_exo),x_species)


########################################################################################################################


def aerosol_data_read(aerosol_file,dec_first,space,species,name,directory,Save=True) :

    data = open(aerosol_file,'r')
    aero_data = data.readlines()

    n_bande = np.int(aero_data[dec_first+1][0:4])
    n_r = np.int(aero_data[dec_first+3][0:5])

    print 'Number of wavelength : %i'%(n_bande)
    print 'Number of radius : %i'%(n_r)

    Q = np.zeros((n_r,n_bande))
    bande = np.zeros(n_bande)
    radius = np.zeros(n_r)

    dec_line = dec_first + 8 + n_bande/5 + n_r/5
    if n_bande%5 != 0 :
        dec_line += 1
    if n_r%5 != 0 :
        dec_line += 1

    bar = ProgressBar(n_bande*n_r,'Aerosol data for %s record'%(species))

    for i in range(n_r) :

        for j in range(n_bande) :

            i_line = int(dec_line + int(j/5) + i*(n_bande/5 + 2))
            i_col = int((j%5)*space)

            Q[i,j] = np.float(aero_data[i_line][i_col:i_col+space])

            if i == 0 :

                i_b_line = dec_first +int(5 + int(j/5))
                i_b_col = int((j%5)*space)
                bande[j] = np.float(aero_data[i_b_line][i_b_col:i_b_col+space])

            if j == 0 :

                i_r_line = dec_first + int(7 + n_bande/5 + int(i/5))
                i_r_col = int((i%5)*space)
                radius[i] = np.float(aero_data[i_r_line][i_r_col:i_r_col+space])

            bar.animate(i*n_bande+j+1)

    if Save == True :
        np.save('%sQsingle_%s_%s.npy'%(directory,species,name),Q)
        np.save('%sbande_cloud_%s.npy'%(directory,name),bande)
        np.save('%sradius_cloud_%s.npy'%(directory,name),radius)

    return Q,bande,radius


########################################################################################################################


def line_search(line) :

    size = len(line)
    words = np.array([])
    i_d, i_f = 0, 0
    for i_l in range(size) :
        if line[i_l] == ' ' :
            if i_l != 0 :
                if line[i_l-1] != ' ' :
                    words = np.append(words,line[i_d:i_f])
                    i_d = i_f + 1
                    i_f += 1
                else :
                    i_d += 1
                    i_f += 1
            else :
                i_d += 1
                i_f += 1
        else :
            if i_l == size-1 :
                words = np.append(words,line[i_d:size])
            else :
                i_f += 1
    return words


########################################################################################################################


def flux_script(path,name_source,source,save_name,I,error,Rs,Rp,r_step,Kcorr=False,Middle=True,Noise=False) :

    sh_I = np.shape(I)

    if error.size == 1 :
        error_cor = np.ones(sh_I[0])*error[0]
        error = error_cor

    from pytransfert import atmospectre

    bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,source))

    R_eff_bar,R_eff,ratio_bar,ratR_bar,bande_bar,flux_bar,flux = atmospectre(I,bande_sample,Rs,Rp,r_step,0,\
                                                                                        False,Kcorr,Middle)

    print flux
    file_name = '%s.dat'%(save_name)

    output = open(file_name,'w')
    if Kcorr == True :
        bsize = bande_sample.size-1
    else :
        bsize = bande_sample.size

    for i_wr in range(bsize) :
        if bande_sample[bsize - i_wr - 1] != 0 :
            if str(error[bsize - i_wr - 2]) != 'nan' :
                output.write('%.18E '%(1/(100.*bande_sample[bsize - i_wr - 1])*10**6))
                if Noise == False :
                    output.write('%.18E '%(flux[bsize - i_wr - 1]))
                else :
                    output.write('%.18E '%(flux[bsize - i_wr - 1]+\
                                           np.random.normal(0,error[bsize - i_wr - 2])))
                output.write('%.18E \n'%(error[bsize - i_wr - 2]))


########################################################################################################################


def spectrum_output_taurex_read(file) :

    data = open('%s'%(file),'r')
    data = data.readlines()
    size = len(data)
    bande_sample = np.zeros(size)
    flux = np.zeros(size)
    for i_s in range(size) :
        don = line_search(data[i_s])
        bande_sample[i_s] = don[0]
        flux[i_s] = don[1]
    return bande_sample,flux

