from pyfunction import *
from pyconstant import *


########################################################################################################################
########################################################################################################################

"""
    K_CORRELATED_INTERP, K_CORRELATED_INTERP_BOUCLE

    La fonction exploite les profils en temperature, pression et fraction molaire pour fournir en chaque point une valeur
    d'opacite qui depend de la temperature locale, de la pression locale, de l'abondance relative locale ainsi que de la
    bande et du point de Gauss considere. Elle effectue une interpolation lineaire par rapport a la temperature et la
    fraction molaire, ainsi qu'une interpolation lineaire par rapport au logarithme de la pression.

    En somme, elle cherhe les 8 points dans l'espace T,P,x pour une bande et un point de gauss donne, retrouve les
    opacites correspondante et en deduit une valeur locale par interpolation. Cette fonction s'applique sur des tableaux
    de donnees. Une premiere iteration permet de generer un ensemble de coefficients d'interpolation et les indices
    correspondants, il est ensuite reintroduit dans la fonction _BOUCLE pour eviter d'en reiterer les calculs.

"""

########################################################################################################################
########################################################################################################################


def Ssearcher(T_array,P_array,compo_array,sigma_array,P_sample,T_sample,rank,rank_ref,Kcorr=False,Optimal=False,Script=True) :

    n_sp, P_size, T_size, dim_bande = np.shape(sigma_array)

    sigma_data = sigma_array[0,:,:,:]

    sigma_array_t = np.transpose(sigma_array,(1,2,3,0))
    compo_array_t = np.transpose(compo_array)

    k_inter,size,i_Tu_arr,i_pu_arr,coeff_1_array,coeff_3_array = \
    k_correlated_interp(sigma_data,P_array,T_array,0,P_sample,T_sample,rank,rank_ref,Kcorr,Optimal,Script)

    zz, = np.where(i_Tu_arr == 0)
    i_Td_arr = i_Tu_arr - 1
    i_Td_arr[zz] = np.zeros(zz.size)
    zz, = np.where(i_pu_arr == 0)
    i_pd_arr = i_pu_arr - 1
    i_pd_arr[zz] = np.zeros(zz.size)

    k_rmd = np.zeros((i_Tu_arr.size,dim_bande),dtype=np.float64)

    if Script == True :
        if rank == rank_ref :
            bar = ProgressBar(i_Tu_arr.size,'Ssearcher progression')

    for i in xrange( i_Tu_arr.size ):

        i_Tu = i_Tu_arr[i]
        i_Td = i_Td_arr[i]
        i_pu = i_pu_arr[i]
        i_pd = i_pd_arr[i]

        if Optimal == False :

            coeff_1 = coeff_1_array[i]
            coeff_2 = 1. - coeff_1
            coeff_3 = coeff_3_array[i]
            coeff_4 = 1. - coeff_3

            c13 = coeff_1 * coeff_3 * 0.0001
            c23 = coeff_2 * coeff_3 * 0.0001
            c14 = coeff_1 * coeff_4 * 0.0001
            c24 = coeff_2 * coeff_4 * 0.0001

            k_pd_Td = sigma_array_t[i_pd, i_Td, :, :]
            k_pd_Tu = sigma_array_t[i_pd, i_Tu, :, :]
            k_pu_Td = sigma_array_t[i_pu, i_Td, :, :]
            k_pu_Tu = sigma_array_t[i_pu, i_Tu, :, :]

            comp = compo_array_t[i,:]

            k_1 = k_pd_Td * c24 + k_pd_Tu * c23
            k_2 = k_pu_Td * c14 + k_pu_Tu * c13

            k_rmd[i, :] = np.dot( k_1 + k_2, comp )

        else :

            b_m = coeff_1_array[0,i]
            a_m = coeff_1_array[1,i]
            T = coeff_1_array[2,i]
            coeff_3 = coeff_3_array[i] * 0.0001
            coeff_4 = (1. - coeff_3) * 0.0001

            k_pd_Tu = sigma_array_t[i_pd, i_Tu, :, :]
            k_pu_Tu = sigma_array_t[i_pu, i_Tu, :, :]

            comp = compo_array_t[i,:]

            k_u = k_pd_Tu * coeff_4 + k_pu_Tu * coeff_3
            k_d = k_pd_Td * coeff_4 + k_pu_Td * coeff_3

            b_mm = b_m * np.log(k_u/k_d)
            a_mm = np.exp(b_mm/a_m)

            k_rmd[i, :] = np.dot( k_u*a_mm*np.exp(-b_mm/T), comp )

        if Script == True :
            if rank == rank_ref :
                bar.animate( i+1 )

    return k_rmd


########################################################################################################################


def Ksearcher(T_array,P_array,dim_gauss,dim_bande,k_corr_data_grid,P_sample,T_sample,rank,rank_ref,Kcorr,Optimal=False,Script=True) :

    k_rmd = np.zeros((P_array.size,dim_bande,dim_gauss),dtype=np.float64)
    layer = int(T_array.size/10.)
    T_size = T_array.size

    if Script == True :
        if rank == rank_ref :
            bar = ProgressBar(dim_bande*dim_gauss,'K-correlated coefficients computation')

    for i_bande in range(dim_bande) :
        k_corr_data = k_corr_data_grid[:,:,:,i_bande,:]

        if i_bande == 0 :
            for i_gauss in range(dim_gauss) :
                if i_gauss == 0 :
                    k_inter,size,i_Tu_array,i_pu_array,coeff_1_array,coeff_3_array = \
                    k_correlated_interp(k_corr_data,P_array,T_array,i_gauss,P_sample,T_sample,rank,rank_ref,Kcorr,Optimal,Script)

                    k_rmd[:,i_bande,i_gauss] = k_inter[:]
                else :
                    for lay in range(10) :
                        if lay != 9 :
                            k_inter = k_correlated_interp_boucle(k_corr_data,size,i_Tu_array[lay*layer:(lay+1)*layer],i_pu_array[lay*layer:(lay+1)*layer],\
                                            coeff_1_array[lay*layer:(lay+1)*layer],coeff_3_array[lay*layer:(lay+1)*layer],i_gauss)

                            k_rmd[lay*layer:(lay+1)*layer,i_bande,i_gauss] = k_inter[:]
                        else :
                            k_inter = k_correlated_interp_boucle(k_corr_data,size,i_Tu_array[lay*layer:T_size],i_pu_array[lay*layer:T_size],\
                                            coeff_1_array[lay*layer:T_size],coeff_3_array[lay*layer:T_size],i_gauss)

                            k_rmd[lay*layer:T_size,i_bande,i_gauss] = k_inter[:]

                if Script == True :
                    if rank == rank_ref :
                        bar.animate(i_bande*dim_gauss + i_gauss + 1)

        else :
            for i_gauss in range(dim_gauss) :
                for lay in range(10) :
                    if lay != 9 :
                        k_inter = k_correlated_interp_boucle(k_corr_data,size,i_Tu_array[lay*layer:(lay+1)*layer],i_pu_array[lay*layer:(lay+1)*layer],\
                                            coeff_1_array[lay*layer:(lay+1)*layer],coeff_3_array[lay*layer:(lay+1)*layer],i_gauss)

                        k_rmd[lay*layer:(lay+1)*layer,i_bande,i_gauss] = k_inter[:]
                    else :
                        k_inter = k_correlated_interp_boucle(k_corr_data,size,i_Tu_array[lay*layer:T_size],i_pu_array[lay*layer:T_size],\
                                            coeff_1_array[lay*layer:T_size],coeff_3_array[lay*layer:T_size],i_gauss)

                        k_rmd[lay*layer:T_size,i_bande,i_gauss] = k_inter[:]

                if Script == True :
                    if rank == rank_ref :
                        bar.animate(i_bande*dim_gauss + i_gauss + 1)

    return k_rmd


########################################################################################################################


def k_correlated_interp(k_corr_data,P_array,T_array,i_gauss,P_sample,T_sample,rank,rank_ref,Kcorr=True,Optimal=False,Script=True) :

    size = P_array.size
    k_inter = np.zeros(size)

    i_Tu_array = np.zeros(size, dtype = "int")
    i_pu_array = np.zeros(size, dtype = "int")
    if Optimal == True :
        coeff_1_array = np.zeros((3,size))
    else :
        coeff_1_array = np.zeros(size)
    coeff_3_array = np.zeros(size)

    if Script == True :
        if rank == rank_ref :
            bar = ProgressBar(size,'Module interpolates cross sections')

    for i in range(size) :

        P = np.log10(P_array[i])-2
        T = T_array[i]

        if T == 0 or P == 0 :
            i_Tu = 0
            i_pu = 0
            coeff_1 = 0
            coeff_3 = 0
            k_inter[i] = 0.
        else :
            if T == T_array[i-1] and P == P_array[i-1] and i != 0 :
                k_inter[i] = k_inter[i-1]
            else :
                if Kcorr == True :
                    res,c_grid,i_grid = interp2olation_opti_uni(T,P,T_sample,P_sample,k_corr_data[:,:,0,i_gauss],False,False)
                    k_inter[i] = res
                    i_Td, i_Tu, i_pd, i_pu = i_grid[0], i_grid[1], i_grid[2], i_grid[3]
                    coeff_3, coeff_1 = c_grid[0], c_grid[2]
                else :
                    if Optimal == True :
                        res,c_grid,i_grid = interp2olation_opti_uni(P,T,P_sample,T_sample,k_corr_data[:,:,0],False,True)
                        b_m, a_m, T = c_grid[0], c_grid[1], c_grid[2]
                        coeff_3 = c_grid[3]
                        k_inter[i] = res
                    else :
                        res,c_grid,i_grid = interp2olation_opti_uni(P,T,P_sample,T_sample,k_corr_data[:,:,0],False,False)
                        coeff_1, coeff_3 = c_grid[0], c_grid[2]
                        k_inter[i] = res
                    i_pd, i_pu, i_Td, i_Tu = i_grid[0], i_grid[1], i_grid[2], i_grid[3]

        i_Tu_array[i] = int(i_Tu)
        i_pu_array[i] = int(i_pu)
        if Optimal == False :
            coeff_1_array[i] = coeff_1
        else :
            coeff_1_array[0,i] = b_m
            coeff_1_array[1,i] = a_m
            coeff_1_array[2,i] = T
        coeff_3_array[i] = coeff_3

        if Script == True :
            if rank == rank_ref :
                if i%100 == 0. or i == size - 1 :
                    bar.animate(i + 1)

    return k_inter*0.0001,size,i_Tu_array,i_pu_array,coeff_1_array,coeff_3_array


########################################################################################################################


def k_correlated_interp_boucle(k_corr_data,size,i_Tu_array,i_pu_array,coeff_1_array,coeff_3_array,i_gauss):

    coeff_2_array = 1 - coeff_1_array
    coeff_4_array = 1 - coeff_3_array
    zz, = np.where(i_Tu_array == 0)
    i_Td_array = i_Tu_array - 1
    i_Td_array[zz] = np.zeros(zz.size)
    zz, = np.where(i_pu_array == 0)
    i_pd_array = i_pu_array - 1
    i_pd_array[zz] = np.zeros(zz.size)

    gauss = np.ones(i_Tu_array.size,dtype='int')*i_gauss

    k_pd_Td = k_corr_data[i_Td_array,i_pd_array,0,gauss]
    k_pd_Tu = k_corr_data[i_Tu_array,i_pd_array,0,gauss]
    k_pu_Td = k_corr_data[i_Td_array,i_pu_array,0,gauss]
    k_pu_Tu = k_corr_data[i_Tu_array,i_pu_array,0,gauss]

    k_d = k_pd_Td*coeff_4_array + k_pu_Td*coeff_3_array
    k_u = k_pd_Tu*coeff_4_array + k_pu_Tu*coeff_3_array

    k_inter = k_d*coeff_2_array + k_u*coeff_1_array

    return k_inter*0.0001


########################################################################################################################


def Ksearcher_M(T_array,P_array,Q_array,dim_gauss,dim_bande,k_corr_data_grid,P_sample,T_sample,Q_sample,rank,rank_ref,Kcorr,Optimal=False,Script=True) :

    k_rmd = np.zeros((P_array.size,dim_bande,dim_gauss),dtype=np.float64)

    if Script == True :
        if rank == rank_ref :
            bar = ProgressBar(dim_bande*dim_gauss,'K-correlated coefficients computation')

    layer = int(T_array.size/10.)
    T_size = T_array.size

    for i_bande in range(dim_bande) :
        k_corr_data = k_corr_data_grid[:,:,:,i_bande,:]

        if i_bande == 0 :
            for i_gauss in range(dim_gauss) :
                if i_gauss == 0 :
                    k_inter,size,i_Tu_array,i_pu_array,i_Qu_array,coeff_1_array,coeff_3_array,coeff_5_array = \
                        k_correlated_interp_M(k_corr_data,P_array,T_array,Q_array,i_gauss,P_sample,T_sample,Q_sample,rank,rank_ref,Kcorr,Optimal,Script=True)

                    k_rmd[:,i_bande,i_gauss] = k_inter[:]
                else :
                    for lay in range(5) :
                        if lay != 4 :
                            k_inter = k_correlated_interp_boucle_M(k_corr_data,size,i_Tu_array[lay*layer:(lay+1)*layer],i_pu_array[lay*layer:(lay+1)*layer],i_Qu_array[lay*layer:(lay+1)*layer],\
                                                    coeff_1_array[lay*layer:(lay+1)*layer],coeff_3_array[lay*layer:(lay+1)*layer],coeff_5_array[lay*layer:(lay+1)*layer],i_gauss)

                            k_rmd[lay*layer:(lay+1)*layer,i_gauss,i_bande] = k_inter[:]
                        else :
                            k_inter = k_correlated_interp_boucle_M(k_corr_data,size,i_Tu_array[lay*layer:T_size],i_pu_array[lay*layer:T_size],i_Qu_array[lay*layer:T_size],\
                                                    coeff_1_array[lay*layer:T_size],coeff_3_array[lay*layer:T_size],coeff_5_array[lay*layer:T_size],i_gauss)

                            k_rmd[lay*layer:T_size,i_bande,i_gauss] = k_inter[:]

                if Script == True :
                    if rank == rank_ref :
                        bar.animate(i_bande*dim_gauss + i_gauss + 1)

        else :
            for i_gauss in range(dim_gauss) :
                for lay in range(5) :
                    if lay != 4 :
                        k_inter = k_correlated_interp_boucle_M(k_corr_data,size,i_Tu_array[lay*layer:(lay+1)*layer],i_pu_array[lay*layer:(lay+1)*layer],i_Qu_array[lay*layer:(lay+1)*layer],\
                                                    coeff_1_array[lay*layer:(lay+1)*layer],coeff_3_array[lay*layer:(lay+1)*layer],coeff_5_array[lay*layer:(lay+1)*layer],i_gauss)

                        k_rmd[lay*layer:(lay+1)*layer,i_bande,i_gauss] = k_inter[:]
                    else :
                        k_inter = k_correlated_interp_boucle_M(k_corr_data,size,i_Tu_array[lay*layer:T_size],i_pu_array[lay*layer:T_size],i_Qu_array[lay*layer:T_size],\
                                                    coeff_1_array[lay*layer:T_size],coeff_3_array[lay*layer:T_size],coeff_5_array[lay*layer:T_size],i_gauss)

                        k_rmd[lay*layer:T_size,i_bande,i_gauss] = k_inter[:]

                if Script == True :
                    if rank == rank_ref :
                        bar.animate(i_bande*dim_gauss + i_gauss + 1)

    return k_rmd


########################################################################################################################


def k_correlated_interp_M(k_corr_data,P_array,T_array,Q_array,i_gauss,P_sample,T_sample,Q_sample,rank,rank_ref,Kcorr=True,Optimal=False,Script=True) :

    size = P_array.size
    k_inter = np.zeros(size)

    i_Tu_array = np.zeros(size, dtype = "int")
    i_pu_array = np.zeros(size, dtype = "int")
    i_qu_array = np.zeros(size, dtype = "int")
    if Optimal == True :
        coeff_1_array = np.zeros((3,size),dtype=np.float64)
    else :
        coeff_1_array = np.zeros(size,dtype=np.float64)
    coeff_3_array = np.zeros(size,dtype=np.float64)
    coeff_5_array = np.zeros(size,dtype=np.float64)

    if Script == True :
        if rank == rank_ref :
            bar = ProgressBar(size,'Module interpolates cross sections')

    for i in range(size) :

        P = np.log10(P_array[i])-2
        T = T_array[i]
        Q = np.log10(Q_array[i])

        if T == 0 or P == 0 :
            i_Tu = 0
            i_pu = 0
            i_qu = 0
            coeff_1 = 0
            coeff_3 = 0
            coeff_5 = 0
            k_inter[i] = 0.

        else :

            if T == T_array[i-1] and P == P_array[i-1] and Q == Q_array[i-1] and i != 0 :
                k_inter[i] = k_inter[i-1]
            else :
                if Kcorr == True :
                    res,c_grid,i_grid = interp3olation_opti_uni(T,P,Q,T_sample,P_sample,Q_sample,k_corr_data[:,:,:,0,i_gauss],False,False,False)
                    k_inter[i] = res
                    i_Td, i_Tu, i_pd, i_pu, i_qd, i_qu = i_grid[0], i_grid[1], i_grid[2], i_grid[3], i_grid[4], i_grid[5]
                    coeff_3, coeff_1, coeff_5 = c_grid[0], c_grid[2], c_grid[4]
                else :
                    if Optimal == True :
                        res,c_grid,i_grid = interp2olation_opti_uni(P,T,P_sample,T_sample,k_corr_data[:,:,0],False,True)
                        b_m, a_m, T = c_grid[0], c_grid[1], c_grid[2]
                        coeff_3 = c_grid[3]
                        k_inter[i] = res
                    else :
                        res,c_grid,i_grid = interp2olation_opti_uni(P,T,P_sample,T_sample,k_corr_data[:,:,0],False,False)
                        coeff_1, coeff_3 = c_grid[0], c_grid[2]
                        k_inter[i] = res
                    i_pd, i_pu, i_Td, i_Tu = i_grid[0], i_grid[1], i_grid[2], i_grid[3]

        i_Tu_array[i] = int(i_Tu)
        i_pu_array[i] = int(i_pu)
        i_qu_array[i] = int(i_qu)
        if Optimal == False :
            coeff_1_array[i] = coeff_1
        else :
            coeff_1_array[0,i] = b_m
            coeff_1_array[1,i] = a_m
            coeff_1_array[2,i] = T
        coeff_3_array[i] = coeff_3
        coeff_5_array[i] = coeff_5

        if Script == True :
            if rank == rank_ref :
                if i%100 == 0. or i == size - 1 :
                    bar.animate(i + 1)

    return k_inter*0.0001,size,i_Tu_array,i_pu_array,i_qu_array,coeff_1_array,coeff_3_array,coeff_5_array


########################################################################################################################


def k_correlated_interp_boucle_M(k_corr_data,size,i_Tu_array,i_pu_array,i_Qu_array,coeff_1_array,coeff_3_array,coeff_5_array,i_gauss):

    coeff_2_array = 1 - coeff_1_array
    coeff_4_array = 1 - coeff_3_array
    coeff_6_array = 1 - coeff_5_array
    zz, = np.where(i_Tu_array == 0)
    i_Td_array = i_Tu_array - 1
    i_Td_array[zz] = np.zeros(zz.size)
    zz, = np.where(i_pu_array == 0)
    i_pd_array = i_pu_array - 1
    i_pd_array[zz] = np.zeros(zz.size)
    zz, = np.where(i_Qu_array == 0)
    i_Qd_array = i_Qu_array - 1
    i_Qd_array[zz] = np.zeros(zz.size)

    gauss = np.ones(i_Tu_array.size,dtype='int')*i_gauss

    k_pd_Td_Qd = k_corr_data[i_Td_array,i_pd_array,i_Qd_array,gauss]
    k_pd_Tu_Qd = k_corr_data[i_Tu_array,i_pd_array,i_Qd_array,gauss]
    k_pu_Td_Qd = k_corr_data[i_Td_array,i_pu_array,i_Qd_array,gauss]
    k_pu_Tu_Qd = k_corr_data[i_Tu_array,i_pu_array,i_Qd_array,gauss]
    k_pd_Td_Qu = k_corr_data[i_Td_array,i_pd_array,i_Qu_array,gauss]
    k_pd_Tu_Qu = k_corr_data[i_Tu_array,i_pd_array,i_Qu_array,gauss]
    k_pu_Td_Qu = k_corr_data[i_Td_array,i_pu_array,i_Qu_array,gauss]
    k_pu_Tu_Qu = k_corr_data[i_Tu_array,i_pu_array,i_Qu_array,gauss]

    k_1 = k_pd_Td_Qd*coeff_6_array + k_pd_Td_Qu*coeff_5_array
    k_2 = k_pu_Td_Qd*coeff_6_array + k_pu_Td_Qu*coeff_5_array
    k_3 = k_pd_Tu_Qd*coeff_6_array + k_pd_Tu_Qu*coeff_5_array
    k_4 = k_pu_Tu_Qd*coeff_6_array + k_pu_Tu_Qu*coeff_5_array

    k_d = k_1*coeff_4_array + k_2*coeff_3_array
    k_u = k_3*coeff_4_array + k_4*coeff_3_array
    k_inter = k_d*coeff_2_array + k_u*coeff_1_array

    return k_inter*0.0001


########################################################################################################################


def Rayleigh_scattering (P_array,T_array,bande_sample,x_mol_species,n_species,zero,rank,rank_ref,Kcorr=True,MarcIngo=False,Script=True) :

    if Kcorr == True :
        dim_bande = bande_sample.size-1
    else :
        dim_bande = bande_sample.size

    k_sca_rmd = np.zeros((P_array.size,dim_bande),dtype=np.float64)

    fact = 24*np.pi**3/((101325/(R_gp*273.15)*N_A)**2)

    n_mol_tot = P_array/(R_gp*T_array)*N_A

    if zero.size != 0 :

        n_mol_tot[zero] = np.zeros((zero.size))

    if Script == True :
        if rank == rank_ref :
            bar = ProgressBar(dim_bande,'Scattering computation progression')

    for i_bande in range(dim_bande) :

        if Kcorr == True :
            w_n = (bande_sample[i_bande] + bande_sample[i_bande + 1])/2.
        else :
            w_n = bande_sample[i_bande]

        wl = 1./(w_n)*10**4

        for sp in range(n_species.size) :

            if n_species[sp] == 'H2' :

                f_K = 1.

                if wl < 0.300 :

                    sig = f_K*8.49e-33/(wl**4)

                else :

                    sig = f_K*(8.14e-33/(wl**4) + 1.28e-35/(wl**6) + 1.61e-37/(wl**8))

            elif n_species[sp] == 'He' :

                f_K = 1.

                if MarcIngo == True :

                    sig = f_K*(5.484e-34/(wl**4) + 1.33e-36/(wl**6))

                else :

                    index = 1 + (2.283e-5 + 1.8102e-3/(1.532e+2-1/(wl**2)))
                    pol = (index**2-1)/(index**2+2)**2


            elif n_species[sp] == 'O2' :

                f_K = 1.096 + 1.385e-3/(wl**2) + 1.448e-4/(wl**4)

                if wl < 0.221 :

                    index = 1 + (2.37967e-4 + 1.689884e-3/(4.09e+1-1/wl**2))

                if wl > 0.221 and wl < 0.288 :

                    index = 1 + (2.21204e-4 + 2.03187e-3/(4.09e+1-1/wl**2))

                if w_n > 0.228 and wl < 0.546 :

                    index = 1 + (2.0564e-4 + 2.480899e-3/(4.09e+1-1/wl**2))

                if w_n > 0.546 :

                    index = 1 + (2.1351e-4 + 2.18567e-3/(4.09e+1-1/wl**2))

                pol = (index**2-1.)/(index**2+2.)**2

            elif n_species[sp] == 'N2' :

                f_K = 1.034 + 3.17e-4/wl**2

                if wl > 0.468 :

                    index = 1 + (6.4982e-5 + (3.0743305e-2)/(1.44e+2-1/wl**2))

                if wl > 0.254 and w_n < 0.468 :

                    index = 1 + (5.677465e-5 + (3.1881874e-2)/(1.44e+2-1/wl**2))

                if wl < 0.254 :

                    index = 1 + (6.998749e-5 + (3.23358e-2)/(1.44e+2-1/wl**2))

                pol = (index**2-1.)/(index**2+2.)**2

            elif n_species[sp] == 'CO2' :

                f_K = 1.1364 + 2.53e-3/wl**2
                index = 1 + 1.1427e-5*(5.79925e+3/(1.66175e+2-1/wl**2) + 1.2005e+2/(7.9608e+1-1/wl**2) + 5.3334/(5.6306e+1-1/wl**2) + \
                                    4.3244/(4.619e+1-1/wl**2) + 0.12181e-4/(5.8474e-2-1/wl**2))
                pol = (index**2-1)/(index**2+2)**2

            elif n_species[sp] == 'CO' :

                f_K = 1.016
                index = 1 + (2.2851e-4 + (4.56e-5)/(5.101816329e+1-1/wl**2))
                pol = (index**2-1.)/(index**2+2.)**2

            elif n_species[sp] == 'CH4' :

                f_K = 1.
                index = 1 + (4.6662e-4 + (4.02e-6)/(wl**2))
                pol = (index**2-1.)/(index**2+2.)**2

            elif n_species[sp] == 'Ar' :

                f_K = 1.
                index = 1 + (6.432135e-5 + (2.8606e-2)/(1.44e+2 - 1/(wl**2)))
                pol = (index**2-1.)/(index**2+2.)**2

            elif n_species[sp] == 'H2O' :

                f_K = (6+3*0.17)/(6-7*0.17)

                if w_n > 0.230 :

                    index = 1 + (4.92303e-2/(2.380185e+2 - 1/wl**2) + 1.42723e-3/(5.7362e+1 - 1/wl**2))

                else :

                    index = 1 + 0.85*(8.06051e-4 + 2.48099e-2/(132.274e+2 - 1/wl**2) + 1.74557e-2/(3.932957e+1 - 1/wl**2))
                pol = (index**2-1.)/(index**2+2.)**2

            else :

                sig = 0.
                pol = 0.
                f_K = 0.

            if n_species[sp] != 'H2' and n_species[sp] != 'He':
                sig = fact*1.e+24/(wl**4)*pol*f_K
            if n_species[sp] == 'He' and MarcIngo == False :
                sig = fact*1.e+24/(wl**4)*pol*f_K
            wll = 1./(w_n*10**(2))

            k_sca_rmd[:,i_bande] += sig*n_mol_tot*x_mol_species[sp,:]

        if Script == True :
            if rank == rank_ref :
                bar.animate(i_bande + 1)

    return k_sca_rmd


########################################################################################################################


def cloud_scattering(Qext,bande_cloud,P,T,bande_sample,M,rho_p,gen,r_eff,r_cloud,zero,rank,rank_ref,Script=True) :

    wh, = np.where(r_cloud == r_eff)

    if wh.size == 0 :
        whu, = np.where(r_cloud > r_eff)
        if whu.size != 0 :
            if whu[0] != 0 :
                i_r_u = whu[0]
                r_u = r_cloud[whu[0]]
                i_r_d = i_r_u - 1
                r_d = r_cloud[whu[0]-1]

                coeff1 = (r_eff - r_d)/(r_u - r_d)
                coeff2 = 1 - coeff1
            else :
                i_r_u,i_r_d = 0,0
                coeff1 = 1
                coeff2 = 0
        else :
            i_r_u,i_r_d = r_cloud.size - 1, r_cloud.size - 1
            coeff1 = 1
            coeff2 = 0
    else :
        i_r_u,i_r_d = wh[0], wh[0]
        coeff1 = 1
        coeff2 = 0

    k_cloud_rmd = np.zeros((P.size,bande_sample.size))
    Q_int = Qext[i_r_u,:]*coeff1 + Qext[i_r_d,:]*coeff2

    if Script == True :
        if rank == rank_ref :
            bar = ProgressBar(bande_sample.size,'Clouds scattering computation progression')

    for i_bande in range(bande_sample.size) :
        wh, = np.where(bande_cloud == bande_sample[i_bande])
        if wh.size == 0 :
            whu, = np.where(bande_cloud > bande_sample[i_bande])
            if whu.size != 0 :
                if whu[0] != 0 :
                    i_b_u = whu[0]
                    b_u = bande_cloud[whu[0]]
                    i_b_d = i_b_u - 1
                    b_d = bande_cloud[whu[0]-1]

                    coeff3 = (bande_sample[i_bande] - b_d)/(b_u - b_d)
                    coeff4 = 1 - coeff3
                else :
                    i_b_u,i_b_d = 0,0

                    coeff3 = 1
                    coeff4 = 0
            else :
                i_b_u,i_b_d = bande_cloud.size - 1, bande_cloud.size - 1

                coeff3 = 1
                coeff4 = 0
        else :
            i_b_u,i_b_d = wh[0], wh[0]

            coeff3 = 1
            coeff4 = 0

        Q_fin = Q_int[i_b_u]*coeff3 + Q_int[i_b_d]*coeff4

        k_cloud_rmd[:,i_bande] = 3/4.*(Q_fin*gen*P*M/(rho_p*r_eff*R_gp*T))

        if Script == True :
            if rank == rank_ref :
                bar.animate(i_bande + 1)

    if zero.size != 0 :

        k_cloud_rmd[zero,:] = np.zeros((zero.size,bande_sample.size))

    return k_cloud_rmd


########################################################################################################################


def refractive_index (P_array,T_array,bande_sample,x_mol_species,n_species,Kcorr) :

    if Kcorr == True :
        dim_bande = bande_sample.size-1
    else :
        dim_bande = bande_sample.size

    for i_bande in range(dim_bande) :

        if Kcorr == True :
            w_n = (bande_sample[i_bande] + bande_sample[i_bande + 1])/2.
        else :
            w_n = bande_sample[i_bande]

        wl = 1./(w_n)*10**4

        for sp in range(n_species.size) :

            if n_species[sp] == 'He' :

                index = 1 + (2.283e-5 + 1.8102e-3/(1.532e+2-1/(wl**2)))

            elif n_species[sp] == 'O2' :

                if wl < 0.221 :

                    index = 1 + (2.37967e-4 + 1.689884e-3/(4.09e+1-1/wl**2))

                if wl > 0.221 and wl < 0.288 :

                    index = 1 + (2.21204e-4 + 2.03187e-3/(4.09e+1-1/wl**2))

                if w_n > 0.228 and wl < 0.546 :

                    index = 1 + (2.0564e-4 + 2.480899e-3/(4.09e+1-1/wl**2))

                if w_n > 0.546 :

                    index = 1 + (2.1351e-4 + 2.18567e-3/(4.09e+1-1/wl**2))

            elif n_species[sp] == 'N2' :

                if wl > 0.468 :

                    index = 1 + (6.4982e-5 + (3.0743305e-2)/(1.44e+2-1/wl**2))

                if wl > 0.254 and w_n < 0.468 :

                    index = 1 + (5.677465e-5 + (3.1881874e-2)/(1.44e+2-1/wl**2))

                if wl < 0.254 :

                    index = 1 + (6.998749e-5 + (3.23358e-2)/(1.44e+2-1/wl**2))

            elif n_species[sp] == 'CO2' :

                index = 1 + 1.1427e-5*(5.79925e+3/(1.66175e+2-1/wl**2) + 1.2005e+2/(7.9608e+1-1/wl**2) + 5.3334/(5.6306e+1-1/wl**2) + \
                                    4.3244/(4.619e+1-1/wl**2) + 0.12181e-4/(5.8474e-2-1/wl**2))

            elif n_species[sp] == 'CO' :

                index = 1 + (2.2851e-4 + (4.56e-5)/(5.101816329e+1-1/wl**2))

            elif n_species[sp] == 'CH4' :

                index = 1 + (4.6662e-4 + (4.02e-6)/(wl**2))

            elif n_species[sp] == 'Ar' :

                index = 1 + (6.432135e-5 + (2.8606e-2)/(1.44e+2 - 1/(wl**2)))

            elif n_species[sp] == 'H2O' :

                if w_n > 0.230 :

                    index = 1 + (4.92303e-2/(2.380185e+2 - 1/wl**2) + 1.42723e-3/(5.7362e+1 - 1/wl**2))

                else :

                    index = 1 + 0.85*(8.06051e-4 + 2.48099e-2/(132.274e+2 - 1/wl**2) + 1.74557e-2/(3.932957e+1 - 1/wl**2))

            else :

                sig = 0.
                pol = 0.
                f_K = 0.

    return index


########################################################################################################################


def k_cont_interp_h2h2_integration(K_cont_h2h2,wavelength_cont_h2h2,T_array,bande_array,T_cont_h2h2,rank,rank_ref,Kcorr=True,Script=True) :


    losch = 2.6867774e19
    size = T_array.size
    if Kcorr == True :
        dim_bande = bande_array.size - 1
    else :
        dim_bande = bande_array.size

    k_interp_h2h2 = np.zeros((dim_bande,size))

    T_min = T_cont_h2h2[0]
    T_max = T_cont_h2h2[T_cont_h2h2.size-1]
    end = T_cont_h2h2.size - 1
    stop = 0

    if Script == True :
        if rank == rank_ref :
            bar = ProgressBar(dim_bande,'Continuum H2/H2 computation progression')

    for i_bande in range(dim_bande) :

        if Kcorr == True :
            k_interp = np.zeros(size)
        else :
            k_interp = np.zeros((2,size))

        if i_bande == 0 :

            i_Tu_array = np.zeros(size, dtype='int')
            coeff_array = np.zeros(size)

            if Kcorr == True :

                wave_max = bande_array[1]
                wave_min = bande_array[0]
                zone_wave, = np.where((wavelength_cont_h2h2 >= wave_min)*(wavelength_cont_h2h2 <= wave_max))
                fact = zone_wave.size

                for i_wave in zone_wave :
                    if i_wave == zone_wave[0] :
                        for i in range(size) :
                            T = T_array[i]
                            if T < T_min or T > T_max :
                                if T < T_min :
                                    if T != 0 :
                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[i] = K_cont_h2h2[0,i_wave]
                                    else :
                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[i] = 0.
                                if T > T_max :
                                    i_Tu, i_Td, coeff = end, end, 1
                                    k_interp[i] = K_cont_h2h2[T_cont_h2h2.size-1,i_wave]
                            else :
                                wh, = np.where(T_cont_h2h2 > T)
                                i_Tu = wh[0]
                                Tu = T_cont_h2h2[i_Tu]
                                i_Td = i_Tu - 1
                                Td = T_cont_h2h2[i_Td]

                                k_h2h2_Tu = K_cont_h2h2[i_Tu,i_wave]
                                k_h2h2_Td = K_cont_h2h2[i_Td,i_wave]

                                coeff = (T - Td)/(float(Tu + Td))

                                k_interp[i] = k_h2h2_Tu*coeff +  k_h2h2_Td*(1 - coeff)

                            i_Tu_array[i] = i_Tu
                            i_Td_array = i_Tu_array - 1
                            coeff_array[i] = coeff
                    else :
                        zer, = np.where(i_Tu_array == 0)
                        i_Td_array[zer] = np.zeros(zer.size, dtype='int')

                        ex, = np.where(i_Tu_array == end)
                        i_Td_array[ex] = np.ones(ex.size, dtype='int')*end

                        k_h2h2_Tu = K_cont_h2h2[i_Tu_array,i_wave]
                        k_h2h2_Td = K_cont_h2h2[i_Td_array,i_wave]

                        k_interp += k_h2h2_Tu*coeff_array + k_h2h2_Td*(1 - coeff_array)

                    if i_wave == zone_wave[fact-1] :

                        k_interp = k_interp/(float(fact))
                        k_interp_h2h2[i_bande,:] = k_interp

            else :

                zone_wave_up, = np.where(wavelength_cont_h2h2 >= bande_array[i_bande])
                wave_up = wavelength_cont_h2h2[zone_wave_up[0]]
                wave_down = wavelength_cont_h2h2[zone_wave_up[0]-1]

                coef = (bande_array[i_bande] - wave_down)/(float(wave_up + wave_down))

                for i_wave in [zone_wave_up[0]-1,zone_wave_up[0]] :
                    if i_wave == zone_wave_up[0]-1 :
                        if i_wave == -1 :
                            i_wave = 0
                        for i in range(size) :
                            T = T_array[i]
                            if T < T_min or T > T_max :
                                if T < T_min :
                                    if T != 0 :
                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[0,i] = K_cont_h2h2[0,i_wave]
                                    else :
                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[0,i] = 0.
                                if T > T_max :
                                    i_Tu, i_Td, coeff = end, end, 1
                                    k_interp[0,i] = K_cont_h2h2[T_cont_h2h2.size-1,i_wave]
                            else :
                                wh, = np.where(T_cont_h2h2 > T)
                                i_Tu = wh[0]
                                Tu = T_cont_h2h2[i_Tu]
                                i_Td = i_Tu - 1
                                Td = T_cont_h2h2[i_Td]

                                k_h2h2_Tu = K_cont_h2h2[i_Tu,i_wave]
                                k_h2h2_Td = K_cont_h2h2[i_Td,i_wave]

                                coeff = (T - Td)/(float(Tu + Td))

                                k_interp[0,i] = k_h2h2_Tu*coeff +  k_h2h2_Td*(1 - coeff)

                            i_Tu_array[i] = i_Tu
                            i_Td_array = i_Tu_array - 1
                            coeff_array[i] = coeff
                    else :
                        zer, = np.where(i_Tu_array == 0)
                        i_Td_array[zer] = np.zeros(zer.size, dtype='int')

                        ex, = np.where(i_Tu_array == end)
                        i_Td_array[ex] = np.ones(ex.size, dtype='int')*end

                        k_h2h2_Tu = K_cont_h2h2[i_Tu_array,i_wave]
                        k_h2h2_Td = K_cont_h2h2[i_Td_array,i_wave]

                        k_interp[1,:] = k_h2h2_Tu*coeff_array + k_h2h2_Td*(1 - coeff_array)

                        k_interp_h2h2[i_bande,:] = k_interp[1,:]*coef + k_interp[0,:]*(1 - coef)

        else :

            if Kcorr == True :

                wave_max = bande_array[i_bande + 1]
                wave_min = bande_array[i_bande]
                zone_wave, = np.where((wavelength_cont_h2h2 >= wave_min)*(wavelength_cont_h2h2 <= wave_max))
                fact = zone_wave.size

                for i_wave in zone_wave :
                    k_h2h2_Tu = K_cont_h2h2[i_Tu_array,i_wave]
                    k_h2h2_Td = K_cont_h2h2[i_Td_array,i_wave]

                    k_interp += k_h2h2_Tu*coeff_array + k_h2h2_Td*(1 - coeff_array)

                    if i_wave == zone_wave[fact-1] :
                        k_interp = k_interp/(float(fact))
                        k_interp_h2h2[i_bande,:] = k_interp

            else :

                zone_wave_up, = np.where(wavelength_cont_h2h2 >= bande_array[i_bande])
                if zone_wave_up.size == 0 :
                    if stop == 0 :
                        i_wave = wavelength_cont_h2h2.size-1

                        k_h2h2_Tu = K_cont_h2h2[i_Tu_array,i_wave]
                        k_h2h2_Td = K_cont_h2h2[i_Td_array,i_wave]

                        k_interp_h2h2[i_bande,:] = k_h2h2_Tu*coeff_array + k_h2h2_Td*(1 - coeff_array)

                        stop += 1
                    else :
                        k_interp_h2h2[i_bande,:] = k_interp_h2h2[i_bande-1,:]
                else :
                    wave_up = wavelength_cont_h2h2[zone_wave_up[0]]
                    wave_down = wavelength_cont_h2h2[zone_wave_up[0]]

                    coef = (bande_array[i_bande] - wave_down)/(float(wave_up + wave_down))

                    i = 0

                    for i_wave in [zone_wave_up[0]-1,zone_wave_up[0]] :
                        k_h2h2_Tu = K_cont_h2h2[i_Tu_array,i_wave]
                        k_h2h2_Td = K_cont_h2h2[i_Td_array,i_wave]

                        k_interp[i,:] = k_h2h2_Tu*coeff_array + k_h2h2_Td*(1 - coeff_array)

                        i += 1

                    k_interp_h2h2[i_bande,:] = k_interp[1,:]*coef + k_interp[0,:]*(1 - coef)

        if Script == True :
            if rank == rank_ref :
                bar.animate(i_bande+1)

    return k_interp_h2h2*100*losch**2


########################################################################################################################


def k_cont_interp_h2he_integration(K_cont_h2he,wavelength_cont_h2he,T_array,bande_array,T_cont_h2he,rank,rank_ref,Kcorr=True,Script=True) :

    losch = 2.6867774e19
    size = T_array.size
    if Kcorr == True :
        dim_bande = bande_array.size - 1
    else :
        dim_bande = bande_array.size
    k_interp_h2he = np.zeros((dim_bande,size))

    T_min = T_cont_h2he[0]
    T_max = T_cont_h2he[T_cont_h2he.size-1]
    end = T_cont_h2he.size - 1
    stop = 0

    if Script == True :
        if rank == rank_ref :
            bar = ProgressBar(dim_bande, 'Continuum H2/He computation progression')

    for i_bande in range(dim_bande) :

        if Kcorr == True :
            k_interp = np.zeros(size)
        else :
            k_interp = np.zeros((2,size))

        if i_bande == 0 :
            i_Tu_array = np.zeros(size, dtype='int')
            coeff_array = np.zeros(size)

            if Kcorr == True :

                wave_max = bande_array[1]
                wave_min = bande_array[0]
                zone_wave, = np.where((wavelength_cont_h2he >= wave_min)*(wavelength_cont_h2he <= wave_max))
                fact = zone_wave.size

                for i_wave in zone_wave :
                    if i_wave == zone_wave[0] :
                        for i in range(size) :
                            T = T_array[i]
                            if T < T_min or T > T_max :
                                if T < T_min :
                                    if T != 0 :
                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[i] = K_cont_h2he[0,i_wave]
                                    else :
                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[i] = 0.
                                if T > T_max :
                                    i_Tu, i_Td, coeff = end, end, 1
                                    k_interp[i] = K_cont_h2he[T_cont_h2he.size-1,i_wave]
                            else :
                                wh, = np.where(T_cont_h2he > T)
                                i_Tu = wh[0]
                                Tu = T_cont_h2he[i_Tu]
                                i_Td = i_Tu - 1
                                Td = T_cont_h2he[i_Td]

                                k_h2he_Tu = K_cont_h2he[i_Tu,i_wave]
                                k_h2he_Td = K_cont_h2he[i_Td,i_wave]

                                coeff = (T - Td)/(float(Tu + Td))

                                k_interp[i] = k_h2he_Tu*coeff +  k_h2he_Td*(1 - coeff)

                            i_Tu_array[i] = i_Tu
                            i_Td_array = i_Tu_array - 1
                            coeff_array[i] = coeff
                    else :
                        zer, = np.where(i_Tu_array == 0)
                        i_Td_array[zer] = np.zeros(zer.size, dtype='int')

                        ex, = np.where(i_Tu_array == end)
                        i_Td_array[ex] = np.ones(ex.size, dtype='int')*end

                        k_h2he_Tu = K_cont_h2he[i_Tu_array,i_wave]
                        k_h2he_Td = K_cont_h2he[i_Td_array,i_wave]

                        k_interp += k_h2he_Tu*coeff_array + k_h2he_Td*(1 - coeff_array)

                    if i_wave == zone_wave[fact-1] :

                        k_interp = k_interp/(float(fact))
                        k_interp_h2he[i_bande,:] = k_interp

            else :

                zone_wave_up, = np.where(wavelength_cont_h2he >= bande_array[i_bande])
                wave_up = wavelength_cont_h2he[zone_wave_up[0]]
                wave_down = wavelength_cont_h2he[zone_wave_up[0]-1]

                coef = (bande_array[i_bande] - wave_down)/(float(wave_up + wave_down))

                for i_wave in [zone_wave_up[0]-1,zone_wave_up[0]] :
                    if i_wave == zone_wave_up[0]-1 :
                        if i_wave == -1 :
                            i_wave = 0
                        for i in range(size) :
                            T = T_array[i]
                            if T < T_min or T > T_max :
                                if T < T_min :
                                    if T != 0 :
                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[0,i] = K_cont_h2he[0,i_wave]
                                    else :
                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[0,i] = 0.
                                if T > T_max :
                                    i_Tu, i_Td, coeff = end, end, 1
                                    k_interp[0,i] = K_cont_h2he[T_cont_h2he.size-1,i_wave]
                            else :
                                wh, = np.where(T_cont_h2he > T)
                                i_Tu = wh[0]
                                Tu = T_cont_h2he[i_Tu]
                                i_Td = i_Tu - 1
                                Td = T_cont_h2he[i_Td]

                                k_h2he_Tu = K_cont_h2he[i_Tu,i_wave]
                                k_h2he_Td = K_cont_h2he[i_Td,i_wave]

                                coeff = (T - Td)/(float(Tu + Td))

                                k_interp[0,i] = k_h2he_Tu*coeff +  k_h2he_Td*(1 - coeff)

                            i_Tu_array[i] = i_Tu
                            i_Td_array = i_Tu_array - 1
                            coeff_array[i] = coeff
                    else :
                        zer, = np.where(i_Tu_array == 0)
                        i_Td_array[zer] = np.zeros(zer.size, dtype='int')

                        ex, = np.where(i_Tu_array == end)
                        i_Td_array[ex] = np.ones(ex.size, dtype='int')*end

                        k_h2he_Tu = K_cont_h2he[i_Tu_array,i_wave]
                        k_h2he_Td = K_cont_h2he[i_Td_array,i_wave]

                        k_interp[1,:] = k_h2he_Tu*coeff_array + k_h2he_Td*(1 - coeff_array)

                        k_interp_h2he[i_bande,:] = k_interp[1,:]*coef + k_interp[0,:]*(1 - coef)

        else :

            if Kcorr == True :

                wave_max = bande_array[i_bande + 1]
                wave_min = bande_array[i_bande]
                zone_wave, = np.where((wavelength_cont_h2he >= wave_min)*(wavelength_cont_h2he <= wave_max))
                fact = zone_wave.size

                for i_wave in zone_wave :
                    k_h2he_Tu = K_cont_h2he[i_Tu_array,i_wave]
                    k_h2he_Td = K_cont_h2he[i_Td_array,i_wave]

                    k_interp += k_h2he_Tu*coeff_array + k_h2he_Td*(1 - coeff_array)

                    if i_wave == zone_wave[fact-1] :

                        k_interp = k_interp/(float(fact))
                        k_interp_h2he[i_bande,:] = k_interp

            else :

                zone_wave_up, = np.where(wavelength_cont_h2he >= bande_array[i_bande])

                if zone_wave_up.size == 0 :
                    if stop == 0 :
                        i_wave = wavelength_cont_h2he.size-1

                        k_h2he_Tu = K_cont_h2he[i_Tu_array,i_wave]
                        k_h2he_Td = K_cont_h2he[i_Td_array,i_wave]

                        k_interp_h2he[i_bande,:] = k_h2he_Tu*coeff_array + k_h2he_Td*(1 - coeff_array)

                        stop += 1
                    else :
                        k_interp_h2he[i_bande,:] = k_interp_h2he[i_bande-1,:]
                else :
                    wave_up = wavelength_cont_h2he[zone_wave_up[0]]
                    wave_down = wavelength_cont_h2he[zone_wave_up[0]]

                    coef = (bande_array[i_bande] - wave_down)/(float(wave_up + wave_down))

                    i = 0

                    for i_wave in [zone_wave_up[0]-1,zone_wave_up[0]] :
                        k_h2he_Tu = K_cont_h2he[i_Tu_array,i_wave]
                        k_h2he_Td = K_cont_h2he[i_Td_array,i_wave]

                        k_interp[i,:] = k_h2he_Tu*coeff_array + k_h2he_Td*(1 - coeff_array)

                        i += 1

                    k_interp_h2he[i_bande,:] = k_interp[1,:]*coef + k_interp[1,:]*(1 - coef)

        if Script == True :
            if rank == rank_ref :
                bar.animate(i_bande+1)

    return k_interp_h2he*100*losch**2


########################################################################################################################


def k_cont_interp_spespe_integration(K_cont_spespe,wavelength_cont_spespe,T_array,bande_array,T_cont_spespe,rank,rank_ref,species,Kcorr=True,H2O=False,Script=True) :

    losch = 2.6867774e19
    size = T_array.size
    if Kcorr == True :
        dim_bande = bande_array.size - 1
    else :
        dim_bande = bande_array.size
    k_interp_spespe = np.zeros((dim_bande,size))

    T_min = T_cont_spespe[0]
    T_max = T_cont_spespe[T_cont_spespe.size-1]
    end = T_cont_spespe.size - 1
    stop = 0

    if Script == True :
        if rank == rank_ref :
            bar = ProgressBar(dim_bande,'Continuum %s computation progression'%(species))

    for i_bande in range(dim_bande) :

        if Kcorr == True :
            k_interp = np.zeros(size)
        else :
            k_interp = np.zeros((2,size))

        if i_bande == 0 :

            i_Tu_array = np.zeros(size, dtype='int')
            coeff_array = np.zeros(size)

            if Kcorr == True :

                wave_max = bande_array[1]
                wave_min = bande_array[0]
                zone_wave, = np.where((wavelength_cont_spespe >= wave_min)*(wavelength_cont_spespe <= wave_max))
                fact = zone_wave.size

                for i_wave in zone_wave :
                    if i_wave == zone_wave[0] :
                        for i in range(size) :
                            T = T_array[i]
                            if T < T_min or T > T_max :
                                if T < T_min :
                                    if T != 0 :
                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[i] = K_cont_spespe[0,i_wave]
                                    else :
                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[i] = 0.
                                if T > T_max :
                                    i_Tu, i_Td, coeff = end, end, 1
                                    k_interp[i] = K_cont_spespe[T_cont_spespe.size-1,i_wave]
                            else :
                                wh, = np.where(T_cont_spespe > T)
                                i_Tu = wh[0]
                                Tu = T_cont_spespe[i_Tu]
                                i_Td = i_Tu - 1
                                Td = T_cont_spespe[i_Td]

                                k_spespe_Tu = K_cont_spespe[i_Tu,i_wave]
                                k_spespe_Td = K_cont_spespe[i_Td,i_wave]

                                coeff = (T - Td)/(float(Tu + Td))

                                k_interp[i] = k_spespe_Tu*coeff +  k_spespe_Td*(1 - coeff)

                            i_Tu_array[i] = i_Tu
                            i_Td_array = i_Tu_array - 1
                            coeff_array[i] = coeff
                    else :
                        zer, = np.where(i_Tu_array == 0)
                        i_Td_array[zer] = np.zeros(zer.size, dtype='int')

                        ex, = np.where(i_Tu_array == end)
                        i_Td_array[ex] = np.ones(ex.size, dtype='int')*end

                        k_spespe_Tu = K_cont_spespe[i_Tu_array,i_wave]
                        k_spespe_Td = K_cont_spespe[i_Td_array,i_wave]

                        k_interp += k_spespe_Tu*coeff_array + k_spespe_Td*(1 - coeff_array)

                    if i_wave == zone_wave[fact-1] :

                        k_interp = k_interp/(float(fact))
                        k_interp_spespe[i_bande,:] = k_interp

            else :

                zone_wave_up, = np.where(wavelength_cont_spespe >= bande_array[i_bande])
                wave_up = wavelength_cont_spespe[zone_wave_up[0]]
                wave_down = wavelength_cont_spespe[zone_wave_up[0]-1]

                coef = (bande_array[i_bande] - wave_down)/(float(wave_up + wave_down))

                for i_wave in [zone_wave_up[0]-1,zone_wave_up[0]] :
                    if i_wave == zone_wave_up[0]-1 :
                        if i_wave == -1 :
                            i_wave = 0

                        for i in range(size) :
                            T = T_array[i]
                            if T < T_min or T > T_max :
                                if T < T_min :
                                    if T != 0 :
                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[0,i] = K_cont_spespe[0,i_wave]
                                    else :
                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[0,i] = 0.
                                if T > T_max :
                                    i_Tu, i_Td, coeff = end, end, 1
                                    k_interp[0,i] = K_cont_spespe[T_cont_spespe.size-1,i_wave]
                            else :
                                wh, = np.where(T_cont_spespe > T)
                                i_Tu = wh[0]
                                Tu = T_cont_spespe[i_Tu]
                                i_Td = i_Tu - 1
                                Td = T_cont_spespe[i_Td]

                                k_spespe_Tu = K_cont_spespe[i_Tu,i_wave]
                                k_spespe_Td = K_cont_spespe[i_Td,i_wave]

                                coeff = (T - Td)/(float(Tu + Td))

                                k_interp[0,i] = k_spespe_Tu*coeff +  k_spespe_Td*(1 - coeff)

                            i_Tu_array[i] = i_Tu
                            i_Td_array = i_Tu_array - 1
                            coeff_array[i] = coeff
                    else :
                        zer, = np.where(i_Tu_array == 0)
                        i_Td_array[zer] = np.zeros(zer.size, dtype='int')

                        ex, = np.where(i_Tu_array == end)
                        i_Td_array[ex] = np.ones(ex.size, dtype='int')*end

                        k_spespe_Tu = K_cont_spespe[i_Tu_array,i_wave]
                        k_spespe_Td = K_cont_spespe[i_Td_array,i_wave]

                        k_interp[1,:] = k_spespe_Tu*coeff_array + k_spespe_Td*(1 - coeff_array)

                        k_interp_spespe[i_bande,:] = k_interp[1,:]*coef + k_interp[0,:]*(1 - coef)

        else :

            if Kcorr == True :

                wave_max = bande_array[i_bande + 1]
                wave_min = bande_array[i_bande]
                zone_wave, = np.where((wavelength_cont_spespe >= wave_min)*(wavelength_cont_spespe <= wave_max))
                fact = zone_wave.size

                for i_wave in zone_wave :
                    k_spespe_Tu = K_cont_spespe[i_Tu_array,i_wave]
                    k_spespe_Td = K_cont_spespe[i_Td_array,i_wave]

                    k_interp += k_spespe_Tu*coeff_array + k_spespe_Td*(1 - coeff_array)

                    if i_wave == zone_wave[fact-1] :

                        k_interp = k_interp/(float(fact))
                        k_interp_spespe[i_bande,:] = k_interp
            else :
                zone_wave_up, = np.where(wavelength_cont_spespe >= bande_array[i_bande])
                if zone_wave_up.size == 0 :

                    if stop == 0 :
                        i_wave = wavelength_cont_spespe.size-1

                        k_spespe_Tu = K_cont_spespe[i_Tu_array,i_wave]
                        k_spespe_Td = K_cont_spespe[i_Td_array,i_wave]

                        k_interp_spespe[i_bande,:] = k_spespe_Tu*coeff_array + k_spespe_Td*(1 - coeff_array)

                        stop += 1
                    else :
                        k_interp_spespe[i_bande,:] = k_interp_spespe[i_bande-1,:]
                else :
                    wave_up = wavelength_cont_spespe[zone_wave_up[0]]
                    wave_down = wavelength_cont_spespe[zone_wave_up[0]]
                    coef = (bande_array[i_bande] - wave_down)/(float(wave_up + wave_down))

                    i = 0

                    for i_wave in [zone_wave_up[0]-1,zone_wave_up[0]] :
                        k_spespe_Tu = K_cont_spespe[i_Tu_array,i_wave]
                        k_spespe_Td = K_cont_spespe[i_Td_array,i_wave]

                        k_interp[i,:] = k_spespe_Tu*coeff_array + k_spespe_Td*(1 - coeff_array)

                        i += 1

                    k_interp_spespe[i_bande,:] = k_interp[1,:]*coef + k_interp[1,:]*(1 - coef)

        if Script == True :
            if rank == rank_ref :
                bar.animate(i_bande+1)

    if H2O == True :
        return k_interp_spespe*0.0001
    else :
        return k_interp_spespe*100*losch**2
