from pyopacities import *

########################################################################################################################
########################################################################################################################

"""
    PYCONVERT

    Cette bibliotheque contient l'ensemble des routines permettant l'interpolation des sections efficaces ou des
    k coefficients dans les atmospheres qui nous interessent. Certaines fonctions sont dupliquees en fonction de la
    presence ou non d'un marqueur et de l'utilisation de sections efficaces plutot que de k coefficients. Ces routines
    sont executees par le script Parameters (qui peut etre precede d'un acronyme de l'exoplanete etudiee, par exemple :
    GJParameters).

    Les tableaux d'opacites ainsi generes sont directement utilises dans les modules de pytransfert et de pyremind afin
    de resoudre le transfert radiatif. Certaines fonctions sont utilisees directement dans la routine de transfert 1D.
    La cle de voute de cette bibliotheque, a savoir convertator, permet la generation de l'ensemble des opacites
    (moleculaire, continuum, diffusion Rayleigh et diffusion de Mie)

    Version : 6.3

    Recentes mise a jour :

    >> Modification totale de l'identification des couples P,T ou P,T,Q dans les simulations 3D dans les routines
    convertator

    Date de derniere modification : 07.07.2016

    >> Refonte de convertator et convertator1D

    Date de derniere modification : 03.04.2018

"""

########################################################################################################################
########################################################################################################################


def convertator (P,T,gen,c_species,Q,compo,ind_active,ind_cross,K,K_cont,Qext,P_sample,T_sample,\
                 Q_sample,bande_sample,bande_cloud,x_step,r_eff,r_cloud,rho_p,name,t,phi_rot,phi_obli,n_species,domain,ratio,directory,name_exo,reso_long,reso_lat,\
                 name_source,Tracer=False,Molecular=False,Continuum=False,Clouds=False,Scattering=False,Kcorr=True,Optimal=False,Script=True) :

    rank, rank_ref = 0, 0

    if Clouds == True :

        (c_number,a,b,c) = np.shape(gen)

    P_rmd, T_rmd, Q_rmd, gen_cond_rmd, composit_rmd, wher, indices, liste = sort_set_param(P,T,Q,gen,compo,rank,Tracer,Clouds)

    p = np.log10(P_rmd)
    p_min = int(np.amin(p))-1
    p_max = int(np.amax(p))+1
    rmind = np.zeros((2,p_max - p_min+1),dtype='int')
    rmind[0,0] = 0

    for i in xrange(p_max - p_min) :

        wh, = np.where((p >= p_min + i)*(p <= p_min + (i+1)))

        if wh.size != 0 :
            rmind[0,i+1] = wh[wh.size-1]
            rmind[1,i] = p_min + i
        else :
            rmind[0,i+1] = 0
            rmind[1,i] = p_min + i

    rmind[1,i+1] = p_max

    zero, = np.where(P_rmd == 0.)

    del P,T,Q,gen,compo

    K = np.load(K)

    if Kcorr == True :
        dim_T,dim_P,dim_x,dim_bande,dim_gauss = np.shape(K)
    else :
        K = K[ind_cross]
        dim_T,dim_P,dim_x,dim_bande = np.shape(K)

    if Kcorr == True :
        np.save("%s%s/P_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s.npy"\
        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),P_rmd)
        np.save("%s%s/T_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s.npy"\
        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),T_rmd)
        np.save("%s%s/rmind_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s.npy"\
        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),rmind)
        if Tracer == True :
            np.save("%s%s/Q_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),Q_rmd)
    else :
        np.save("%s%s/P_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),P_rmd)
        np.save("%s%s/T_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),T_rmd)
        np.save("%s%s/rmind_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy" \
        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),rmind)
        if Tracer ==True :
            np.save("%s%s/Q_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),Q_rmd)

    if Clouds == True :

        if Kcorr == True :
            np.save("%s%s/gen_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),gen_cond_rmd)
        else :
            np.save("%s%s/gen_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),gen_cond_rmd)

    if Kcorr == True :
        np.save("%s%s/compo_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),composit_rmd)
    else :
        np.save("%s%s/compo_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),composit_rmd)

    zero, = np.where(P_rmd == 0.)

    if Molecular == True :

        if Kcorr == True :
            if Tracer == True :
                t_size,p_size,q_size,dim_bande,dim_gauss = np.shape(K)
            else :
                t_size,p_size,dim_bande,dim_gauss = np.shape(K)
        else :
            K = K[ind_cross]
            np_size,p_size,t_size,dim_bande = np.shape(K)

        if Kcorr == True :
            if Tracer == False :
                k_rmd = Ksearcher(T_rmd,P_rmd,dim_gauss,dim_bande,K,P_sample,T_sample,rank,rank_ref,Kcorr,Optimal,True)
                np.save("%s%s/k_corr_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),k_rmd)
            else :
                k_rmd = Ksearcher_M(T_rmd,P_rmd,Q_rmd,dim_gauss,dim_bande,K,P_sample,T_sample,Q_sample,rank,rank_ref,Kcorr,Optimal,True)
                np.save("%s%s/k_corr_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),k_rmd)

            if Script == True :
                print "Ksearcher finished with success"

        else :
            compo_active = composit_rmd[ind_active,:]
            k_rmd = Ssearcher(T_rmd,P_rmd,compo_active,K,P_sample,T_sample,rank,rank_ref,Kcorr,Optimal,True)

            if Optimal == False :
                np.save("%s%s/k_cross_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                    %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),k_rmd)
            else :
                np.save("%s%s/k_cross_opt_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                    %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),k_rmd)

            if Script == True :
                print "Ssearcher finished with success"

        del k_rmd,K

    else :

        if Script == True :
            print 'There is no molecular absoprtion'
        del K

    if Continuum == True :

        if Kcorr == True :
            dim_cont = bande_sample.size-1
        else :
            dim_cont = bande_sample.size
        cont_species = K_cont.species
        H2, He, Other = H2HeO(cont_species)

        bar = ProgressBar(rmind[0].size-1,'Continuum %s computation progression : ')

        for lay in range(rmind[0].size-1) :

            decont = 0
            P_rmind = P_rmd[rmind[0,lay]:rmind[0,lay+1]]
            T_rmind = T_rmd[rmind[0,lay]:rmind[0,lay+1]]
            composit_rmind = composit_rmd[:,rmind[0,lay]:rmind[0,lay+1]]
            amagat = 2.69578e-3*P_rmind/T_rmind

            k_cont_rmd = np.zeros((dim_cont,P_rmind.size))

            if H2 == True :

                decont += 1
                if lay == 0 :
                    K_cont_h2h2 = np.load('%s%s/k_cont_%s.npy'%(directory,name_source,K_cont.associations[0]))
                    K_cont_nu_h2h2 = np.load('%s%s/k_cont_nu_%s.npy'%(directory,name_source,K_cont.associations[0]))
                    T_cont_h2h2 = np.load('%s%s/T_cont_%s.npy'%(directory,name_source,K_cont.associations[0]))

                k_interp_h2h2 = k_cont_interp_h2h2_integration(K_cont_h2h2,K_cont_nu_h2h2,\
                                                T_rmind,bande_sample,T_cont_h2h2,rank,rank_ref,Kcorr,False)

                amagat_h2h2 = amagat*composit_rmind[0,:]

                for i_bande in range(dim_cont) :
                    k_cont_rmd[i_bande,:] = amagat_h2h2**2*k_interp_h2h2[i_bande,:]

                del amagat_h2h2,k_interp_h2h2

            if He == True :

                decont += 1
                if lay == 0 :
                    K_cont_h2he = np.load('%s%s/k_cont_%s.npy'%(directory,name_source,K_cont.associations[1]))
                    K_cont_nu_h2he = np.load('%s%s/k_cont_nu_%s.npy'%(directory,name_source,K_cont.associations[1]))
                    T_cont_h2he = np.load('%s%s/T_cont_%s.npy'%(directory,name_source,K_cont.associations[1]))

                k_interp_h2he = k_cont_interp_h2he_integration(K_cont_h2he,K_cont_nu_h2he,\
                                                T_rmind,bande_sample,T_cont_h2he,rank,rank_ref,Kcorr,False)

                amagat_self = amagat*composit_rmind[0,:]
                amagat_foreign = amagat*composit_rmind[1,:]

                for i_bande in range(dim_bande) :
                    k_cont_rmd[i_bande,:] += amagat_foreign*amagat_self*k_interp_h2he[i_bande,:]

                del amagat_foreign,k_interp_h2he

            if Other == False :

                if Kcorr == True :
                    np.save("%s%s/k_cont_all_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                    %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain,lay),k_cont_rmd)
                else :
                    np.save("%s%s/k_cont_all_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                    %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain,lay),k_cont_rmd)
                del k_cont_rmd

            else :
                for i_cont in range(decont,cont_species.size) :

                    K_cont_spespe = np.load('%s%s/k_cont_%s.npy'%(directory,name_source,K_cont.associations[i_cont]))
                    K_cont_nu_spespe = np.load('%s%s/k_cont_nu_%s.npy'%(directory,name_source,K_cont.associations[i_cont]))
                    T_cont_spespe = np.load('%s%s/T_cont_%s.npy'%(directory,name_source,K_cont.associations[i_cont]))

                    if cont_species[i_cont] != 'H2O' and cont_species[i_cont] != 'H2Os':
                        wh_c, = np.where(n_species == cont_species[i_cont])
                        amagat_spefor = amagat*composit_rmind[0,:]
                        amagat_speself = amagat*composit_rmind[wh_c[0],:]
                        amagat_spe = amagat_spefor*amagat_speself
                    else :
                        wh_c, = np.where(n_species == 'H2O')
                        H2O = True
                        N_mol = P_rmind/(k_B*T_rmind)
                        if cont_species[i_cont] == 'H2O' :
                            amagat_spe = amagat*(1.-composit_rmind[wh_c[0],:])*composit_rmind[wh_c[0],:]*N_mol
                        if cont_species[i_cont] == 'H2Os' :
                            amagat_spe = amagat*composit_rmind[wh_c[0],:]**2*N_mol

                    k_interp_spespe = k_cont_interp_spespe_integration(K_cont_spespe,K_cont_nu_spespe,\
                                    T_rmind,bande_sample,T_cont_spespe,rank,rank_ref,K_cont.associations[i_cont],Kcorr,H2O,False)

                    for i_bande in range(dim_bande) :

                        k_cont_rmd[i_bande,:] += amagat_spe*k_interp_spespe[i_bande,:]

                    del amagat_spe,k_interp_spespe

                    if decont == cont_species.size -1 :
                        if Kcorr == True :
                            np.save("%s%s/k_cont_all_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                            %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain,lay),k_cont_rmd)
                        else :
                            np.save("%s%s/k_cont_all_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                            %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain,lay),k_cont_rmd)
                        del k_cont_rmd
                    decont += 1

            bar.animate(lay+1)

        del K_cont_h2h2, K_cont_h2he, K_cont_spespe

        k_cont_rmd = np.zeros((P_rmd.size,dim_cont))
        for lay in range(rmind[0].size-1) :

            if Kcorr == True :
                k = np.load("%s%s/k_cont_all_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain,lay))
                k_cont_rmd[rmind[0,lay]:rmind[0,lay+1],:] = np.transpose(k)
                os.remove("%s%s/k_cont_all_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain,lay))
            else :
                k = np.load("%s%s/k_cont_all_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain,lay))
                k_cont_rmd[rmind[0,lay]:rmind[0,lay+1],:] = np.transpose(k)
                os.remove("%s%s/k_cont_all_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%i.npy"\
                %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain,lay))
            del k

        if Kcorr == True :
            np.save("%s%s/k_cont_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s.npy"\
                    %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),k_cont_rmd)
        else :
            np.save("%s%s/k_cont_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                    %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),k_cont_rmd)
        del k_cont_rmd

        if Script == True :
            print "Integration of the continuum finished with success"

    else :
        if Script == True :
            print "There is no continuum"

    x_mol_species = composit_rmd[0:n_species.size,:]

    if Scattering == True :

        k_sca_rmd = Rayleigh_scattering(P_rmd,T_rmd,bande_sample,x_mol_species,n_species,zero,rank,rank_ref,Kcorr,True)

        if Kcorr == True :
            np.save("%s%s/k_sca_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),k_sca_rmd)
        else :
            np.save("%s%s/k_sca_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),k_sca_rmd)

        print "Rayleigh_scattering finished with success"

        del k_sca_rmd,x_mol_species

    else :

        if Script == True :
            print "There is no scattering"

    if Clouds == True :

        zer_n, = np.where(bande_sample != 0.)
        zer_p, = np.where(bande_sample == 0.)

        if Kcorr == True :
            wl = np.zeros(bande_sample.size-1)
            for i in range(bande_sample.size - 1) :
                wl[i] = (bande_sample[i+1] + bande_sample[i])/2.
            wl = 1./wl
        else :
            wl = np.zeros(bande_sample.size)
            wl[zer_n] = 1./(100*bande_sample[zer_n])
            wl[zer_p] = 0.

        n_size,rmd_size = np.shape(composit_rmd)
        c_number = c_species.size

        Qext = np.load(Qext)

        k_cloud_rmd = np.zeros((c_number,P_rmd.size,dim_bande))

        for c_num in range(c_number) :

            k_cloud_rmd[c_num] = cloud_scattering(Qext[c_num,:,:],bande_cloud,P_rmd,T_rmd,wl,composit_rmd[n_size-1,:],rho_p[c_num],gen_cond_rmd[c_num,:],r_eff[c_num],r_cloud,zero,rank,rank_ref,True)

        if Script == True :
            print "Cloud_scattering finished with success, process are beginning to save data remind"

        r_enn = ''
        for i_r in range(r_eff.size) :
            if i_r != r_eff.size-1 :
                r_enn += '%.2f_'%(r_eff[i_r]*10**6)
            else :
                r_enn += '%.2f'%(r_eff[i_r]*10**6)

        if Kcorr == True :
            np.save("%s%s/k_cloud_%ix%i_%s_%i_%ix%i_%i_rmd_%.2f_%.2f_%s_%s.npy" \
                %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,r_enn,domain),k_cloud_rmd)
        else :
            np.save("%s%s/k_cloud_%ix%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%s.npy" \
                %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,r_enn,domain),k_cloud_rmd)

        del Qext,k_cloud_rmd

    else :

        if Script == True :
            print "There is no clouds"



########################################################################################################################


def convertator1D (P_col,T_col,gen_col,c_species,Q_col,compo_col,ind_active,K,K_cont,Qext,P_sample,\
                   T_sample,Q_sample,bande_sample,bande_cloud,x_step,r_eff,r_cloud,rho_p,name,t,phi_rot,\
                   n_species,domain,ratio,directory,Tracer=False,Continuum=False,Scattering=False,Clouds=False,\
                   Kcorr=True,Optimal=False,Script=True) :

    rank, rank_ref = 0, 0

    c_number = c_species.size

    if Kcorr == True :
        dim_P,dim_T,dim_x,dim_bande,dim_gauss = np.shape(K)
    else :
        n_spe,dim_P,dim_T,dim_bande = np.shape(K)
        dim_gauss = 0

    no_zero, = np.where(P_col != 0)
    T_rmd = T_col[no_zero]
    P_rmd = P_col[no_zero]
    compo_rmd = compo_col[:,no_zero]
    zero = np.array([])

    if Kcorr == True :

        if Tracer == True :
            Q_rmd = Q_col[no_zero]
            k_rmd = Ksearcher_M(T_rmd,P_rmd,Q_rmd,dim_gauss,dim_bande,K,P_sample,T_sample,Q_sample,rank,rank_ref,Kcorr,Optimal,Script)
            if Script == True :
                print "Ssearcher_M finished with success"
        else :
            k_rmd = Ksearcher(T_rmd,P_rmd,dim_gauss,dim_bande,K,P_sample,T_sample,rank,rank_ref,Kcorr,Optimal,Script)
            if Script == True :
                print "Ksearcher finished with success"
        Q_rmd = np.array([])

    else :
        compo = compo_col[ind_active,:]
        k_rmd = Ssearcher(T_rmd,P_rmd,compo,K,P_sample,T_sample,rank,rank_ref,Kcorr,Optimal,Script)
        if Script == True :
            print "Ssearcher finished with success"
        Q_rmd = Q_col

    if Continuum == True :

        cont_species = K_cont.species
        H2, He, Other = H2HeO(cont_species)

        decont = 0
        amagat = 2.69578e-3*P_rmd/T_rmd
        k_cont_rmd = np.zeros((dim_bande,P_rmd.size))

        if H2 == True :

            decont += 1
            K_cont_h2h2 = np.load('%sSource/k_cont_%s.npy'%(directory,K_cont.associations[0]))
            K_cont_nu_h2h2 = np.load('%sSource/k_cont_nu_%s.npy'%(directory,K_cont.associations[0]))
            T_cont_h2h2 = np.load('%sSource/T_cont_%s.npy'%(directory,K_cont.associations[0]))

            k_interp_h2h2 = k_cont_interp_h2h2_integration(K_cont_h2h2,K_cont_nu_h2h2,\
                                        T_rmd,bande_sample,T_cont_h2h2,rank,rank_ref,Kcorr,Script)

            amagat_h2h2 = amagat*compo_rmd[0,:]

            for i_bande in range(dim_bande) :

                k_cont_rmd[i_bande,:] = amagat_h2h2**2*k_interp_h2h2[i_bande,:]

            del amagat_h2h2,k_interp_h2h2

        if He == True :

            decont += 1
            K_cont_h2he = np.load('%sSource/k_cont_%s.npy'%(directory,K_cont.associations[1]))
            K_cont_nu_h2he = np.load('%sSource/k_cont_nu_%s.npy'%(directory,K_cont.associations[1]))
            T_cont_h2he = np.load('%sSource/T_cont_%s.npy'%(directory,K_cont.associations[1]))

            k_interp_h2he = k_cont_interp_h2he_integration(K_cont_h2he,K_cont_nu_h2he,\
                                    T_rmd,bande_sample,T_cont_h2he,rank,rank_ref,Kcorr,Script)

            amagat_self = amagat*compo_rmd[0,:]
            amagat_foreign = amagat*compo_rmd[1,:]

            for i_bande in range(dim_bande) :

                k_cont_rmd[i_bande,:] += amagat_foreign*amagat_self*k_interp_h2he[i_bande,:]

            del amagat_foreign,amagat_self,k_interp_h2he

        if Other == True :

            for i_cont in range(decont,cont_species.size) :

                K_cont_spespe = np.load('%sSource/k_cont_%s.npy'%(directory,K_cont.associations[i_cont]))
                K_cont_nu_spespe = np.load('%sSource/k_cont_nu_%s.npy'%(directory,K_cont.associations[i_cont]))
                T_cont_spespe = np.load('%sSource/T_cont_%s.npy'%(directory,K_cont.associations[i_cont]))

                if cont_species[i_cont] != 'H2O' and cont_species[i_cont] != 'H2Os':
                    wh_c, = np.where(n_species == cont_species[i_cont])
                    amagat_spefor = amagat*compo_rmd[0,:]
                    amagat_speself = amagat*compo_rmd[wh_c[0],:]
                    amagat_spe = amagat_spefor*amagat_speself
                else :
                    wh_c, = np.where(n_species == 'H2O')
                    H2O = True
                    N_mol = P_rmd/(k_B*T_rmd)
                    if cont_species[i_cont] == 'H2O' :
                        amagat_spe = amagat*(1.-compo_rmd[wh_c[0],:])*compo_rmd[wh_c[0],:]*N_mol
                    if cont_species[i_cont] == 'H2Os' :
                        amagat_spe = amagat*compo_rmd[wh_c[0],:]**2*N_mol

                k_interp_spespe = k_cont_interp_spespe_integration(K_cont_spespe,K_cont_nu_spespe,\
                            T_rmd,bande_sample,T_cont_spespe,rank,rank_ref,K_cont.associations[i_cont],Kcorr,H2O,Script)

                for i_bande in range(dim_bande) :

                    k_cont_rmd[i_bande,:] += amagat_spe*k_interp_spespe[i_bande,:]

                del amagat_spe,k_interp_spespe

        k_cont_rmd = np.transpose(k_cont_rmd)

    else :

        if Script == True :
            print 'There is no continuum'

        k_cont_rmd = np.zeros((T_rmd.size,dim_bande))

    order,len = np.shape(compo_rmd)
    x_mol_species = compo_rmd[0:order-1,:]

    if Scattering == True :

        k_sca_rmd = Rayleigh_scattering(P_rmd,T_rmd,bande_sample,x_mol_species,n_species,zero,rank,rank_ref,Kcorr,False,Script)

        if Script == True :
            print "Rayleigh_scattering finished with success"

    else :

        k_sca_rmd = np.zeros((T_rmd.size,dim_bande))

    if Clouds == True :

        gen_rmd = gen_col[:,no_zero]
        zer_n, = np.where(bande_sample != 0.)
        zer_p, = np.where(bande_sample == 0.)

        if Kcorr == True :
            wl = np.zeros(bande_sample.size-1)
            for i in range(bande_sample.size - 1) :
                wl[i] = (bande_sample[i+1] + bande_sample[i])/2.
            wl = 1./wl
        else :
            wl = np.zeros(bande_sample.size)
            wl[zer_n] = 1./(100*bande_sample[zer_n])
            wl[zer_p] = 0.

        n_size,rmd_size = np.shape(compo_rmd)

        k_cloud_rmd = np.zeros((c_number,P_rmd.size,dim_bande))

        for c_num in range(c_number) :

            k_cloud_rmd[c_num,:,:] = cloud_scattering(Qext[c_num,:,:],bande_cloud,P_rmd,T_rmd,wl,compo_rmd[n_size-1,:],rho_p[c_num],gen_rmd[c_num,:],r_eff[c_num],r_cloud,zero,rank,rank_ref,Script)

        if Script == True :
            print "Cloud_scattering finished with success, process are beginning to save data remind \n"

    else :

        if Script == True :
            print "There is no clouds"

        k_cloud_rmd = np.zeros((T_rmd.size,dim_bande))

    return P_rmd,T_rmd,Q_rmd,k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd


