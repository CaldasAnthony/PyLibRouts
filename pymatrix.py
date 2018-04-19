from pyfunction import *


########################################################################################################################
########################################################################################################################

"""
    ALTITUDE_LINE_ARRAY2D

    Cette fonction genere les profils en pression, temperature et en fraction molaire pour un rayon incident qui se
    propagerait rectilignement a travers une atmosphere. Les effets de refraction ou de diffusion ne sont pas pris en
    compte dans cette fonction. Elle effectue une interpolation a partir des donnees produites par le LMDZ_GCM a la
    resolution adoptee pour la grille de transmitance. Pour ne pas alourdir l'interpolation, la fonction ne conserve
    que les donnees utiles et extrapole sur les coordonnees realistes des proprietes enregistrees.

    Elle retourne les donnees necessaires au calcul de transfert radiatif, completee par la fonction k_correlated_interp
    qui se charge de produire un tableau d'opacite a partir duquel la profondeur optique locale est estimee.

"""

########################################################################################################################
########################################################################################################################


def altitude_line_array2D_cyl_optimized_correspondance (r_line,theta_line,dx_grid,order_grid,Rp,h,P,T,Q_vap,\
                                    r_step,x_step,lim_alt,Tracer=False,Clouds=False,Cut=False) :

    zone, = np.where(order_grid >= 0)

    D = np.nansum(dx_grid[zone])

    if Tracer == True :
        Q_vap_ref = np.zeros(zone.size)

    T_ref = T[r_line,theta_line,order_grid[zone]]
    P_ref = P[r_line,theta_line,order_grid[zone]]

    if Tracer == True :
        Q_vap_ref =Q_vap[r_line,theta_line,order_grid[zone]]

    dx_ref = dx_grid[zone]

    Cn_mol_ref = P_ref/(R_gp*T_ref)*N_A

    if Cut == True :

        zero, = np.where(T_ref == 0)

        D -= 2*np.nansum(dx_grid[zero])
        h = lim_alt*1000.

    l = (np.sqrt((Rp+h)**2 - (Rp+r_line*r_step)**2)*2 - D)/2.

    # Pinter est en Pa, tandis que Cn_mol_inter est deja converti en densite moleculaire (m^-3)

    if Tracer == True :

        return zone,l,dx_ref,Cn_mol_ref,T_ref,P_ref,Q_vap_ref

    else :

        return zone,l,dx_ref,Cn_mol_ref,T_ref,P_ref


########################################################################################################################


def altitude_line_array1D_cyl_optimized_correspondance (r_line,theta_line,dx_grid,alt_grid,order_grid,Rp,h,P_col,T_col,\

                                Q_vap_col,r_step,x_step,lim_alt,Tracer=False) :

    zone, = np.where(dx_grid >= 0)

    D = np.nansum(dx_grid[zone])

    T_ref = T_col[alt_grid[order_grid[zone]]]
    P_ref = P_col[alt_grid[order_grid[zone]]]

    if Tracer == True :
        Q_vap_ref = Q_vap_col[alt_grid[order_grid[zone]]]

    dx_ref = dx_grid[zone]

    zero, = np.where(T_ref == 0)
    no_zero, = np.where(T_ref != 0)

    Cn_mol_ref = np.zeros(P_ref.size)
    Cn_mol_ref[no_zero] = P_ref[no_zero]/(R_gp*T_ref[no_zero])*N_A

    if zero.size != 0 :

        D -= 2*np.nansum(dx_grid[zero])
        h = lim_alt*1000.

    l = (np.sqrt((Rp+h)**2 - (Rp+r_line*r_step)**2)*2 - D)/2.

    # Pinter est en Pa, tandis que Cn_mol_inter est deja converti en densite moleculaire (m^-3)

    if Tracer == True :

        return zone,l,dx_ref,Cn_mol_ref,T_ref,P_ref,Q_vap_ref

    else :

        return zone,l,dx_ref,Cn_mol_ref,T_ref,P_ref

########################################################################################################################
########################################################################################################################

"""
    ATMOSPHERIC_MATRIX_EARTH

    Produit les matrices cylindriques de temperature, pression, fraction molaire, fraction massique, de concentration
    moalire et de concentration massique a la resolution adoptee par la matrice de reference.

    A ameliorer en lui permettant n'importe quelle resolution finale malgre la resolution de la matrice de reference
    initiale

"""

########################################################################################################################
########################################################################################################################

def atmospheric_matrix_3D(order,data,t,Rp,c_species,rank,Tracer=False,Clouds=False) :

    sp,reso_t,reso_z,reso_lat,reso_long = np.shape(data)
    T_file = data[1,:,:,:,:]
    P_file = data[0,:,:,:,:]
    c_number = c_species.size

    if Clouds == True :
        if Tracer == True :
            Q_vap = data[2,:,:,:,:]
            gen_cond = data[3:3+c_number,:,:,:,:]
            num = 3+c_number
        else :
            gen_cond = data[2:2+c_number,:,:,:,:]
            num = 2+c_number
    else :
        if Tracer == True :
            Q_vap = data[2,:,:,:,:]
            num = 3
        else :
            num = 2

    composit = data[num : sp,:,:,:,:]

    shape = np.shape(order)
    T = np.zeros((shape[1],shape[2],shape[3]),dtype=np.float64)
    P = np.zeros((shape[1],shape[2],shape[3]),dtype=np.float64)
    Cn = np.zeros((shape[1],shape[2],shape[3]),dtype=np.float64)

    if Tracer == True :
        Xm_Q = np.zeros((shape[1],shape[2],shape[3]),dtype=np.float64)

    if Clouds == True :
        gen = np.zeros((c_number,shape[1],shape[2],shape[3]),dtype=np.float64)

    compo = np.zeros((sp-num,shape[1],shape[2],shape[3]))

    bar = ProgressBar(shape[1],'Parametric recording')

    for i in range(shape[1]) :

        for j in range(shape[2]) :

            wh, = np.where(order[0,i,j,:] > 0)

            T[i,j,wh] = T_file[t,order[0,i,j,wh],order[1,i,j,wh],order[2,i,j,wh]]
            P[i,j,wh] = P_file[t,order[0,i,j,wh],order[1,i,j,wh],order[2,i,j,wh]]
            Cn[i,j,wh] = P[i,j,wh]/(R_gp*T[i,j,wh])*N_A

            if Tracer == True :
                Xm_Q[i,j,wh] = Q_vap[t,order[0,i,j,wh],order[1,i,j,wh],order[2,i,j,wh]]

            if Clouds == True :
                gen[:,i,j,wh] = gen_cond[:,t,order[0,i,j,wh],order[1,i,j,wh],order[2,i,j,wh]]

            compo[:,i,j,wh] = composit[:,t,order[0,i,j,wh],order[1,i,j,wh],order[2,i,j,wh]]

        if rank == 0 :
            bar.animate(i + 1)

    if Tracer == True :
        if Clouds == False :
            return P,T,Xm_Q,Cn,compo
        else :
            return P,T,Xm_Q,Cn,gen,compo
    else :
        if Clouds == False :
            return P,T,Cn,compo
        else :
            return P,T,Cn,gen,compo


########################################################################################################################


def atmospheric_matrix_1D(z_file,P_col,T_col,Q_col) :

    z_grid = np.load("%s.npy"%(z_file))

    shape = np.shape(z_grid)
    T = np.zeros(shape,dtype=np.float64)
    P = np.zeros(shape,dtype=np.float64)
    Xm_Q = np.zeros(shape,dtype=np.float64)
    Cn = np.zeros(shape,dtype=np.float64)

    j = 0

    for i in range(shape[0]) :
        z_ref = -1

        for k in range(shape[2]) :
            z = z_grid[i,j,k]

            if z >= 0 :
                if z == z_ref :
                    T[i,j,k] = T[i,j,k-1]
                    P[i,j,k] = P[i,j,k-1]
                    Xm_Q[i,j,k] = Xm_Q[i,j,k-1]
                    Cn[i,j,k] = Cn[i,j,k-1]
                else :
                    T[i,j,k] = T_col[z]
                    P[i,j,k] = P_col[z]
                    Xm_Q[i,j,k] = Q_col[z]
                    Cn[i,j,k] = P_col[z]/(R_gp*T_col[z])*N_A

                    z_ref = z

    for j in range(1,shape[1]) :
        T[:,j,:] = T[:,0,:]
        P[:,j,:] = P[:,0,:]
        Xm_Q[:,j,:] = Xm_Q[:,0,:]
        Cn[:,j,:] = Cn[:,0,:]

    return P,T,Xm_Q


########################################################################################################################


def PTprofil1D(Rp,g0,M,P_surf,T_iso,n_species,x_ratio_species,r_step,delta_z,dim,number,Middle,Origin,Gravity) :

    data_convert = np.zeros((number,1,dim,1,1))

    data_convert[number - 1,:,:,:,:] += M
    data_convert[0,:,0,:,:] = P_surf
    data_convert[1,:,:,:,:] += T_iso
    for i in range(n_species.size) :
        data_convert[2+i,:,:,:,:] = x_ratio_species[i]

    bar = ProgressBar(dim,'Computation of the atmospheric dataset')

    for i_z in range(1,dim) :

        if Middle == False :
            z_ref = i_z*delta_z
        else :
            if i_z != dim-1 :
                z_ref = (i_z - 0.5)*delta_z
            else :
                z_ref = (i_z - 1)*delta_z

        if Origin == True :
            if i_z != 1 :
                data_convert[0,0,i_z,0,0] = data_convert[0,0,i_z-1,0,0]*np.exp(-data_convert[number-1,0,i_z-1,0,0]*g0*\
                                    delta_z/(R_gp*data_convert[1,0,i_z-1,0,0])*1/((1+(z_ref-1*r_step)/Rp)*(1+z_ref/Rp)))
            else :
                data_convert[0,0,i_z,0,0] = data_convert[0,0,i_z-1,0,0]*np.exp(-data_convert[number-1,0,i_z-1,0,0]*g0*\
                                    delta_z/(2*R_gp*data_convert[1,0,i_z-1,0,0])*1/((1+(z_ref-0.5*r_step)/Rp)*(1+z_ref/Rp)))
        else :
            if Gravity == False :
                data_convert[0,0,i_z,0,0] = P_surf*np.exp(-data_convert[number-1,0,i_z-1,0,0]*g0/(R_gp*data_convert[1,0,i_z-1,0,0])*((z_ref/(1+z_ref/Rp))))
            else :
                data_convert[0,0,i_z,0,0] = P_surf*np.exp(-data_convert[number-1,0,i_z-1,0,0]*g0/(R_gp*data_convert[1,0,i_z-1,0,0])*z_ref)

        bar.animate(i_z + 1)

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

    return data_convert