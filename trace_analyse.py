import criticality as crfn

#================================    
class trace_analyse: 
#================================    
    """
    Class to analyse trace datasets. 
    
    """
    
    #========================
    def __init__(self, name):
    #========================
        self.name = name #dataset name

    #====================================
    def criticality(self,trace, bind, coord, n_neigh, dim):
    #====================================

        """
        This functions runs all criticality analysis on your data.
        
   
        
    Inputs:
        trace (np array): cells x timepoints, raw or normalised fluorescence values
        bind (np array): cells x time, binarised state vector
        coord (np array): cells x XYZ coordinates
        n_neigh (int): number of closest neigbours to find
        dim (np array): 1d vector of floats to convert XYZ coordinates into microns
        
    Returns:
        dict: Dictionary containing all critical statistics
        
        """
        self.nnb = crfn.neighbour(coord, n_neigh, dim) #Calculate nearest neighbours
        print('Nearest neighbours found')
        
        self.av, self.pkg = crfn.avalanche(self.nnb, bind) #Calculate avalanches
        print('Avalanches calculated')
        
        self.llr_s, self.llr_d = crfn.LLR(self.av, 2000) #Calculate loglikelihood ratio
        self.exp_s, self.exp_d = crfn.power_exponent(self.av, 2000) #Calculate power law exponents
        self.dcc = crfn.DCC(self.av) #Calculate exponent relation
        print('Avalanche statistics calculated')
        
        self.br = crfn.branch(self.pkg, self.av) #Calculate branching ratio
        print('Branching ratio calculated')
        
        #self.corr = np.corrceoff
        #self.corrdis_bin()
        #print('Correlation function calculated')
        
        return(self)
        
    
#================================================
def select_region(trace, bind, coord, region):
#================================================
    
    """
    This function slices data to include only those within a specific brain region.

    Inputs:
        trace (np array): cells x timepoints, raw or normalised fluorescence values
        bind (np array): cells x time, binarised state vector
        coord (np array): cells x XYZ coordinates and labels
        region (str): 'all', 'Diencephalon', 'Midbrain', 'Hindbrain' or 'Telencephalon'
    
    Returns:
        sub_trace (np array): cells x timepoints, raw or normalised fluorescence values for subregion
        sub_bind (np array): cells x time, binarised state vector for subregion
        sub_coord (np array): cells x XYZ coordinates for subregion
    
    
    """
    
    import numpy as np

    if coord.shape[0] != trace.shape[0]:
        print('Trace and coordinate data not same shape')
        return()


    if region == 'all':
        locs = np.where(coord[:,4] != 'nan')

    else: 
        locs = np.where(coord[:,4] == region)

    sub_coord = coord[locs][:,:3].astype(float)

    sub_trace, sub_bind = trace[locs], bind[locs]


    return(sub_trace, sub_bind, sub_coord)