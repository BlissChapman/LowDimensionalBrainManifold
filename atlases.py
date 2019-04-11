import numpy as np
from nilearn import plotting

class DesikanAtlas:        
    
    def plot(connectome, title, axes):
        plotting.plot_connectome(connectome,
                                 DesikanAtlas.coordinates(), 
                                 title=title,
                                 edge_threshold='99%',
                                 node_size=20,
                                 colorbar=True,
                                 axes=axes,
                                 annotate=True)
#         for coord, name in zip(DesikanAtlas.coordinates(), DesikanAtlas.names()):
#             axes.annotate(name, coord)
        
    
    def coordinates():
        desikan_x_coords = [-56,-2,-45,-1,-16,-24,-47,-56,-1,-43,-35.98,-3,-4,-68,-18,-3,-57,-44,-50,-12,-50,-1,-44,-9,-2,-34,-21,-25,-63,-60,-9,-26,-49,-41,50,3,43,3,21,33,48,61,1,38,20,5,7,69,20,4,57,49,55,14,51,1,47,7,4,43,16,17,67,65,9,37,45,43]
        desikan_y_coords = [-44,21,18,-82,-10,-54,-70,-32,-48,-87,30.71,-73,44,-29,-30,-28,22,48,35,-89,-26,-18,-12,-59,39,53,38,-62,-7,-41,65,15,-23,13,-42,21,14,-81,-4,-52,-67,-31,-48,-88,38,-69,45,-24,-29,-25,15,40,31,-85,-23,-16,-9,-57,38,45,34,-65,-12,-36,64,21,-25,12]
        desikan_z_coords = [5,27,46,20,-29,-16,31,-24,25,1,-12.11,-1,-14,-13,-18,62,18,-13,3,3,58,38,59,46,6,17,50,63,-1,39,-12,-35,9,-6,6,27,43,19,-32,-19,29,-27,26,1,-21,-1,-13,-14,-17,61,17,-13,5,5,56,37,54,41,4,21,53,59,-1,35,-13,-38,11,-6]
        desikan_coordinates = np.vstack((desikan_x_coords, desikan_y_coords, desikan_z_coords)).T
        return desikan_coordinates
    
    def names():
        return ["lBSTS","lcACC","lcMFG","lCUN","lENT","lFUS","lIPL","lITG","liCC","lLOG","lLOF","lLING","lMOF","lMTG","lPARH","lparaC","lpOPER","lpORB","lpTRI","lperiCAL","lpostC","lPCC","lpreC","lPCUN","lrACC","lrMFG","lSFG","lSPL","lSTG","lSMAR","lFP","lTP","lTT","lINS","rBSTS","rcACC","rcMFG","rCUN","rENT","rFUS","rIPL","rITG","riCC","rLOG","rLOF","rLING","rMOF","rMTG","rPARH","rparaC","rpOPER","rpORB","rpTRI","rperiCAL","rpostC","rPCC","rpreC","rPCUN","rrACC","rrMFG","rSFG","rSPL","rSTG","rSMAR","rFP","rTP","rTT","rINS"]
    
    def full_names():
        return ["L bank of the superior temporal sulcus","L caudal anterior cingulate","L caudal middle frontal gyrus","L cuneus","L entorhinal","L fusiform","L inferior parietal lobule","L inferior temporal gyrus","L isthmus cingulate cortex","L lateral occipital gyrus","L lateral orbitofrontal","L lingual","L medial orbitofrontal","L middle temporal gyrus","L parahippocampal","L paracentral","L pars opercularis","L pars orbitalis","L pars triangularis","L pericalcarine","L postcentral","L posterior cingulate cortex","L precentral","L precuneus","L rostral anterior cingulate cortex","L rostral middle frontal gyrus","L superior frontal gyrus","L superior parietal lobule","L superior temporal gyrus","L supramarginal gyrus","L frontal pole","L temporal pole","L transverse temporal","L insula","R bank of the superior temporal sulcus","R caudal anterior cingulate","R caudal middle frontal gyrus","R cuneus","R entorhinal","R fusiform","R inferior parietal lobule","R inferior temporal gyrus","R isthmus cingulate cortex","R lateral occipital gyrus","R lateral orbitofrontal","R lingual","R medial orbitofrontal","R middle temporal gyrus","R parahippocampal","R paracentral","R pars opercularis","R pars orbitalis","R pars triangularis","R pericalcarine","R postcentral","R posterior cingulate cortex","R precentral","R precuneus","R rostral anterior cingulate cortex","R rostral middle frontal gyrus","R superior frontal gyrus","R superior parietal lobule","R superior temporal gyrus","R supramarginal gyrus","R frontal pole","R temporal pole","R transverse temporal","R insula"]
    