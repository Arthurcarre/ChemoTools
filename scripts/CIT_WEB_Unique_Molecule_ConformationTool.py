"""
#####################################################
##                                                 ##
##           -- CIT CONFORMATIONTOOL --            ##
##                                                 ##
#####################################################
"""

import os, copy, random, py3Dmol
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from stmol import showmol
from rdkit import Chem
from rdkit.Chem import AllChem, Draw 
from rdkit.Chem.rdMolAlign import CalcRMS
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

class ConformationTool :
    """
    -- DESCRIPTION --
    This class aims to quantify the number of different conformations adopted by the set of all the poses of a molecule,
    or several molecules sharing the same structural backbone, resulting from docking simulations.
    """
    def __init__(self, 
                 sdf_file,
                 score = "Gold.PLP.Fitness"):
        """
        -- DESCRIPTION --
        This class takes as input the sdf file containing the results of the docking simulations.
        The name of the column in the sdf that contains the name of the molecule (not the pose) must be specified,
        by default "Compound Name". The name of the column in the sdf that uses the score result must also be specified,
        by default "Gold.PLP.Fitness".
            PARAMS:
                - sdf_file (string): path to a SDF file
                - score (string): Name of the column in the sdf
        """
        
        self.mols = [x for x in Chem.SDMolSupplier(sdf_file)]
        st.session_state.mols_brut = len(self.mols)
        self.score = score
    
    def get_heatmap_sample(self, individuals = 200) :
        """
        -- DESCRIPTION --
        This function takes as input a number that constitutes the amount of the sample of poses from the filtered sdf file
        taken randomly and without rebate. The RMSD is then calculated between each individual and the result is reported
        in a dataframe that forms the basis for the construction of a heatmap that is returned.
            PARAMS:
                - individuals (int): Amount of the sample
        """
        try :
            sample = random.sample(range(len(self.mols)), individuals)
        except ValueError as e :
            st.error("OOPS ! You're trying to define a sample more larger than the numbers of molecules/poses in your sdf.",
                  " This is impossible, please redefine a size of sample equal or smaller to your sdf")
        
        array = np.ones(shape=(len(sample),len(sample)))
        for i, indivduali in enumerate(sample) :
            for j, indivdualj in enumerate(sample) :
                array[i, j] = CalcRMS(self.mols[indivduali],
                                      self.mols[indivdualj])

        df = pd.DataFrame(
            array, index=list(range(len(sample))), columns=list(range(len(sample))))

        fig, ax = plt.subplots(figsize=(20, 10))
        sns.set_context('talk')
        sns.heatmap(df, fmt='d', ax= ax)
        st.pyplot(fig)
        st.session_state.heatmap = fig
    
    def get_sorted_heatmap(self, individuals = 200, RMSDthreshold = 2.0, loop = 3, p = 0.10) :
        """
        -- DESCRIPTION --
        This function aims to build a heatmap where the individuals constituting the sample have been previously sorted
        in order toreveal the different existing conformations. It takes, thus, in input the amount of this sample (by default: 200),
        and the threshold of the RMSD allowing to gather the poses in groups where each pose shares an RMSD around this threshold
        or below.
            PARAMS:
                - individuals (int): Amount of the sample
                - RMSDthreshold (float): Threshold for which the poses come together
                - loop (int): Times for which the sorting process is repeated
        """
        try :
            sample = random.sample(range(len(self.mols)), individuals)
            st.session_state.sample = sample
            self.sample = sample
        except TypeError :
            st.error("OOPS ! You're trying to define a sample more larger",
                  " than the numbers of molecules/poses in your sdf.",
                  " This is impossible, please redefine a size of sample",
                  "equal or smaller to your sdf")
            
        def RMSD_listes(sample) :
            """
            -- DESCRIPTION --
            For each individual in the sample, the function creates a list for which it inserts the index of all other individuals
            which together share an RMSD below the selected threshold. There are as many lists as there are individuals.
            Afterwards, the empty lists are deleted.
                PARAMS:
                    - sample (list): liste including each individuals of the sample
                RETURNS:
                    - output_lists (list) : liste including listes containing the index of some individuals  
            """            
            remove_list = list(range(len(sample)))
            output_lists = []

            for individual in sample :
                subliste = []
                for i in remove_list :
                    if CalcRMS(self.mols[individual],
                               self.mols[sample[i]]) <  RMSDthreshold :
                        subliste.append(i)
                        remove_list.remove(i)
                output_lists.append(subliste[:])

            while [] in output_lists :
                for i, output_list in enumerate(output_lists) :
                    try :
                        if output_list == [] :
                            del output_lists[i]
                    except :
                        pass
            return output_lists
        
        def gather_groups_RMSD(input_list) :
            """
            -- DESCRIPTION --
            This function takes as input the list returned by the function "RMSD_lists". For each group (list) within the input list,
            the function calculates the RMSD between the individuals in its group and those in the other groups.
            If the average of each RMSD calculated between two groups is less than the RMSD threshold, then the two groups are merged.
                PARAMS:
                    - input_list (list) : List from the fonction "RMSD_listes"
                RETURNS:
                    - output_list (list) : List includind merged listes according to the RMSD threshold shared between individuals
            """    
            output_list = []
            for i, groupei in enumerate(input_list) :
                p = 0
                for j, groupej in enumerate(input_list) :
                    RMSD = []
                    n = -1
                    if i != j :
                        if len(groupej) <= len(groupei):
                            for k in groupej :
                                n += 1
                                RMSD.append(CalcRMS(self.mols[self.sample[k]],
                                                    self.mols[self.sample[input_list[i][n]]]))
                        if len(groupej) > len(groupei):
                            for k in groupei :
                                n += 1
                                RMSD.append(CalcRMS(self.mols[self.sample[k]],
                                                    self.mols[self.sample[input_list[j][n]]]))
                    if i != j :
                        #print(f"La moyenne des RMSD entre le groupe {i} et le groupe {j} est de {np.mean(RMSD)}")
                        if np.mean(RMSD) < RMSDthreshold :
                            p = 1
                            copy1 = copy.deepcopy(groupei)   
                            copy2 = copy.deepcopy(groupej)
                            for l in range(len(copy2)) :
                                copy1.append(copy2.pop())
                            output_list.append(copy1)
                if p == 0 : 
                    copy1 = copy.deepcopy(groupei)
                    output_list.append(copy1)
            return output_list

        def sorted_list_lengroups(input_list) :
            """
            -- DESCRIPTION --
            This function takes as input the list returned by the "gather_groups_RMSD" function.
            The function sorts the list so that the groups with the most individuals are first
            and the groups with the least individuals are at the end of the list. 
                PARAMS:
                    - input_list (list) : List from the fonction "gather_groups_RMSD"
                RETURNS:
                    - output_list (list) : List including all groups sorted.
            """ 
            lengroups_list = [len(i) for i in input_list]
            lengroups_list.sort(reverse=True)

            return [x for y in lengroups_list for x in input_list if y == len(x)]


        def get_filtered_liste(input_list) :
            """
            -- DESCRIPTION --
            This function takes as input the list returned by the "sorted_list_lengroups" function.
            This function deletes all the groups inside the list, leaving only the previously sorted individuals without any duplicates.
                PARAMS:
                    - input_list (list) : List from the fonction "sorted_list_lengroups"
                RETURNS:
                    - output_list (list) : List including all individuals sorted.
            """
            def get_unique_numbers(numbers):
                'This function allows to have a number list without duplicates and conserving the original order.'
                unique = []

                for number in numbers:
                    if number in unique:
                        continue
                    else:
                        unique.append(number)
                return unique
            
            filtered_numbersduplicate_liste = []

            for liste in input_list : 
                for individual in liste :
                    filtered_numbersduplicate_liste.append(individual)

            filtered_liste = get_unique_numbers(filtered_numbersduplicate_liste)
            return filtered_liste


        def get_groups_inside_list(liste) :
            """
            -- DESCRIPTION --
            This function takes as input the list from the "get_filtered_list" function.
            It allows to restore the groups (as new lists inside the original list)
            by going through the different individuals of the original list and calculating the RMSD
            two by two by defining the separation between a group and another with a RMSD higher
            than 3 Ångstroms.
                PARAMS:
                    - input_list (list) : List from the fonction "get_filtered_list"
                RETURNS:
                    - output_list (list) : List with grouped individuals sharing an RMSD below 3 Å
            """
            liste1 = copy.deepcopy(liste)
            liste2 = copy.deepcopy(liste)
            output_list = []
            for i, n in enumerate(liste1) :
                try :
                    r = CalcRMS(self.mols[self.sample[n]],
                                self.mols[self.sample[liste1[i+1]]])
                except IndexError :
                    pass
                if r > 3 :
                    try :
                        output_list.append(liste2[:liste2.index(liste1[i+1])])
                        del liste2[:liste2.index(liste1[i+1])]
                    except IndexError :
                        pass
            return output_list

        def improve_sort(liste, n) :
            """
            -- DESCRIPTION --
            This function recombines the different functions used in the sorting process so that the process
            can be repeated n times (n being an integer that the function takes as input).
                PARAMS:
                    - input_list (list) : Corresponds initially to the sample,
                    it also corresponds to the list returned by the "get_filtered_liste" function. 
                    - n (int) : Defines the number of times the sorting process should be repeated
                RETURNS:
                    - output_list (list) : Corresponds to the list containing the individuals sorted n times
            """
            for i in range(n) :
                liste = get_filtered_liste(
                    sorted_list_lengroups(
                        gather_groups_RMSD(
                            get_groups_inside_list(liste))))
            return liste
        
        def get_predominant_poses(finallyliste, p) :
            """
            -- DESCRIPTION --
            This function takes as input the list from the "sorted_list_lengroups" function at the end of
            the sorting process. Then, it retrieves in a new list the groups (lists) of individuals that
            constitute more than p th of the total number of individuals. 
                PARAMS:
                    - input_list (list): List resulting from the sorting process function "sorted_list_lengroups"
                    - p (float): Minimum proportion (value between 0 and 1) of individuals in a group within the
                    sample to consider that group large enough to be representative of a full conformation.
                RETURNS:
                    - output_list (list): List including only the largest groups (lists) from the input_list
            """
            k = 0 
            for i in finallyliste : 
                if len(i)/len(self.sample)*100 > p*len(self.sample) :
                    k += 1

            predominant_poses = finallyliste[:k]
            st.info(f"There is (are) {k} predominant pose(s) among all poses.\n")
            st.session_state.numbers_conformation = k
            st.session_state.predominant_poses = predominant_poses

            for i, predominant_pose in enumerate(predominant_poses) :
                st.write(f"The predominant pose n°{i+1} represents {len(predominant_pose)/len(self.sample)*100:.1f}", 
                      f"% of the sample, i.e. {len(predominant_pose)} on {len(self.sample)} poses in total.")
            return predominant_poses

        def get_sample_indice_best_score(predominant_poses) :
            """
            -- DESCRIPTION --
            This function takes as input the list from the "get_predominant_poses" function. This function
            determines which individuals have the best score (according to the scoring function from the Docking
            simulations) in each predominant group and inserts in a new list that will be returned the index
            of those individuals who have the best score and then become the representatives of their group,
            the representatives of their conformation.
                PARAMS:
                    - input_list (list) : List from the fonction "get_predominant_poses"
                RETURNS:
                    - index_best_score (list) : List including the index of poses in the sample which represent
                    their conformation.
            """
            index_best_score = []
            m = -1

            for group in predominant_poses :
                m += 1
                n = -1
                suppl = []
                for individual in group :
                    n += 1
                    suppl.append(self.mols[self.sample[predominant_poses[m][n]]])
                PLPscore = [x.GetProp(self.score) for x in suppl]
                best_score = 0
                for k in PLPscore :
                    if float(k) > float(best_score) :
                        best_score = k
                        indice = PLPscore.index(k)
                index_best_score.append(indice)
            return index_best_score
        
        def get_SDF_Sample_and_Best_Score_Poses(sample_predominant_poses, sample_indice_best_score) :
            """
            -- DESCRIPTION --
            This function makes it possible to obtain the sdf file which contains the poses which have
            the best score (according to the scoring function resulting from the docking simulation) in
            one of the various conformations existing among the individuals of the sample BUT ALSO, in a
            different sdf file, those among all the poses of the sdf file deposited in input of this class
            then filtered. This function also provides a sdf file for each conformation containing all the
            individuals from the sample within the groups that characterise these conformations.
                PARAMS:
                    - sample_predominant_poses (list): List including all individuals of the largest groups
                    from the sorting process 
                    - sample_indice_best_score (list): List including the index of poses in the sample which
                    represent their conformation
            """
            conformations = (self.mols[self.sample[group[indice]]]
                             for group, indice in zip(sample_predominant_poses, sample_indice_best_score))

            with Chem.SDWriter("Sample_Best_PLPScore_Poses.sdf") as w:
                for m in conformations:
                    w.write(m)
            w.close()

            for k, group in enumerate(sample_predominant_poses) :
                with Chem.SDWriter(f"Sample_Conformation{k+1}.sdf") as w : 
                    for m in group :
                        mol = self.mols[self.sample[m]]
                        w.write(mol)
                w.close()

            a_supprimer = []
            for i, indice in enumerate(sample_indice_best_score) :
                subliste = []
                with Chem.SDWriter(f"À_supprimer{i+1}.sdf") as w:
                    for j, mol in enumerate(self.mols) :
                        if CalcRMS(self.mols[self.sample[sample_predominant_poses[i][indice]]],
                                   mol) < 2 :
                            subliste.append(self.mols[j])
                            w.write(self.mols[j]) 
                    copy1 = copy.deepcopy(subliste)
                    a_supprimer.append(copy1)

            indice_best_score = []
            for group in a_supprimer : 
                PLPscore = [x.GetProp(self.score) for x in group]
                best_score = 0
                for k in PLPscore :
                    if float(k) > float(best_score) :
                        best_score = k
                        indice = PLPscore.index(k)
                indice_best_score.append(indice)

            best_PLPScore_poses = (group[indice] for group, indice in zip(a_supprimer, indice_best_score))

            with Chem.SDWriter(f"Best_PLPScore_Poses.sdf") as w:
                for m in best_PLPScore_poses:
                    w.write(m)
            w.close()
            
            for i, indice in enumerate(sample_indice_best_score) :
                os.remove(f'À_supprimer{i+1}.sdf')

        def get_data_frame_best_poses(input_list) :
            """
            -- DESCRIPTION --
            This function takes as input the list of poses (structural backbone) of the representatives of
            each conformation and outputs a table that presents the calculated RMSD between all. This table allows
            to check that each group is different from each other.
                PARAMS:
                    - input_list (list) : list of poses of the representatives of each conformation
            """
            
            columns = list(range(len(input_list)))
            for i in columns :
                columns[i] = f"Pose n°{i+1}"

            index = list(range(len(input_list)))
            for i in index :
                index[i] = f"Pose n°{i+1}"
            
            array = np.ones(shape=(len(input_list),len(input_list)))

            for i, moli in enumerate(input_list) :
                for j, molj in enumerate(input_list) :
                    array[i, j] = CalcRMS(moli,
                                          molj)

            data_frame = pd.DataFrame(array, index=index, columns=columns)
            st.write("\nIn order to check that each group is different from each other, a table taking " 
              "the first individual from each group and calculating the RMSD between each was constructed :\n")
            st.dataframe(data_frame)
            st.write("RMSD value between each representative of each conformation.")
            st.session_state.df1 = data_frame
        
        def get_histogramme_sample_bestPLP(input_list) :
            """
            -- DESCRIPTION --
            This function takes as input the list of poses (structural backbone) of the representatives of
            each conformation and calculates the RMSD between each representative and all other poses from
            the original sdf file deposited at the initialization of the class. Subsequently, the function
            constructs a histogram, for each representative of a conformation, which presents the distribution
            of the RMSD between the representative and all other poses.
                PARAMS:
                    - input_list (list) : list of poses of the representatives of each conformation.
            """
            sns.set_style('whitegrid')
            sns.set_context('paper')
            
            if len(input_list) == 1 :
                sdf_to_hist = [CalcRMS(GetScaffoldForMol(input_list[0]), GetScaffoldForMol(mol)) for mol in self.mols]
                fig, ax = plt.subplots(len(input_list), 1, figsize=(15, 0.2*len(input_list)*9))
                a, b, c = 0, 0, 0
                for RMSD in sdf_to_hist : 
                    if RMSD < 2 :
                        a += 1
                    if RMSD < 3 :
                        b += 1
                    if RMSD < 4 :
                        c += 1
                ax.hist(sdf_to_hist, bins =100, label = "Conformation n°1")
                ax.axvline(x=2, ymin=0, ymax=1, color="black", linestyle="--")
                ax.annotate(a, (1.5, 0.05*len(self.mols)), fontsize=15)
                ax.axvline(x=3, ymin=0, ymax=1, color="black", linestyle="--")
                ax.annotate(b-a, (2.5, 0.05*len(self.mols)), fontsize=15)
                ax.axvline(x=4, ymin=0, ymax=1, color="black", linestyle="--")
                ax.annotate(c-b, (3.5, 0.05*len(self.mols)), fontsize=15)
                ax.legend(loc='upper left', shadow=True, markerfirst = False)
            else :
                sdf_to_hist = ([CalcRMS(GetScaffoldForMol(representative_conf),
                                        GetScaffoldForMol(mol)) for mol in self.mols] for representative_conf in input_list)
                fig, ax = plt.subplots(len(input_list), 1, figsize=(15, 0.2*len(best_PLP_poses)*9))
                for z, group in enumerate(sdf_to_hist) :
                    a, b, c = 0, 0, 0
                    for i in group : 
                        if i < 2 :
                            a += 1
                        if i < 3 :
                            b += 1
                        if i < 4 :
                            c += 1
                    ax[z].hist(group, bins =100, label =f"Conformation n°{z+1}") #Create an histogram to see the distribution of the RMSD of the sample
                    ax[z].axvline(x=2, ymin=0, ymax=1, color="black", linestyle="--")
                    ax[z].annotate(a, (1.5, 0.05*len(self.mols)), fontsize=15)
                    ax[z].axvline(x=3, ymin=0, ymax=1, color="black", linestyle="--")
                    ax[z].annotate(b-a, (2.5, 0.05*len(self.mols)), fontsize=15)
                    ax[z].axvline(x=4, ymin=0, ymax=1, color="black", linestyle="--")
                    ax[z].annotate(c-b, (3.5, 0.05*len(self.mols)), fontsize=15)
                    ax[z].legend(loc='upper left', shadow=True, markerfirst = False)               

            st.pyplot(fig)
            st.write("Density of the number of poses as a function of the RMSD calculated between the representative of each conformation"
             " and all poses of all molecules in the docking solutions of the filtered incoming sdf file.")
            st.session_state.histplot = fig
            fig.savefig("Histograms_Best_Score.jpeg", dpi=300, bbox_inches='tight')
            #st.write("RMSD distribution between all docking solutions", 
            #      " and the pose with the highest score of all solutions for a given conformation.")
        
        
        #OUT THE FUNCTIUN "get_histogramme_sample_bestPLP".
        output_liste = sorted_list_lengroups(get_groups_inside_list(improve_sort(
            get_filtered_liste(sorted_list_lengroups(gather_groups_RMSD(RMSD_listes(self.sample)))), loop)))
        
        finallyliste = get_filtered_liste(output_liste)
        
        if len(finallyliste) != individuals :
            st.warning(f"Attention. The sorting process discarded {individuals-len(finallyliste)} individuals")
        st.session_state.indviduals_deleted = individuals-len(finallyliste)
        
        array = np.ones(shape=(len(finallyliste),len(finallyliste)))
        
        for i, indivduali in enumerate(finallyliste) :
            for j, indivdualj in enumerate(finallyliste) :
                array[i, j] = CalcRMS(self.mols[self.sample[indivduali]],
                                      self.mols[self.sample[indivdualj]])

        data_frame = pd.DataFrame(array, index=finallyliste,
                                  columns=finallyliste)
        fig, ax = plt.subplots(figsize=(20, 10))
        
        try : 
            sns.set_context('talk')
            g = sns.heatmap(data_frame, fmt='d', ax= ax, cmap = "rocket")
            fig = g.get_figure()
            fig.savefig("Sorted_Heatmap.jpeg", dpi=300, bbox_inches='tight')
            st.pyplot(fig)
            st.session_state.sorted_heatmap = fig
            with open("Sorted_Heatmap.jpeg", "rb") as file:
                 btn = st.download_button(
                         label="Download PLOT sorted heatmap",
                         data=file,
                         file_name="Sorted_Heatmap.jpeg",
                         mime="image/jpeg")
            sample_predominant_poses = get_predominant_poses(output_liste, p)
            sample_indice_best_score = get_sample_indice_best_score(sample_predominant_poses)
            get_SDF_Sample_and_Best_Score_Poses(sample_predominant_poses, sample_indice_best_score)
            best_PLP_poses = [x for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
            
            if 'pdb' in st.session_state :
                style = st.selectbox('Style',['cartoon','cross','stick','sphere','line','clicksphere'])
                #bcolor = st.color_picker('Pick A Color', '#ffffff')
                pdb_file = Chem.MolFromPDBFile('pdb_file.pdb')
                best_mols = [x for x in Chem.SDMolSupplier('Best_PLPScore_Poses.sdf')]
                for i, mol in enumerate(best_mols) :
                    merged = Chem.CombineMols(pdb_file, mol)
                    Chem.MolToPDBFile(merged, f'Conformation n°{i+1}.pdb')
                    xyz_pdb = open(f'Conformation n°{i+1}.pdb', 'r', encoding='utf-8')
                    pdb = xyz_pdb.read().strip()
                    xyz_pdb.close()
                    xyzview = py3Dmol.view(width=700,height=500)
                    xyzview.addModel(pdb, 'pdb')
                    xyzview.setStyle({style:{'color':'spectrum'}})
                    #xyzview.setBackgroundColor(bcolor)#('0xeeeeee')
                    xyzview.setStyle({'resn':'UNL'},{'stick':{}})
                    xyzview.zoomTo({'resn':'UNL'})
                    showmol(xyzview, height = 500,width=1000)
                    os.remove(f'Conformation n°{i+1}.pdb')
                    st.write(f'Conformation n°{i+1}')
                    with open(f"Sample_Conformation{i+1}.sdf", "rb") as file:
                         btn = st.download_button(
                                    label=f"Download all the poses of the conformation n°{i+1} from the SAMPLE",
                                    data=file,
                                     file_name=f"Sample_Conformation{i+1}.sdf")
            
            os.remove(f'Sample_Best_PLPScore_Poses.sdf')
            with open("Best_PLPScore_Poses.sdf", "rb") as file:
                 btn = st.download_button(
                            label="Download the SDF file including each of the representatives of a conformation",
                            data=file,
                            file_name="Best_Score_Poses.sdf")
            
            get_data_frame_best_poses(best_PLP_poses)
            
            get_histogramme_sample_bestPLP(best_PLP_poses)
            self.best_PLP_poses = best_PLP_poses
            with open("Histograms_Best_Score.jpeg", "rb") as file:
                 btn = st.download_button(
                            label="Download PLOT Histograms",
                            data=file,
                             file_name="Histograms_Best_Score.jpeg",
                             mime="image/jpeg")
            
        except ValueError :
            st.error("OOPS ! The selected RMSD threshold does not allow all individuals to be grouped "
                  "into distinct groups that share a low RMSD. Please change the RMSD threshold.")

    
    def get_sdf_conformations(self, k, RMSDtarget) :
        """
        -- DESCRIPTION --
        From the histograms resulting from the "get_histogram_sample_bestPLP" function, it is possible
        to observe Gaussian curves for low value RMSDs constituting the group of poses aligned to
        the conformation. Thus, this function allows to create sdf files containing for a conformation,
        given by the integer k in input, all the poses which are below a selected RMSD value (-> RMSDtarget).
            PARAMS:
                - k (int) : Number of the selected conformation
                - RMSDtarget (float) : Target value of the RMSD below which all poses will be written to an sdf file.
        """
        with Chem.SDWriter(f'Conformation{k}.sdf') as w: 
            for j, mol in enumerate(self.mols) :
                if CalcRMS(GetScaffoldForMol(self.best_PLP_poses[k-1]),
                           GetScaffoldForMol(mol)) < float(RMSDtarget) :
                    w.write(self.mols[j])
        w.close()

###############################################
#                                             #
#            APPLICATION SECTION              #
#                                             #                                                      
###############################################

def makeblock(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mblock = Chem.MolToMolBlock(mol)
    return mblock

def render_mol(xyz):
    xyzview = py3Dmol.view()#(width=400,height=400)
    xyzview.addModel(xyz,'mol')
    xyzview.setStyle({'stick':{}})
    xyzview.setBackgroundColor('white')
    xyzview.zoomTo()
    showmol(xyzview, height=500, width=1000)

def main():
    
    st.header('Unique Molecule ConformationTool !')
    
    st.markdown('Welcome to Unique Molecule ConformationTool ! Here, the processing and analysis of the docking simulation'
                ' results when your sdf file include ONLY ONE molecule (in many different 3D positions).')

    #SDF FILE SECTION#
    sdf = st.file_uploader("Upload the coordinates of the docked ligand in SDF format:",
                                        type = ["sdf"])
    if sdf:
        score = st.text_input(
                     'What is the scoring function used in your sdf file ?',
                     'Gold.PLP.Fitness')
        st.session_state.score = score
        if 'sdf_file_stock' not in st.session_state :
            st.session_state.sdf_file_stock = sdf
            with open("sdf_file.sdf", "wb") as f:
                f.write(st.session_state.sdf_file_stock.getbuffer())
            mols = [x for x in Chem.SDMolSupplier("sdf_file.sdf")]
            st.session_state.mol1 = mols[0]
            del mols
            os.remove("sdf_file.sdf")
    else:
        try : 
            del st.session_state.sdf_file_stock
            del st.session_state.mol1
        except :
            pass

    #BENCHMARK MOLECULE SECTION#
    
    if 'molref_stock' not in st.session_state:
        if 'mol1' in st.session_state :
            smiles = Chem.MolToSmiles(st.session_state.mol1)
            st.session_state.molref_stock = smiles
            blk=makeblock(st.session_state.molref_stock)
            render_mol(blk)
    else : 
        blk=makeblock(st.session_state.molref_stock)
        render_mol(blk)

    #PDB PROTEIN SECTION#

    pdb = st.file_uploader("Upload a pdb file for viewing purposes. (FACULTATIVE)",
                                        type = ["pdb"])
    if pdb :
        st.session_state.pdb = pdb
        with open("pdb_file.pdb", "wb") as pdb_file:
            pdb_file.write(pdb.getbuffer())


    ###############################################
    #--        CHECKBOX "CHECK YOUR SDF"        --#                                                     
    ###############################################

    first_checkbox = st.checkbox(
        'Check your sdf (Attention ! Before closing this app, please, UNCHECK THIS BOX)')
    if first_checkbox :
        try : 
            if 'ConformationClass' not in st.session_state :
                with open("sdf_file.sdf", "wb") as sdf:
                    sdf.write(st.session_state.sdf_file_stock.getbuffer())
                st.session_state.sdf_file = "sdf_file.sdf"        
                st.session_state.ConformationClass = ConformationTool(st.session_state.sdf_file,
                                                                      st.session_state.score)

        except AttributeError:
            st.error("OOPS ! Did you forget to input the SDF file ?")
            #pass

        try :
            st.info(f"There are {st.session_state.mols_brut} poses in the sdf file.\n")
        except AttributeError :
            st.error('OOPS ! Have you submitted your sdf file? ?')

        try :
            os.remove("sdf_file.sdf")
        except :
            pass

        if st.session_state.mols_brut > 200 :
            size_sample = 200
        else :
            size_sample = st.session_state.mols_brut
        individuals = st.slider('Select the size of your sample. Default size of sample = 200', 0, 500, size_sample,
                                help='If you want to change this setting during the program, make sure the box below is unchecked!')

        #SECOND BOX
        second_checkbox = st.checkbox('Get the sample heatmap')
        if second_checkbox :
            try:
                st.pyplot(st.session_state.heatmap)
            except:
                pass

            if 'heatmap' not in st.session_state :
                st.session_state.ConformationClass.get_heatmap_sample(individuals)

        else:
            if 'heatmap' in st.session_state:
                del st.session_state.heatmap

        RMSD_Target = st.slider('RMSD threshold: Select the maximum RMSD threshold that should constitute a conformation.'
                                ' Default RMSD threshold = 2 A',
                                 0.0, 15.0, 2.0,
                                help='If you want to change this setting during the program, make sure the box below'
                                ' is unchecked!')

        Proportion = st.slider('Minimum size of the sample defining a conformation. Default proportion = 0.05',
                             0.0, 1.0, 0.05,
                               help=('This setting define the minimum proportion (value between 0 and 1) of individuals'
                                     ' in a group within the sample to consider that group large enough to be'
                                     ' representative of a full conformation.'))

    ###############################################
    #--   CHECKBOX "GET THE SORTED HEATMAP"     --#                                                     
    ############################################### 

        third_checkbox = st.checkbox('Get the sorted heatmap')
        if third_checkbox :
            try:
                st.warning(
                    f"Attention. The sorting process discarded {st.session_state.indviduals_deleted} individuals")

                st.pyplot(st.session_state.sorted_heatmap)
                with open("Sorted_Heatmap.jpeg", "rb") as file:
                     btn = st.download_button(
                             label="Download PLOT sorted heatmap",
                             data=file,
                             file_name="Sorted_Heatmap.jpeg",
                             mime="image/jpeg")
                st.info(f"There is (are) {st.session_state.numbers_conformation} predominant pose(s) among all poses.\n")

                if 'pdb' in st.session_state :
                    style = st.selectbox('Style',['cartoon','cross','stick','sphere','line','clicksphere'])
                    #bcolor = st.color_picker('Pick A Color', '#ffffff')
                    pdb_file = Chem.MolFromPDBFile('pdb_file.pdb')
                    best_mols = [x for x in Chem.SDMolSupplier('Best_PLPScore_Poses.sdf')]
                    for i, mol in enumerate(best_mols) :
                        merged = Chem.CombineMols(pdb_file, mol)
                        Chem.MolToPDBFile(merged, f'Conformation n°{i+1}.pdb')
                        xyz_pdb = open(f'Conformation n°{i+1}.pdb', 'r', encoding='utf-8')
                        pdb = xyz_pdb.read().strip()
                        xyz_pdb.close()
                        xyzview = py3Dmol.view(width=700,height=500)
                        xyzview.addModel(pdb, 'pdb')
                        xyzview.setStyle({style:{'color':'spectrum'}})
                        #xyzview.setBackgroundColor(bcolor)#('0xeeeeee')
                        xyzview.setStyle({'resn':'UNL'},{'stick':{}})
                        xyzview.zoomTo({'resn':'UNL'})
                        showmol(xyzview, height = 500,width=1000)
                        os.remove(f'Conformation n°{i+1}.pdb')
                        st.write(f'Conformation n°{i+1}')
                        with open(f"Sample_Conformation{i+1}.sdf", "rb") as file:
                             btn = st.download_button(
                                        label=f"Download all the poses of the conformation n°{i+1} from the SAMPLE",
                                        data=file,
                                         file_name=f"Sample_Conformation{i+1}.sdf")

                with open("Best_PLPScore_Poses.sdf", "rb") as file:
                     btn = st.download_button(
                                label="Download the SDF file including each of the representatives of a conformation",
                                data=file,
                                file_name="Best_Score_Poses.sdf")

                for i, predominant_pose in enumerate(st.session_state.predominant_poses) :
                    st.write(
                        f"The predominant pose n°{i+1} represents {len(predominant_pose)/len(st.session_state.sample)*100:.1f}" 
                        f"% of the sample, i.e. {len(predominant_pose)} on {len(st.session_state.sample)} poses in total.")

                st.write("\nIn order to check that each group is different from each other, a table taking " 
                          "the first individual from each group and calculating the RMSD between each was constructed :\n")

                st.dataframe(st.session_state.df1)

                st.write("RMSD value between each representative of each conformation.")

                st.pyplot(st.session_state.histplot)

                st.write("Density of the number of poses as a function of the RMSD calculated between the representative of each conformation"
                 " and all poses of all molecules in the docking solutions of the filtered incoming sdf file.")      

                with open("Histograms_Best_Score.jpeg", "rb") as file:
                     btn = st.download_button(
                                label="Download PLOT Histograms",
                                data=file,
                                 file_name="Histograms_Best_Score.jpeg",
                                 mime="image/jpeg")

                ###############################################
                #--    BUTTON "PREPARE YOUR SDF FILE"       --#                                                     
                ###############################################

                if st.session_state.numbers_conformation != 1 :
                    temp_options = range(1, st.session_state.numbers_conformation + 1)
                    st.session_state.temp = st.select_slider("You want a sdf file or plots including molecules in the conformation n°",
                                                             options=temp_options, value = st.session_state.temp)
                    st.write(f"The Conformation selected is {st.session_state.temp}")

                    st.session_state.RMSD_Target_conformation = st.slider('... With all poses under a RMSD =', 0.0, 15.0, 2.0)
                    st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")

                    if st.button('Prepare your sdf file and build plots'):             
                        st.session_state.ConformationClass.get_sdf_conformations(
                            st.session_state.temp,
                            st.session_state.RMSD_Target_conformation
                            )

                        with open(f'Conformation{st.session_state.temp}.sdf', "rb") as file:
                             btn = st.download_button(
                                        label="Download your sdf file",
                                        data=file,
                                         file_name=f"Conformation n°{st.session_state.temp}.sdf")
                    else :
                        try :
                            with open(f'Conformation{st.session_state.temp}.sdf', "rb") as file:
                                 btn = st.download_button(
                                            label="Download your sdf file",
                                            data=file,
                                             file_name=f"Conformation n°{st.session_state.temp}.sdf")
                        except :
                            pass

                else :                              
                    st.session_state.RMSD_Target_conformation = st.slider(
                        'You want a sdf file and/or plot analysis including molecules in the unique predominant conformation with all poses under a RMSD =',
                        0.0, 15.0, 2.0)

                    st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")

                    st.info('There is only one predominant conformation. Do you want to have the sdf file of poses in this conformation OR see analysis of this conformation ? ')

                    bouton = st.button('I want to have the sdf file of poses in this conformation AND/OR the plots.')
                    if bouton :
                        st.session_state.ConformationClass.get_sdf_conformations(
                            1, st.session_state.RMSD_Target_conformation)

                        with open('Conformation1.sdf', "rb") as file:
                         btn = st.download_button(
                                    label="Download your sdf file",
                                    data=file,
                                     file_name=f"Unique Conformation.sdf")
                    else :
                        try :
                            with open(f'Conformation1.sdf', "rb") as file:
                                 btn = st.download_button(
                                            label="Download your sdf file",
                                            data=file,
                                             file_name=f"Conformation n°1.sdf")
                        except :
                            pass
                        
            except AttributeError as e :
                st.exception(e)
                #pass

    ###############################################
    #--        BEGINNING THIRDCHECKBOX          --#                                                     
    ###############################################

            if 'sorted_heatmap' not in st.session_state:
                with st.spinner('Please wait, the sorted heatmap is coming...'):
                    st.session_state.ConformationClass.get_sorted_heatmap(
                        individuals,
                        RMSD_Target,
                        loop = 1,
                        p = Proportion)

                if st.session_state.numbers_conformation != 1:
                    temp_options = range(1, st.session_state.numbers_conformation + 1)
                    st.session_state.temp = st.select_slider("You want a sdf file and/or a anlysis plots including molecules in the conformation n°",
                                                             options=temp_options)
                    st.write(f"The Conformation selected is {st.session_state.temp}")

                    st.session_state.RMSD_Target_conformation = st.slider('... With all poses under a RMSD =', 0.0, 15.0, 2.0)
                    st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")

                    if st.button('Prepare your sdf file'):             
                        st.session_state.ConformationClass.get_sdf_conformations(
                            st.session_state.temp,
                            st.session_state.RMSD_Target_conformation
                            )
                else :
                    st.info('There is only one predominant conformation. Do you want to have the sdf file of poses in this conformation OR see analysis of this conformation ? ')
                    st.session_state.RMSD_Target_conformation = st.slider('You want a sdf file and/or a analysis plots including molecules in the unique predominant conformation with all poses under a RMSD =', 0.0, 15.0, 2.0)

                    if st.button('I want to have the sdf file of poses in this conformation.'):             
                        st.session_state.ConformationClass.get_sdf_conformations(
                            1, st.session_state.RMSD_Target_conformation)
        else :
            if 'numbers_conformation' in st.session_state:
                del st.session_state.numbers_conformation
                try :
                    os.remove('Sorted_Heatmap.jpeg')
                    os.remove('Histograms_Best_Score.jpeg')
                    os.remove('Sample_Best_PLPScore_Poses.sdf')
                    os.remove('Best_PLPScore_Poses.sdf')
                    for i in range(st.session_state.numbers_conformation):
                        os.remove(f'Sample_Conformation{i+1}.sdf')
                        try :
                            os.remove(f'Conformation{i+1}.sdf')
                            os.remove(f'Barplot_Conformation{i+1}.jpeg')
                            os.remove(f'Scatter_Plot{i+1}.jpeg')
                            os.remove(f'Box_Plot{i+1}.jpeg')
                        except :
                            pass
                except :
                    pass        

            if 'indviduals_deleted' in st.session_state:
                del st.session_state.indviduals_deleted
            if 'sorted_heatmap' in st.session_state:
                del st.session_state.sorted_heatmap
            if 'predominant_poses' in st.session_state:
                del st.session_state.predominant_poses
            if 'df1' in st.session_state:
                del st.session_state.df1
            if 'histplot' in st.session_state:
                del st.session_state.histplot
            if 'output_liste' in st.session_state:
                del st.session_state.output_liste 
            if 'sample_predominant_poses' in st.session_state:
                del st.session_state.sample_predominant_poses
            if 'sample_indice_best_score' in st.session_state:
                del st.session_state.sample_indice_best_score
            if 'best_PLP_poses' in st.session_state:
                del st.session_state.best_PLP_poses

    else :
        if 'ConformationClass' in st.session_state:
            del st.session_state.ConformationClass
        if 'pdb' in st.session_state:
            del st.session_state.pdb
        if 'error_mols' in st.session_state:
            del st.session_state.error_mols
        if 'output_name_prefix' in st.session_state:
            del st.session_state.output_name_prefix        
        if 'sdf_file' in st.session_state:
            del st.session_state.sdf_file
        if 'mols_brut' in st.session_state:
            del st.session_state.mols_brut
        if 'molref_stock' in st.session_state:
            del st.session_state.molref_stock
        if 'mols' in st.session_state:
            del st.session_state.mols
        if 'numbers_conformation' in st.session_state:
            os.remove('Sorted_Heatmap.jpeg')
            os.remove('Histograms_Best_Score.jpeg')
            try : 
                os.remove('pdb_file.pdb')
            except :
                pass
            os.remove('Best_PLPScore_Poses.sdf')
            for i in range(st.session_state.numbers_conformation):
                os.remove(f'Sample_Conformation{i+1}.sdf')
                try :
                    os.remove(f'Conformation{i+1}.sdf')
                except :
                    pass

            del st.session_state.numbers_conformation
        if 'indviduals_deleted' in st.session_state:
            del st.session_state.indviduals_deleted
        if 'sorted_heatmap' in st.session_state:
            del st.session_state.sorted_heatmap
        if 'predominant_poses' in st.session_state:
            del st.session_state.predominant_poses
        if 'df1' in st.session_state:
            del st.session_state.df1
        if 'histplot' in st.session_state:
            del st.session_state.histplot
        if 'output_liste' in st.session_state:
            del st.session_state.output_liste 
        if 'sample_predominant_poses' in st.session_state:
            del st.session_state.sample_predominant_poses
        if 'sample_indice_best_score' in st.session_state:
            del st.session_state.sample_indice_best_score
        if 'best_PLP_poses' in st.session_state:
            del st.session_state.best_PLP_poses
