"""
#####################################################
##                                                 ##
##         -- CIT CONFORMATIONTOOL V1.1 --         ##
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
from rdkit.Chem.rdFMCS import FindMCS
from rdkit.Chem.rdMolAlign import CalcRMS

def Get_MCS_Fusion(self, mol):
    """
    -- DESCRIPTION --
    This function is used to obtain the MCS between a selected molecule and the MCS (called "fusion") from the sdf file
    of the ConformationTool class.
        PARAMS:
            - mol (mol): Molecule selected to obtain its MCS
        RETURNS:
            - _ (mol): MCS of the selected molecule
    """
    moledit = Chem.EditableMol(mol)
    match = mol.GetSubstructMatch(self.fusion)
    j = 0
    for i in range(len(mol.GetAtoms())):
        if i not in match:
            moledit.RemoveAtom(i-j)
            j += 1
    return moledit.GetMol() 

def preprocess(liste_mols):
    """
    -- DESCRIPTION --
    This function is a preprocessing of molecules based on 2 points: the MCS must be as large as possible so that the RMSD is as
    representative as possible AND only the points as coordinates are important for calculating the RMSD.
    Thus, this function consists in, firstly, removing the hydrogens, secondly, removing all the double and triple bonds.
        PARAMS:
            - liste_mols (list): list including all molecules intended to be preprocessed 
        RETURNS:
            - score (list): list of all molecules preprocessed
    """
    mol_withoutHs = (Chem.RemoveHs(mol) for mol in liste_mols)
    with Chem.SDWriter('out.sdf') as w:
        for mol in mol_withoutHs:
            w.write(mol)
    with open('out.sdf', "r") as f:
        content = f.read()
        f.close()
    os.remove('out.sdf')
    mols = content.split("$$$$")

    with open('output.sdf', 'w') as output_file:
         for mol in mols:
                for line in mol.strip().split("\n") :
                    if len(line) == 12:
                        if line.strip()[6] == "2" or line.strip()[7] == "2":
                            output_file.write(line[:8] + "1" + line[9:] + "\n")
                        else:
                            output_file.write(line + "\n")
                    else :
                        output_file.write(line + "\n")
                output_file.write("\n$$$$\n")
    mols_preprocessed = [x for x in Chem.SDMolSupplier('output.sdf')]
    os.remove('output.sdf')
    return mols_preprocessed

class ConformationTool :
    """
    -- DESCRIPTION --
    This class aims to isolate, within the results of docking simulations, the different consensus conformations and
    to quantify the consistency of the poses for each molecule of the same family.
    """
    def __init__(self, 
                 sdf_file,
                 score = "Gold.PLP.Fitness"):
        """
        -- DESCRIPTION --
        This class takes as input the sdf file containing the results of the docking simulations.
        The name of the column in the sdf that uses the score result must also be specified,
        by default "Gold.PLP.Fitness".
            PARAMS:
                - sdf_file (string): path to a SDF file
                - score (string): Name of the column in the sdf
        """
        
        self.mols_brut = [x for x in Chem.SDMolSupplier(sdf_file)]
        st.session_state.mols_brut = len(self.mols_brut)
        sdf_preprocessed = preprocess(self.mols_brut)
        self.sdf_preprocessed = sdf_preprocessed
        self.score = score
        unique_molecules = []
        smiles_unique_molecules = []
        for mol in sdf_preprocessed :
            if Chem.MolToSmiles(mol) not in smiles_unique_molecules :
                smiles_unique_molecules.append(Chem.MolToSmiles(mol))
                unique_molecules.append(mol)
        print(f"There are {len(unique_molecules)} different molecules and {len(self.mols_brut)}"
              " poses in your sdf file.")
        del smiles_unique_molecules
        self.unique_molecules = unique_molecules

    def get_MCS_SDF(self, ringMatchesRingOnly=False, percentage=100):
        """
        -- DESCRIPTION --
        This function allows to build the MCS from a proportion of the molecules in the sdf file. If the proportion is not 1,
        the algorithm will choose the molecules for which the MCS is the most larger. 
            PARAMS:
                - ringMatchesRingOnly (bool): If True, the MCS take into account the rings.
                - percentage (int) : Set the proportion of molecules for which the algorithm will build the MCS.
        """
        
        try :
            mcs = FindMCS(self.unique_molecules, 
                      ringMatchesRingOnly=ringMatchesRingOnly, 
                      threshold=percentage / 100.0)
        except RuntimeError :
            st.error('ERROR : Your sdf file contains only one unique molecule (in many different poses). In this case, please, use "CIT: Unique Molecule ConformationTool"')
        del self.unique_molecules
        fusion = Chem.MolFromSmarts(mcs.smartsString)
        st.session_state.fusion = fusion
        im = Draw.MolToImage(fusion)
        st.session_state.fusion_im = im
        
        mols = []
        MCS_mols = []
        error_mols = []
        for mol, mol_preprocessed in zip(self.mols_brut, self.sdf_preprocessed) :
            if mol_preprocessed.HasSubstructMatch(fusion) is True :
                mols.append(mol)
                moledit = Chem.EditableMol(mol_preprocessed)
                match = mol_preprocessed.GetSubstructMatch(fusion)
                j = 0
                for i in range(len(mol_preprocessed.GetAtoms())):
                    if i not in match:
                        moledit.RemoveAtom(i-j)
                        j += 1
                MCS_mols.append(moledit.GetMol())
            else :
                error_mols.append(mol)

        st.session_state.error_mols = error_mols
        st.session_state.mols = len(mols)
        
        self.mols = mols
        self.MCS_mols = MCS_mols        
        self.fusion = fusion
        
        del error_mols
        del self.mols_brut
        del self.sdf_preprocessed
    
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
                try :
                    array[i, j] = CalcRMS(self.MCS_mols[indivduali],
                                          self.MCS_mols[indivdualj])
        
                except RuntimeError :
                    array[i, j] = CalcRMS(self.MCS_mols[indivdualj],
                                          self.MCS_mols[indivduali])
                    

        df = pd.DataFrame(
            array, index=list(range(len(sample))), columns=list(range(len(sample))))

        fig, ax = plt.subplots(figsize=(20, 10))
        sns.set_context('talk')
        sns.heatmap(df, fmt='d', ax= ax)
        st.pyplot(fig)
        st.session_state.heatmap = fig
    
    def get_sorted_heatmap(self, individuals = 200, RMSDthreshold = 3.0, loop = 3, p = 0.05) :
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
        except ValueError as e :
            st.error("OOPS ! You're trying to define a sample more larger",
                     " than the numbers of molecules/poses in your sdf.",
                     " This is impossible, please redefine a size of sample",
                     " equal or smaller to your sdf")  
        
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
                    try :
                        result = CalcRMS(self.MCS_mols[individual],
                                         self.MCS_mols[sample[i]])
                    except RuntimeError:
                        result = CalcRMS(self.MCS_mols[sample[i]],
                                         self.MCS_mols[individual])                       
                    if result <  RMSDthreshold :
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
                                try:
                                    RMSD.append(CalcRMS(self.MCS_mols[self.sample[k]],
                                                        self.MCS_mols[self.sample[input_list[i][n]]]))
                                except RuntimeError:
                                    RMSD.append(CalcRMS(self.MCS_mols[self.sample[input_list[i][n]]],
                                                        self.MCS_mols[self.sample[k]]))
                        if len(groupej) > len(groupei):
                            for k in groupei :
                                n += 1
                                try:
                                    RMSD.append(CalcRMS(self.MCS_mols[self.sample[k]],
                                                        self.MCS_mols[self.sample[input_list[j][n]]]))
                                except RuntimeError:
                                    RMSD.append(CalcRMS(self.MCS_mols[self.sample[input_list[j][n]]],
                                                        self.MCS_mols[self.sample[k]]))
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
            than 3 ??ngstroms.
                PARAMS:
                    - input_list (list) : List from the fonction "get_filtered_list"
                RETURNS:
                    - output_list (list) : List with grouped individuals sharing an RMSD below 3 ??
            """
            liste1 = copy.deepcopy(liste)
            liste2 = copy.deepcopy(liste)
            output_list = []
            for i, n in enumerate(liste1) :
                try :
                    r = CalcRMS(self.MCS_mols[self.sample[n]],
                                self.MCS_mols[self.sample[liste1[i+1]]])
                except RuntimeError :
                    r = CalcRMS(self.MCS_mols[self.sample[liste1[i+1]]],
                                self.MCS_mols[self.sample[n]])
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
                print(f"The predominant conformation n??{i+1} represents {len(predominant_pose)/len(self.sample)*100:.1f}", 
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
                with Chem.SDWriter(f"??_supprimer{i+1}.sdf") as w:
                    for j, mol in enumerate(self.MCS_mols) :
                        try:
                            result = CalcRMS(self.MCS_mols[self.sample[sample_predominant_poses[i][indice]]], mol)
                        except RuntimeError:
                            result = CalcRMS(mol, self.MCS_mols[self.sample[sample_predominant_poses[i][indice]]])
                        if result < 2 :
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
                os.remove(f'??_supprimer{i+1}.sdf')

        def get_data_frame_best_poses(input_list) :
            """
            -- DESCRIPTION --
            This function takes as input the list of poses (structural backbone) of the representatives of
            each conformation and outputs a table that presents the calculated RMSD between all. This table allows
            to check that each group is different from each other.
                PARAMS:
                    - input_list (list) : list of poses of the representatives of each conformation
            """
            
            def MCS_RMSD(mol1, mol2):
                mcs = FindMCS([mol1, mol2], completeRingsOnly=True)
                fusion = Chem.MolFromSmarts(mcs.smartsString)
                def subfunction(mol):
                    moledit = Chem.EditableMol(mol)
                    match = mol.GetSubstructMatch(fusion)
                    j = 0
                    for i in range(len(mol.GetAtoms())):
                        if i not in match:
                            moledit.RemoveAtom(i-j)
                            j += 1
                    return moledit.GetMol()
                mcsmol1 = subfunction(mol1)
                mcsmol2 = subfunction(mol2)
                try :
                    result = CalcRMS(mcsmol1, mcsmol2)
                except RuntimeError:
                    result = CalcRMS(mcsmol2, mcsmol1)
                return result
            
            preprocess_list = preprocess(input_list)
            columns = list(range(len(preprocess_list)))
            for i in columns :
                columns[i] = f"Conformation n??{i+1}"

            index = list(range(len(preprocess_list)))
            for i in index :
                index[i] = f"Conformation n??{i+1}"
            
            array = np.ones(shape=(len(preprocess_list),len(preprocess_list)))

            for i, moli in enumerate(preprocess_list) :
                for j, molj in enumerate(preprocess_list) :
                    array[i, j] = MCS_RMSD(moli, molj)

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
                try :
                    sdf_to_hist = [CalcRMS(Get_MCS_Fusion(self, input_list[0]), mol) for mol in self.MCS_mols]
                except RuntimeError:
                    sdf_to_hist = [CalcRMS(mol, Get_MCS_Fusion(self, input_list[0])) for mol in self.MCS_mols]
                
                fig, ax = plt.subplots(len(input_list), 1, figsize=(15, 0.2*len(input_list)*9))
                a, b, c = 0, 0, 0
                for RMSD in sdf_to_hist : 
                    if RMSD < 2 :
                        a += 1
                    if RMSD < 3 :
                        b += 1
                    if RMSD < 4 :
                        c += 1
                ax.hist(sdf_to_hist, bins =100, label = "Conformation n??1")
                ax.axvline(x=2, ymin=0, ymax=1, color="black", linestyle="--")
                ax.annotate(a, (1.5, 0.05*len(self.mols)), fontsize=15)
                ax.axvline(x=3, ymin=0, ymax=1, color="black", linestyle="--")
                ax.annotate(b-a, (2.5, 0.05*len(self.mols)), fontsize=15)
                ax.axvline(x=4, ymin=0, ymax=1, color="black", linestyle="--")
                ax.annotate(c-b, (3.5, 0.05*len(self.mols)), fontsize=15)
                ax.legend(loc='upper left', shadow=True, markerfirst = False)
            else :
                try:
                    sdf_to_hist = ([CalcRMS(Get_MCS_Fusion(self, representative_conf),
                                            mol) for mol in self.MCS_mols] for representative_conf in input_list)
                except RuntimeError:
                    sdf_to_hist = ([CalcRMS(mol,Get_MCS_Fusion(self, representative_conf)) for mol in self.MCS_mols] for representative_conf in input_list)
                
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
                    ax[z].hist(group, bins =100, label =f"Conformation n??{z+1}")
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
                try:
                    array[i, j] = CalcRMS(self.MCS_mols[self.sample[indivduali]],
                                          self.MCS_mols[self.sample[indivdualj]])
                except RuntimeError:
                    array[i, j] = CalcRMS(self.MCS_mols[self.sample[indivdualj]],
                                          self.MCS_mols[self.sample[indivduali]])


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
            best_PLP_poses_preprocessed = preprocess(best_PLP_poses)
            
            if 'pdb' in st.session_state :
                style = st.selectbox('Style',['cartoon','cross','stick','sphere','line','clicksphere'])
                #bcolor = st.color_picker('Pick A Color', '#ffffff')
                pdb_file = Chem.MolFromPDBFile('pdb_file.pdb')
                best_mols = [x for x in Chem.SDMolSupplier('Best_PLPScore_Poses.sdf')]
                for i, mol in enumerate(best_mols) :
                    merged = Chem.CombineMols(pdb_file, mol)
                    Chem.MolToPDBFile(merged, f'Conformation n??{i+1}.pdb')
                    xyz_pdb = open(f'Conformation n??{i+1}.pdb', 'r', encoding='utf-8')
                    pdb = xyz_pdb.read().strip()
                    xyz_pdb.close()
                    xyzview = py3Dmol.view(width=700,height=500)
                    xyzview.addModel(pdb, 'pdb')
                    xyzview.setStyle({style:{'color':'spectrum'}})
                    #xyzview.setBackgroundColor(bcolor)#('0xeeeeee')
                    xyzview.setStyle({'resn':'UNL'},{'stick':{}})
                    xyzview.zoomTo({'resn':'UNL'})
                    showmol(xyzview, height = 500,width=1000)
                    os.remove(f'Conformation n??{i+1}.pdb')
                    st.write(f'Conformation n??{i+1}')
                    with open(f"Sample_Conformation{i+1}.sdf", "rb") as file:
                         btn = st.download_button(
                                    label=f"Download all the poses of the conformation n??{i+1} from the SAMPLE",
                                    data=file,
                                     file_name=f"Sample_Conformation{i+1}.sdf")
            
            os.remove(f'Sample_Best_PLPScore_Poses.sdf')
            with open("Best_PLPScore_Poses.sdf", "rb") as file:
                 btn = st.download_button(
                            label="Download the SDF file including each of the representatives of a conformation",
                            data=file,
                            file_name="Best_Score_Poses.sdf")
            
            get_data_frame_best_poses(best_PLP_poses_preprocessed)
            
            get_histogramme_sample_bestPLP(best_PLP_poses_preprocessed)
            self.best_PLP_poses = best_PLP_poses
            self.best_PLP_poses_preprocessed = best_PLP_poses_preprocessed
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
            for j, mol in enumerate(self.MCS_mols) :
                try:
                    result = CalcRMS(Get_MCS_Fusion(self, self.best_PLP_poses_preprocessed[k-1]), mol)
                except RuntimeError:
                    result = CalcRMS(mol, Get_MCS_Fusion(self, self.best_PLP_poses_preprocessed[k-1]))
                if result < float(RMSDtarget) :
                    w.write(self.mols[j])
        w.close()
    

    def get_plots(self,
                     k,
                     molecule_name,
                     aspect_plot = 1.75,
                     height_plot = 18,
                     size_xlabels = 25,
                     size_ylabels = 15) :
        
        """
        -- DESCRIPTION --
        After the "get_sdf_conformations" function, this function allows you to use the previously created
        sdf file and draw plots from it. The barplot shows for each molecule the ratio between the number of
        poses for which the molecule adheres to the conformation and the total number of poses and presents
        them in a decreasing order of the ratio. Subsequently, a boxplot is drawn and
        illustrate the distribution of the score (according to the scoring function from the docking algorithms)
        for each molecule with respect to the poses that are in the given confirmation. The order of the molecules
        is the same as that of the barplot. Finally, a new boxplot is again drawn,
        but this time in the descending order of the median of each distribution of the score for each molecule.
            PARAMS:
                - k (int): Number of the selected conformation
                - molecule_name (string): Name of the column in the sdf that contains the name of the molecule
                - aspect_plot (float): Aspect ratio of each facet, regarding the barplot and the boxplots,
                so that aspect * height gives the width of each facet in inches
                - height_plot (float): Height (in inches) of each facet, regarding the barplot and the boxplots
                - size_xlabels (float): Number characterising the size of the axis X labels
                - size_ylabels (float): Number characterising the size of the axis Y labels
        """
            
        # BAR PLOT
        file_in = [x for x in Chem.SDMolSupplier(f"Conformation{k}.sdf")]
        ensemble = {mol.GetProp(molecule_name) for mol in self.mols}
        
        index = list(ensemble)
        index.sort()
        
        columns = "Good Poses", "Total Poses", "Ratio"
        
        array = np.zeros(shape=(len(ensemble),3))

        for i, indice in enumerate(index) :
            for mol in file_in :
                if mol.GetProp(molecule_name) == indice : 
                    array[i, 0] += 1
            for mol in self.mols :
                if mol.GetProp(molecule_name) == indice : 
                    array[i, 1] += 1        
            array[i, 2] = array[i, 0]/array[i, 1]
            
        data_frame = pd.DataFrame(array, index=index, columns=columns)
        table = data_frame.sort_values("Ratio", ascending=False)       
        
        sns.set_context('talk')
        sns.set_style('whitegrid')
        g = sns.catplot(x="Ratio", y=list(table.index), data = table, kind="bar", aspect=aspect_plot,
                        height=height_plot, palette = "rocket")
        g.set_ylabels("")
        xlocs, xlabs = plt.xticks()
        
        for i, v in enumerate(list(table["Ratio"])):
            plt.text(float(v)+0.047, float(i)+0.35, f"{round(v*100, 1)}%", horizontalalignment = "center",
                     fontsize=size_ylabels+3)
        
        plt.yticks(fontsize=18)
        plt.axvline(x=0.5, color="black", linestyle="--")
        plt.axvline(x=0.25, color="black", linestyle=":")
        plt.axvline(x=0.75, color="black", linestyle=":")
        g.set_axis_labels("Ratio", "", fontsize = 25)
        g.set_yticklabels(fontsize = size_ylabels)
        g.set_xticklabels(fontsize = size_xlabels)
        g.fig.savefig(f"Barplot_Conformation{k}.jpeg", dpi=300, bbox_inches='tight')
        st.session_state.barplot = g

        
        # Preparation
        file_synt = (mol for name in list(table.index) for mol in file_in if mol.GetProp(molecule_name) == name)
        
        index2 = []
        
        array2 = np.zeros(shape=(len(file_in),2))
        
        for i, mol in enumerate(file_synt):
                    index2.append(mol.GetProp(molecule_name))
                    array2[i, 0] = mol.GetProp(self.score)
        
        data_frame2 = pd.DataFrame(array2, index=index2, columns=[self.score, "Zero"])
        
        file_synt = (mol for name in list(table.index) for mol in file_in if mol.GetProp(molecule_name) == name)
        for i, mol in enumerate(file_synt):
            array2[i, 1] = np.median(data_frame2.loc[mol.GetProp(molecule_name), self.score])
        
        data_frame3 = pd.DataFrame(array2, index=index2, columns=[self.score, "Median"])
        df3 = copy.copy(data_frame3)
        
        data_frame2.reset_index(inplace=True)
        data_frame2.rename(columns={'index': 'Name'}, inplace=True)

        #Reset index and Sort the DF3 Median column in a ascending order 
        data_frame3.reset_index(inplace=True)
        data_frame3.rename(columns={'index': 'Name'}, inplace=True)
        data_frame3.sort_values("Median", ascending=False, inplace=True)
        
        ### BOX PLOT
        sns.set_style('whitegrid')
        sns.set_context('talk')
        
        g = sns.catplot(x=self.score, y='Name', data=data_frame3, kind = "box", palette="rocket",
                        aspect=aspect_plot, height=height_plot)
        g.set_axis_labels(self.score, "")
        g.set_yticklabels(fontsize = size_ylabels)
        g.set_xticklabels(fontsize = size_xlabels)
        g.fig.savefig(f"Box_Plot{k}.jpeg", dpi=300, bbox_inches='tight')
        st.session_state.box_plot = g

        #SCATTERPLOT
        
        df4 = df3.groupby(df3.index).aggregate({'Median': 'first'})
        df5 = pd.concat([data_frame, df4], axis=1, verify_integrity=True)
        df5.dropna(inplace=True)
        df5.reset_index(inplace=True)
        df5.rename(columns={'index': 'Name'}, inplace=True)

        def label_point(x, y, val, ax):
            a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
            for i, point in a.iterrows():
                ax.text(point['x'], point['y'], str(point['val']), fontsize = 7.5)
  
        sns.set_style('darkgrid')
        sns.set_context('talk')
        g = sns.relplot(x='Median', y='Ratio', data = df5, aspect=2,
                        height=5)
        label_point(df5.Median, df5.Ratio, df5.Name, plt.gca())
        g.set_axis_labels(self.score, "Ratio")
        g.set(ylim=(0, 1), yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        g.fig.savefig(f"Scatter_Plot{k}.jpeg", dpi=300, bbox_inches='tight')
        st.session_state.scatterplot = g

###############################################
#                                             #
#            APPLICATION SECTION              #
#                                             #                                                      
###############################################

def render_mol(xyz):
    xyzview = py3Dmol.view()#(width=400,height=400)
    xyzview.addModel(xyz,'mol')
    xyzview.setStyle({'stick':{}})
    xyzview.setBackgroundColor('white')
    xyzview.zoomTo()
    showmol(xyzview, height=500, width=1000)

def main():

    st.header('MCS ConformationTool !')

    st.markdown('Welcome to **MCS ConformationTool** ! Here, the processing and analysis of the docking simulation'
                ' results is done through the **Maximum Common Substructure** (MCS).')

    #SDF FILE SECTION#
    sdf = st.file_uploader("Upload the coordinates of the docked ligand in SDF format:",
                                        type = ["sdf"], key = 'CIT_WEB_MCS_ConformationTool')
    if sdf:
        molecule_name = st.text_input("What is the name of the column in your sdf file that contains the names of the molecules"
                          " (and not the names of each poses resulting from the docking simulations)?", 'Compound Name',
                                       key = 'CIT_WEB_MCS_ConformationTool')
        st.session_state.molecule_name = molecule_name
        score = st.text_input(
             'What is the scoring function used in your sdf file ?',
             'Gold.PLP.Fitness', key = 'CIT_WEB_MCS_ConformationTool')
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

    #PDB PROTEIN SECTION#

    pdb = st.file_uploader("Upload a pdb file for viewing purposes. (FACULTATIVE)",
                                        type = ["pdb"],  key = 'CIT_WEB_MCS_ConformationTool')
    if pdb :
        st.session_state.pdb = pdb
        with open("pdb_file.pdb", "wb") as pdb_file:
            pdb_file.write(pdb.getbuffer())

    #PERCENTAGE PARAMETER SECTION#

    help_ringMatchesRingOnly = ('If "no", depending on the molecules used, it is possible that the RMSD cannot be'
                                ' calculated afterwards. "Yes" may solve this problem in return for having a less'
                                ' relevant RMSD.')

    ringMatchesRingOnly = st.radio(
         "Should the Maximum Common Substructure take into account uniquely complete rings ?",
         ('No', 'Yes'), help=help_ringMatchesRingOnly)

    if  ringMatchesRingOnly == "Yes":
        st.session_state.ringMatchesRingOnly = True
    else:
        st.session_state.ringMatchesRingOnly = False

    help_percentage = ('In case you consider that the maximum common substructure is not enough to be effective,'
                       ' you can lower this parameter. This will result in removing from your sdf those molecules'
                       ' that share THE LEAST common substructure with the other molecules.')

    st.session_state.percentage = st.slider('Selects the percentage of molecules (not poses) for which a maximum'
                                            ' common substructure will be determined.', 0, 100, 100,
                                            help=help_percentage,  key = 'CIT_WEB_MCS_ConformationTool')

    ###############################################
    #--         CHECKBOX "GET MCS SDF"          --#                                                     
    ###############################################

    first_checkbox = st.checkbox(
        'Get the Maximum Common Substructure between molecules of your SDF file (Attention ! Before closing this app,'
        ' please, UNCHECK THIS BOX)',  key = 'CIT_WEB_MCS_ConformationTool')

    if first_checkbox :
        with st.spinner('Please wait, the maximum common substructure is coming...'):
            try : 
                if 'ConformationClass' not in st.session_state :
                    with open("sdf_file.sdf", "wb") as sdf:
                        sdf.write(st.session_state.sdf_file_stock.getbuffer())
                    st.session_state.sdf_file = "sdf_file.sdf"        
                    st.session_state.ConformationClass = ConformationTool(st.session_state.sdf_file,
                                                                          st.session_state.score)
                if 'error_mols' not in st.session_state:
                    st.session_state.ConformationClass.get_MCS_SDF(percentage = st.session_state.percentage,
                                                                   ringMatchesRingOnly=st.session_state.ringMatchesRingOnly)

            except AttributeError:
                st.error("OOPS ! Did you forget to upload your SDF file ?")
                #pass

            render_mol(Chem.MolToMolBlock(st.session_state.fusion))
            st.write("Maximum common substructure between the different molecules of your SDF file")
            #st.image(st.session_state.fusion_im, caption='Maximum common substructure between the different molecules'
            #         ' of your SDF file')   

        show_molecules = st.checkbox('Show molecules which will be not included in the algorithm')
        if show_molecules:
            if st.session_state.error_mols == None :
                st.write('All molecules are good !')
            if st.session_state.error_mols :
                unique_name = []
                unique_mol = []
                for mol in st.session_state.error_mols:
                    if mol.GetProp(st.session_state.molecule_name) in unique_name:
                        continue
                    else:
                        unique_name.append(mol.GetProp(st.session_state.molecule_name))
                        unique_mol.append(mol)
                col1, col2 = st.columns(2)
                with col1 :
                    st.write("Molecule Name")
                with col2 :
                    st.write("Chemical structure of the molecule")          
                for mol in unique_mol :
                    AllChem.Compute2DCoords(mol)
                    col1, col2 = st.columns(2)
                    with col1 :
                        st.write(mol.GetProp(st.session_state.molecule_name))
                    with col2 :
                        st.image(Draw.MolToImage(mol))
                st.warning("These molecules are the ones which share the least common substructure with"
                           " the other molecules in your SDF file.")

        try :
            st.info(f"There are {st.session_state.mols_brut} molecule poses in the sdf file.\n")
        except AttributeError :
            st.error('OOPS ! Have you submitted your sdf file? ?')
        st.info(f"There are {st.session_state.mols} molecule poses which will be used by the algorithm.\n")

        try :
            os.remove("sdf_file.sdf")
            os.remove("mol_ref.mol")
        except :
            pass

        if st.session_state.mols > 200 :
            size_sample = 200
        else :
            size_sample = st.session_state.mols
        individuals = st.slider('Select the size of your sample. Default size of sample = 200', 0, 500, size_sample,
                                help='If you want to change this setting during the program, make sure the box below is unchecked!',
                                key = 'CIT_WEB_MCS_ConformationTool')

        #SECOND BOX
        second_checkbox = st.checkbox('Get the sample heatmap',  key = 'CIT_WEB_MCS_ConformationTool')
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
                                ' is unchecked!',  key = 'CIT_WEB_MCS_ConformationTool')

        Proportion = st.slider('Minimum size of the sample defining a conformation. Default proportion = 0.05',
                             0.0, 1.0, 0.05,
                               help=('This setting define the minimum proportion (value between 0 and 1) of individuals'
                                     ' in a group within the sample to consider that group large enough to be'
                                     ' representative of a full conformation.'),  key = 'CIT_WEB_MCS_ConformationTool')

    ###############################################
    #--   CHECKBOX "GET THE SORTED HEATMAP"     --#                                                     
    ############################################### 

        third_checkbox = st.checkbox('Get the sorted heatmap',  key = 'CIT_WEB_MCS_ConformationTool')
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
                        Chem.MolToPDBFile(merged, f'Conformation n??{i+1}.pdb')
                        xyz_pdb = open(f'Conformation n??{i+1}.pdb', 'r', encoding='utf-8')
                        pdb = xyz_pdb.read().strip()
                        xyz_pdb.close()
                        xyzview = py3Dmol.view(width=700,height=500)
                        xyzview.addModel(pdb, 'pdb')
                        xyzview.setStyle({style:{'color':'spectrum'}})
                        #xyzview.setBackgroundColor(bcolor)#('0xeeeeee')
                        xyzview.setStyle({'resn':'UNL'},{'stick':{}})
                        xyzview.zoomTo({'resn':'UNL'})
                        showmol(xyzview, height = 500,width=1000)
                        os.remove(f'Conformation n??{i+1}.pdb')
                        st.write(f'Conformation n??{i+1}')
                        with open(f"Sample_Conformation{i+1}.sdf", "rb") as file:
                             btn = st.download_button(
                                        label=f"Download all the poses of the conformation n??{i+1} from the SAMPLE",
                                        data=file,
                                         file_name=f"Sample_Conformation{i+1}.sdf")

                with open("Best_PLPScore_Poses.sdf", "rb") as file:
                     btn = st.download_button(
                                label="Download the SDF file including each of the representatives of a conformation",
                                data=file,
                                file_name="Best_Score_Poses.sdf")

                for i, predominant_pose in enumerate(st.session_state.predominant_poses) :
                    st.write(
                        f"The predominant conformation n??{i+1} represents {len(predominant_pose)/len(st.session_state.sample)*100:.1f}" 
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
                    st.session_state.temp = st.select_slider("You want a sdf file or plots including molecules in the conformation n??",
                                                             options=temp_options, value = st.session_state.temp)
                    st.write(f"The Conformation selected is {st.session_state.temp}")

                    st.session_state.RMSD_Target_conformation = st.slider('... With all poses under a RMSD =', 0.0, 15.0, 2.0)
                    st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")

                    settings_checkbox = st.checkbox('Plot Settings (to configure size and some elements of the plots) *Facultative',
                                                    help=st.session_state.help_paragraph)
                    if settings_checkbox :
                        st.session_state.aspect_plot = st.slider(
                            'Configure the aspect ratio of the plots', 0.0, 10.0, 1.75,
                            help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")

                        st.session_state.height_plot = st.slider(
                            'Configure the height of the plots', 0, 50, 18,
                            help="Height (in inches) of each facet.")

                        st.session_state.size_xlabels = st.slider('Configure the size of the axis X labels', 0, 50, 25)
                        st.session_state.size_ylabels = st.slider('Configure the size of the axis Y labels', 0, 50, 15)

                    else :
                        pass


                    if st.button('Prepare your sdf file and build plots'):             
                        st.session_state.ConformationClass.get_sdf_conformations(
                            st.session_state.temp,
                            st.session_state.RMSD_Target_conformation
                            )

                        with open(f'Conformation{st.session_state.temp}.sdf', "rb") as file:
                             btn = st.download_button(
                                        label="Download your sdf file",
                                        data=file,
                                         file_name=f"Conformation n??{st.session_state.temp}.sdf")

                        if "barplot" in st.session_state :
                            del st.session_state.barplot
                        if "box_plot" in st.session_state :
                            del st.session_state.box_plot
                        if "scatterplot" in st.session_state :
                            del st.session_state.scatterplot 

                        st.session_state.ConformationClass.get_plots(
                            st.session_state.temp,
                            st.session_state.molecule_name,
                            aspect_plot = st.session_state.aspect_plot,
                            height_plot = st.session_state.height_plot,
                            size_xlabels = st.session_state.size_xlabels,
                            size_ylabels = st.session_state.size_ylabels)

                        st.pyplot(st.session_state.barplot)
                        st.write("Ratio of each compounds between the number of poses in the conformation selected and the number of total poses.")

                        with open(f"Barplot_Conformation{st.session_state.temp}.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Bar PLOT Conformation n??{st.session_state.temp} ",
                                     data=file,
                                     file_name=f"Barplot_Conformation n??{st.session_state.temp}.jpeg",
                                     mime="image/jpeg")

                        st.write(f"Boxplot built following the descending order of the {st.session_state.score}.")
                        st.pyplot(st.session_state.box_plot)
                        with open(f"Box_Plot{st.session_state.temp}.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Box PLOT Conformation n??{st.session_state.temp} ",
                                     data=file,
                                     file_name=f"Boxplot2_Conformation n??{st.session_state.temp}.jpeg",
                                     mime="image/jpeg")
                        st.write(f"Scatter plot built with the ratio as function of the {st.session_state.score}.")
                        st.pyplot(st.session_state.scatterplot)
                        with open(f"Scatter_Plot{st.session_state.temp}.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Scatter Plot Conformation n??{st.session_state.temp} ",
                                     data=file,
                                     file_name=f"Scatter_Plot n??{st.session_state.temp}.jpeg",
                                     mime="image/jpeg")
                    else :
                        try :
                            with open(f'Conformation{st.session_state.temp}.sdf', "rb") as file:
                                 btn = st.download_button(
                                            label="Download your sdf file",
                                            data=file,
                                             file_name=f"Conformation n??{st.session_state.temp}.sdf")

                            st.pyplot(st.session_state.barplot)

                            st.write("Ratio of each compounds between the number of poses in the conformation selected and the number of total poses.")
                            with open(f"Barplot_Conformation{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Bar PLOT Conformation n??{st.session_state.temp} ",
                                         data=file,
                                         file_name=f"Barplot_Conformation n??{st.session_state.temp}.jpeg",
                                         mime="image/jpeg")

                            st.write(f"Boxplot built following the descending order of the {st.session_state.score}.")
                            st.pyplot(st.session_state.box_plot)
                            with open(f"Box_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Box Plot Conformation n??{st.session_state.temp} ",
                                         data=file,
                                         file_name=f"Boxplot2_Conformation n??{st.session_state.temp}.jpeg",
                                         mime="image/jpeg")
                            st.write(f"Scatter plot built with the ratio as function of the {st.session_state.score}.")
                            st.pyplot(st.session_state.scatterplot)
                            with open(f"Scatter_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Scatter Plot Conformation n??{st.session_state.temp} ",
                                         data=file,
                                         file_name=f"Scatter_Plot n??{st.session_state.temp}.jpeg",
                                         mime="image/jpeg")

                        except :
                            pass

                else :                              
                    st.session_state.RMSD_Target_conformation = st.slider(
                        'You want a sdf file and/or plot analysis including molecules in the unique predominant conformation with all poses under a RMSD =',
                        0.0, 15.0, 2.0)

                    st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")

                    st.info('There is only one predominant conformation. Do you want to have the sdf file of poses in this conformation OR see analysis of this conformation ? ')

                    settings_checkbox = st.checkbox('Plot Settings (to configure size and some elements of the plots) *Facultative',
                                                    help=st.session_state.help_paragraph)
                    if settings_checkbox :
                        st.session_state.aspect_plot = st.slider(
                            'Configure the aspect ratio of the plts', 0.0, 10.0, 1.75,
                            help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")

                        st.session_state.height_plot = st.slider(
                            'Configure the height of the plots', 0, 50, 18,
                            help="Height (in inches) of each facet.")

                        st.session_state.size_xlabels = st.slider('Configure the size of the axis X labels', 0, 50, 25)
                        st.session_state.size_ylabels = st.slider('Configure the size of the axis Y labels', 0, 50, 15)


                    bouton = st.button('I want to have the sdf file of poses in this conformation AND/OR the plots.')
                    if bouton :
                        st.session_state.ConformationClass.get_sdf_conformations(
                            1, st.session_state.RMSD_Target_conformation)

                        with open('Conformation1.sdf', "rb") as file:
                         btn = st.download_button(
                                    label="Download your sdf file",
                                    data=file,
                                     file_name=f"Unique Conformation.sdf")

                        if "barplot" in st.session_state :
                            del st.session_state.barplot
                        if "box_plot" in st.session_state :
                            del st.session_state.box_plot
                        if "scatterplot" in st.session_state :
                            del st.session_state.scatterplot 


                        st.session_state.ConformationClass.get_plots(
                            1,
                            st.session_state.molecule_name,
                            aspect_plot = st.session_state.aspect_plot,
                            height_plot = st.session_state.height_plot,
                            size_xlabels = st.session_state.size_xlabels,
                            size_ylabels = st.session_state.size_ylabels)

                        st.pyplot(st.session_state.barplot)
                        st.write("Ratio of each compounds between the number of poses in the conformation selected and the number of total poses.")

                        with open(f"Barplot_Conformation1.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Bar PLOT Conformation n??1 ",
                                     data=file,
                                     file_name=f"Barplot_Conformation n??1.jpeg",
                                     mime="image/jpeg")

                        st.write(f"Boxplot built following the descending order of the {st.session_state.score}.")
                        st.pyplot(st.session_state.box_plot)
                        with open(f"Box_Plot1.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Box Plot Conformation n??1 ",
                                     data=file,
                                     file_name=f"Boxplot2_Conformation n??1.jpeg",
                                     mime="image/jpeg")

                        st.write(f"Scatterplot built with the ratio as a function of the {st.session_state.score}.")
                        st.pyplot(st.session_state.scatterplot)
                        with open(f"Scatter_Plot1.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Scatter Plot Conformation n??1 ",
                                     data=file,
                                     file_name=f"Scatter_Plot n??1.jpeg",
                                     mime="image/jpeg")
                    else :
                        try :
                            with open(f'Conformation1.sdf', "rb") as file:
                                 btn = st.download_button(
                                            label="Download your sdf file",
                                            data=file,
                                             file_name=f"Conformation n??1.sdf")

                            st.pyplot(st.session_state.barplot)
                            st.write("Ratio of each compounds between the number of poses in the conformation selected and the number of total poses.")

                            with open(f"Barplot_Conformation1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Bar PLOT Conformation n??1 ",
                                         data=file,
                                         file_name=f"Barplot_Conformation n??1.jpeg",
                                         mime="image/jpeg")

                            st.write(f"Boxplot built following the descending order of the {st.session_state.score}.")
                            st.pyplot(st.session_state.box_plot)
                            with open(f"Box_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Box Plot Conformation n??1 ",
                                         data=file,
                                         file_name=f"Boxplot2_Conformation n??1.jpeg",
                                         mime="image/jpeg")

                            st.write(f"Scatterplot built with the ratio as a function of the {st.session_state.score}.")
                            st.pyplot(st.session_state.scatterplot)
                            with open(f"Scatter_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Scatter Plot Conformation n??1 ",
                                         data=file,
                                         file_name=f"Scatter_Plot n??1.jpeg",
                                         mime="image/jpeg")
                        except :
                            pass


            except KeyError :
                st.error("The name of the column in your sdf file that contains the names of the molecules doesn't seem to be "
                           f"'{st.session_state.molecule_name}'. Please correct it.")
            except AttributeError as e :
                #st.exception(e)
                pass


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
                    st.session_state.temp = st.select_slider("You want a sdf file and/or a anlysis plots including molecules in the conformation n??",
                                                             options=temp_options)
                    st.write(f"The Conformation selected is {st.session_state.temp}")

                    st.session_state.RMSD_Target_conformation = st.slider('... With all poses under a RMSD =', 0.0, 15.0, 2.0)
                    st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")

                    if "aspect_plot" not in st.session_state :
                        st.session_state.aspect_plot = 1.75
                    if "height_plot" not in st.session_state :
                        st.session_state.height_plot = 18
                    if "size_xlabels" not in st.session_state :
                        st.session_state.size_xlabels = 25
                    if "size_ylabels" not in st.session_state :
                        st.session_state.size_ylabels = 15

                    st.session_state.help_paragraph = (
                        "To give an idea, if your number of molecules (not number of poses) = 15 :\n"
                        "- Aspect ratio = 3, Height = 5, Xlabels Size = 25, Ylabels Size = 30\n "
                        "\nif your number of molecules (not number of poses) = 75 :\n - Aspect ratio = 1.75,"
                        " Height = 18, Xlabels Size = 25, Ylabels Size = 15")

                    settings_checkbox = st.checkbox('Plot Settings (to configure size and some elements of the plots) *Facultative',
                                                    help=st.session_state.help_paragraph)
                    if settings_checkbox :
                        st.session_state.aspect_plot = st.slider(
                            'Configure the aspect ratio of the plots', 0.0, 10.0, 1.75,
                            help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")

                        st.session_state.height_plot = st.slider(
                            'Configure the height of the plots', 0, 50, 18,
                            help="Height (in inches) of each facet.")

                        st.session_state.size_xlabels = st.slider('Configure the size of the axis X labels', 0, 50, 25)
                        st.session_state.size_ylabels = st.slider('Configure the size of the axis Y labels', 0, 50, 15)

                    if st.button('Prepare your sdf file and build plots'):             
                        st.session_state.ConformationClass.get_sdf_conformations(
                            st.session_state.temp,
                            st.session_state.RMSD_Target_conformation
                            )
                else :
                    st.info('There is only one predominant conformation. Do you want to have the sdf file of poses in this conformation OR see analysis of this conformation ? ')
                    st.session_state.RMSD_Target_conformation = st.slider('You want a sdf file and/or a analysis plots including molecules in the unique predominant conformation with all poses under a RMSD =', 0.0, 15.0, 2.0)

                    if "aspect_plot" not in st.session_state :
                        st.session_state.aspect_plot = 1.75
                    if "height_plot" not in st.session_state :
                        st.session_state.height_plot = 18
                    if "size_xlabels" not in st.session_state :
                        st.session_state.size_xlabels = 25
                    if "size_ylabels" not in st.session_state :
                        st.session_state.size_ylabels = 15

                    st.session_state.help_paragraph = (
                        "To give an idea, if your number of molecules (not number of poses) = 15 :\n"
                        "- Aspect ratio = 3, Height = 5, Xlabels Size = 25, Ylabels Size = 30\n "
                        "\nif your number of molecules (not number of poses) = 75 :\n - Aspect ratio = 1.75,"
                        " Height = 18, Xlabels Size = 25, Ylabels Size = 15")

                    settings_checkbox = st.checkbox('Plot Settings (to configure size and some elements of the plots) *Facultative',
                                                    help=st.session_state.help_paragraph)
                    if settings_checkbox :
                        st.session_state.aspect_plot = st.slider(
                            'Configure the aspect ratio of the plots', 0.0, 10.0, 2.5,
                            help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")

                        st.session_state.height_plot = st.slider(
                            'Configure the height of the plots', 0, 50, 7,
                            help="Height (in inches) of each facet.")

                        st.session_state.size_xlabels = st.slider('Configure the size of the axis X labels', 0, 50, 25)
                        st.session_state.size_ylabels = st.slider('Configure the size of the axis Y labels', 0, 50, 25)


                    if st.button('I want to have the sdf file of poses in this conformation AND/OR the plots.'):             
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
            if 'barplot' in st.session_state:
                del st.session_state.barplot
            if 'box_plot' in st.session_state:
                del st.session_state.box_plot
            if 'scatterplot' in st.session_state:
                del st.session_state.scatterplot



    else :
        if 'ConformationClass' in st.session_state:
            del st.session_state.ConformationClass
        if 'fusion' in st.session_state:
            del st.session_state.fusion
        if 'percentage' in st.session_state:
            del st.session_state.percentage
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
        if 'barplot' in st.session_state:
            del st.session_state.barplot
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
                    os.remove(f'Barplot_Conformation{i+1}.jpeg')
                    os.remove(f'Scatter_Plot{i+1}.jpeg')
                    os.remove(f'Box_Plot{i+1}.jpeg')
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
        if 'barplot' in st.session_state:
            del st.session_state.barplot
        if 'box_plot' in st.session_state:
            del st.session_state.box_plot
        if 'scatterplot' in st.session_state:
            del st.session_state.scatterplot
