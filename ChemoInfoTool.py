"""
#####################################################
##                                                 ##
##        -- CIT CONFORMATIONTOOL CLASS --         ##
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
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from scipy.cluster import hierarchy

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
                 score = "Gold.PLP.Fitness",
                 conformation_tool = False,
                 streamlit = False):
        """
        -- DESCRIPTION --
        This class takes as input the sdf file containing the results of the docking simulations.
        The name of the column in the sdf that uses the score result must also be specified,
        by default "Gold.PLP.Fitness".
            PARAMS:
                - sdf_file (string): path to a SDF file
                - score (string): Name of the column in the sdf
        """
        
        
        self.conformation_tool = conformation_tool
        self.streamlit = streamlit
        
        if self.conformation_tool == "Unique":
            self.mols = [x for x in Chem.SDMolSupplier(sdf_file)]
            st.session_state.mols_brut = len(self.mols)
            self.score = score
            
        else :
            self.mols_brut = [x for x in Chem.SDMolSupplier(sdf_file)]
            st.session_state.mols_brut = len(self.mols_brut)
            self.score = score
            
            if self.conformation_tool == "MCS" :
                sdf_preprocessed = preprocess(self.mols_brut)
                self.sdf_preprocessed = sdf_preprocessed        
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

        if self.streamlit == True:       
            st.session_state.error_mols = error_mols
            st.session_state.mols = len(mols)
        
        self.mols = mols
        self.MCS_mols = MCS_mols        
        self.fusion = fusion
        
        del error_mols
        del self.mols_brut
        del self.sdf_preprocessed
    
    def check_sdf_file(self, benchmark_molecule):
        """
        -- DESCRIPTION --
        This function allows to check, from a reference molecule (whose pathway will be provided),
        that all molecules in the sdf file share the same structural backbone with the reference molecule.
        Molecules that do not meet this criterion will not be included in the rest of the algorithm.
            PARAMS:
                - benchmark_molecule (string): code SMILES
        """
        
        mol_benchmark = Chem.MolFromSmiles(benchmark_molecule)
        AllChem.EmbedMolecule(mol_benchmark)
        mols = []
        error_mols = []
        for mol in self.mols_brut :
            try :
                CalcRMS(GetScaffoldForMol(mol_benchmark), GetScaffoldForMol(mol))
                CalcRMS(GetScaffoldForMol(mol), GetScaffoldForMol(mol_benchmark))
                mols.append(mol)
            except RuntimeError as e :
                error_mols.append(mol)

        if self.streamlit == True:       
            st.session_state.error_mols = error_mols
            st.session_state.mols = len(mols)

        self.mols = mols
        
        del self.mols_brut
        
            
    def get_RMSD_dataframe_sample(self) :
        
        array = np.ones(shape=(len(self.sample),len(self.sample)))
        for i, indivduali in enumerate(self.sample) :
            for j, indivdualj in enumerate(self.sample) :
                if self.conformation_tool == "Unique":
                    array[i, j] = CalcRMS(self.mols[indivduali],
                                          self.mols[indivdualj])
                elif self.conformation_tool == "Murcko":
                    try :
                        array[i, j] = CalcRMS(GetScaffoldForMol(self.mols[indivduali]),
                                              GetScaffoldForMol(self.mols[indivdualj]))
                    except RuntimeError :
                        array[i, j] = CalcRMS(GetScaffoldForMol(self.mols[indivdualj]),
                                              GetScaffoldForMol(self.mols[indivduali]))

                elif self.conformation_tool == "MCS":
                    try :
                        array[i, j] = CalcRMS(self.MCS_mols[indivduali],
                                              self.MCS_mols[indivdualj])
            
                    except RuntimeError :
                        array[i, j] = CalcRMS(self.MCS_mols[indivdualj],
                                              self.MCS_mols[indivduali])
                    

        df = pd.DataFrame(
            array, index=list(range(len(self.sample))), columns=list(range(len(self.sample))))
        
        if self.streamlit == True:
            st.session_state.df_sample = df
            
        return df
    
    def get_heatmap_sample(self, individuals = 200, print_plot = True) :
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
                    
        df = self.get_RMSD_dataframe_sample()

        if print_plot == True :  
            fig, ax = plt.subplots(figsize=(20, 10))
            sns.set_context('talk')
            sns.heatmap(df, fmt='d', ax= ax)
        
        if self.streamlit == True:
            st.session_state.df_sample = df
            st.pyplot(fig)
            st.session_state.heatmap = fig
    
    def get_predominant_poses(self, output_liste, p) :
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

        predominant_poses = [group for group in output_liste if len(group)/len(self.sample)*100 > p*len(self.sample)]
        
        if self.streamlit == True :
            st.info(f"There is (are) {len(predominant_poses)} predominant pose(s) among all poses.\n")
            st.session_state.n_conformations = len(predominant_poses)
            st.session_state.predominant_poses = predominant_poses
            for i, predominant_pose in enumerate(st.session_state.predominant_poses) :
                st.write(
                    f"The predominant pose n°{i+1} represents {len(predominant_pose)/len(st.session_state.sample)*100:.1f}" 
                    f"% of the sample, i.e. {len(predominant_pose)} on {len(st.session_state.sample)} poses in total.")

        for i, predominant_pose in enumerate(predominant_poses) :
            print(f"The predominant conformation n°{i+1} represents {len(predominant_pose)/len(self.sample)*100:.1f}", 
                  f"% of the sample, i.e. {len(predominant_pose)} on {len(self.sample)} poses in total.")
        return predominant_poses

    def get_sample_indice_best_score(self, predominant_poses) :
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
    
    def get_SDF_Sample_and_Best_Score_Poses(self, sample_predominant_poses, sample_indice_best_score) :
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
                if self.conformation_tool == "Unique":
                    for j, mol in enumerate(self.mols) :
                        result = CalcRMS(self.mols[self.sample[sample_predominant_poses[i][indice]]],
                                         mol)
                        if result < 2 :
                            subliste.append(self.mols[j])
                            w.write(self.mols[j]) 
                    copy1 = copy.deepcopy(subliste)
                    a_supprimer.append(copy1)
                        
                elif self.conformation_tool == "Murcko":
                    for j, mol in enumerate(self.mols) :
                        try:
                            result = CalcRMS(GetScaffoldForMol(self.mols[self.sample[sample_predominant_poses[i][indice]]]),
                                                       GetScaffoldForMol(mol))
                        except RuntimeError:
                            result = CalcRMS(GetScaffoldForMol(mol),
                                             GetScaffoldForMol(self.mols[self.sample[sample_predominant_poses[i][indice]]]))
                        if result < 2 :
                            subliste.append(self.mols[j])
                            w.write(self.mols[j]) 
                    copy1 = copy.deepcopy(subliste)
                    a_supprimer.append(copy1)

                elif self.conformation_tool == "MCS":
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
            PLPscore = [mol.GetProp(self.score) for mol in group]
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

    def get_data_frame_best_poses(self, input_list) :
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
            columns[i] = f"Conformation n°{i+1}"

        index = list(range(len(preprocess_list)))
        for i in index :
            index[i] = f"Conformation n°{i+1}"
        
        array = np.ones(shape=(len(preprocess_list),len(preprocess_list)))

        for i, moli in enumerate(preprocess_list) :
            for j, molj in enumerate(preprocess_list) :
                
                if self.conformation_tool == "Unique":
                    array[i, j] = CalcRMS(moli, molj)
                
                elif self.conformation_tool == "Murcko":
                    try :
                        array[i, j] = CalcRMS(GetScaffoldForMol(moli),
                                      GetScaffoldForMol(molj))
                    except RuntimeError :
                        array[i, j] = CalcRMS(GetScaffoldForMol(molj),
                                      GetScaffoldForMol(moli))
                
                elif self.conformation_tool == "MCS":
                    array[i, j] = MCS_RMSD(moli, molj)

        data_frame = pd.DataFrame(array, index=index, columns=columns)
        
        if self.streamlit == True :
            st.write("\nIn order to check that each group is different from each other, a table taking " 
              "the individual **with the best score** from each group and calculating the RMSD between each was constructed :\n")
            st.dataframe(data_frame)
            st.write("RMSD value between each representative of each conformation.")
            st.session_state.df1 = data_frame

    def get_histogramme_sample_bestPLP(self, input_list) :
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
                if self.conformation_tool == "Unique":
                    sdf_to_hist = [CalcRMS(input_list[0], mol) for mol in self.mols]
                    
                elif self.conformation_tool == "Murcko":
                    try :
                        sdf_to_hist = [CalcRMS(GetScaffoldForMol(input_list[0]), GetScaffoldForMol(mol)) for mol in self.mols]
                    except RuntimeError:
                        sdf_to_hist = [CalcRMS(GetScaffoldForMol(mol), GetScaffoldForMol(input_list[0])) for mol in self.mols]
                        
                elif self.conformation_tool == "MCS":
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
                ax.hist(sdf_to_hist, bins =100, label = "Conformation n°1")
                ax.axvline(x=2, ymin=0, ymax=1, color="black", linestyle="--")
                ax.annotate(a, (1.5, 0.05*len(self.mols)), fontsize=15)
                ax.axvline(x=3, ymin=0, ymax=1, color="black", linestyle="--")
                ax.annotate(b-a, (2.5, 0.05*len(self.mols)), fontsize=15)
                ax.axvline(x=4, ymin=0, ymax=1, color="black", linestyle="--")
                ax.annotate(c-b, (3.5, 0.05*len(self.mols)), fontsize=15)
                ax.legend(loc='upper left', shadow=True, markerfirst = False)
            
            else :
                if self.conformation_tool == "Unique":
                    sdf_to_hist = ([CalcRMS(representative_conf,
                                            mol) for mol in self.mols] for representative_conf in input_list)
                    
                elif self.conformation_tool == "Murcko":
                    try:
                        sdf_to_hist = ([CalcRMS(GetScaffoldForMol(representative_conf),
                                                GetScaffoldForMol(mol)) for mol in self.mols] for representative_conf in input_list)
                    except RuntimeError:
                        sdf_to_hist = ([CalcRMS(GetScaffoldForMol(mol),
                                                GetScaffoldForMol(representative_conf)) for mol in self.mols] for representative_conf in input_list)
                
                elif self.conformation_tool == "MCS":
                    try:
                        sdf_to_hist = ([CalcRMS(Get_MCS_Fusion(self, representative_conf),
                                                mol) for mol in self.MCS_mols] for representative_conf in input_list)
                    except RuntimeError:
                        sdf_to_hist = ([CalcRMS(mol, Get_MCS_Fusion(self, representative_conf)) for mol in self.MCS_mols] for representative_conf in input_list)
                
                fig, ax = plt.subplots(len(input_list), 1, figsize=(15, 0.2*len(self.best_PLP_poses)*9))
                for z, group in enumerate(sdf_to_hist) :
                    a, b, c = 0, 0, 0
                    for i in group : 
                        if i < 2 :
                            a += 1
                        if i < 3 :
                            b += 1
                        if i < 4 :
                            c += 1
                    ax[z].hist(group, bins =100, label =f"Conformation n°{z+1}")
                    ax[z].axvline(x=2, ymin=0, ymax=1, color="black", linestyle="--")
                    ax[z].annotate(a, (1.5, 0.05*len(self.mols)), fontsize=15)
                    ax[z].axvline(x=3, ymin=0, ymax=1, color="black", linestyle="--")
                    ax[z].annotate(b-a, (2.5, 0.05*len(self.mols)), fontsize=15)
                    ax[z].axvline(x=4, ymin=0, ymax=1, color="black", linestyle="--")
                    ax[z].annotate(c-b, (3.5, 0.05*len(self.mols)), fontsize=15)
                    ax[z].legend(loc='upper left', shadow=True, markerfirst = False)               

            if self.streamlit == True :
                st.pyplot(fig)
                st.write("Density of the number of poses as a function of the RMSD calculated between the representative of each conformation"
                 " and all poses of all molecules in the docking solutions of the filtered incoming sdf file.")
                st.session_state.histplot = fig
                fig.savefig("Histograms_Best_Score.jpeg", dpi=300, bbox_inches='tight')
                
                with open("Histograms_Best_Score.jpeg", "rb") as file:
                     btn = st.download_button(
                                label="Download PLOT Histograms",
                                data=file,
                                 file_name="Histograms_Best_Score.jpeg",
                                 mime="image/jpeg")
    
    def get_cluster_heatmap(self, individuals, method = 'canberra', metric = 'average'):      
        try :
            sample = random.sample(range(len(self.mols)), individuals)
            if self.streamlit == True:
                st.session_state.sample = sample
            self.sample = sample
        except ValueError :
            st.error("OOPS ! You're trying to define a sample more larger"
                     " than the numbers of molecules/poses in your sdf."
                     " This is impossible, please redefine a size of sample"
                     " equal or smaller to your sdf")
        
        self.get_RMSD_dataframe_sample()
        with st.spinner('Process in progress. Please wait.'):
            sns.set_context('talk')
            fig = sns.clustermap(st.session_state.df_sample, figsize=(20, 15),method=method,
                                 metric=metric, dendrogram_ratio=0.3, tree_kws = {"linewidths": 1.5},
                                 cbar_pos=(1, 0.1, .03, .5))
            fig.savefig("Cluster_Hierarchy_Heatmap.jpeg", dpi=300, bbox_inches='tight')
            st.pyplot(fig)
            st.session_state.cluster_hierarchy_heatmap = fig
        
            with open("Cluster_Hierarchy_Heatmap.jpeg", "rb") as file:
                 btn = st.download_button(
                         label="Download PLOT : Cluster Hierarchy Heatmap",
                         data=file,
                         file_name="Cluster_Hierarchy_Heatmap.jpeg",
                         mime="image/jpeg",
                         key = 'CIT_WEB_Unique_Molecule_ConformationTool')
             
        cluster = hierarchy.linkage(st.session_state.df_sample, metric = st.session_state.metric,
                                    method = st.session_state.method, optimal_ordering=True)
        st.session_state.cluster = cluster
        
    def analyze_cluster_heatmap(self, n_clusters, p = 0.05) :
        
        cutree = hierarchy.cut_tree(st.session_state.cluster, n_clusters=st.session_state.n_clusters)
        liste = [int(x) for x in cutree]
        tuples = list(zip(liste, self.mols))
        dataframe = pd.DataFrame(np.array(tuples), columns=["Clusters", "Poses"])
        clusters = []
        for i in range(st.session_state.n_clusters) :
            clusters.append([j for j, mol in enumerate(dataframe.loc[:, 'Poses'])
                             if dataframe.loc[dataframe.index[j], 'Clusters'] == i])
        
        sample_predominant_poses = self.get_predominant_poses(clusters, p)
        sample_indice_best_score = self.get_sample_indice_best_score(sample_predominant_poses)
        self.get_SDF_Sample_and_Best_Score_Poses(sample_predominant_poses, sample_indice_best_score)
        
        if self.conformation_tool == "Unique":
            best_PLP_poses = [x for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
        elif self.conformation_tool == "Murcko":
            best_PLP_poses = [GetScaffoldForMol(x) for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
        elif self.conformation_tool == "MCS":
            best_PLP_poses = [x for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
            self.best_PLP_poses = best_PLP_poses
            best_PLP_poses_preprocessed = preprocess(best_PLP_poses)
        
        if self.streamlit == True :
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
        
        if self.conformation_tool == "Unique" or self.conformation_tool == "Murcko" :
            self.get_data_frame_best_poses(best_PLP_poses)
            self.best_PLP_poses = best_PLP_poses
            self.get_histogramme_sample_bestPLP(best_PLP_poses)
        
        if self.conformation_tool == "MCS":
            self.get_data_frame_best_poses(best_PLP_poses_preprocessed)
            self.get_histogramme_sample_bestPLP(best_PLP_poses_preprocessed)
            self.best_PLP_poses_preprocessed = best_PLP_poses_preprocessed
            
    
    def get_sorted_heatmap(self, individuals = 200, RMSDthreshold = 3.0, loop = 3, p = 0.05, cluster_hierarchy_heatmap = False) :
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
            if self.streamlit == True:
                st.session_state.sample = sample
            self.sample = sample
        except ValueError :
            st.error("OOPS ! You're trying to define a sample more larger"
                     " than the numbers of molecules/poses in your sdf."
                     " This is impossible, please redefine a size of sample"
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
                    if self.conformation_tool == "Unique":
                        result = CalcRMS(self.mols[individual],
                                   self.mols[sample[i]])
                    elif self.conformation_tool == "Murcko":
                        try :
                            result = CalcRMS(GetScaffoldForMol(self.mols[individual]),
                                             GetScaffoldForMol(self.mols[sample[i]]))
                        except RuntimeError:
                            result = CalcRMS(GetScaffoldForMol(self.mols[sample[i]]),
                                             GetScaffoldForMol(self.mols[individual]))
                    elif self.conformation_tool == "MCS":
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
                                if self.conformation_tool == "Unique":
                                    RMSD.append(CalcRMS(self.mols[self.sample[k]],
                                                        self.mols[self.sample[input_list[i][n]]]))
                                    
                                elif self.conformation_tool == "Murcko":
                                    try:
                                        RMSD.append(CalcRMS(GetScaffoldForMol(self.mols[self.sample[k]]),
                                                            GetScaffoldForMol(self.mols[self.sample[input_list[i][n]]])))
                                    except RuntimeError:
                                        RMSD.append(CalcRMS(GetScaffoldForMol(self.mols[self.sample[input_list[i][n]]]),
                                                            GetScaffoldForMol(self.mols[self.sample[k]])))

                                elif self.conformation_tool == "MCS":
                                    try:
                                        RMSD.append(CalcRMS(self.MCS_mols[self.sample[k]],
                                                            self.MCS_mols[self.sample[input_list[i][n]]]))
                                    except RuntimeError:
                                        RMSD.append(CalcRMS(self.MCS_mols[self.sample[input_list[i][n]]],
                                                            self.MCS_mols[self.sample[k]]))
                        if len(groupej) > len(groupei):
                            for k in groupei :
                                n += 1
                                if self.conformation_tool == "Unique":
                                    RMSD.append(CalcRMS(self.mols[self.sample[k]],
                                                        self.mols[self.sample[input_list[j][n]]]))
                                    
                                elif self.conformation_tool == "Murcko":
                                    try:
                                        RMSD.append(CalcRMS(GetScaffoldForMol(self.mols[self.sample[k]]),
                                                            GetScaffoldForMol(self.mols[self.sample[input_list[j][n]]])))
                                    except RuntimeError:
                                        RMSD.append(CalcRMS(GetScaffoldForMol(self.mols[self.sample[input_list[j][n]]]),
                                                            GetScaffoldForMol(self.mols[self.sample[k]])))

                                elif self.conformation_tool == "MCS":
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
                
                if self.conformation_tool == "Unique":
                    try :
                        r = CalcRMS(self.mols[self.sample[n]],
                                    self.mols[self.sample[liste1[i+1]]])
                    except IndexError :
                        pass
                
                elif self.conformation_tool == "Murcko":
                    try :
                        r = CalcRMS(GetScaffoldForMol(self.mols[self.sample[n]]),
                                    GetScaffoldForMol(self.mols[self.sample[liste1[i+1]]]))
                    except RuntimeError :
                        r = CalcRMS(GetScaffoldForMol(self.mols[self.sample[liste1[i+1]]]),
                                    GetScaffoldForMol(self.mols[self.sample[n]]))
                    except IndexError :
                        pass
                
                elif self.conformation_tool == "MCS":
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
        
        
        ###############################################
        #--      Function "Get sorted Heatmap"      --#                                                     
        ###############################################
        output_liste = sorted_list_lengroups(get_groups_inside_list(improve_sort(
            get_filtered_liste(sorted_list_lengroups(gather_groups_RMSD(RMSD_listes(self.sample)))), loop)))
        finallyliste = get_filtered_liste(output_liste)
        
        if len(finallyliste) != individuals :
            st.warning(f"Attention. The sorting process discarded {individuals-len(finallyliste)} individuals")
        st.session_state.indviduals_deleted = individuals-len(finallyliste)
        array = np.ones(shape=(len(finallyliste),len(finallyliste)))
        
        for i, indivduali in enumerate(finallyliste) :
            for j, indivdualj in enumerate(finallyliste) :
                if self.conformation_tool == "Unique":
                    array[i, j] = CalcRMS(self.mols[self.sample[indivduali]],
                                          self.mols[self.sample[indivdualj]])
                
                elif self.conformation_tool == "Murcko":
                    try :
                        array[i, j] = CalcRMS(GetScaffoldForMol(self.mols[self.sample[indivduali]]),
                                              GetScaffoldForMol(self.mols[self.sample[indivdualj]]))
                    except RuntimeError :
                        array[i, j] = CalcRMS(GetScaffoldForMol(self.mols[self.sample[indivdualj]]),
                                              GetScaffoldForMol(self.mols[self.sample[indivduali]]))
                        
                elif self.conformation_tool == "MCS":
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
            if self.streamlit == True :
                st.pyplot(fig)
                st.session_state.sorted_heatmap = fig
                with open("Sorted_Heatmap.jpeg", "rb") as file:
                     btn = st.download_button(
                             label="Download PLOT sorted heatmap",
                             data=file,
                             file_name="Sorted_Heatmap.jpeg",
                             mime="image/jpeg")
        except ValueError :
            st.error("OOPS ! The selected RMSD threshold does not allow all individuals to be grouped "
                  "into distinct groups that share a low RMSD. Please change the RMSD threshold.")
                     
        sample_predominant_poses = self.get_predominant_poses(output_liste, p)
        sample_indice_best_score = self.get_sample_indice_best_score(sample_predominant_poses)
        self.get_SDF_Sample_and_Best_Score_Poses(sample_predominant_poses, sample_indice_best_score)
        
        if self.conformation_tool == "Unique":
            best_PLP_poses = [x for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
        elif self.conformation_tool == "Murcko":
            best_PLP_poses = [GetScaffoldForMol(x) for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
        elif self.conformation_tool == "MCS":
            best_PLP_poses = [x for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
            self.best_PLP_poses = best_PLP_poses
            best_PLP_poses_preprocessed = preprocess(best_PLP_poses)
        
        if self.streamlit == True :
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
        
        if self.conformation_tool == "Unique" or self.conformation_tool == "Murcko" :
            self.get_data_frame_best_poses(best_PLP_poses)
            self.best_PLP_poses = best_PLP_poses
            self.get_histogramme_sample_bestPLP(best_PLP_poses)
        
        elif self.conformation_tool == "MCS":
            self.get_data_frame_best_poses(best_PLP_poses_preprocessed)
            self.get_histogramme_sample_bestPLP(best_PLP_poses_preprocessed)
            self.best_PLP_poses_preprocessed = best_PLP_poses_preprocessed
        
    
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
            if self.conformation_tool == "Unique":
                for j, mol in enumerate(self.mols) :
                    result = CalcRMS(self.best_PLP_poses[k-1], mol)
                    if result < float(RMSDtarget) :
                        w.write(self.mols[j])
                
            elif self.conformation_tool == "Murcko":
                for j, mol in enumerate(self.mols) :
                    try:
                        result = CalcRMS(GetScaffoldForMol(self.best_PLP_poses[k-1]), GetScaffoldForMol(mol))
                    except RuntimeError:
                        result = CalcRMS(GetScaffoldForMol(mol), GetScaffoldForMol(self.best_PLP_poses[k-1]))
                    if result < float(RMSDtarget) :
                        w.write(self.mols[j])

            elif self.conformation_tool == "MCS":
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
        if self.streamlit == True :
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
        if self.streamlit == True :
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
        if self.streamlit == True :
            st.session_state.scatterplot = g
