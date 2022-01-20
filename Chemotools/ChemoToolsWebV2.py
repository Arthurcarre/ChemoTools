import os
import copy, random
import shutil
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from PIA.PIA import PIA as PIA
from PIA.PIA import Preparation as Preparation

class Conformations :
    def __init__(self, 
                 sdf_file,
                 RMSDtarget,
                 individuals):
        self.sdf_file = sdf_file
        self.suppl_brut = [x for x in Chem.SDMolSupplier(self.sdf_file)]
        self.premols = [GetScaffoldForMol(x) for x in self.suppl_brut]
        self.RMSDtarget = RMSDtarget
        self.individuals = individuals

    def check_sdf_file(self, molref):
        """
    Petit script qui séléctionne une molécule de référence 
    et qui retire toutes les molécules qui posent problème.
    molref : pathway to the filename of the benchmark molecule
    moleculename : Name of the column who contains the name of the molecule
        """
        self.molref = Chem.MolFromMolFile(molref)
        mols = []
        suppl = []
        error_molecules = []
        for i, mol in enumerate(self.premols) :
            try :
                rdMolAlign.CalcRMS(mol, self.molref)
                mols.append(mol)
                suppl.append(self.suppl_brut[i])
            except RuntimeError as e :
                error_molecules.append(mol.GetProp("_Name"))

        if st.checkbox('Show molecules name which are not included'):
            if error_molecules == False :
                st.write('All molecules are good !')
            if error_molecules :
                for i in error_molecules : 
                    st.write(i)
                st.write("All theses molecules have not sub-structure which match between the reference and probe mol.\n",
                          "An RMSD can't be calculated with these molecules, they will therefore not be taken into", 
                          "account by the algorithm.\n")

        self.mols = mols
        self.suppl = suppl
        self.sample = random.sample(range(len(self.mols)), self.individuals)
        st.write(f"There are {len(self.premols)} molecules' poses in the sdf file.\n")
        st.write(f"There are {len(self.mols)} molecules' poses which will be used by the algorithm.\n")
    
    def get_heatmap_sample(self) :
    #Commenter cette fonction
        array = np.ones(shape=(len(self.sample),len(self.sample)))
        for i, indivduali in enumerate(self.sample) :
            for j, indivdualj in enumerate(self.sample) :
                array[i, j] = rdMolAlign.CalcRMS(self.mols[indivduali],self.mols[indivdualj])

        df = pd.DataFrame(
            array, index=list(range(len(self.sample))), columns=list(range(len(self.sample))))

        fig, ax = plt.subplots(figsize=(40, 20))
        sns.heatmap(df, fmt='d', ax= ax)

        print("The table, and therefore the heatmap, was constructed in the order in which the sample was constructed", 
              "(i.e. randomly). To make this heatmap more readable, the different individuals have been grouped", 
              f"according to whether the RMSD between each individual in a group is less than {self.RMSDtarget}.\n")
        st.pyplot(fig)
    
    def get_heatmap(self) :
        'Commenter cette fonction'

        def RMSD_listes(sample) :
            'Create a list containing the index lists of molecules that share an RMSD less than the RMSDtarget.'            
            remove_list = list(range(len(sample)))
            output_listes = []

            for individuali in sample :
                subliste = []
                for j in remove_list :
                    if rdMolAlign.CalcRMS(self.mols[individuali],self.mols[sample[j]]) < self.RMSDtarget + 1 :
                        subliste.append(j)
                        remove_list.remove(j)
                output_listes.append(subliste[:])

            while [] in output_listes :
                for i, output_liste in enumerate(output_listes) :
                    try :
                        if output_liste == [] :
                            del output_listes[i]
                    except :
                        pass
            return output_listes
        
        def gather_groups_RMSD(input_liste) :
            'Commenter cette fonction'
            liste = []
            for i, groupei in enumerate(input_liste) :
                p = 0
                for j, groupej in enumerate(input_liste) :
                    RMSD = []
                    n = -1
                    if i != j :
                        if len(groupej) <= len(groupei):
                            for k in groupej :
                                n += 1
                                RMSD.append(rdMolAlign.CalcRMS(self.mols[self.sample[k]],self.mols[self.sample[input_liste[i][n]]]))
                        if len(groupej) > len(groupei):
                            for k in groupei :
                                n += 1
                                RMSD.append(rdMolAlign.CalcRMS(self.mols[self.sample[k]],self.mols[self.sample[input_liste[j][n]]]))
                    if i != j :
                        #print(f"La moyenne des RMSD entre le groupe {i} et le groupe {j} est de {np.mean(RMSD)}")
                        if np.mean(RMSD) < self.RMSDtarget + 1 :
                            p = 1
                            copy1 = copy.deepcopy(groupei)   
                            copy2 = copy.deepcopy(groupej)
                            for l in range(len(copy2)) :
                                copy1.append(copy2.pop())
                            liste.append(copy1)
                if p == 0 : 
                    copy1 = copy.deepcopy(groupei)
                    liste.append(copy1)
            return liste

        def sorted_list_lengroups(filtered_gather_groups_RMSD) :
            'Commenter cette fonction'
            lengroups_list = [len(i) for i in filtered_gather_groups_RMSD]
            lengroups_list.sort(reverse=True)

            return [x for y in lengroups_list for x in filtered_gather_groups_RMSD if y == len(x)]

        def get_unique_numbers(numbers):
            'Commenter cette fonction'
            unique = []

            for number in numbers:
                if number in unique:
                    continue
                else:
                    unique.append(number)
            return unique

        def get_filtered_liste(sorted_list_lengroups_var) :
            'Commenter cette fonction'
            filtered_numbersduplicate_liste = []

            for sorted_list_lengroup in sorted_list_lengroups_var : 
                for j in sorted_list_lengroup :
                    filtered_numbersduplicate_liste.append(j)

            filtered_liste = get_unique_numbers(filtered_numbersduplicate_liste)
            return filtered_liste


        def get_groups_inside_list(liste) :
            'Commenter cette fonction'
            liste1 = copy.deepcopy(liste)
            liste2 = copy.deepcopy(liste)
            liste3 = []
            for i, n in enumerate(liste1) :
                try :
                    r = rdMolAlign.CalcRMS(self.mols[self.sample[n]], self.mols[self.sample[liste1[i+1]]])
                except IndexError :
                    pass
                if r > 3 :
                #try :
                    #print(f"En position {i}, une séparation de groupe se trouve entre {n} et {liste1[i+1]}", 
                    #      f"pour un RMSD de {r} A")
                #except IndexError :
                    #pass
                    try :
                        liste3.append(liste2[:liste2.index(liste1[i+1])])
                        del liste2[:liste2.index(liste1[i+1])]
                    except IndexError :
                        pass
                    #print(f"liste 2 --> {liste2}")
            return liste3

        def improve_sort(liste, n) :
            'Commenter cette fonction'
            for i in range(n) :
                liste = get_filtered_liste(sorted_list_lengroups(gather_groups_RMSD(get_groups_inside_list(liste))))
            return liste
        
        output_liste = sorted_list_lengroups(get_groups_inside_list(improve_sort(
            get_filtered_liste(sorted_list_lengroups(gather_groups_RMSD(RMSD_listes(self.sample)))), 5)))
        finallyliste = get_filtered_liste(output_liste)
        array = np.ones(shape=(len(finallyliste),len(finallyliste)))
        
        for i, indivduali in enumerate(finallyliste) :
            for j, indivdualj in enumerate(finallyliste) :
                array[i, j] = rdMolAlign.CalcRMS(self.mols[self.sample[indivduali]],
                                                 self.mols[self.sample[indivdualj]])

        data_frame = pd.DataFrame(array, index=finallyliste,
                                  columns=finallyliste)
        fig, ax = plt.subplots(figsize=(40, 20))
        g = sns.heatmap(data_frame, fmt='d', ax= ax, cmap = "rocket")
        fig = g.get_figure()
        fig.savefig("/Users/carrearthur/Desktop/Results_Conformation/Heatmap.jpeg", dpi=300)
        plt.show()
        self.output_liste = output_liste
        st.pyplot(fig)


    def analyse_heatmap(self) :
        'Commenter cette fonction'
        def get_predominant_poses(finallyliste) :
            'Commenter cette fonction'
            k = 0 
            for i in finallyliste : 
                if len(i)/len(self.sample)*100 > 1/17*len(self.sample) :
                    k += 1

            predominant_poses = finallyliste[:k]
            print(f"There are {k} predominant poses among all poses.\n")

            for i, predominant_pose in enumerate(predominant_poses) :
                st.write(f"The predominant pose n°{i+1} represents {len(predominant_pose)/len(self.sample)*100:.1f}", 
                      f" % of the sample, i.e. {len(predominant_pose)} on {len(self.sample)} poses in total.")
            return predominant_poses

        def get_sample_indice_best_score(predominant_poses) :
            'Commenter cette fonction'
            indice_best_score = []
            m = -1

            for i in predominant_poses :
                m += 1
                n = -1
                suppl_2 = []
                for j in i :
                    n += 1
                    suppl_2.append(self.suppl[self.sample[predominant_poses[m][n]]])
                PLPscore = [x.GetProp("Gold.PLP.Fitness") for x in suppl_2]
                best_score = 0
                for k in PLPscore :
                    if float(k) > float(best_score) :
                        best_score = k
                        indice = PLPscore.index(k)
                indice_best_score.append(indice)
            return indice_best_score

        def get_data_frame_best_poses(predominant_poses, indice_best_score) :
            "Commenter cette fonction"
            columns = list(range(len(predominant_poses)))
            for i in columns :
                columns[i] = f"Pose n°{i+1}"

            index = list(range(len(predominant_poses)))
            for i in index :
                index[i] = f"Pose n°{i+1}"

            array = np.ones(shape=(len(predominant_poses),len(predominant_poses)))

            i = -1
            for groupi, indicei in zip(predominant_poses, indice_best_score) :
                i += 1
                j = 0
                for groupj, indicej in zip(predominant_poses, indice_best_score) :
                    array[i, j] = rdMolAlign.CalcRMS(self.mols[self.sample[groupi[indicei]]],
                                                      self.mols[self.sample[groupj[indicej]]])
                    j += 1

            data_frame = pd.DataFrame(array, index=index, columns=columns)

            print("\nIn order to check that each group is different from each other, a table taking", 
                  "the first individual from each group and calculating the RMSD between each was constructed :\n")
            print(data_frame)
            st.write(data_frame)
            return data_frame
    
        self.sample_predominant_poses = get_predominant_poses(self.output_liste)
        self.sample_indice_best_score = get_sample_indice_best_score(self.sample_predominant_poses)
        get_data_frame_best_poses(self.sample_predominant_poses, self.sample_indice_best_score)
    
    def get_sdf_files(self) :
            #Commenter cette fonction
        conformations = (self.suppl[self.sample[group[indice]]]
                         for group, indice in zip(self.sample_predominant_poses, self.sample_indice_best_score))

        with Chem.SDWriter(f'/Users/carrearthur/Desktop/Results_Conformation/Sample_Best_PLPScore_Poses.sdf') as w:
            for m in conformations:
                w.write(m)
        w.close()

        for k, group in enumerate(self.sample_predominant_poses) :
            with Chem.SDWriter(f'/Users/carrearthur/Desktop/Results_Conformation/Sample_Conformation{k+1}.sdf') as w : 
                for m in group :
                    mol = self.suppl[self.sample[m]]
                    w.write(mol)
            w.close()

        a_supprimer = []
        for i, indice in enumerate(self.sample_indice_best_score) :
            subliste = []
            with Chem.SDWriter(f'/Users/carrearthur/Desktop/Results_Conformation/À_supprimer{i+1}.sdf') as w: #Create a sdf file
                for j, mol in enumerate(self.suppl) :
                    if rdMolAlign.CalcRMS(self.mols[self.sample[self.sample_predominant_poses[i][indice]]], mol) < 2 :
                        subliste.append(self.suppl[j])
                        w.write(self.suppl[j]) #Then write this molecule pose in the new sdf file.
                copy1 = copy.deepcopy(subliste)
                a_supprimer.append(copy1)

        indice_best_score = []
        for group in a_supprimer : 
            PLPscore = [x.GetProp("Gold.PLP.Fitness") for x in group]
            best_score = 0
            for k in PLPscore :
                if float(k) > float(best_score) :
                    best_score = k
                    indice = PLPscore.index(k)
            indice_best_score.append(indice)

        best_PLPScore_poses = (group[indice] for group, indice in zip(a_supprimer, indice_best_score))

        with Chem.SDWriter(f'/Users/carrearthur/Desktop/Results_Conformation/Best_PLPScore_Poses.sdf') as w:
            for m in best_PLPScore_poses:
                w.write(m)
        w.close()

        best_PLP_poses = [GetScaffoldForMol(x) for x in Chem.SDMolSupplier("/Users/carrearthur/Desktop/Results_Conformation/Best_PLPScore_Poses.sdf")]
        for i, best_PLP_pose in enumerate(best_PLP_poses) :
            with Chem.SDWriter(f'/Users/carrearthur/Desktop/Results_Conformation/Conformation{i+1}.sdf') as w: #Create a sdf file
                for j, mol in enumerate(self.mols) :
                    if rdMolAlign.CalcRMS(best_PLP_pose, mol) < 2 :
                        w.write(self.suppl[j]) #Then write this molecule pose in the new sdf file.
        w.close()
        
        def get_histogramme_sample_bestPLP(self) :
            sns.set_style('whitegrid')
            sns.set_context('paper')
            best_PLP_poses = [GetScaffoldForMol(x) for x in Chem.SDMolSupplier("/Users/carrearthur/Desktop/Results_Conformation/Best_PLPScore_Poses.sdf")]
            sdf_to_hist = ([rdMolAlign.CalcRMS(best_PLP_pose, mol) for mol in self.mols] for best_PLP_pose in best_PLP_poses)
            fig, ax = plt.subplots(len(best_PLP_poses), 1, figsize=(15, 0.2*len(best_PLP_poses)*9))
            for z, group in enumerate(sdf_to_hist) :
                a, b, c = 0, 0, 0
                for i in group : 
                    if i < 2 :
                        a += 1
                    if i < 3 :
                        b += 1
                    if i < 4 :
                        c += 1
                ax[z].hist(group, bins =100) #Create an histogram to see the distribution of the RMSD of the sample
                ax[z].axvline(x=2, ymin=0, ymax=1, color="black", linestyle="--")
                ax[z].annotate(a, (1.5, 0.05*len(self.mols)), fontsize=10)
                ax[z].axvline(x=3, ymin=0, ymax=1, color="black", linestyle="--")
                ax[z].annotate(b-a, (2.5, 0.05*len(self.mols)), fontsize=10)
                ax[z].axvline(x=4, ymin=0, ymax=1, color="black", linestyle="--")
                ax[z].annotate(c-b, (3.5, 0.05*len(self.mols)), fontsize=10)

            plt.show()
            fig.savefig("/Users/carrearthur/Desktop/Results_Conformation/Histograms_BestPLP.jpeg", dpi=300)
            print("RMSD distribution between all docking solutions", 
                  " and the pose with the highest ChemPLP score of all solutions for a given conformation.")

        get_histogramme_sample_bestPLP(self)
        st.pyplot(fig)
    
    def get_barsplot(self, k) :
        sns.set_context('talk')
        sns.set_style('whitegrid')
        file_in = [x for x in Chem.SDMolSupplier(f"/Users/carrearthur/Desktop/Results_Conformation/Conformation{k}.sdf")]
        ensemble = {mol.GetProp("Molecule Name 2") for mol in self.suppl}
        index = list(ensemble)
        index.sort()


        columns = "Good Poses", "Total Poses", "Ratio"

        array = np.zeros(shape=(len(ensemble),3))

        for i, indice in enumerate(index) :
            for mol in file_in :
                if mol.GetProp("Molecule Name 2") == indice : 
                    array[i, 0] += 1
            for mol in self.suppl :
                if mol.GetProp("Molecule Name 2") == indice : 
                    array[i, 1] += 1        
            array[i, 2] = array[i, 0]/array[i, 1]

        data_frame = pd.DataFrame(array, index=index, columns=columns)
        table = data_frame.sort_values("Ratio", ascending=False)
        sns.set_context('talk')
        g = sns.catplot(x="Ratio", y=list(table.index), data = table, kind="bar", height=19, aspect=15/8, palette = "rocket")
        g.set_ylabels("")
        g.fig.savefig(f"/Users/carrearthur/Desktop/Results_Conformation/Barplot_Conformation{k}.jpeg", dpi=300)
        plt.yticks(fontsize=18)
        plt.show()










st.title = "ChemoTools !"
if 'sdf_file_stock' not in st.session_state:
    st.session_state.sdf_file_stock = None

sdf_file = st.file_uploader("Upload docked ligand coordinates in SDF format:",
                                    type = ["sdf"])
if sdf_file:
    st.session_state.sdf_file_stock = sdf_file

if "RMSD_Target" not in st.session_state:
    # set the initial default value of the slider widget
    st.session_state.RMSD_Target = 2.5

st.slider('A group of poses representing a conformation should be formed for poses sharing a RMSD of less than what ?',
                         0.0, 15.0, key = "RMSD_Target")
    
if "individuals" not in st.session_state:
    # set the initial default value of the slider widget
    st.session_state.individuals = 200

st.slider('Select the size of your sample',
                         0, 500, key = "individuals")

if 'molref_stock' not in st.session_state:
    st.session_state.molref_stock = None

molref = st.file_uploader("Input your benchmark molecule (mol file in 2D or 3D) here.")

if molref :
    st.session_state.molref_stock = molref

if molref != None :
    if 'output_name_prefix' not in st.session_state :
        st.session_state.output_name_prefix = st.session_state.sdf_file_stock.name.split(".sdf")[0] + datetime.now().strftime("%b-%d-%Y_%H-%M-%S") + "_" + str(random.randint(10000, 99999))
    if 'structures_directory' not in st.session_state :
        st.session_state.structures_directory = st.session_state.output_name_prefix + "_structures"
    if 'structures_path' not in st.session_state :
        st.session_state.structures_path = os.path.join(os.getcwd(), st.session_state.structures_directory)
        os.mkdir(st.session_state.structures_path)
        with open(st.session_state.output_name_prefix + "_sdf_file.sdf", "wb") as f1:
            f1.write(st.session_state.sdf_file_stock.getbuffer())
        with open(st.session_state.output_name_prefix + "_mol_ref.mol", "wb") as f2:
            f2.write(st.session_state.molref_stock.getbuffer())
    
first_checkbox = st.checkbox('Check your sdf')
if first_checkbox :
    if 'conformation' not in st.session_state:
        st.session_state.conformation = Conformations(st.session_state.output_name_prefix + "_sdf_file.sdf", st.session_state.RMSD_Target, st.session_state.individuals)
    if 'first_step' not in st.session_state:
        st.session_state.first_step = st.session_state.conformation.check_sdf_file(st.session_state.output_name_prefix + "_mol_ref.mol")
    else:
        st.write("L'application s'est relancer mais n'a pas modifier conformation")
if first_checkbox == False :
    st.session_state.conformation == False
    st.session_state.first_step == False

second_checkbox = st.checkbox('Get the heatmap')
if second_checkbox :
    st.session_state.entre_deux = st.session_state.conformation.get_heatmap_sample()
if second_checkbox == False :
    st.session_state.entre_deux = None

third_checkbox = st.checkbox('Get the heatmap sorted')
if third_checkbox :
    st.session_state.second_step = st.session_state.conformation.get_heatmap()
    st.session_state.second_step = st.session_state.conformation.analyse_heatmap()
if third_checkbox == False :
    st.session_state.second_step = None

#cgha





class Counter:
    def __init__(self):
        if 'count' not in st.session_state:
            st.session_state.count = 0
        #if 'sdf_file' not in st.session_state :
         #   st.session_state.sdf_file = st.file_uploader("Upload docked ligand coordinates in SDF format:",
          #                          type = ["sdf"])
        #if 'RMSD_Target' not in st.session_state :
         #   st.session_state.RMSD_Target = st.slider('A group of poses representing a conformation should be formed for poses sharing a RMSD of less than what ?',
          #               0.0, 15.0, 2.5)
        #if 'individuals' not in st.session_state :
         #   st.session_state.individuals = st.slider('Select the size of your sample',
          #               0, 500, 200)
        #if 'molref' not in st.session_state :
         #   st.session_state.molref = st.file_uploader("Input your benchmark molecule (mol file in 2D or 3D) here.")
        self.col1, self.col2 = st.columns(2)
        #if st.session_state.molref != None :
         #   if 'output_name_prefix' not in st.session_state :
          #      st.session_state.output_name_prefix = st.session_state.sdf_file.name.split(".sdf")[0] + datetime.now().strftime("%b-%d-%Y_%H-%M-%S") + "_" + str(random.randint(10000, 99999))
           # if 'structures_directory' not in st.session_state :
            #    st.session_state.structures_directory = st.session_state.output_name_prefix + "_structures"
            #if 'structures_path' not in st.session_state :
            #    st.session_state.structures_path = s.path.join(os.getcwd(), st.session_state.structures_directory)
            #os.mkdir(st.session_state.structures_path)
            #with open(st.session_state.output_name_prefix + "_sdf_file.sdf", "wb") as f1:
             #   f1.write(st.session_state.sdf_file.getbuffer())
            #with open(st.session_state.output_name_prefix + "_mol_ref.mol", "wb") as f2:
             #   f2.write(st.session_state.molref.getbuffer())
    
    def add(self):
        st.session_state.count += 1

    def subtract(self):
        st.session_state.count -= 1

    def tester(self):
        if 'tester' not in st.session_state:
            st.session_state.tester = "Tester"
        else:
            st.session_state.tester = "Already tested"
        with self.col2:
            st.write(f"Hello from the {st.session_state.tester}!")

   # def conformation(self):
    #    if 'conformation' not in st.stession_state:
     #       st.session_state.conformation = Conformations(st.session_state.output_name_prefix + "_sdf_file.sdf", st.session_state.RMSD_Target, st.session_state.individuals)
      #  else:
       #     st.write("L'application s'est relancer mais n'a pas modifier conformation")
    
   # def check_sdf(self):
    #    if 'check_sdf' not in st.stession_state:
     #       st.session_state.check_sdf = conformation.check_sdf_file(st.session_state.output_name_prefix + "_mol_ref.mol")
    
   ## def heatmap_sample(self):
    #    if 'heatmap_sample' not in st.stession_state:
     #       st.session_state.heatmap_sample = conformation.get_heatmap_sample()

    #def heatmap_sorted(self):
     #   if 'heatmap_sorted' not in st.stession_state:
      #      st.session_state.heatmap_sorted = conformation.get_heatmap()

    def window(self):

        with self.col1:
            st.button("Increment", on_click=self.add)
            st.button("Subtract", on_click=self.subtract)
            st.write(f'Count = {st.session_state.count}')
        with self.col2:
            st.button('test me', on_click=self.tester)
    #        st.button('Check your sdf', on_click=self.check_sdf)
     #       st.button('Get the heatmap of the sample', on_click=self.heatmap_sample)
      #      st.button('Get the heatmap sorted', on_click=self.heatmap_sorted)



if __name__ == '__main__':
    ct = Counter()
    ct.window()






#if st.session_state.sdf_file is not None or st.session_state.molref is not None :
#    if 'output_name_prefix' not in st.session_state :
#        st.session_state.output_name_prefix = st.session_state.sdf_file.name.split(".sdf")[0] + datetime.now().strftime("%b-%d-%Y_%H-%M-%S") + "_" + str(random.randint(10000, 99999))
 #   if 'structures_directory' not in st.session_state :
  #      st.session_state.structures_directory = st.session_state.output_name_prefix + "_structures"
   # if 'structures_path' not in st.session_state :
    #    st.session_state.structures_path = s.path.join(os.getcwd(), st.session_state.structures_directory)
    #os.mkdir(st.session_state.structures_path)
    #with open(st.session_state.output_name_prefix + "_sdf_file.sdf", "wb") as f1:
    #    f1.write(st.session_state.sdf_file.getbuffer())
    #with open(st.session_state.output_name_prefix + "_mol_ref.mol", "wb") as f2:
    #    f2.write(st.session_state.molref.getbuffer())
#test = Conformations(st.session_state.output_name_prefix + "_sdf_file.sdf", st.session_state.RMSD_Target, st.session_state.individuals)
#if 'test' not in st.session_state:
#    st.session_state.test = test

#if st.button('Check your sdf') :

 #   test.check_sdf_file(st.session_state.output_name_prefix + "_mol_ref.mol")
    #shutil.rmtree(structures_directory)
    #os.remove(output_name_prefix + "_mol_ref")
    #os.remove(output_name_prefix + "_sdf_file.sdf")
   # if st.button('Get the heatmap of the sample'):
  #      st.session_state.test.get_heatmap_sample()
    #if st.button('Get the heatmap sorted'):
     #   st.session_state.test.get_heatmap()
      #  st.session_state.test.analyse_heatmap()

#AR_DBD.get_sdf_files()
#AR_DBD.get_barsplot(3)
