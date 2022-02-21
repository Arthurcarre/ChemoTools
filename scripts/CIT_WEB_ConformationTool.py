"""
#####################################################
##                                                 ##
##       -- STREAMLIT CHEMOTOOLS V1.0 --           ##
##                                                 ##
#####################################################
"""


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

def RMSD_listes(sample) :
    'Create a list containing the index lists of molecules that share an RMSD less than the RMSDtarget.'            
    remove_list = list(range(len(sample)))
    output_listes = []

    for individuali in sample :
        subliste = []
        for j in remove_list :
            if rdMolAlign.CalcRMS(st.session_state.mols[individuali],st.session_state.mols[sample[j]]) < RMSD_Target + 1 :
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
                        RMSD.append(rdMolAlign.CalcRMS(st.session_state.mols[st.session_state.samples[k]],
                                                       st.session_state.mols[st.session_state.samples[input_liste[i][n]]]))
                if len(groupej) > len(groupei):
                    for k in groupei :
                        n += 1
                        RMSD.append(rdMolAlign.CalcRMS(st.session_state.mols[st.session_state.samples[k]],
                                                       st.session_state.mols[st.session_state.samples[input_liste[j][n]]]))
            if i != j :
                #print(f"La moyenne des RMSD entre le groupe {i} et le groupe {j} est de {np.mean(RMSD)}")
                if np.mean(RMSD) < RMSD_Target + 1 :
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
            r = rdMolAlign.CalcRMS(st.session_state.mols[st.session_state.samples[n]],
                                   st.session_state.mols[st.session_state.samples[liste1[i+1]]])
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

def get_predominant_poses(finallyliste) :
    'Commenter cette fonction'
    k = 0 
    for i in finallyliste : 
        if len(i)/len(st.session_state.samples)*100 > 1/17*len(st.session_state.samples) :
            k += 1

    predominant_poses = finallyliste[:k]
    st.info(f"There are {k} predominant poses among all poses.\n")
    st.session_state.numbers_conformation = k
    st.session_state.numbers_conformationA = k
    st.session_state.predominant_poses = predominant_poses

    for i, predominant_pose in enumerate(predominant_poses) :
        st.write(f"The predominant pose n°{i+1} represents {len(predominant_pose)/len(st.session_state.samples)*100:.1f}" 
              f"% of the sample, i.e. {len(predominant_pose)} on {len(st.session_state.samples)} poses in total.")
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
            suppl_2.append(st.session_state.suppl[st.session_state.samples[predominant_poses[m][n]]])
        try :
            PLPscore = [x.GetProp("Gold.PLP.Fitness") for x in suppl_2]
        except :
            PLPscore = [x.GetProp("Gold.Goldscore.Fitness") for x in suppl_2]
        best_score = 0
        for k in PLPscore :
            if float(k) > float(best_score) :
                best_score = k
                indice = PLPscore.index(k)
        indice_best_score.append(indice)
    return indice_best_score

def get_Sample_Best_PLPScore_Poses() :
    #Commenter cette fonction
    conformations = (st.session_state.suppl[st.session_state.samples[group[indice]]]
                     for group, indice in zip(st.session_state.sample_predominant_poses,
                                              st.session_state.sample_indice_best_score))

    with Chem.SDWriter('Sample_Best_PLPScore_Poses.sdf') as w:
        for m in conformations:
            w.write(m)
    w.close()

    for k, group in enumerate(st.session_state.sample_predominant_poses) :
        with Chem.SDWriter(f'Sample_Conformation{k+1}.sdf') as w : 
            for m in group :
                mol = st.session_state.suppl[st.session_state.samples[m]]
                w.write(mol)
        w.close()

    a_supprimer = []
    for i, indice in enumerate(st.session_state.sample_indice_best_score) :
        subliste = []
        with Chem.SDWriter(f'À_supprimer{i+1}.sdf') as w: #Create a sdf file
            for j, mol in enumerate(st.session_state.suppl) :
                if rdMolAlign.CalcRMS(st.session_state.mols[st.session_state.samples[st.session_state.sample_predominant_poses[i][indice]]], mol) < 2 :
                    subliste.append(st.session_state.suppl[j])
                    w.write(st.session_state.suppl[j]) #Then write this molecule pose in the new sdf file.
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

    with Chem.SDWriter(f'Best_PLPScore_Poses.sdf') as w:
        for m in best_PLPScore_poses:
            w.write(m)
    w.close()
    
    for i, indice in enumerate(st.session_state.sample_indice_best_score) :
        os.remove(f'À_supprimer{i+1}.sdf')

def get_data_frame_best_poses(best_PLP_poses) :
    "Commenter cette fonction"
    
    columns = list(range(len(best_PLP_poses)))
    for i in columns :
        columns[i] = f"Pose n°{i+1}"

    index = list(range(len(best_PLP_poses)))
    for i in index :
        index[i] = f"Pose n°{i+1}"
    
    array = np.ones(shape=(len(best_PLP_poses),len(best_PLP_poses)))

    for i, moli in enumerate(best_PLP_poses) :
        for j, molj in enumerate(best_PLP_poses) :
            array[i, j] = rdMolAlign.CalcRMS(moli, molj)

    data_frame = pd.DataFrame(array, index=index, columns=columns)

    st.write("\nIn order to check that each group is different from each other, a table taking " 
          "the first individual from each group and calculating the RMSD between each was constructed :\n")
    st.dataframe(data_frame)
    st.write("RMSD value between each representative of each conformation.")
    st.session_state.df1 = data_frame
    return data_frame

def get_histogramme_sample_bestPLP(best_PLP_poses) :
    sns.set_style('whitegrid')
    sns.set_context('paper')
    
    if len(best_PLP_poses) == 1 :
        sdf_to_hist = [rdMolAlign.CalcRMS(best_PLP_poses[0], mol) for mol in st.session_state.mols]
        fig, ax = plt.subplots(len(best_PLP_poses), 1, figsize=(15, 0.2*len(best_PLP_poses)*9))
        a, b, c = 0, 0, 0
        for RMSD in sdf_to_hist : 
            if RMSD < 2 :
                a += 1
            if RMSD < 3 :
                b += 1
            if RMSD < 4 :
                c += 1
        ax.hist(sdf_to_hist, bins =100) #Create an histogram to see the distribution of the RMSD of the sample
        ax.axvline(x=2, ymin=0, ymax=1, color="black", linestyle="--")
        ax.annotate(a, (1.5, 0.05*len(st.session_state.mols)), fontsize=15)
        ax.axvline(x=3, ymin=0, ymax=1, color="black", linestyle="--")
        ax.annotate(b-a, (2.5, 0.05*len(st.session_state.mols)), fontsize=15)
        ax.axvline(x=4, ymin=0, ymax=1, color="black", linestyle="--")
        ax.annotate(c-b, (3.5, 0.05*len(st.session_state.mols)), fontsize=15)
    else :
        sdf_to_hist = ([rdMolAlign.CalcRMS(best_PLP_pose, mol) for mol in st.session_state.mols] for best_PLP_pose in best_PLP_poses)
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
            ax[z].annotate(f"Conformation n°{z+1}", (0, 0.05*len(st.session_state.mols)), fontsize=10)
            ax[z].axvline(x=2, ymin=0, ymax=1, color="black", linestyle="--")
            ax[z].annotate(a, (1.5, 0.05*len(st.session_state.mols)), fontsize=15)
            ax[z].axvline(x=3, ymin=0, ymax=1, color="black", linestyle="--")
            ax[z].annotate(b-a, (2.5, 0.05*len(st.session_state.mols)), fontsize=15)
            ax[z].axvline(x=4, ymin=0, ymax=1, color="black", linestyle="--")
            ax[z].annotate(c-b, (3.5, 0.05*len(st.session_state.mols)), fontsize=15)

    st.pyplot(fig)
    st.write("Density of the number of poses as a function of the RMSD calculated between the representative of each conformation"
             " and all poses of all molecules in the docking solutions of the filtered incoming sdf file.")
    st.session_state.fig3 = fig
    fig.savefig("Histograms_Best_Score.jpeg", dpi=300)
    print("RMSD distribution between all docking solutions", 
          " and the pose with the highest ChemPLP score of all solutions for a given conformation.")
    
def get_barsplot(k) :
    sns.set_context('talk')
    sns.set_style('whitegrid')
    file_in = [x for x in Chem.SDMolSupplier(f"Conformation{k}.sdf")]
    ensemble = {mol.GetProp("SourceTag") for mol in st.session_state.suppl}
    index = list(ensemble)
    index.sort()

    columns = "Good Poses", "Total Poses", "Ratio"

    array = np.zeros(shape=(len(ensemble),3))

    for i, indice in enumerate(index) :
        for mol in file_in :
            if mol.GetProp("SourceTag") == indice : 
                array[i, 0] += 1
        for mol in st.session_state.suppl :
            if mol.GetProp("SourceTag") == indice : 
                array[i, 1] += 1        
        array[i, 2] = array[i, 0]/array[i, 1]

    data_frame = pd.DataFrame(array, index=index, columns=columns)
    table = data_frame.sort_values("Ratio", ascending=False)
    sns.set_context('talk')
    g = sns.catplot(x="Ratio", y=list(table.index), data = table, kind="bar", height=19, aspect=15/8, palette = "rocket")
    g.set_ylabels("")
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(list(table["Ratio"])):
        plt.text(float(v)+0.014, float(i)+0.35, f"{round(v*100, 1)}%", horizontalalignment = "center")
    plt.yticks(fontsize=18)
    plt.axvline(x=0.5, color="black", linestyle="--")
    plt.axvline(x=0.25, color="black", linestyle=":")
    plt.axvline(x=0.75, color="black", linestyle=":")
    plt.annotate("50%", (0.49, i), fontsize=35)
    plt.annotate("25%", (0.24, i), fontsize=35)
    plt.annotate("75%", (0.74, i), fontsize=35)
    g.fig.savefig(f"Barplot_Conformation{k}.jpeg", dpi=300)
    st.session_state.fig4 = g
        

#################################################################################################################


#################################################################################################################

def main():

    st.header('Chemotools ! V1.0')

    st.markdown('Welcome to Chemotools first version ! This version is compatible with the docking results files from the GOLD software'
                ' using the scoring functions: ChemPLP Score and GoldScore.\n' 'In this version, make sure, through the DataWarrior '
                'software, that the column containing the name of the molecule (and not the name of the pose itself) is named "SourceTag".')

    sdf = st.file_uploader("Upload docked ligand coordinates in SDF format:",
                                        type = ["sdf"])
    if sdf:
        st.session_state.sdf_file_stock = sdf
    else:
        try : 
            del st.session_state.sdf_file_stock
        except :
            pass

    if 'molref_stock' not in st.session_state:
        st.session_state.molref_stock = None

    molref = st.file_uploader("Input your benchmark molecule (mol file in 2D or 3D) here.")

    if molref :
        st.session_state.molref_stock = molref
    else:
        try : 
            del st.session_state.molref_stock
        except :
            pass


    individuals = st.slider('Select the size of your sample. Default size of sample = 200', 0, 500, 200,
                            help='If you want to change this setting during the program, make sure the box below is unchecked!')


    first_checkbox = st.checkbox('Check your sdf')
    if first_checkbox :
        if 'output_name_prefix' not in st.session_state :
            st.session_state.output_name_prefix = "Chemotools_Work_" + datetime.now().strftime("%b-%d-%Y_%H-%M-%S") + "_" + str(random.randint(10000, 99999))
        if 'structures_directory' not in st.session_state :
            st.session_state.structures_directory = st.session_state.output_name_prefix + "_structures"
            with open(st.session_state.output_name_prefix + "_sdf_file.sdf", "wb") as f1:
                f1.write(st.session_state.sdf_file_stock.getbuffer())
            with open(st.session_state.output_name_prefix + "_mol_ref.mol", "wb") as f2:
                f2.write(st.session_state.molref_stock.getbuffer())
        if 'sdf_file' not in st.session_state:
            st.session_state.sdf_file = st.session_state.output_name_prefix + "_sdf_file.sdf"
        if 'suppl_brut' not in st.session_state:
            st.session_state.suppl_brut = [x for x in Chem.SDMolSupplier(st.session_state.output_name_prefix + "_sdf_file.sdf")]
        if 'premols' not in st.session_state:
            st.session_state.premols = [GetScaffoldForMol(x) for x in st.session_state.suppl_brut]
        if 'molref' not in st.session_state:
            st.session_state.molref = Chem.MolFromMolFile(st.session_state.output_name_prefix + "_mol_ref.mol")


        mols = []
        suppl = []
        error_molecules = []

        for i, mol in enumerate(st.session_state.premols) :
            try :
                rdMolAlign.CalcRMS(mol, st.session_state.molref)
                mols.append(mol)
                suppl.append(st.session_state.suppl_brut[i])
            except RuntimeError as e :
                error_molecules.append(mol.GetProp("_Name"))

        if 'mols' not in st.session_state:
            st.session_state.mols = mols
        if 'suppl' not in st.session_state:
            st.session_state.suppl = suppl
        if 'error_molecules' not in st.session_state:
            st.session_state.error_molecules = error_molecules
        if 'samples' not in st.session_state:
            st.session_state.samples = random.sample(range(len(st.session_state.mols)), individuals)

        show_molecules = st.checkbox('Show molecules name which will be not included in the algorithm')
        if show_molecules:
            if error_molecules == False :
                st.write('All molecules are good !')
            if error_molecules :
                for i in error_molecules : 
                    st.write(i)
                st.warning("All theses molecules have not sub-structure which match between the reference and probe mol.\n"
                          "An RMSD can't be calculated with these molecules, they will therefore not be taken into " 
                          "account by the algorithm.\n")

        st.info(f"There are {len(st.session_state.premols)} molecule poses in the sdf file.\n")
        st.info(f"There are {len(st.session_state.mols)} molecule poses which will be used by the algorithm.\n")
        try :
            os.remove(st.session_state.output_name_prefix + "_sdf_file.sdf")
            os.remove(st.session_state.output_name_prefix + "_mol_ref.mol")
        except :
            pass

        second_checkbox = st.checkbox('Get the sample heatmap')
        if second_checkbox :
            try:
                st.pyplot(st.session_state.fig1)
            except:
                pass

            if 'fig1' not in st.session_state:
                array = np.ones(shape=(len(st.session_state.samples),len(st.session_state.samples)))
                for i, indivduali in enumerate(st.session_state.samples) :
                    for j, indivdualj in enumerate(st.session_state.samples) :
                        array[i, j] = rdMolAlign.CalcRMS(st.session_state.mols[indivduali],st.session_state.mols[indivdualj])

                df = pd.DataFrame(
                    array, index=list(range(len(st.session_state.samples))), columns=list(range(len(st.session_state.samples))))

                fig, ax = plt.subplots(figsize=(20, 10))
                sns.heatmap(df, fmt='d', ax= ax)

                print("The table, and therefore the heatmap, was constructed in the order in which the sample was constructed", 
                      "(i.e. randomly). To make this heatmap more readable, the different individuals have been grouped", 
                      f"according to whether the RMSD between each individual in a group is less than {RMSD_Target}.\n")
                st.pyplot(fig)
                st.session_state.fig1 = fig
        else:
            if 'fig1' in st.session_state:
                del st.session_state.fig1

        RMSD_Target = st.slider('RMSD threshold: Select the maximum RMSD threshold that should constitute a conformation. Default RMSD threshold = 1',
                             0.0, 15.0, 1.0, help='If you want to change this setting during the program, make sure the box below is unchecked!')

        loop = st.slider('Number of Loops', 0, 20, 1, help="In the aim to build the sorted heatmap, the sorting process may requires to"
                         " be repeating many times, this is the number of loops, in order to have more resolution. However, especially"
                         " in the case of there is only one predominant conformation adopted by the large majority of the poses, you may"
                         " be invited to reduce the number of loops as the sorting process may remove a lot of individuals of the sample.")

        third_checkbox = st.checkbox('Get the sorted heatmap (Attention ! Before closing this app, please, UNCHECK THIS BOX)')
        if third_checkbox :
            try:
                st.warning(f"ATTENTION !!! {st.session_state.indviduals_deleted} INDIVIDUALS HAVE BEEN DELETED DURING THE TREATMENT")
                st.write("In order to increase the resolution of your sorted heatmap you can play with 2 setting :\n"
                         "- Decrease the RMSD threshold (Do not hesitate!)\n"
                         "- Increase the number of loops")
                st.pyplot(st.session_state.fig2)
                with open("Sorted_Heatmap.jpeg", "rb") as file:
                     btn = st.download_button(
                             label="Download PLOT sorted heatmap",
                             data=file,
                             file_name="Sorted_Heatmap.jpeg",
                             mime="image/jpeg")
                st.info(f"There are {st.session_state.numbers_conformation} predominant poses among all poses.\n")
                for i, predominant_pose in enumerate(st.session_state.predominant_poses) :
                    st.write(f"The predominant pose n°{i+1} represents {len(predominant_pose)/len(st.session_state.samples)*100:.1f}" 
                             f"% of the sample, i.e. {len(predominant_pose)} on {len(st.session_state.samples)} poses in total.")
                st.write("\nIn order to check that each group is different from each other, a table taking " 
                          "the first individual from each group and calculating the RMSD between each was constructed :\n")
                st.dataframe(st.session_state.df1)
                st.write("RMSD value between each representative of each conformation.")
                with open("Sample_Best_PLPScore_Poses.sdf", "rb") as file:
                     btn = st.download_button(
                                label="Download SDF Best Score Poses of each Conformation from the SAMPLE",
                                data=file,
                                 file_name="Sample_Best_Score_Poses.sdf")
                with open("Best_PLPScore_Poses.sdf", "rb") as file:
                     btn = st.download_button(
                                label="Download SDF Best Score Poses of each Conformation from the filtered INPUT SDF FILE",
                                data=file,
                                 file_name="Best_Score_Poses.sdf")
                for i in range(st.session_state.numbers_conformation):
                    with open(f"Sample_Conformation{i+1}.sdf", "rb") as file:
                         btn = st.download_button(
                                    label=f"Download all the poses of the conformation n°{i+1} from the SAMPLE",
                                    data=file,
                                     file_name=f"Sample_Conformation{i+1}.sdf")
                st.pyplot(st.session_state.fig3)
                st.write("Density of the number of poses as a function of the RMSD calculated between the representative of each conformation"
                 " and all poses of all molecules in the docking solutions of the filtered incoming sdf file.")      

                with open("Histograms_Best_Score.jpeg", "rb") as file:
                     btn = st.download_button(
                                label="Download PLOT Histograms",
                                data=file,
                                 file_name="Histograms_Best_Score.jpeg",
                                 mime="image/jpeg")

                if st.session_state.numbers_conformation != 1 :
                    temp_options = range(1, st.session_state.numbers_conformation + 1)
                    st.session_state.temp = st.select_slider("You want a sdf file or a barplot including molecules in the conformation n°",
                                                             options=temp_options)
                    st.write(f"The Conformation selected is {st.session_state.temp}")

                    st.session_state.RMSD_Target_conformation = st.slider('... With all poses under a RMSD =', 0.0, 15.0, 2.0)
                    st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")


                    if st.button('Prepare your sdf file and build the barplot'):             
                        best_PLP_poses = [GetScaffoldForMol(x) for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
                        with Chem.SDWriter(f'Conformation{st.session_state.temp}.sdf') as w: #Create a sdf file
                            for j, mol in enumerate(st.session_state.mols) :
                                if rdMolAlign.CalcRMS(best_PLP_poses[int(st.session_state.temp - 1)], mol) < float(st.session_state.RMSD_Target_conformation) :
                                    w.write(st.session_state.suppl[j]) #Then write this molecule pose in the new sdf file.
                        w.close()


                        with open(f'Conformation{st.session_state.temp}.sdf', "rb") as file:
                             btn = st.download_button(
                                        label="Download your sdf file",
                                        data=file,
                                         file_name=f"Conformation n°{st.session_state.temp}.sdf")


                        with st.spinner('Please wait, the barplot is coming...'):              
                            if "fig4" in st.session_state :
                                del st.session_state.fig4
                            get_barsplot(st.session_state.temp)
                            st.pyplot(st.session_state.fig4)
                            st.write("Ratio of each compounds between the number of poses in the conformation selected and the number of total poses.")
                            with open(f"Barplot_Conformation{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download PLOT Barplot Conformation n°{st.session_state.temp} ",
                                         data=file,
                                         file_name=f"Barplot_Conformation n°{st.session_state.temp}.jpeg",
                                         mime="image/jpeg")

                else :                              
                    st.session_state.RMSD_Target_conformation = st.slider('You want a sdf file or a barplot including molecules in the unique predominant conformation with all poses under a RMSD =', 0.0, 15.0, 2.0)
                    st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")

                    st.info('There is only one predominant conformation. Do you want to have the sdf file of poses in this conformation OR see analysis of this conformation ? ')
                    bouton = st.button('I want to have the sdf file of poses in this conformation OR the barplot analysis.')
                    if bouton :
                        best_PLP_poses = [GetScaffoldForMol(x) for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
                        for i, best_PLP_pose in enumerate(best_PLP_poses):
                            with Chem.SDWriter('Conformation1.sdf') as w: #Create a sdf file
                                for j, mol in enumerate(st.session_state.mols) :
                                    if rdMolAlign.CalcRMS(best_PLP_pose, mol) < float(st.session_state.RMSD_Target_conformation) :
                                        w.write(st.session_state.suppl[j]) #Then write this molecule pose in the new sdf file.
                            w.close()

                        with open('Conformation1.sdf', "rb") as file:
                         btn = st.download_button(
                                    label="Download your sdf file",
                                    data=file,
                                     file_name=f"Unique Conformation.sdf")

                        with st.spinner('Please wait, the barplot is coming...'):
                            if "fig4" in st.session_state :
                                del st.session_state.fig4
                            get_barsplot(1)
                            st.pyplot(st.session_state.fig4)
                            st.write("Ratio of each compounds between the number of poses in the conformation selected and the number of total poses.")
                            with open(f"Barplot_Conformation1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label="Download PLOT Barplot Unique Conformation  ",
                                         data=file,
                                         file_name="Barplot Unique Conformation.jpeg",
                                         mime="image/jpeg")


            except Exception as e:
                #st.exception(e)
                pass

            if 'fig2' not in st.session_state:
                with st.spinner('Please wait, the sorted heatmap is coming...'):
                    output_liste = sorted_list_lengroups(get_groups_inside_list(improve_sort(
                        get_filtered_liste(sorted_list_lengroups(gather_groups_RMSD(RMSD_listes(st.session_state.samples)))), loop)))
                    finallyliste = get_filtered_liste(output_liste)
                    if len(finallyliste) != individuals :
                        st.warning(f"ATTENTION !!! {individuals-len(finallyliste)} INDIVIDUALS HAVE BEEN DELETED DURING THE TREATMENT")
                    st.write("In order to increase the resolution of your sorted heatmap you can play with 2 settings :\n"
                         "- Decrease the RMSD threshold (Do not hesitate!)\n"
                         "- Increase the number of loops")
                    st.session_state.indviduals_deleted = individuals-len(finallyliste)
                    array = np.ones(shape=(len(finallyliste),len(finallyliste)))

                    for i, indivduali in enumerate(finallyliste) :
                        for j, indivdualj in enumerate(finallyliste) :
                            array[i, j] = rdMolAlign.CalcRMS(st.session_state.mols[st.session_state.samples[indivduali]],
                                                             st.session_state.mols[st.session_state.samples[indivdualj]])

                    data_frame = pd.DataFrame(array, index=finallyliste,
                                              columns=finallyliste)
                    fig, ax = plt.subplots(figsize=(20, 10))
                    try : 
                        g = sns.heatmap(data_frame, fmt='d', ax= ax, cmap = "rocket")
                        fig = g.get_figure()
                        fig.savefig("Sorted_Heatmap.jpeg", dpi=300)
                        st.pyplot(fig)
                        with open("Sorted_Heatmap.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label="Download PLOT sorted heatmap",
                                     data=file,
                                     file_name="Sorted_Heatmap.jpeg",
                                     mime="image/jpeg")
                        st.session_state.fig2 = fig
                        if 'output_liste' not in st.session_state:
                            st.session_state.output_liste = output_liste
                        if 'sample_predominant_poses' not in st.session_state:
                            st.session_state.sample_predominant_poses = get_predominant_poses(st.session_state.output_liste)
                        if 'sample_indice_best_score' not in st.session_state:
                            st.session_state.sample_indice_best_score = get_sample_indice_best_score(st.session_state.sample_predominant_poses)
                        get_Sample_Best_PLPScore_Poses()
                        if 'best_PLP_poses' not in st.session_state:
                            st.session_state.best_PLP_poses = [GetScaffoldForMol(x) for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
                        if 'df1' not in st.session_state:
                            get_data_frame_best_poses(st.session_state.best_PLP_poses)
                        with open("Sample_Best_PLPScore_Poses.sdf", "rb") as file:
                             btn = st.download_button(
                                        label="Download SDF Best Score Poses of each Conformation from the SAMPLE",
                                        data=file,
                                         file_name="Sample_Best_Score_Poses.sdf")
                        with open("Best_PLPScore_Poses.sdf", "rb") as file:
                             btn = st.download_button(
                                        label="Download SDF Best Score Poses of each Conformation from the filtered INPUT SDF FILE",
                                        data=file,
                                         file_name="Best_Score_Poses.sdf")
                        for i in range(st.session_state.numbers_conformation):
                            with open(f"Sample_Conformation{i+1}.sdf", "rb") as file:
                                 btn = st.download_button(
                                            label=f"Download all the poses of the conformation n°{i+1} from the SAMPLE",
                                            data=file,
                                             file_name=f"Sample_Conformation{i+1}.sdf")
                        if 'fig3' not in st.session_state:
                            get_histogramme_sample_bestPLP(st.session_state.best_PLP_poses)
                        with open("Histograms_Best_Score.jpeg", "rb") as file:
                             btn = st.download_button(
                                        label="Download PLOT Histograms",
                                        data=file,
                                         file_name="Histograms_Best_Score.jpeg",
                                         mime="image/jpeg")

                    except ValueError :
                        st.error("OOPS ! The number of loops is too high. Please uncheck the sorted heatmap box and then lower it")

                if st.session_state.numbers_conformation != 1 :
                    temp_options = range(1, st.session_state.numbers_conformation + 1)
                    st.session_state.temp = st.select_slider("You want a sdf file or a barplot including molecules in the conformation n°",
                                                             options=temp_options)
                    st.write(f"The Conformation selected is {st.session_state.temp}")

                    st.session_state.RMSD_Target_conformation = st.slider('... With all poses under a RMSD =', 0.0, 15.0, 2.0)
                    st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")

                    if st.button('Prepare your sdf file and build the barplot'):             
                        best_PLP_poses = [GetScaffoldForMol(x) for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
                        with Chem.SDWriter(f'Conformation{st.session_state.temp}.sdf') as w: #Create a sdf file
                            for j, mol in enumerate(st.session_state.mols) :
                                if rdMolAlign.CalcRMS(best_PLP_poses[int(st.session_state.temp - 1)], mol) < float(st.session_state.RMSD_Target_conformation) :
                                    w.write(st.session_state.suppl[j]) #Then write this molecule pose in the new sdf file.
                        w.close()
                else :
                    st.info('There is only one predominant conformation. Do you want to have the sdf file of poses in this conformation OR see analysis of this conformation ? ')
                    st.session_state.RMSD_Target_conformation = st.slider('You want a sdf file or a barplot including molecules in the unique predominant conformation with all poses under a RMSD =', 0.0, 15.0, 2.0)
                    bouton = st.button('I want to have the sdf file of poses in this conformation OR the barplot analysis.')
                    if bouton :
                        best_PLP_poses = [GetScaffoldForMol(x) for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
                        for i, best_PLP_pose in enumerate(best_PLP_poses):
                            with Chem.SDWriter('Unique_Conformation.sdf') as w: #Create a sdf file
                                for j, mol in enumerate(st.session_state.mols) :
                                    if rdMolAlign.CalcRMS(best_PLP_pose, mol) < float(st.session_state.RMSD_Target_conformation) :
                                        w.write(st.session_state.suppl[j]) #Then write this molecule pose in the new sdf file.
                            w.close()




        else :
            if 'numbers_conformation' in st.session_state:
                os.remove('Sorted_Heatmap.jpeg')
                os.remove('Histograms_Best_Score.jpeg')
                os.remove('Sample_Best_PLPScore_Poses.sdf')
                os.remove('Best_PLPScore_Poses.sdf')
                for i in range(st.session_state.numbers_conformation):
                    os.remove(f'Sample_Conformation{i+1}.sdf')
                    try :
                        os.remove(f'Conformation{i+1}.sdf')
                        os.remove(f'Barplot_Conformation{i+1}.jpeg')
                    except :
                        pass

                del st.session_state.numbers_conformation
            if 'indviduals_deleted' in st.session_state:
                del st.session_state.indviduals_deleted
            if 'fig2' in st.session_state:
                del st.session_state.fig2
            if 'predominant_poses' in st.session_state:
                del st.session_state.predominant_poses
            if 'df1' in st.session_state:
                del st.session_state.df1
            if 'fig3' in st.session_state:
                del st.session_state.fig3
            if 'output_liste' in st.session_state:
                del st.session_state.output_liste 
            if 'sample_predominant_poses' in st.session_state:
                del st.session_state.sample_predominant_poses
            if 'sample_indice_best_score' in st.session_state:
                del st.session_state.sample_indice_best_score
            if 'best_PLP_poses' in st.session_state:
                del st.session_state.best_PLP_poses
            if 'fig4' in st.session_state:
                del st.session_state.fig4



    else :
        try :
            if 'output_name_prefix' in st.session_state:
                del st.session_state.output_name_prefix        
            if 'structures_directory' in st.session_state:
                del st.session_state.structures_directory        
            if 'sdf_file' in st.session_state:
                del st.session_state.sdf_file
            if 'suppl_brut' in st.session_state:
                del st.session_state.suppl_brut 
            if 'premols' in st.session_state:
                del st.session_state.premols
            if 'molref' in st.session_state:
                del st.session_state.molref
            if 'mols' in st.session_state:
                del st.session_state.mols
            if 'suppl' in st.session_state:
                del st.session_state.suppl
            if 'error_molecules' in st.session_state:
                del st.session_state.error_molecules
            if 'samples' in st.session_state:
                del st.session_state.samples
        except :
            pass


