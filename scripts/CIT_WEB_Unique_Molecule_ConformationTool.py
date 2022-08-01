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
from rdkit.Chem import AllChem 
from scipy.cluster import hierarchy
from ChemoInfoTool import ConformationTool

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

###############################################
#                                             #
#            APPLICATION SECTION              #
#                                             #                                                      
###############################################

def main():
    
    st.header('Unique Molecule ConformationTool !')
    
    st.markdown('Welcome to Unique Molecule ConformationTool ! Here, the processing and analysis of the docking simulation'
                ' results when your sdf file include ONLY ONE molecule (in many different 3D positions).')

    #SDF FILE SECTION#
    sdf = st.file_uploader("Upload the coordinates of the docked ligand in SDF format:",
                                        type = ["sdf"], key = 'CIT_WEB_Unique_Molecule_ConformationTool')
    if sdf:
        score = st.text_input(
                     'What is the scoring function used in your sdf file ?',
                     'Gold.PLP.Fitness', key = 'CIT_WEB_Unique_Molecule_ConformationTool')
        st.session_state.score = score
        if 'sdf_file_stock_unique' not in st.session_state :
            st.session_state.sdf_file_stock_unique = sdf
            with open("sdf_file.sdf", "wb") as f:
                f.write(st.session_state.sdf_file_stock_unique.getbuffer())
            mols = [x for x in Chem.SDMolSupplier("sdf_file.sdf")]
            st.session_state.mol1_unique = mols[0]
            del mols
            os.remove("sdf_file.sdf")
    else:
        if "sdf_file_stock_unique" in st.session_state :
            del st.session_state.sdf_file_stock_unique
        if "mol1_unique" in st.session_state :
            del st.session_state.mol1_unique

    #BENCHMARK MOLECULE SECTION#
    
    if 'molref_stock_unique' not in st.session_state:
        if 'mol1_unique' in st.session_state :
            smiles = Chem.MolToSmiles(st.session_state.mol1_unique)
            st.session_state.molref_stock_unique = smiles
            blk=makeblock(st.session_state.molref_stock_unique)
            render_mol(blk)
    else : 
        blk=makeblock(st.session_state.molref_stock_unique)
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
        st.session_state.uniquemol_delete_activated = True
        try : 
            if 'ConformationClass' not in st.session_state :
                with open("sdf_file.sdf", "wb") as sdf:
                    sdf.write(st.session_state.sdf_file_stock_unique.getbuffer())
                st.session_state.sdf_file = "sdf_file.sdf"        
                st.session_state.ConformationClass = ConformationTool(st.session_state.sdf_file,
                                                                      st.session_state.score,
                                                                      conformation_tool = "Unique",
                                                                      streamlit = True)

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
                                help='If you want to change this setting during the program, make sure the box below is unchecked!',
                                key = 'CIT_WEB_Unique_Molecule_ConformationTool')

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
                                ' is unchecked!', key = 'CIT_WEB_Unique_Molecule_ConformationTool')

        Proportion = st.slider('Minimum size of the sample defining a conformation. Default proportion = 0.05',
                             0.0, 1.0, 0.05,
                               help=('This setting define the minimum proportion (value between 0 and 1) of individuals'
                                     ' in a group within the sample to consider that group large enough to be'
                                     ' representative of a full conformation.'), key = 'CIT_WEB_Unique_Molecule_ConformationTool')

    ###############################################
    #--   CHECKBOX "GET THE SORTED HEATMAP"     --#                                                     
    ############################################### 

        tab1, tab2 = st.tabs(["Sorting Process", "Cluster Hierarchy"])

        with tab1:
            third_checkbox = st.checkbox('Get the sorted heatmap')
            if third_checkbox :
                st.session_state.premier_passage = True
                if 'sorted_heatmap' not in st.session_state:
                    with st.spinner('Please wait, the sorted heatmap is coming...'):
                        st.session_state.ConformationClass.get_sorted_heatmap(
                            individuals,
                            RMSD_Target,
                            loop = 1,
                            p = Proportion)
                
                else :
                    if 'indviduals_deleted' in st.session_state:
                        st.warning(
                            f"Attention. The sorting process discarded {st.session_state.indviduals_deleted} individuals")
                    st.pyplot(st.session_state.sorted_heatmap)
                    with open("Sorted_Heatmap.jpeg", "rb") as file:
                         btn = st.download_button(
                                 label="Download PLOT sorted heatmap",
                                 data=file,
                                 file_name="Sorted_Heatmap.jpeg",
                                 mime="image/jpeg", key = 'CIT_WEB_Unique_Molecule_ConformationTool')
                    st.info(f"There is (are) {st.session_state.n_conformations} predominant pose(s) among all poses.\n")

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
                                     file_name=f"Sample_Conformation{i+1}.sdf",
                                     key = 'CIT_WEB_Unique_Molecule_ConformationTool')

                    with open("Best_PLPScore_Poses.sdf", "rb") as file:
                         btn = st.download_button(
                                    label="Download the SDF file including each of the representatives of a conformation",
                                    data=file,
                                    file_name="Best_Score_Poses.sdf",
                                    key = 'CIT_WEB_Unique_Molecule_ConformationTool')

                    for i, predominant_pose in enumerate(st.session_state.predominant_poses) :
                        st.write(
                            f"The predominant pose n°{i+1} represents {len(predominant_pose)/len(st.session_state.sample)*100:.1f}" 
                            f"% of the sample, i.e. {len(predominant_pose)} on {len(st.session_state.sample)} poses in total.")

                    st.write("\nIn order to check that each group is different from each other, a table taking " 
                              "the individual **with the best score** from each group and calculating the RMSD between each was constructed :\n")

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
                                    mime="image/jpeg", key = 'CIT_WEB_Unique_Molecule_ConformationTool')

                if st.session_state.n_conformations != 1:
                    temp_options = range(1, st.session_state.n_conformations + 1)
                    if "temp" not in st.session_state :
                        st.session_state.temp = 1
                    st.session_state.temp = st.select_slider("You want a sdf file and/or a anlysis plots including molecules in the conformation n°",
                                                             options=temp_options,
                                                             value = st.session_state.temp,
                                                             key = 'CIT_WEB_Unique_Molecule_ConformationTool')
                    
                    st.write(f"The Conformation selected is {st.session_state.temp}")

                    st.session_state.RMSD_Target_conformation = st.slider('... With all poses under a RMSD =', 0.0, 15.0, 2.0,
                                                                          key = 'CIT_WEB_Unique_Molecule_ConformationTool')
                    st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")
               
                    if st.button('Prepare your sdf file', key = 'CIT_WEB_Unique_Molecule_ConformationTool'):
                        st.session_state.ConformationClass.get_sdf_conformations(
                            st.session_state.temp,
                            st.session_state.RMSD_Target_conformation)
                        
                        with open(f'Conformation{st.session_state.temp}.sdf', "rb") as file:
                             btn = st.download_button(
                                        label="Download your sdf file",
                                        data=file,
                                        file_name=f"Conformation n°{st.session_state.temp}.sdf",
                                        key = 'CIT_WEB_Unique_Molecule_ConformationTool')
                    else :
                        try :
                            with open(f'Conformation{st.session_state.temp}.sdf', "rb") as file:
                                 btn = st.download_button(
                                            label="Download your sdf file",
                                            data=file,
                                            file_name=f"Conformation n°{st.session_state.temp}.sdf",
                                            key = 'CIT_WEB_Unique_Molecule_ConformationTool')
                        except :
                            pass
                
                
                else :
                    st.info('There is only one predominant conformation. Do you want to have the sdf file of poses in this conformation OR see analysis of this conformation ? ')
                    st.session_state.RMSD_Target_conformation = st.slider(
                        'You want a sdf file and/or a analysis plots including molecules in the unique predominant conformation'
                        ' with all poses under a RMSD =', 0.0, 15.0, 2.0, key = 'CIT_WEB_Unique_Molecule_ConformationTool')
                    
                    if st.button('Prepare your sdf file', key = 'CIT_WEB_Unique_Molecule_ConformationTool'):             
                        st.session_state.ConformationClass.get_sdf_conformations(
                            1, st.session_state.RMSD_Target_conformation)
                        
                        with open('Conformation1.sdf', "rb") as file:
                         btn = st.download_button(
                                    label="Download your sdf file",
                                    data=file,
                                    file_name=f"Unique Conformation.sdf",
                                    key = 'CIT_WEB_Unique_Molecule_ConformationTool')
                    else :
                        try :
                            with open(f'Conformation1.sdf', "rb") as file:
                                 btn = st.download_button(
                                            label="Download your sdf file",
                                            data=file,
                                            file_name=f"Conformation n°1.sdf",
                                            key = 'CIT_WEB_Unique_Molecule_ConformationTool')
                        except :
                            pass
            
            else :
                if 'premier_passage' in st.session_state:
                    del st.session_state.premier_passage
                    try :
                        os.remove('Sorted_Heatmap.jpeg')
                        os.remove('Histograms_Best_Score.jpeg')
                        os.remove('Sample_Best_PLPScore_Poses.sdf')
                        os.remove('Best_PLPScore_Poses.sdf')
                        for i in range(st.session_state.n_conformations):
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
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1 :                
                metric = st.selectbox('Which metric do you want to use to compute the pairwise distances between observations in n-dimensional space ?',
                                      ('canberra', 'euclidean', 'hamming', 'braycurtis','chebyshev', 'cityblock', 'correlation',
                                       'cosine', 'dice', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis',
                                       'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
                                       'sokalsneath', 'sqeuclidean', 'yule'), key = 'CIT_WEB_Unique_Molecule_ConformationTool',
                                       help = "Before changing this setting, make sure the following checkbox is unchecked")
                st.session_state.metric = metric
                
            with col2 :
                method = st.selectbox('Which method do you want to use to compute the distance  between two clusters ?',
                                        ('average', 'single', 'complete', 'weighted', 'centroid', 'median', 'ward'),
                                       key = 'CIT_WEB_Unique_Molecule_ConformationTool',
                                       help = "Before changing this setting, make sure the following checkbox is unchecked")
                st.session_state.method = method
            third_checkbox_2 = st.checkbox('Get the cluster hierarchy heatmap', key = 'CIT_WEB_Unique_Molecule_ConformationTool')
            if third_checkbox_2 :
                if "cluster_hierarchy_heatmap" not in st.session_state :
                    st.session_state.ConformationClass.get_cluster_heatmap(individuals,
                                                                           method = st.session_state.method,
                                                                           metric = st.session_state.metric)
                    
                    if "n_clusters" not in st.session_state :
                        st.session_state.n_clusters = 2
                        
                    n_clusters = st.slider('In how many clusters do you want to cut the tree (dendrogram) ?', 2, 30, st.session_state.n_clusters,
                                            key = 'CIT_WEB_Unique_Molecule_ConformationTool')
                    st.session_state.n_clusters = n_clusters
                    st.session_state.n_clusters_selected = n_clusters
                    
                    if st.button('Get analysis', key = 'CIT_WEB_Unique_Molecule_ConformationTool'):
                        st.session_state.ConformationClass.analyze_cluster_heatmap(st.session_state.n_clusters, p = Proportion)
                else :
                    if "cluster_hierarchy_heatmap" in st.session_state :
                        st.pyplot(st.session_state.cluster_hierarchy_heatmap)
                        with open("Cluster_Hierarchy_Heatmap.jpeg", "rb") as file:
                            btn = st.download_button(
                                     label="Download PLOT : Cluster Hierarchy Heatmap",
                                     data=file,
                                     file_name="Cluster_Hierarchy_Heatmap.jpeg",
                                     mime="image/jpeg",
                                     key = 'CIT_WEB_Unique_Molecule_ConformationTool')                         
                        n_clusters = st.slider('In how many clusters do you want to cut the tree (dendrogram) ?', 2, 30,
                                               st.session_state.n_clusters, key = 'CIT_WEB_Unique_Molecule_ConformationTool')
                        st.session_state.n_clusters = n_clusters
                        
                        if st.button('Get analysis', key = 'CIT_WEB_Unique_Molecule_ConformationTool'):
                            st.session_state.n_clusters_selected = st.session_state.n_clusters
                            st.session_state.ConformationClass.analyze_cluster_heatmap(st.session_state.n_clusters, p = Proportion)
                            if st.button('Continue', key = 'CIT_WEB_Unique_Molecule_ConformationTool'):
                                "Rerun"
                            
                        else :
                            if "n_conformations" in st.session_state and st.session_state.n_clusters_selected == st.session_state.n_clusters:
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
                                                 file_name=f"Sample_Conformation{i+1}.sdf",
                                                 key = 'CIT_WEB_Unique_Molecule_ConformationTool')

                                with open("Best_PLPScore_Poses.sdf", "rb") as file:
                                     btn = st.download_button(
                                                label="Download the SDF file including each of the representatives of a conformation",
                                                data=file,
                                                file_name="Best_Score_Poses.sdf",
                                                key = 'CIT_WEB_Unique_Molecule_ConformationTool')

                                st.info(f"There is (are) {st.session_state.n_conformations} predominant pose(s) among all poses.\n")
                                for i, predominant_pose in enumerate(st.session_state.predominant_poses) :
                                    st.write(
                                        f"The predominant pose n°{i+1} represents {len(predominant_pose)/len(st.session_state.sample)*100:.1f}" 
                                        f"% of the sample, i.e. {len(predominant_pose)} on {len(st.session_state.sample)} poses in total.")

                                st.write("\nIn order to check that each group is different from each other, a table taking " 
                                          "the **with the best score** from each group and calculating the RMSD between each was constructed :\n")

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
                                                mime="image/jpeg", key = 'CIT_WEB_Unique_Molecule_ConformationTool')
                                
                                temp_options = range(1, st.session_state.n_conformations + 1)
                                if "temp" not in st.session_state :
                                    st.session_state.temp = 1
                                st.session_state.temp = st.select_slider("You want a sdf file and/or a analysis plots including molecules in the conformation n°",
                                                                         options=temp_options,
                                                                         value = st.session_state.temp,
                                                                         key = 'CIT_WEB_Unique_Molecule_ConformationTool')
                                
                                st.write(f"The Conformation selected is {st.session_state.temp}")

                                st.session_state.RMSD_Target_conformation = st.slider('... With all poses under a RMSD =', 0.0, 15.0, 2.0,
                                                                                      key = 'CIT_WEB_Unique_Molecule_ConformationTool')
                                st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")
                           
                                if st.button('Prepare your sdf file', key = 'CIT_WEB_Unique_Molecule_ConformationTool'):
                                    st.session_state.ConformationClass.get_sdf_conformations(
                                        st.session_state.temp,
                                        st.session_state.RMSD_Target_conformation)
                                    
                                    with open(f'Conformation{st.session_state.temp}.sdf', "rb") as file:
                                         btn = st.download_button(
                                                    label="Download your sdf file",
                                                    data=file,
                                                    file_name=f"Conformation n°{st.session_state.temp}.sdf",
                                                    key = 'CIT_WEB_Unique_Molecule_ConformationTool')
                                else :
                                    if os.path.exists(f'Conformation{st.session_state.temp}.sdf') == True :
                                        with open(f'Conformation{st.session_state.temp}.sdf', "rb") as file:
                                             btn = st.download_button(
                                                        label="Download your sdf file",
                                                        data=file,
                                                         file_name=f"Conformation n°{st.session_state.temp}.sdf")
                                    
            else:
                if "cluster_hierarchy_heatmap" in st.session_state :
                    del st.session_state.cluster_hierarchy_heatmap
                if "n_conformations" in st.session_state :
                    del st.session_state.n_conformations
                
    else :
        if 'uniquemol_delete_activated' in st.session_state:
            if os.path.exists(f'Sorted_Heatmap.jpeg') == True :
                os.remove('Sorted_Heatmap.jpeg')
            if os.path.exists(f'Cluster_Hierarchy_Heatmap.jpeg.jpeg') == True :
                os.remove('Cluster_Hierarchy_Heatmap.jpeg.jpeg')
            if os.path.exists(f'Histograms_Best_Score.jpeg') == True :
                os.remove('Histograms_Best_Score.jpeg')
            if os.path.exists(f'Sample_Best_PLPScore_Poses.sdf') == True :
                os.remove('Sample_Best_PLPScore_Poses.sdf')
            if os.path.exists(f'Best_PLPScore_Poses.sdf') == True :
                os.remove('Best_PLPScore_Poses.sdf')

            if "n_conformations" in st.session_state or "n_clusters" in st.session_state :
                for i in range(st.session_state.n_conformations):
                    if os.path.exists(f'Sample_Conformation{i+1}.sdf') == True :
                        os.remove(f'Sample_Conformation{i+1}.sdf')
                    if os.path.exists(f'Conformation{i+1}.sdf') == True :
                        os.remove(f'Conformation{i+1}.sdf')
                    if os.path.exists(f'Barplot_Conformation{i+1}.jpeg') == True :
                        os.remove(f'Barplot_Conformation{i+1}.jpeg')
                    if os.path.exists(f'Box_Plot{i+1}.jpeg') == True :
                        os.remove(f'Box_Plot{i+1}.jpeg')
                    if os.path.exists(f'Scatter_Plot{i+1}.jpeg') == True :
                        os.remove(f'Scatter_Plot{i+1}.jpeg')

            for key in st.session_state.keys():
                del st.session_state[key]
 
