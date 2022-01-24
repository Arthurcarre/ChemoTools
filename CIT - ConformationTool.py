import copy, random, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import rdMolAlign
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
        
        self.mols_brut = [x for x in Chem.SDMolSupplier(sdf_file)]
        self.score = score
        
        output_name_prefix = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
        structures_directory = output_name_prefix + "_Results_Conformation"
        structures_path = os.path.join(os.getcwd(), structures_directory)
        os.mkdir(structures_path)


    def check_sdf_file(self, benchmark_molecule):
        """
        -- DESCRIPTION --
        This function allows to check, from a reference molecule (whose pathway will be provided),
        that all molecules in the sdf file share the same structural backbone with the reference molecule.
        Molecules that do not meet this criterion will not be included in the rest of the algorithm.
            PARAMS:
                - benchmark_molecule (string): path to a MOL file
        """
        
        mols = []
        error_mols = []
        for i, mol in enumerate(self.mols_brut) :
            try :
                rdMolAlign.CalcRMS(GetScaffoldForMol(mol), GetScaffoldForMol(Chem.MolFromMolFile(benchmark_molecule)))
                mols.append(self.mols_brut[i])
            except RuntimeError as e :
                error_mols.append(mol.GetProp("_Name"))

        for i in error_mols : 
            print(i)
        print("All theses molecules have not sub-structure which match between the reference and probe mol.\n"
              "An RMSD can't be calculated with these molecules, they will therefore not be taken into", 
              "account by the algorithm.\n")

        self.mols = mols
        
        print(f"There are {len(self.mols_brut)} molecules' poses in the sdf file.\n")
        print(f"There are {len(self.mols)} molecules' poses which will be used by the algorithm.\n")
        
        del error_mols
        del self.mols_brut
    
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
            print("OOPS ! You're trying to define a sample more larger than the numbers of molecules/poses in your sdf.",
                  " This is impossible, please redefine a size of sample equal or smaller to your sdf")
        
        array = np.ones(shape=(len(sample),len(sample)))
        for i, indivduali in enumerate(sample) :
            for j, indivdualj in enumerate(sample) :
                array[i, j] = rdMolAlign.CalcRMS(GetScaffoldForMol(self.mols[indivduali]),
                                                 GetScaffoldForMol(self.mols[indivdualj]))

        df = pd.DataFrame(
            array, index=list(range(len(sample))), columns=list(range(len(sample))))

        fig, ax = plt.subplots(figsize=(20, 10))
        sns.set_context('talk')
        sns.heatmap(df, fmt='d', ax= ax)
        plt.show()
    
    def get_sorted_heatmap(self, individuals = 200, RMSDthreshold = 3.0, loop = 1, p = 0.05) :
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
            self.sample = random.sample(range(len(self.mols)), individuals)
        except ValueError as e :
            print("OOPS ! You're trying to define a sample more larger",
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
                    if rdMolAlign.CalcRMS(GetScaffoldForMol(self.mols[individual]),
                                          GetScaffoldForMol(self.mols[sample[i]])) <  RMSDthreshold :
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
                                RMSD.append(rdMolAlign.CalcRMS(GetScaffoldForMol(self.mols[self.sample[k]]),
                                                               GetScaffoldForMol(self.mols[self.sample[input_list[i][n]]])))
                        if len(groupej) > len(groupei):
                            for k in groupei :
                                n += 1
                                RMSD.append(rdMolAlign.CalcRMS(GetScaffoldForMol(self.mols[self.sample[k]]),
                                                               GetScaffoldForMol(self.mols[self.sample[input_list[j][n]]])))
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
                    r = rdMolAlign.CalcRMS(GetScaffoldForMol(self.mols[self.sample[n]]),
                                           GetScaffoldForMol(self.mols[self.sample[liste1[i+1]]]))
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
            print(f"There is (are) {k} predominant pose(s) among all poses.\n")

            for i, predominant_pose in enumerate(predominant_poses) :
                print(f"The predominant pose n°{i+1} represents {len(predominant_pose)/len(self.sample)*100:.1f}", 
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
                        if rdMolAlign.CalcRMS(GetScaffoldForMol(self.mols[self.sample[sample_predominant_poses[i][indice]]]),
                                              GetScaffoldForMol(mol)) < 2 :
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
                    array[i, j] = rdMolAlign.CalcRMS(GetScaffoldForMol(moli), GetScaffoldForMol(molj))

            data_frame = pd.DataFrame(array, index=index, columns=columns)
            print(data_frame)
        
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
                sdf_to_hist = [rdMolAlign.CalcRMS(GetScaffoldForMol(input_list[0]), GetScaffoldForMol(mol)) for mol in self.mols]
                fig, ax = plt.subplots(len(input_list), 1, figsize=(15, 0.2*len(input_list)*9))
                a, b, c = 0, 0, 0
                for RMSD in sdf_to_hist : 
                    if RMSD < 2 :
                        a += 1
                    if RMSD < 3 :
                        b += 1
                    if RMSD < 4 :
                        c += 1
                ax.hist(sdf_to_hist, bins =100, label = "Conformation n°1") #Create an histogram to see the distribution of the RMSD of the sample
                ax.axvline(x=2, ymin=0, ymax=1, color="black", linestyle="--")
                ax.annotate(a, (1.5, 0.05*len(self.mols)), fontsize=15)
                ax.axvline(x=3, ymin=0, ymax=1, color="black", linestyle="--")
                ax.annotate(b-a, (2.5, 0.05*len(self.mols)), fontsize=15)
                ax.axvline(x=4, ymin=0, ymax=1, color="black", linestyle="--")
                ax.annotate(c-b, (3.5, 0.05*len(self.mols)), fontsize=15)
                ax.legend(loc='upper left', shadow=True, markerfirst = False)
            else :
                sdf_to_hist = ([rdMolAlign.CalcRMS(GetScaffoldForMol(representative_conf),
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

            plt.show()
            fig.savefig("Histograms_BestPLP.jpeg", dpi=300)
            print("RMSD distribution between all docking solutions", 
                  " and the pose with the highest ChemPLP score of all solutions for a given conformation.")
        
        
        #OUT THE FUNCTIUN "get_histogramme_sample_bestPLP".
        output_liste = sorted_list_lengroups(get_groups_inside_list(improve_sort(
            get_filtered_liste(sorted_list_lengroups(gather_groups_RMSD(RMSD_listes(self.sample)))), loop)))
        
        finallyliste = get_filtered_liste(output_liste)
        
        if len(finallyliste) != individuals :
            print(f"ATTENTION ! The sorting process discarded {individuals-len(finallyliste)} individuals")
        array = np.ones(shape=(len(finallyliste),len(finallyliste)))
        
        for i, indivduali in enumerate(finallyliste) :
            for j, indivdualj in enumerate(finallyliste) :
                array[i, j] = rdMolAlign.CalcRMS(GetScaffoldForMol(self.mols[self.sample[indivduali]]),
                                                 GetScaffoldForMol(self.mols[self.sample[indivdualj]]))

        data_frame = pd.DataFrame(array, index=finallyliste,
                                  columns=finallyliste)
        fig, ax = plt.subplots(figsize=(20, 10))
        
        try : 
            sns.set_context('talk')
            g = sns.heatmap(data_frame, fmt='d', ax= ax, cmap = "rocket")
            fig = g.get_figure()
            fig.savefig("Heatmap.jpeg", dpi=300)
            plt.show()
            sample_predominant_poses = get_predominant_poses(output_liste, p)
            sample_indice_best_score = get_sample_indice_best_score(sample_predominant_poses)
            get_SDF_Sample_and_Best_Score_Poses(sample_predominant_poses, sample_indice_best_score)
            best_PLP_poses = [GetScaffoldForMol(x) for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
            get_data_frame_best_poses(best_PLP_poses)
            get_histogramme_sample_bestPLP(best_PLP_poses)
            self.best_PLP_poses = best_PLP_poses
            
        except ValueError :
            print("OOPS ! The selected RMSD threshold does not allow all individuals to be grouped "
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
                if rdMolAlign.CalcRMS(GetScaffoldForMol(self.best_PLP_poses[k-1]),
                                      GetScaffoldForMol(mol)) < float(RMSDtarget) :
                    w.write(self.mols[j])
        w.close()
    

    def get_plots(self,
                     k,
                     molecule_name = "Compound Name",
                     aspect_plot = 1.75,
                     height_plot = 18,
                     size_xlabels = 25,
                     size_ylabels = 15,
                     aspect_density_plot = 25,
                     height_density_plot = 0.5,
                     gap_density_plot = -0.45) :
        
        """
        -- DESCRIPTION --
        After the "get_sdf_conformations" function, this function allows you to use the previously created
        sdf file and draw plots from it. The barplot shows for each molecule the ratio between the number of
        poses for which the molecule adheres to the conformation and the total number of poses and presents
        them in a decreasing order of the ratio. Subsequently, a boxplot and a densityplot are drawn and
        illustrate the distribution of the score (according to the scoring function from the docking algorithms)
        for each molecule with respect to the poses that are in the given confirmation. The order of the molecules
        is the same as that of the barplot. Finally, a new boxplot and a new density plot are again drawn,
        but this time in the descending order of the median of each distribution of the score for each molecule.
            PARAMS:
                - k (int): Number of the selected conformation
                - molecule_name (string): Name of the column in the sdf that contains the name of the molecule
                - aspect_plot (float): Aspect ratio of each facet, regarding the barplot and the boxplots,
                so that aspect * height gives the width of each facet in inches
                - height_plot (float): Height (in inches) of each facet, regarding the barplot and the boxplots
                - size_xlabels (float): Number characterising the size of the axis X labels
                - size_ylabels (float): Number characterising the size of the axis Y labels
                - aspect_density_plot (float): Aspect ratio of each facet, regarding the density plots,
                so that aspect * height gives the width of each facet in inches
                - height_density_plot (float): Height (in inches) of each facet, regarding the density plots
                - gap_density_plot (float): Number characterising the gap between each density plot.
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
        g.fig.savefig(f"Barplot_Conformation{k}.jpeg", dpi=300)
        
        # BOX PLOT
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
        
        data_frame2.reset_index(inplace=True)
        data_frame2.rename(columns={'index': 'Name'}, inplace=True)
        
        sns.set_style('whitegrid')
        sns.set_context('talk')
        g = sns.catplot(x=self.score, y='Name', data=data_frame2, kind = "box", palette="rocket",
                        aspect=aspect_plot, height=height_plot)
        g.set_axis_labels(self.score, "", fontsize = 20)
        g.set_yticklabels(fontsize = size_ylabels)
        g.set_xticklabels(fontsize = size_xlabels)
        g.fig.savefig(f"Box_Plot{k}.jpeg", dpi=300)
        
        # DENSITY PLOT
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        sns.set_context('talk')

        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(len(data_frame2.index), rot=-.25, light=.7)
        g = sns.FacetGrid(data_frame2, row="Name", hue="Name", aspect=aspect_density_plot,
                          height=height_density_plot, palette="rocket")

        # Draw the densities in a few steps
        g.map(sns.kdeplot, self.score,
              bw_adjust=.5, clip_on=False,
              fill=True, alpha=1, linewidth=1.5)
        g.map(sns.kdeplot, self.score, clip_on=False, color="w", lw=2, bw_adjust=.5)

        # passing color=None to refline() uses the hue mapping
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)

        g.map(label, self.score)

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=gap_density_plot)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        g.fig.savefig(f"Density_Plot{k}.jpeg", dpi=300)
        
        
        
        ### SECOND BOX PLOT AND DENSITY PLOT

        #Reset index and Sort the DF3 Median column in a ascending order 
        data_frame3.reset_index(inplace=True)
        data_frame3.rename(columns={'index': 'Name'}, inplace=True)
        data_frame3.sort_values("Median", ascending=False, inplace=True)
        
        # Second Box Plot
        sns.set_style('whitegrid')
        sns.set_context('talk')
        
        g = sns.catplot(x=self.score, y='Name', data=data_frame3, kind = "box", palette="rocket",
                        aspect=aspect_plot, height=height_plot)
        g.set_axis_labels(self.score, "")
        g.set_yticklabels(fontsize = size_ylabels)
        g.set_xticklabels(fontsize = size_xlabels)
        g.fig.savefig(f"Box2_Plot{k}.jpeg", dpi=300)
        
        
        #Second Density Plot
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        sns.set_context('talk')
        
        pal = sns.cubehelix_palette(len(data_frame3.index), rot=-.25, light=.7)
        g = sns.FacetGrid(data_frame3, row="Name", hue="Name", aspect=aspect_density_plot,
                          height=height_density_plot, palette="rocket")
        g.map(sns.kdeplot, self.score,
              bw_adjust=.5, clip_on=False,
              fill=True, alpha=1, linewidth=1.5)
        g.map(sns.kdeplot, self.score, clip_on=False, color="w", lw=2, bw_adjust=.5)
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
        g.map(label, self.score)
        g.figure.subplots_adjust(hspace=gap_density_plot)
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        g.fig.savefig(f"Density2_Plot{k}.jpeg", dpi=300)
