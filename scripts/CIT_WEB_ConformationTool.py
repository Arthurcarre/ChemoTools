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

"Hello from conda"

b = st.button("Click me")

if b:
  st.balloons()
