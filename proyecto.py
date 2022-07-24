
#Primero importamos las librerias
####################################################
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.model_selection import cross_val_score
import random
random.seed(42)
import os
print(os.listdir("../input"))



#Necesitamos los siguientes datos
####################################################

pot_energy=pd.read_csv('../input/potential_energy.csv')
mulliken_charges=pd.read_csv('../input/mulliken_charges.csv')
train_df=pd.read_csv('../input/train.csv')
scalar_coupling_cont=pd.read_csv('../input/scalar_coupling_contributions.csv')
test_df=pd.read_csv('../input/test.csv')
magnetic_shield_tensor=pd.read_csv('../input/magnetic_shielding_tensors.csv')
dipole_moment=pd.read_csv('../input/dipole_moments.csv')
structures=pd.read_csv('../input/structures.csv')




print('Shape of potential energy dataset:',pot_energy.shape)
print('Shape of mulliken_charges dataset:',mulliken_charges.shape)
print('Shape of train dataset:',train_df.shape)
print('Shape of scalar coupling contributions dataset:',scalar_coupling_cont.shape)
print('Shape of test dataset:',test_df.shape)
print('Shape of magnetic shielding tensors dataset:',magnetic_shield_tensor.shape)
print('Shape of dipole moments dataset:',dipole_moment.shape)
print('Shape of structures dataset:',structures.shape)




#Exploramos los datasetes
####################################################


#Dataset de energia
print('Data Types:\n',pot_energy.dtypes)
print('Descriptive statistics:\n',np.round(pot_energy.describe(),3))
pot_energy.head(6)



#Datasetes de la carga
print('Data Types:\n',mulliken_charges.dtypes)
print('Descriptive statistics:\n',np.round(mulliken_charges.describe(),3))
mulliken_charges.head(6)


#Datasetes 
print('Data Types:\n',train_df.dtypes)
print('Descriptive statistics:\n',np.round(train_df.describe(),3))
train_df.head(6)



#Datasetes del acople escalar
print('Data Types:\n',scalar_coupling_cont.dtypes)
print('Descriptive statistics:\n',np.round(scalar_coupling_cont.describe(),3))
scalar_coupling_cont.head(6)

#Dataset estadistico
print('Data Types:\n',test_df.dtypes)
print('Descriptive statistics:\n',np.round(test_df.describe(),3))
test_df.head(6)

#Dataset de tensor de campo magnetico
print('Data Types:\n',magnetic_shield_tensor.dtypes)
print('Descriptive statistics:\n',np.round(magnetic_shield_tensor.describe(),3))
magnetic_shield_tensor.head(6)


#Dataset de la estructura
print('Data Types:\n',structures.dtypes)
print('Descriptive statistics:\n',np.round(structures.describe(),3))
structures.head(6)

#######################################################################################
#Mapa de la estructura atomica y prueba

def map_atom_data(df,atom_idx):
    df=pd.merge(df,structures,how='left',
               left_on=['molecule_name',f'atom_index_{atom_idx}'],
               right_on=['molecule_name','atom_index'])
    df=df.drop('atom_index',axis=1)
    df=df.rename(columns={'atom':f'atom_{atom_idx}',
                         'x':f'x_{atom_idx}',
                         'y':f'y_{atom_idx}',
                         'z':f'z_{atom_idx}'})
    return df

train_df['type_0']=train_df['type'].apply(lambda x:x)
test_df['type_0']=test_df['type'].apply(lambda x : x)

train_df=train_df.drop(columns=['molecule_name','type'],axis=1)
display(train_df.head(6))


test_df=test_df.drop(columns=['molecule_name','type'],axis=1)
display(test_df.head(10))


##############################################
#Histograma de visualizacion

train_df['type_0']=train_df.type_0.astype('category')
train_df['atom_0']=train_df.atom_0.astype('category')
train_df['atom_1']=train_df.atom_1.astype('category')


test_df['type_0']=test_df.type_0.astype('category')
test_df['atom_0']=test_df.atom_0.astype('category')
test_df['atom_1']=test_df.atom_1.astype('category')

plt.hist(train_df['scalar_coupling_constant'])
plt.ylabel('No of times')
plt.xlabel('scalar copling constant')
plt.show()

plt.hist(train_df['dist_vector'])
plt.ylabel('No of times')
plt.xlabel('Distance vector')
plt.show()

plt.hist(train_df['dist_X'])
plt.ylabel('No of times')
plt.xlabel('X distance vector')
plt.show()


plt.hist(train_df['dist_Y'])
plt.ylabel('No of times')
plt.xlabel('Y distance vector')
plt.show()


plt.hist(train_df['dist_Z'])
plt.ylabel('No of times')
plt.xlabel('Z distance vector')
plt.show()

train_df.head(5)






