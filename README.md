# 2019_nCov_predcition
COVID-19 (2019-nCov) has hit not only mainland China but also all over the world. This project is a refined mathematical model of the outbreak of 2019-nCov and prediction.   

## model method   
modified SEIR model:   
1. take incubation period patients' ability to infect other people into consideration.  
2. propose a two stage model.    

## Directory

### data   
input dat for my research   
1. data.h5, data_init.h5, data_new.h5   
three data files containing number of the infected, suspected, dead and cured people.   
2. raw_data.xlsx   
raw data of detailed circumstances in some sample provinces.   
3. detail_dta_store.h5    
HDFStore of detailed data   
4. data_construct.py   
python file for data construction    

### code 
python code for research   
1. SEIR.ipynb  
traditional SEIR model   
2. class_SEIR.py  
encapsulated SEIR model code   
3. class_SEIR_modified.py       
my encapsulated model (modify SEIR)   
4. my_model.ipynb   

### figure   
figure of results, model and mathematical formula.  
21 figures in total  

### research_2019_nCov_YqWu
my research report   
