import numpy as np
import pandas as pd
import os 
import random
import pickle

def dump_to_array(filename):
	atom_list=list()
	with open(filename) as f:
		for i in range(3):
			f.readline()
		atom_num=int(f.readline())
		for i in range(5):
			f.readline()
		for i in range(atom_num):
			atom_id,atom_type,x,y,z=f.readline().split()
			atom_list.append(list([int(atom_type),float(x),float(y),float(z)]))
	return atom_list

def rdf_process(filename,config_length,rdf_length):
	rdf_list=list()
	with open(filename) as f:
		for i in range(3):
			f.readline()
		for i in range(config_length):
			f.readline()
			temp_list=list()
			for j in range(rdf_length):
				number,right_bound,g_r,_=f.readline().split()
				temp_list.append(float(g_r))
			rdf_list.append(temp_list)
	return rdf_list

def data_process(data_path):
	training_data=list()
	rdf_data=list()
	for i in range(101):
		for j in range(101):
			training_data.append(dump_to_array(data_path+'/output'+str(i)+'/dump.'+str(j*10)))
		rdf_data=rdf_data+rdf_process(data_path+'/output'+str(i)+'/rdf.txt',101,256)
	return training_data,rdf_data

config_data=list()
rdf_data=list()
file_list=['../bcc','../fcc','../hcp','../omega']
for item in file_list:
	temp1,temp2=data_process(item)
	config_data=config_data+temp1
	rdf_data=rdf_data+temp2
#print(config_data)
#print(rdf_data)
with open('config_data.pickle','wb') as f:
	pickle.dump(config_data,f)
with open('rdf_data.pickle','wb') as f:
	pickle.dump(rdf_data,f)
