#!/usr/bin/env python
# coding: utf-8

# An implementation of generic algorithm

# In[3]:


import numpy as np
import random
import datetime


# In[6]:


class GenericAlgorithm(object):
    def __init__(self,optimization_function,variable_bounds,population_size=1000,init_values=None,
                 opt_target='max',crossover_rate=0.5,mutation_rate=0.05,variable_types=None):
        self.optimization_function=optimization_function #the function to be optimized
        self.variable_bounds=variable_bounds #the lower and upper bounds of each variable, an np array with size of N(the size of variable)*2
        self.crossover_rate=crossover_rate #the rate that DNA(variable) happens crossover, a float between 0 and 1
        self.mutation_rate=mutation_rate #the rate that DNA(variable) mutates, a float between 0 and 1
        self.variable_size=np.shape(variable_bounds)[0]
        
        #variable type is a list with length N, the value should be 'float' or 'int'
        if variable_types is None:
            self.variable_types=list()
            for i in range(self.variable_size):
                self.variable_types.append('float')
        else:
            assert self.variable_size==len(variable_types)
            self.variable_types=variable_types
        
        #init_values has the size of M*N
        #To use mixed data type, all values are saved as string. Remember to transform the data type when using the values!
        if init_values is not None:
            assert self.variable_size==np.shape(init_values)[1]
            self.population=np.array(init_values,dtype='str')
            self.population_size=np.shape(init_values)[0]
        else:
            self.population_size=population_size
            self.population=np.random.randn(population_size,self.variable_size)
            self.population=np.array(self.population,dtype='str')
            for i in range(population_size):
                for j in range(self.variable_size):
                    if self.variable_types[j]=='int':
                        self.population[i,j]=str(random.randint(self.variable_bounds[j,0],self.variable_bounds[j,1]))
                    else:
                        self.population[i,j]=str(random.uniform(self.variable_bounds[j,0],self.variable_bounds[j,1]))
        
        assert opt_target=='max' or opt_target=='min'
        self.opt_target=opt_target #the optimization target, should be 'min' or 'max'
        
    def select(self): 
        values=self.optimization_function(self.population)
        ret=0.0
        if self.opt_target=='min':
            ret=np.min(values)
            values=-values
        else:
            ret=np.max(values)
        values-=np.max(values)
        temp=np.sum(np.exp(values),axis=0)
        values=np.exp(values)/temp
        self.population=self.population[np.random.choice(np.arange(0,np.shape(self.population)[0],1),size=np.shape(self.population)[0],p=values)]
        return ret
    
    def mutate(self):
        for i in range(self.population_size):
            for j in range(self.variable_size):
                temp=random.uniform(0,1)
                if temp<self.mutation_rate:
                    if self.variable_types[j]=='int':
                        self.population[i,j]=random.randint(self.variable_bounds[j,0],self.variable_bounds[j,1])
                    else:
                        self.population[i,j]=random.uniform(self.variable_bounds[j,0],self.variable_bounds[j,1])
        return 
    
    def crossover(self):
        for i in range(self.population_size):
            temp=random.uniform(0,1)
            if temp<self.crossover_rate:
                pop_choice=random.randint(0,self.population_size-1)
                pop_pos=random.randint(0,self.variable_size-1)
                if pop_choice==i:
                    continue
                temp_var=self.population[pop_choice]
                self.population[pop_choice,pop_pos:]=self.population[i,pop_pos:]
                self.population[i,pop_pos:]=temp_var[pop_pos:]
        return
        
    def evolution(self,num_iterations=1000):
        for i in range(num_iterations):
            time_start=datetime.datetime.now()
            temp=self.select()
            self.crossover()
            self.mutate()
            time_end=datetime.datetime.now()
            print('iteration: ',i,' time= ',time_end-time_start,' min/max value= ',temp)
            
    def get_result(self):
        values=self.optimization_function(self.population)
        seq=np.argsort(values)
        if self.opt_target=='max':
            seq=seq[::-1]
        values=values[seq]
        self.population=self.population[seq]
        return self.population,values


# In[ ]:


if __name__=='__main__':
    print(' ')

