# GCT634/AI613 (Fall 2020)
Code repository for GCT634/AI613 Musical Applications of Machine Learning (Fall 2020)

This contains source code for: 
- Practice sessions
- Homeworks


## Instruction to set up the Course Conda Virtual Environment 


We will use Python throughout the course. For practice session and homework, we will manage all library packages under a Conda virtual environment named "gct634". As a first step, please install the Python 3 version of Anaconda (https://repo.anaconda.com/). It will provide a cross-platform installation package. 

After you have installed conda, close any open terminals you might have. Then open a new terminal and run the following command:

1. Create an environment with dependencies specified in env.yml:
    
   conda env create -f environment.yaml

2. Activate the new environment:
    
   conda activate gct634
    
3. Inside the new environment, install IPython kernel so we can use this environment in jupyter notebook: 
    
   python -m ipykernel install --user --name gct634

4. Run Jupyter Notebook 

   jupyter notebook 	

5. Go to the "practice" folder and open "practice.ipynb". Pleas make sure that the kernel is set to gct634 from the rightmost of the menu. 
 
6. To deactivate an active environment, use
    
   conda deactivate

