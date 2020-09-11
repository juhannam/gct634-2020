# Instruction to set up the course Conda virtual environment 


We will use Python throughout the course. For practice session and homework, we will manage all library packages under a Conda virtual environment named "gct634". As a first step, please install the Python 3 version of Anaconda (https://repo.anaconda.com/). It will provide a cross-platform installation package. 

After you have installed conda, close any open terminals you might have. Then open a new terminal and run the following command:


1. Create a gct634 environment with python 3.8

   conda create -n gct634 python=3.8

2. Activate the new environment:

    conda activate gct634

3. install the following packages by typing one by one

    conda create -y -n gct634 python=3.8 jupyter jupyterlab matplotlib <br>
    conda activate gct634 <br>
    conda install -y -c conda-forge librosa <br>
    python -m ipykernel install —user —name gct634 

4. Run Jupyter Notebook 

   jupyter notebook 	

5. Go to the "practice" folder and open "practice1.ipynb". Pleas make sure that the kernel is set to gct634 from the rightmost of the menu. 
 
6. To deactivate an active environment, use
    
   conda deactivate

