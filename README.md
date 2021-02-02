# Reverse Image Search Engine  
This is a follow along application built on top of the 4th Chapter of *Practical Deep Learning for Cloud, Mobile & Edge* [1]. The `caltech101` dataset was used to train the model.  

## Setup  
The following steps will setup your virtual environment and download the dataset.

### Create Poetry Env  

```bash  
poetry shell && poetry install
```  

### Dataset Download  
Once the poetry environment is setup and active, you can download the dataset.  

```bash
bash training/data_setup.sh
```  

## Training  
From inside the virtual environment, you can train the model by running:  

```bash  
python training/train.py
```

## References  
[1] Koul, A., Ganju, S., & Kasam, M. (2019). Practical Deep Learning for Cloud, Mobile, and Edge: Real-World AI & Computer-Vision Projects Using Python, Keras & TensorFlow (1st ed.). O’Reilly Media.
