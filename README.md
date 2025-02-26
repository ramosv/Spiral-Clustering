# Spiral-Clustering
Using the spiral-dataset.csv to generate a spiral figure.  
Challenging to cluster due to the nature of the data.

Completed all required tasks plus some extra
- Implemented k-means clustering (with Euclidean, cosine, and L3 distances).  
- Evaluated clusters using Sum of Squared Errors and random index.  
- Ran 10 random initializations per metric and selected the best run.  
- plots dir has all the plots
- terminal output as the correspoing output for these plots

Note: For my k-means implementation, and anything inside the cluster_it component I avoided using numpy operations in key parts to dig dipper into how is it that its working behind the scenes.


ramos@Lenovo ~/Desktop/GitHub/Spiral-Clustering (main)
$ python -m venv .venv

activate the virtual environment  
ramos@Lenovo ~/Desktop/GitHub/Spiral-Clustering (main)
$ source .venv/Scripts/activate
(.venv)
ramos@Lenovo ~/Desktop/GitHub/Spiral-Clustering (main)
$
