import os
import sys

def check_ipynb(): 


    notebooks = []
    
    for root, dirs, files in os.walk('.'): 

        if '.git' in dirs: 
            dirs.remove('.git')
        
        dirs[:] = [d for d in dirs if not d.startswith('.')]


        for file in files: 
            if file.endswith('.ipynb'): 
                notebook_path = os.path.join(root, file)
                notebooks.append(notebook_path)

        if notebooks: 
            print("error: jupyter notebook found in the repository")
            for notebook in notebooks: 
                print(f'- {notebook}')

            print("\n Please remove the notebook, or push to the notebooks branch")

        
        return False
    
if __name__ == "__main__": 
    
    notebooks_found = check_ipynb()

    sys.exit(1 if notebooks_found else 0)