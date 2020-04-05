import subprocess
import os.path
    
def download_data(URL, is_colab=1, save_model_drive = False):
    if os.path.isfile("text8"):
        print("File already downloaded!")
        return 
    
    if is_colab:
        subprocess.run(["pip","install", "google"])
        subprocess.run(["apt-get","install", "unzip"])
        subprocess.run(["cd" ,"/content"])
        # to save model on drive
        if save_model_drive:
            from google.colab import drive
            drive.mount('/content/gdrive')
        
    subprocess.run(["wget",URL])
    subprocess.run(["unzip", "text8.zip"])
    subprocess.run(["rm", 'text8.zip'])
