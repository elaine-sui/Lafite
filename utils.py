import gdown
import os

drive_urls = {
    "cc3m_pretrain_ckpt": "https://drive.google.com/u/0/uc?id=17ER7Yl02Y6yCPbyWxK_tGrJ8RKkcieKq&export=download"
}

def download_file_from_google_drive(key, destination):
    url = drive_urls[key]
    gdown.download(url, destination, quiet=False)
    
if __name__ == '__main__':
    dest = "/pasteur/u/esui/data/lafite/"
    os.makedirs(dest, exist_ok="True")
    download_file_from_google_drive("cc3m_pretrain_ckpt", dest)