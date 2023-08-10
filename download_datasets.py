import os
import gdown
import zipfile


if not os.path.exists('Dataset/DAVIS-2017-trainval-480p.zip'):
    url = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip'
    gdown.download(url, output='Dataset/DAVIS-2017-trainval-480p.zip', quiet=False)

if not os.path.exists('Dataset/DAVIS-2017-test-dev-480p.zip'):
    url = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip'
    gdown.download(url, output='Dataset/DAVIS-2017-test-dev-480p.zip', quiet=False)

if not os.path.exists('Dataset/DAVIS-2017-test-challenge-480p.zip'):
    url = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-challenge-480p.zip'
    gdown.download(url, output='Dataset/DAVIS-2017-test-challenge-480p.zip', quiet=False)

if not os.path.exists('Dataset/DAVIS/JPEGImages/480p'):
    print('Extracting DAVIS 2017 trainval...')
    with zipfile.ZipFile('Dataset/DAVIS-2017-trainval-480p.zip', 'r') as zip_file:
        zip_file.extractall('Dataset/')
    print('Extracting DAVIS 2017 test-dev...')
    with zipfile.ZipFile('Dataset/DAVIS-2017-test-dev-480p.zip', 'r') as zip_file:
        zip_file.extractall('Dataset/')
    print('Extracting DAVIS 2017 test-challenge...')
    with zipfile.ZipFile('Dataset/DAVIS-2017-test-challenge-480p.zip', 'r') as zip_file:
        zip_file.extractall('Dataset/')
