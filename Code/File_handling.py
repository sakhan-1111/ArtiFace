####################################################
#
# File Handling
#
####################################################

# Import libraries
import os, shutil

# Make train, validation & test directories
os.mkdir('Datasets')
os.mkdir('Datasets/train')
os.mkdir('Datasets/train/Real')
os.mkdir('Datasets/train/Fake')
os.mkdir('Datasets/validation')
os.mkdir('Datasets/validation/Real')
os.mkdir('Datasets/validation/Fake')
os.mkdir('Datasets/test')
os.mkdir('Datasets/test/Real')
os.mkdir('Datasets/test/Fake')


# Move files from CIPS to train folder
for rn in range(3289):
    src_file = 'Dataset/Real/CIPS/CIPS.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/train/Real'
    shutil.copy(src_file, dst_dir)
    
# Move files from CIPS to test folder
for rn in range(3289, 3701):
    src_file = 'Dataset/Real/CIPS/CIPS.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/test/Real'
    shutil.copy(src_file, dst_dir)

# Move files from CIPS to validation folder
for rn in range(3701, 4113):
    src_file = 'Dataset/Real/CIPS/CIPS.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/validation/Real'
    shutil.copy(src_file, dst_dir)


# Move files from FFHQ to train folder
for rn in range(39405):
    src_file = 'Dataset/Real/FFHQ/FFHQ.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/train/Real'
    shutil.copy(src_file, dst_dir)
    
# Move files from FFHQ to test folder
for rn in range(39405, 44330):
    src_file = 'Dataset/Real/FFHQ/FFHQ.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/test/Real'
    shutil.copy(src_file, dst_dir)

# Move files from FFHQ to validation folder
for rn in range(44330, 49255):
    src_file = 'Dataset/Real/FFHQ/FFHQ.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/validation/Real'
    shutil.copy(src_file, dst_dir)


# Move files from ProjectedGAN to train folder
for rn in range(1313):
    src_file = 'Dataset/Fake/ProjectedGAN/ProjectedGAN.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/train/Fake'
    shutil.copy(src_file, dst_dir)
    
# Move files from ProjectedGAN to test folder
for rn in range(1313, 1477):
    src_file = 'Dataset/Fake/ProjectedGAN/ProjectedGAN.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/test/Fake'
    shutil.copy(src_file, dst_dir)

# Move files from ProjectedGAN to validation folder
for rn in range(1477, 1641):
    src_file = 'Dataset/Fake/ProjectedGAN/ProjectedGAN.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/validation/Fake'
    shutil.copy(src_file, dst_dir)


# Move files from Stable_Diffusion to train folder
for rn in range(1207):
    src_file = 'Dataset/Fake/Stable_Diffusion/Stable_Diffusion.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/train/Fake'
    shutil.copy(src_file, dst_dir)
    
# Move files from Stable_Diffusion to test folder
for rn in range(1207, 1358):
    src_file = 'Dataset/Fake/Stable_Diffusion/Stable_Diffusion.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/test/Fake'
    shutil.copy(src_file, dst_dir)

# Move files from Stable_Diffusion to validation folder
for rn in range(1358, 1509):
    src_file = 'Dataset/Fake/Stable_Diffusion/Stable_Diffusion.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/validation/Fake'
    shutil.copy(src_file, dst_dir)


# Move files from StarGAN to train folder
for rn in range(6008):
    src_file = 'Dataset/Fake/StarGAN/StarGAN.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/train/Fake'
    shutil.copy(src_file, dst_dir)
    
# Move files from StarGAN to test folder
for rn in range(6008, 6760):
    src_file = 'Dataset/Fake/StarGAN/StarGAN.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/test/Fake'
    shutil.copy(src_file, dst_dir)

# Move files from StarGAN to validation folder
for rn in range(6760, 7512):
    src_file = 'Dataset/Fake/StarGAN/StarGAN.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/validation/Fake'
    shutil.copy(src_file, dst_dir)


# Move files from Taming_Transformer to train folder
for rn in range(34096):
    src_file = 'Dataset/Fake/Taming_Transformer/Taming_Transformer.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/train/Fake'
    shutil.copy(src_file, dst_dir)
    
# Move files from Taming_Transformer to test folder
for rn in range(34096, 38358):
    src_file = 'Dataset/Fake/Taming_Transformer/Taming_Transformer.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/test/Fake'
    shutil.copy(src_file, dst_dir)

# Move files from Taming_Transformer to validation folder
for rn in range(38358, 42620):
    src_file = 'Dataset/Fake/Taming_Transformer/Taming_Transformer.'+str(rn)+'.jpg'
    dst_dir = 'Datasets/validation/Fake'
    shutil.copy(src_file, dst_dir)