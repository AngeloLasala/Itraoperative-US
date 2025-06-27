"""
Qualitative visual evaluation for assess 'copy and paste' overfitting.
"""
import os
import cv2
import argparse
import matplotlib.pyplot as plt
from PIL import Image   
import argparse
import numpy as np

def traslation(image, tx, ty):
    """
    Function that traslate the image in the x and y direction
    """
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(np.array(image), translation_matrix, (image.width, image.height))
    return Image.fromarray(translated_image)

if __name__ == '__main__':

    ## subject = {id:13, slide: 215} -> mask: 204 - tx: 0, ty: -10
    ## subject = {id:2, slide: 140} -> mask: 141 - tx: -20, ty: 0
    ## subject = {id:1, slide: 158} -> mask: 336 - tx: 0, ty: 0
    ## subject = {id:4, slide: 168} -> mask: 195 - tx: 10, ty: 10
    ## subject = {id:17, slide: 175} -> mask: 161 - tx: 0, ty: 0
    
    # path or generated data
    gen_path = "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/one_step/Stack_finetuning/split_1/ldm_finetuning/w_-1.0/ddpm/samples_ep_8000"
    gen_id = 161
    ius_gen = os.path.join(gen_path, 'ius', f'x0_{gen_id}.png')
    mask_gen = os.path.join(gen_path, 'masks', f'mask_{gen_id}.png')

    # read generated image
    ius_gen = Image.open(ius_gen)
    mask_gen = Image.open(mask_gen)
    ius_gen = ius_gen.resize((256, 256))
    mask_gen = mask_gen.resize((256, 256), Image.NEAREST)  # Use nearest neighbor for masks
    
    # apply traslation to the generated image
    ius_gen = traslation(ius_gen, 0, 0)
    mask_gen = traslation(mask_gen, 0, 0)

    # real data path
    real_path = "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/RESECT_iUS_dataset/dataset"
    subject = 17
    slide = 175

    ius_real = os.path.join(real_path, f'Case{subject}-US-before', 'volume', f'Case{subject}-US-before_{slide}.png')
    mask = os.path.join(real_path, f'Case{subject}-US-before', 'tumor', f'Case{subject}-US-before_{slide}.png')
   
   
    # read image
    ius_real = Image.open(ius_real)
    mask = Image.open(mask)

    ius_real = ius_real.resize((256, 256))
    mask = mask.resize((256, 256), Image.NEAREST)  # Use nearest neighbor for masks

    # make a trashold of 0.5 fo mask value
    # ius_real = ius_real.convert("L")  # Convert to grayscale
    # mask = mask.convert("L")  # Convert to grayscale

    fig, ax = plt.subplots(2, 3, figsize=(18, 10), tight_layout=True)
    mask = np.array(mask)/255
    mask_gen = np.array(mask_gen)/255
    ax[0, 0].imshow(mask, cmap='gray')
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Mask')
    ax[0, 1].imshow(mask_gen, cmap='gray')
    ax[0, 1].axis('off')
    ax[0, 1].set_title('Generated Mask')
    ax[0, 2].axis('off')
    ax[0, 2].set_title(f'SD maps')
    ax[0, 2].imshow(mask*0.5 + np.abs(mask_gen - mask)*0.8, cmap='bone')
    ## add the cmap colorbar
    cbar = plt.colorbar(ax[0, 2].images[0], ax=ax[0, 2], orientation='vertical')
    cbar.ax.tick_params(labelsize=20)
    

    ius_real = np.array(ius_real)/255
    ius_gen = np.array(ius_gen)/255
    ax[1, 0].imshow(ius_real, cmap='gray')
    ax[1, 0].axis('off')
    ax[1, 0].set_title('Real Image')
    ax[1, 1].imshow(ius_gen, cmap='gray')
    ax[1, 1].axis('off')
    ax[1, 1].set_title('Generated Image')
    ax[1, 2].axis('off')
    ax[1, 2].set_title(f'SD images')

    ax[1, 2].imshow(np.abs(ius_gen - ius_real), cmap='bone')
    ## add the cmap colorbar
    cbar = plt.colorbar(ax[1, 2].images[0], ax=ax[1, 2], orientation='vertical')
    cbar.set_label('', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=20)
    


    plt.show()