import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2

plt.rcParams['image.interpolation'] =  None



def load_images():
    image_dir = 'imagens_pgm'
    input_images_names = sorted(os.listdir(image_dir))
    try:
        input_images = [cv2.imread(os.path.join(image_dir,name),0)\
                                   for name in input_images_names]
    except ValueError as e:
        imageio.plugins.freeimage.download()
        input_images = [imageio.imread(os.path.join(image_dir,name))\
                                   for name in input_images_names]
    return input_images, input_images_names

def load_filters():
    h1 = np.array([[0, 0, -1, 0, 0],
                   [0, -1, -2, -1, 0],
                   [-1, -2, 16, -2, -1],
                   [0, -1, -2, -1, 0],
                   [0, 0, -1, 0, 0]])

    h2 = np.array([[1 ,4 ,6 ,4, 1],
                   [4 ,16 ,24 ,16, 4],
                   [6 ,24 ,36 ,24, 6],
                   [4 ,16 ,24 ,16, 4],
                   [1 ,4 ,6 ,4, 1]])
    h2 = (1./256.)*h2

    h3 = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    h4 = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
    return(h1,h2,h3,h4)


def plot_image(image, figsize=(3,3),grid='on',axis='on',color='gray',
               title='',savename=None,save_fig=True):
    """Plot a image and shows some statistics
    """
    plt.figure(figsize=figsize)
    plt.axis(axis)
    plt.title(title)
    plt.imshow(image, cmap=color)
    if savename is not None:
        if save_fig:
            plt.savefig(savename,bbox_inches='tight', transparent="True", pad_inches=0)
        else:
            plt.imsave(savename, image, cmap='gray')
        print('Saved in:' + savename)
    plt.show()
    

        
    print('Min:', image.min())
    print('Max:', image.max())
    print('Mean:', image.mean())

def plot_images(images, titles=None, resize=10,
                color='gray', axis='off',
                savename=None, num_columns=4,save_fig=True):
    num_images = len(images)
    num_lines = num_images//num_columns + bool(num_images%num_columns)
    fig = plt.figure(figsize=(num_lines*resize, num_columns*resize))
    
    for i in range(1, num_images + 1):
        image = images[i-1]
        fig.add_subplot(num_lines, num_columns, i)
        if titles is not None:
            plt.title(titles[i-1])
        plt.axis(axis)
        plt.imshow(image,cmap=color)
    if savename is not None:
        if save_fig:
            plt.savefig(savename,bbox_inches='tight')
        else:
            plt.imsave(savename, data)
        print('Image saved in: ', savename)
    plt.show()

    

def linear_rescale(image):
    max_intensity = image.max()
    min_intensity = image.min()
    denominator = max_intensity - min_intensity
    if denominator == 0:
        denominator = max_intensity
    image = (image - min_intensity)/(denominator)
    image = image * 255
    return image