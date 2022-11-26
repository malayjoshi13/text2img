def StableDiffusion(input_text):
    import time
    import math
    import keras_cv
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from PIL import Image 
    from tensorflow import keras
    from IPython.display import Image as IImage

    ## Defining functions

    def plot(images):
        """
        Plots every generated image on the screen.
        """
        plt.figure(figsize=(20, 20))

        for index, image in enumerate(images):
            ax = plt.subplot(1, len(images), index + 1)
            plt.imshow(image)
            plt.axis("off")

    def save(image, filename):
        """
        Saves the supplied image to the specified filename.
        """
        im = Image.fromarray(image)
        im.save(filename)

    ## Define the model
    model = keras_cv.models.StableDiffusion(
        img_width=512, 
        img_height=512
    )

    ## Input a **text prompt** and **generate images** corresponding to input text prompt
    images = model.text_to_image(
        input_text,
        batch_size=3
    )

    ## View and save the generated image
    plot(images)
    save(images[0], "/content/stablediff_output1.png")
    save(images[1], "/content/stablediff_output2.png")
    save(images[2], "/content/stablediff_output3.png")