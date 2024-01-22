import matplotlib.pyplot as plt

def show_original_and_reconstructed_image(orig, dec, input_shape, num=10):
    n = num
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1) # display original
        plt.imshow(orig[i].reshape(input_shape))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, i +1 + n) # display reconstruction
        plt.imshow(dec[i].reshape(input_shape))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


