def helper_imshow_output_expected(img1, img2, title1, title2):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplot_mosaic([
        [title1, title2]
    ], figsize=(15, 10))

    ax[title1].imshow(img1)
    ax[title2].imshow(img2)

    plt.show()
