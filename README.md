# Keras_TestTimeAugmentation


The purpose of this repository is to provide Test Time Augmentation abilities to Keras models for application in Image Classification tasks.
""" Test time augmentation (TTA) wrapper for Keras Image Classification models. This makes prediction for one image at
        a time. This can be easily integrated in a loop for making predictions for a sequence of images
     Args:
        model(Keras model): Needs a Keras fitted model with 'predict' method
        use_origimg: Set to 'True' if you want the predictions of original image in the TTA calculation
        fliplr     : Set to 'True' if you want the prediction of left-to-right flipped version of original image in TTA calculation
        flipud     : Set to 'True' if you want the prediction of upside down flipped version of original image in TTA calculation
        rotate     : Provide degrees e.g. 30 for which the image needs to be rotated in TTA calculation
        gaussian_blur: Set to 'True' if you want the prediction of gaussian blur version of original image in TTA calculation
        preserve_edge: Set to 'True' if you want the prediction of edge preserved version of original image in TTA calculation
        



""" 
    Example:
        tta_mod = Keras_TTA(model,use_origimg=True,fliplr=True,flipud =True,rotate=30,gaussian_blur=True,preserve_edge=True)
        predictions = tta_mod.predict(image)
"""
Future versions of this implementation will have options of weighted average of predictions with options of specifying specific weights for each type of augmentation.
