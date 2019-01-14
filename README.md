# Keras_TestTimeAugmentation
The purpose of this repository is to provide Test Time Augmentation abilities to Keras models for application in Image Classification tasks.Test time augmentation (TTA) wrapper for Keras Image Classification models. This makes prediction for one image at a time. This can be easily integrated in a loop for making predictions for a sequence of images. 

Use of Test Time Augmentation has been shown to improve classification accuracy in many cases, however, there maybe instances where it can produce worse results (as compared to predictions without test time augmentation). 

The current implementation does a fixed set of transformations as follows:
        1) Flipped left to right
        2) Flipped  upside down
        3) Rotate 30 degrees
        4) Rotate 45 degrees
        5) Add gaussian blur (parameters for this are fixed in the current version, will be made dynamic in the future versions)
        6) Preserving edges (parameters for this are fixed in the current version, will be made dynamic in the future versions)

""" 
    Example:
        tta_mod = Keras_TTA(model,use_origimg=True,fliplr=True,flipud =True,rotate30=True,rotate45=True,gaussian_blur=True,preserve_edge=True)
        predictions = tta_mod.predict(image)
"""
Future versions of this implementation will have options of weighted average of predictions with options of specifying specific weights for each type of augmentation.
