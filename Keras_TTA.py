
# coding: utf-8

# In[ ]:


import numpy as np
from scipy import ndimage
class Keras_TTA():
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
        
    Example:
        tta_mod = Keras_TTA(model,use_origimg=True,fliplr=True,flipud =True,rotate30=True,rotate45=True,gaussian_blur=True,preserve_edge=True)
        predictions = tta_mod.predict(image)
    """
    def __init__(self,model,use_origimg=False,conv_greyscale=False,fliplr=False,flipud =False,rotate=None,gaussian_blur=False,preserve_edge=False):
        self.model=model
        self.use_origimg = use_origimg
        self.fliplr = fliplr
        self.flipud = flipud
        self.gaussian_blur = gaussian_blur
        self.preserve_edge = preserve_edge
    
        
    def predict(self,X):
        predctr = 0
        arrlist=[]
        if self.use_origimg ==True:
            predctr+=1.0
            score1 = model.predict(X)
            arrlist.append(score1)
        if self.fliplr ==True:
            predctr+=1.0
            img2 = np.fliplr(X)
            score2 = model.predict(img2)
            arrlist.append(score2)
        if self.flipud ==True:
            predctr+=1.0
            img3 = np.flipud(X)
            score3 = model.predict(img3)
            arrlist.append(score3)
        if self.rotate != None:
            rot = self.rotate
            predctr+=1.0
            img4 = ndimage.rotate(X,rot,reshape=False)
            score4= model.predict(img4)
            arrlist.append(score4)
        
        if self.gaussian_blur==True:
            predctr +=1.0
            img5 = ndimage.gaussian_filter(X,sigma=3)
            score5 = model.predict(img5)
            arrlist.append(score5)
        if self.preserve_edge==True:
            predctr+=1.0
            img6 = ndimage.median_filter(X,3)
            score6 = model.predict(img6)
            arrlist.append(score6)
        
        fin_arr = np.array(arrlist)
        score = fin_arr.sum(0)/predctr
        return score

