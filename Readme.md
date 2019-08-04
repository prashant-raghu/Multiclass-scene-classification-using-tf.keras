# Multiclass-scene-classification-using-tf.keras


The goal is to classify around 25k images of size 150x150 distributed under 6 categories.  
namely {'buildings' -> 0, 'forest' -> 1, 'glacier' -> 2, 'mountain' -> 3, 'sea' -> 4, 'street' -> 5 } using  
1)CNN \
2)Transfer Learning \
DataSource: [Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification)

## Usage

```bash
python customCNN.py
```
Or
```bash
python transferLearning.py
```
### Dataset
![Dataset](https://i.ibb.co/cbvfwkZ/Dataset.png)
### Summary of custom CNN Model.
![CNN](https://i.ibb.co/Qmt0mbV/CNNSummary.png)
### Accuracy on training and validation Data.
![CNN](https://i.ibb.co/hdkTvNV/CNNAccuracy.png)
### Summary of Model pre-trained on imagenet using Transfer Learning.
![TL](https://i.ibb.co/4d9Cb43/TLSummary.png)
### Accuracy on training and validation Data.
![TL](https://i.ibb.co/2PZCpsJ/TF1.png)