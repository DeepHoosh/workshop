# <p dir='rtl' align='right'>معرفی کتاب خانه ویژن ترنسفورمر</p> 
<p dir='rtl' align='right' style="text-align: justify;">
در این کتاب خانه کد پیاده سازی مدل هایی که در <a href="https://deephoosh.com/courses/%da%a9%d8%a7%d8%b1%da%af%d8%a7%d9%87-computer-vision/ "> کارگاه ویژن ترنسفورمر </a> در دیپ هوش معرفی شدند قرار دارد. این کتاب خانه یکی از کامل ترین کتاب خانه های فارسی مربوط به ویژن ترنسفورمر ها می باشد که مدل های مهمی که در سال های اخیر معرفی شده اند در این کتاب خانه موجود می باشند. در حال حاضر تمامی مدل ها با استفاده از کتاب خانه تنسرفلو نوشته شده اند اما در آینده ای نسخه پایتورچ این مدل ها نیز در اختیار شما عزیزان قرار خواهد گرفت 
</p>

<!-- <div style="display: flex;
 				 justify-content: center;
  				align-items: center;">
<img src="https://drive.google.com/uc?export=view&id=1kW7rNzishKlPpaSGcnEYXwHKz8TcI93o" width="300px" height="300px">
</div> -->

# <p dir='rtl' align='right'>فهرست مدل ها</p> 

- ###### Vision Transformer (An Image is worth 16 x 16 words)
- ###### Pyramid Vision Transformer
- ###### Pyramid Vision Transforemr V2
- ###### Convolutional Vision Transformer
- ###### DeiT (Training Data Efficient Image Transforemrs)

#  <p dir='rtl' align='right'>نصب کتاب خانه های مورد نیاز برای اجرا</p> 

```bash
pip install tensorflow
```

#  <p dir='rtl' align='right'>استفاده از Vision Transformer</p> 
                
```python
from vit import ViT
import tensorflow as tf

vitClassifier = ViT(
                    num_classes=1000,
                    patch_size=16,
                    num_of_patches=(224//16)**2,
                    d_model=128,
                    heads=2,
                    num_layers=4,
                    mlp_rate=2,
                    dropout_rate=0.1
)

#استفاده از مدل
sampleInput = tf.random.normal(shape=(1 , 224 , 224 , 3))
output = vitClassifier(sampleInput , training=False)
print(output.shape) # (1 , 1000)

#تعلیم مدل
vitClassifier.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=[
                       tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                       tf.keras.metrics.TopKCategoricalAccuracy(k=5 , name="top_5_accuracy"),
              ])

vitClassifier.fit(
trainingData, #داده تعلیم به شکل دیتاست تنسرفلو
validation_data=valData, #داده تست به شکل دیتاست تنسرفلو
epochs=100,)
```

#  <p dir='rtl' align='right'>استفاده از Convolutional Vision Transformer</p> 
                
```python
from cvt import CvT , CvTStage
import tensorflow as tf

cvtModel = CvT(1000 , [
                      CvTStage(projectionDim=64, 
                               heads=1, 
                               embeddingWindowSize=(7 , 7), 
                               embeddingStrides=(4 , 4), 
                               layers=1,
                               projectionWindowSize=(3 , 3), 
                               projectionStrides=(2 , 2), 
                               ffnRate=4),
                      CvTStage(projectionDim=192,
                               heads=3,
                               embeddingWindowSize=(3 , 3), 
                               embeddingStrides=(2 , 2),
                               layers=1, 
                               projectionWindowSize=(3 , 3), 
                               projectionStrides=(2 , 2), 
                               ffnRate=4),
                      CvTStage(projectionDim=384,
                               heads=6,
                               embeddingWindowSize=(3 , 3),
                               embeddingStrides=(2 , 2),
                               layers=1,
                               projectionWindowSize=(3 , 3),
                               projectionStrides=(2 , 2), 
                               ffnRate=4)
])

#استفاده از مدل
sampleInput = tf.random.normal(shape=(1 , 224 , 224 , 3))
output = cvtModel(sampleInput , training=False)
print(output.shape) # (1 , 1000)

#تعلیم مدل
cvtModel.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=[
                       tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                       tf.keras.metrics.TopKCategoricalAccuracy(k=5 , name="top_5_accuracy"),
              ])

cvtModel.fit(
trainingData, #داده تعلیم به شکل دیتاست تنسرفلو
validation_data=valData, #داده تست به شکل دیتاست تنسرفلو
epochs=100,)

```
