from keras import models
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


def get_sample_image():
    img = image.load_img(img_path,target_size=(150,150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor,axis=0)
    img_tensor /= 255.          # r,g,b 값을 0~245 -> 0~1 사이로 조정(값을작게)

    print(img_tensor.shape)

    # plt.imshow(img_tensor[0])
    # plt.show()
    return img_tensor


img_path = './datasets/cats_and_dogs_small/test/cats/cat.1700.jpg'
model = load_model('./predict/5.2-using-convnets-with-small-datasets_learning/cats_and_dogs_small_1.h5')
model.summary()  # 기억을 되살리기 위해서 모델 구조를 출력합니다


img_tensor = get_sample_image()
layer_outputs = [layout.output for layout in model.layers[:8]]
activation_model = models.Model(inputs=model.input,outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0,:,:,19], cmap='viridis')
plt.matshow(first_layer_activation[0,:,:,15], cmap='viridis')
plt.show()

