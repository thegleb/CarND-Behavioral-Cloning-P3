import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Lambda, Cropping2D, Conv2D, Dropout, Dense, Flatten
from keras.models import Model
from keras.preprocessing import image

model = Sequential([
    Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)),
    Cropping2D(cropping=((60, 25), (0, 0))),
    Conv2D(filters=24, kernel_size=(5, 5), activation='relu', strides=(2, 2)),
    Dropout(0.3),
    Conv2D(filters=36, kernel_size=(5, 5), activation='relu', strides=(2, 2)),
    Dropout(0.3),
    Conv2D(filters=48, kernel_size=(5, 5), activation='relu', strides=(2, 2)),
    Dropout(0.3),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    Dropout(0.3),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    Flatten(),
    Dense(100),
    Dense(50),
    Dense(10),
    Dense(1)
])

model.load_weights('model.h5')

layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input

img_path = 'training-data-old/track-1-and-2/IMG/center_2019_07_05_01_19_54_857.jpg'

img = image.load_img(img_path)
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# img_tensor /= 255.

# plt.imshow(img_tensor[0])
# plt.show()
#
# print(img_tensor.shape)

layer_names = []
for layer in model.layers:
    layer_names.append(layer.name)

print(model)
images_per_row = 16

activations = activation_model.predict(img_tensor)

for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
    n_features = layer_activation.shape[-1]  # Number of features in the feature map
    print(n_features)
    shape = (layer_activation.shape[1], layer_activation.shape[2])  # The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
    display_grid = np.zeros((n_cols * shape[0], images_per_row * shape[1]))
    print(display_grid)
    print('images_per_row', images_per_row)
    print('n_cols', n_cols)
    for col in range(n_cols):  # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[
                col * shape[0]: (col + 1) * shape[0],
                row * shape[1]: (row + 1) * shape[1]
            ] = channel_image
    scale = 1. / shape[0]
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
