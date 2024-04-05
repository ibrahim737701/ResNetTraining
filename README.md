# TSAIAssignment10

This assignment focuses on creating a ResNet model and achieve 90% accuracy. 

We have to apply transforms as well such as:-

Padding

Padding is a preprocessing technique used to alter the spatial size of an image. By adding extra pixels around the edges of an image, padding can help maintain the size of the output image after convolutional operations, or ensure that images in a batch have the same dimensions. 

Random Crop

Random cropping is a data augmentation technique where a random part of the image is selected and cropped. This allows the model to focus on different parts of an image during training, improving its ability to recognize patterns and features in various positions. It's a simple yet effective way to increase the diversity of the training data without needing more images.

Flip

Flipping is another augmentation technique where an image is mirrored along a specific axis. This can be a horizontal flip, where the image is mirrored left to right, or a vertical flip, where it's mirrored top to bottom.

Coarse Dropout

Coarse Dropout is a type of data augmentation that involves randomly removing regions of the image. This process introduces rectangular regions of zeros (or a specified value) into an image, forcing the model to rely on less information for making predictions. It's particularly effective for making models more robust to occlusion and encouraging them to pay attention to the entire object rather than just a small set of features.

We also use One cycle policy in it which helps us converge faster and a Learning rate finder which involves gradually increasing the learning rate over a number of iterations or mini-batches and recording the loss at each step.



