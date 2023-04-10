import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

patches_path = './final_visualization'


image_path = os.path.join(patches_path, 'score_final_visualization.png')
image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
image  = image / 100
# target = np.asarray([len(image),len(image[0]),1])
thumbnail = cv2.imread(os.path.join(patches_path,'0','thumbnail.png'))
final_path = os.path.join(patches_path,'final_mask.png')

fig, axes = plt.subplots(1, 1)
im = axes.imshow(image, alpha = 0.95 ,aspect='auto', cmap=plt.get_cmap('RdBu'))
cbar = fig.colorbar(im ,ax = axes,orientation = 'vertical')
plt.axis('off')
# plt.imshow(thumbnail)
axes.imshow(thumbnail,alpha = 0.5)
# plt.imsave(final_path,im)

plt.show()
print('a')