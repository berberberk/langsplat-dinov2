# DinoSplat • 3D Semantic Gaussian Splatting with DINOv2 🦕
***
*Group research project in cooperation with YSDA (Yandex School of Data Analysis) and Center for Cognitive Modeling of MIPT*
***
![изображение](https://github.com/user-attachments/assets/2804759d-7934-4a71-9f1a-c145e31ca7fa)


**Gaussian Splatting** is a cutting-edge rendering technique perfect for complex, detailed, and dynamic scenes. Unlike traditional methods, it delivers smooth visualization of volumetric effects, translucent objects, and realistic blur while maintaining high performance. 

**Semantic 3D Gaussian Splatting** is an algorithm for extracting semantic features in a constructed 3D scene for open-vocabulary segmentation or localization of 3D objects in the scene.

**Related Work:** *Qin et al. 2023 - LangSplat: 3D Language Gaussian Splatting (CVPR2024 Highlight)* [arxiv.org/abs/2312.16084]
![image](https://github.com/user-attachments/assets/76158e54-c835-494d-9062-5cc7edb6e5c6)

**Related Work:** *Barsellotti et al. 2024 - Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation* [arxiv.org/abs/2411.19331]
![image](https://github.com/user-attachments/assets/743ec006-7e85-427a-a8e2-0253765afb7f)

The aim of the research project was to optimize existing SoTA solutions in the field of semantic splatting.

To add semantics to gaussians, we used the features of the DINOv2 model, trained to make high-quality embeddings of images.

This made it possible to achieve a better semantic reconstruction of the scene compared to the approach used in the LangSplat research paper, the authors of which relied on the SAM segmentation model. We have replaced the most computationally demanding stage of the LangSplat pipeline with DINOv2 features, reducing the preprocessing time of each frame by more than 50 times.

The use of the approaches used in Talk2DINO's work, namely the mapping of DINO features with embeddings of text queries using CLIP, allowed us to achieve impressive results in the Natural Language Querying task with Gaussian Splatting.

In addition, we computationally optimized dimensionality reduction when implementing semantic embeddings in splatting, experimenting with autoencoder training and using PCA. Using such a simple approach as PCA made it possible to reduce the time of the last stage of embedding processing by ~10 times at the cost of less confident and dense semantics coverage. However, the rendering showed that this level is sufficient to solve a wide range of tasks in the field.

![image](https://github.com/user-attachments/assets/a7a18110-310a-4be1-9dbe-4326bd32c6d4)
