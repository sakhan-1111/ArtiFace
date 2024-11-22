# ArtiFace: A dataset for deepfake detection.

**Paper:** 

* IEEE Xplore: https://ieeexplore.ieee.org/document/10754706

**Abstract:** Deepfake technology’s rise has led to a surge in false identities, creating a significant and present problem with broad societal ramifications. Concerns over identity theft, harassment, and the dissemination of false information have escalated due to the simplicity with which deepfaked facial images can now be produced and distributed thanks to the broad availability of generative AI tools like Generative Adversarial Networks (GANs). The availability of these tools has political ramifications since it can degrade public opinion and damage institutional trust. As such, the ability to identify deepfake face images has become essential. Ensuring a person’s identity is critical in preventing the dissemination of false information on social media. Detection of deepfake facial images is also necessary for identity verification in border control, law enforcement, and security applications. To effectively and precisely recognize deepfake face images, this study effort has focused on modifying transfer learning models, such as ResNet101V2, MobileNetV2, NASNetLarge, NASNetMobile, DenseNet121, DenseNet169, DenseNet201, and Xception.

**Update:**

* 2024-09-03 The paper has been accepted to IEEE UEMCON 2024
* 2024-08-26 Dataset is available on Kaggle.

**Dataset Description:**

* Total number of images: 106,650
* Number of real images: 53,368
* Number of fake images: 53,282
* Sources of fake images: ProjectedGAN, StarGAN, Stable Diffusion, and Taming Transformer
* Sources of real images: CIPS and FFHQ
* Image Resolution: 200 x 200
* Image format: jpg

**Dataset Folder Structure:**

![2.png](/Images/2.png)

**Samples of Fake Images:**

![1.png](/Images/1.png)

Fig. 1. Example of deepfaked faces using (a) ProjectedGAN, (b) Stable
Diffusion, (c) StarGAN, and (d) Taming Transformer.

**Dataset Download:**

The dataset is hosted on Kaggle. The dataset can be downloaded using the link below:

[ArtiFace Dataset](https://kaggle.com/datasets/8b11f39f58933fbef4f4d0c8c3c5bcb43d7ea09269d2891ca3bc2d834c6b165c)

**How to use:**

Extract the dataset using the command below:

```shell
tar -xvf ArtiFace.tar.xz
```

**Citation:**

TBP

**License:**

The licenses assosicated with the sources of the ArtiFace dataset is listed below:

<details close>
<summary>Data License</summary>

| Method             | License                               |
|:------------------:|:-------------------------------------:|
| ProjectedGAN       | MIT                                   |
| StarGAN            | MIT                                   |
| Stable Diffusion   | Apache-2.0                            |
| Taming Transformer | MIT                                   |
| CIPS               | MIT                                   |
| FFHQ               | Creative Commons BY-NC-SA 4.0 license |

</details>

**Acknowledgment:**

* The authors would like to express their gratitude to the authors of [ArtiFact](https://github.com/awsaf49/artifact) dataset as this dataset is a subset of Artifact.

* The authors also would like to express their gratitude to the authors of the methods that is used for creating ArtiFace dataset.
  
  <details close>
    <summary>Data Method Reference</summary>
  
  | Method             | Reference                                                 |
  |:------------------:|:---------------------------------------------------------:|
  | FFHQ               | [link](https://github.com/NVlabs/ffhq-dataset)            |
  | Taming Transformer | [link](https://github.com/CompVis/taming-transformer)     |
  | Stable Diffusion   | [link](https://github.com/huggingface/diffusers)          |
  | CIPS               | [link](https://github.com/saic-mdal/CIPS)                 |
  | StarGAN            | [link](https://github.com/yunjey/StarGAN)                 |
  | ProjectedGAN       | [link](https://github.com/autonomousvision/projected_gan) |
  
  </details>
