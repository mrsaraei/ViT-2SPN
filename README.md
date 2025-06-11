# ViT-2SPN
**Vision Transformer-Based Dual-Stream Self-Supervised Pretraining Networks for Retinal OCT Classification**

Mohammadreza Saraei [^1] ([Website](https://www.linkedin.com/in/mrsaraei/)), Dr. Igor Kozak ([Website](https://doctors.bannerhealth.com/provider/igor-kozak/2955460?utm_source=gmb&utm_medium=listing&utm_campaign=doc_onlinescheduling&y_source=1_MTA1NzIwNTgzMy03MTUtbG9jYXRpb24ud2Vic2l0ZQ%3D%3D)), Dr. EungJoo Lee ([Website](https://www.brunel.ac.uk/people/sebelan-danishvar](https://ece.engineering.arizona.edu/faculty-staff/faculty/eung-joo-lee)))

**Code [[GitHub](https://github.com/mrsaraei/ViT-2SPN/tree/main)] | Data [[MedMNISTv2](https://github.com/MedMNIST/MedMNIST), [OCTID](https://borealisdata.ca/dataverse/OCTID), [UCSD-OCT](https://data.mendeley.com/datasets/rscbjbr9sj/3)] | [َUnder Review]**

![SSP Approach](https://github.com/mrsaraei/ViT-2SPN/blob/4c5f159f72ea6734c440d6060c65685ede869233/figures/Fig_1.svg)

<p align='justify'>Optical Coherence Tomography (OCT) is a vital imaging modality for diagnosing retinal diseases. While deep learning models, such as convolutional neural networks and vision transformers, have improved OCT-based classification, challenges remain due to the limited availability of datasets, sparse annotations, and privacy concerns. To address these issues, we propose ViT-2SPN—a Vision Transformer-based Dual-Stream Self-Supervised Pretraining Network that enhances feature extraction and classification performance. ViT-2SPN employs a dual-stream architecture with feature concatenation and a contrastive self-supervised learning strategy. The training pipeline includes: (1) supervised ImageNet-1K pretraining, (2) self-supervised pretraining on unlabeled OCTMNIST data with augmented dual views, and (3) fine-tuning on labeled subsets of OCTMNIST, UCSD OCT, and OCTID with cross-validation. Our proposed model, which has 11.68 million parameters and 2.16 billion FLOPs, shows superior performance to self-supervised learning baseline models across various datasets. Specifically, it achieves a mAUC/accuracy of 0.884/0.71 on OCTMNIST, 0.941/0.84 on OCTID, and 0.959/0.86 on UCSD OCT. These results highlight the generalizable and clinical applicability of ViT-2SPN for retinal OCT image analysis. </p>

## Datasets
![Sample Images](https://github.com/mrsaraei/ViT-2SPN/blob/401a5ca26e4f823ba45f4bd9c641420f9a4fe2fd/figures/Fig_2.png)
![Dataset Information](https://github.com/mrsaraei/ViT-2SPN/blob/3bb09ba05784bf58c39c9aaffe2c320b4dfae210/figures/Fig_2a.png)

## ViT-2SPN Architecture
![ViT-2SPN Architecture](https://github.com/mrsaraei/ViT-2SPN/blob/4c5f159f72ea6734c440d6060c65685ede869233/figures/Fig_3.svg)

## Experimental Setup
<p align='justify'>During the SSP phase, the ViT-Tiny model pretrained on ImageNet-1K is trained for 100 epochs on 97K unlabeled, imbalanced OCTMNIST samples using a batch size of 128, a learning rate of 0.0001, and a momentum of 0.999, across six NVIDIA RTX™ 6000 GPUs with CUDA 12.4. In the fine-tuning phase, the model is trained using 5K, 2K, and 0.5K labeled samples from the OCTMNIST, UCSD OCT, and OCTID datasets, respectively, employing a 10-fold cross-validation strategy. This approach enhances stability and generalization while making efficient use of limited labeled data and ensuring a computationally feasible evaluation. Fine-tuning uses the same learning rate, a batch size of 128, a dropout rate of 0.5, and runs for 50 epochs.</p>

## Loss Function
<p align="center">
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/c7acfdbb5f02c4ab89cde6dd91720b43fb57d9fb/figures/Fig_4.png" width="50%"/>
</p>

## Results
<p align="center">
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/cc7ab8c2b0bf45c53b9bd62df1b1776307c749cb/figures/Fig_5a.png" width="45%" style="display: inline-block;"/>
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/cc7ab8c2b0bf45c53b9bd62df1b1776307c749cb/figures/Fig_5b.png" width="45%" style="display: inline-block;"/>
</p>

<p align="center">
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/5cfa06e52a89e785603ccdc704140054b6f8ce7d/figures/Fig_6.png" width="100%"/>
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/5cfa06e52a89e785603ccdc704140054b6f8ce7d/figures/Fig_7.png" width="100%"/>
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/5cfa06e52a89e785603ccdc704140054b6f8ce7d/figures/Fig_8.png" width="100%"/>
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/5cfa06e52a89e785603ccdc704140054b6f8ce7d/figures/Fig_9.png" width="100%"/>
</p>

## Conclusion
+ :white_check_mark: Dual-Stream Network > Single-Stream Network (without increasing the FLOPs)
+ :white_check_mark: ViT Backbones in Dual-Stream Network > CNN only or CNN + ViT Backbones in Dual-Stream Network 
+ :white_check_mark: The Pretraining Strategy > Without Pretraining Strategy
+ :white_check_mark: Self-Supervised Pretraining Strategy (small size data only) > Supervised Pretraining Strategy
+ :white_check_mark: ViT-2SPN Model > Baseline Self-Supervised Learning Models
> **Eventually:** <p align='justify'>ViT-2SPN model consistently performed well across five retraining runs, achieving a high specificity of over 0.8965 with a standard deviation (SD) of 0.0051 across all imbalanced datasets. This underscores its effectiveness in accurately identifying healthy individuals and minimizing false positives, which is advantageous for screening and excluding non-diseased cases. However, its sensitivity, particularly on the OCTMNIST dataset, was relatively lower, suggesting reduced effectiveness in detecting diseased patients. In clinical contexts, high sensitivity is crucial to avoid false negatives and ensure timely diagnosis and treatment.</p>

## Presentation (Please click the cover to view full slides.]
[![Presentation Preview](https://github.com/mrsaraei/ViT-2SPN/blob/a8538c85887ad181197d7041b718402bcab31eb7/presentation/Fig_9.png)](https://github.com/mrsaraei/ViT-2SPN/blob/0de70d45ac6fad31788bd8e045719c42da744890/presentation/Presentation.pdf)

## Citation (BibTeX)

```
*@article{saraei2025vit,
  title={ViT-2SPN: Vision Transformer-based Dual-Stream Self-Supervised Pretraining Networks for Retinal OCT Classification},
  author={Saraei, Mohammadreza and Kozak, Igor and Lee, Eung-Joo},
  journal={arXiv preprint arXiv:2501.17260},
  year={2025}
}*
```

[^1]: Please feel free to if you have any questions: mrsaraei@arizona.edu 
