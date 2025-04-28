# ViT-2SPN
**Vision Transformer-Based Dual-Stream Self-Supervised Pretraining Networks for Retinal OCT Classification**

Mohammadreza Saraei [^1] ([Website](https://www.linkedin.com/in/mrsaraei/)), Dr. Igor Kozak ([Website](https://doctors.bannerhealth.com/provider/igor-kozak/2955460?utm_source=gmb&utm_medium=listing&utm_campaign=doc_onlinescheduling&y_source=1_MTA1NzIwNTgzMy03MTUtbG9jYXRpb24ud2Vic2l0ZQ%3D%3D)), Dr. Eung-Joo Lee ([Website](https://www.brunel.ac.uk/people/sebelan-danishvar](https://ece.engineering.arizona.edu/faculty-staff/faculty/eung-joo-lee)))

**Code [[GitHub](https://github.com/mrsaraei/ViT-2SPN/tree/main)] | Data [[MedMNISTv2](https://github.com/MedMNIST/MedMNIST), [OCTID](https://borealisdata.ca/dataverse/OCTID), [UCSD-OCT](https://data.mendeley.com/datasets/rscbjbr9sj/3)] | Preprint [[ArXiv](https://doi.org/10.48550/arXiv.2501.17260)] | IEEE TBME [َUnder Review]**

![SSP Approach](https://github.com/mrsaraei/ViT-2SPN/blob/4c5f159f72ea6734c440d6060c65685ede869233/figures/Fig_1.svg)

<p align='justify'>Optical Coherence Tomography (OCT) is a key non-invasive tool for diagnosing retinal diseases. However, the development of OCT-based diagnostic models faces challenges such as limited datasets, sparse annotations, and privacy concerns, hindering further progress despite advances in deep learning. To address these challenges, we propose the Vision Transformer-based Dual-Stream Self-Supervised Pretraining Network (ViT-2SPN)—a novel framework designed to enhance feature extraction and improve diagnostic performance. ViT-2SPN features a dual-stream architecture with feature concatenation and incorporates a self-supervised pretraining strategy. The framework operates in three stages: (1) Supervised Learning, (2) Self-Supervised Pretraining, and (3) Run Fine-Tuning. During the pretraining phase, unlabeled OCT images from the OCTMNIST dataset are augmented to generate dual views, enabling effective representation learning via a ViT-Tiny backbone and contrastive loss. Fine-tuning is subsequently performed on annotated subsets of the OCTMNIST, UCSD OCT, and OCTID datasets using cross-validation. Experimental results show that ViT-2SPN outperforms existing self-supervised learning baselines, demonstrating its robustness and clinical potential in retinal OCT classification. The model excels in identifying healthy cases and ruling out disease, though its slightly lower sensitivity suggests room for improvement in detecting all disease instances.</p>

## Datasets
![Sample Images](https://github.com/mrsaraei/ViT-2SPN/blob/801d963f9c28430f45a97978cf19eac90ac69812/figures/Fig_2.svg)
![Dataset Information](https://github.com/mrsaraei/ViT-2SPN/blob/3bb09ba05784bf58c39c9aaffe2c320b4dfae210/figures/Fig_2a.png)

## ViT-2SPN Architecture
![ViT-2SPN Architecture](https://github.com/mrsaraei/ViT-2SPN/blob/4c5f159f72ea6734c440d6060c65685ede869233/figures/Fig_3.svg)

## Experimental Setup
<p align='justify'>During the SSP phase, the model utilizes the unlabeled imbalanced OCTMNIST dataset, which consists of $\sim97k$ training samples. The training process is conducted with a mini-batch size of 128, a learning rate of 0.0001, and a momentum rate of 0.999, spanning 100 epochs. For this phase, the model employs the ViT-Tiny architecture, which has been pretrained on the ImageNet dataset. In the fine-tuning phase, the model takes advantage of 5k, 2k, and 0.5k labeled samples from the OCTMNIST, UCSD OCT, and OCTID datasets, respectively,  using a 10-fold cross-validation strategy. This method was chosen to promote a more stable and generalized learning process, maximizing the utility of the limited labeled data. This setup allows for a robust yet computationally feasible assessment of the model's generalization performance. The fine-tuning process utilizes a batch size of 128, maintains the same learning rate from the pretraining phase, incorporates a dropout rate of 0.5, and also spans 50 epochs.</p>

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
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/f80354f7ba0abb2054c4d79d8cfa860bf56dbe34/figures/Fig_6.png" width="100%"/>
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/f80354f7ba0abb2054c4d79d8cfa860bf56dbe34/figures/Fig_7.png" width="100%"/>
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/f80354f7ba0abb2054c4d79d8cfa860bf56dbe34/figures/Fig_8.png" width="100%"/>
</p>

## Conclusion
- Dual-Stream Network > Single-Stream Network (without increasing the FLOPs)
- ViT Backbones in Dual-Stream Network > CNN only or CNN + ViT Backbones in Dual-Stream Network 
- The Pretraining Strategy > Without Pretraining Strategy
- Self-Supervised Pretraining Strategy (small size data only) > Supervised Pretraining Strategy
- ViT-2SPN Model > Baseline Self-Supervised Learning Models
- <p align='justify'>The ViT-2SPN model demonstrated promising performance in classifying OCT images, achieving high specificity (over 90%) across all datasets. This emphasizes its capacity to accurately identify healthy individuals and reduce false positives, which is particularly beneficial for screening healthy patients and ensuring the correct identification of non-diseased cases. However, its sensitivity, especially on the OCTMNIST dataset, was relatively lower, suggesting that it was less effective at detecting diseased patients. In clinical settings, high sensitivity is essential to minimize false negatives and ensure that patients with conditions are not overlooked, which is critical for timely intervention and treatment.</p>

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
