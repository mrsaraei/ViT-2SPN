# ViT-2SPN
**Vision Transformer-Based Dual-Stream Self-Supervised Pretraining Networks for Retinal OCT Classification**

Mohammadreza Saraei [^1], Dr. Igor Kozak ([Website](https://doctors.bannerhealth.com/provider/igor-kozak/2955460?utm_source=gmb&utm_medium=listing&utm_campaign=doc_onlinescheduling&y_source=1_MTA1NzIwNTgzMy03MTUtbG9jYXRpb24ud2Vic2l0ZQ%3D%3D)), Dr. Eung-Joo Lee ([Website](https://www.brunel.ac.uk/people/sebelan-danishvar](https://ece.engineering.arizona.edu/faculty-staff/faculty/eung-joo-lee)))

**Code [[GitHub](https://github.com/mrsaraei/ViT-2SPN/tree/main)] | Data [[MedMNISTv2](https://github.com/MedMNIST/MedMNIST)] | Preprint [[ArXiv](https://doi.org/10.48550/arXiv.2501.17260)] | Publication [َUnder Review in [MIDL 2025](https://openreview.net/forum?id=2dJkEBWyOE#discussion)]**

![SSP Approach](https://github.com/mrsaraei/ViT-2SPN/blob/4c5f159f72ea6734c440d6060c65685ede869233/figures/Fig_1.svg)

<p align='justify'>Optical Coherence Tomography (OCT) is a non-invasive imaging modality essential for diagnosing various eye diseases. Despite its clinical significance, the development of OCT-based diagnostic tools faces challenges such as limited public datasets, sparse annotations, and privacy concerns. Although deep learning has advanced OCT analysis, these challenges remain unresolved. To address these limitations, we introduce the Vision Transformer-based Dual-Stream Self-Supervised Pretraining Network (ViT-2SPN), a novel framework featuring a dual-stream network, feature concatenation, and a pretraining mechanism designed to enhance feature extraction and improve diagnostic accuracy. ViT-2SPN employs a three-stage workflow: Supervised Learning, Self-Supervised Pretraining, and Supervised Fine-Tuning. The pretraining phase leverages the unlabeled OCTMNIST dataset with data augmentation to create dual-augmented views, enabling effective feature learning through a ViT backbone and contrastive loss. Fine-tuning is then performed on a limited-annotated subset of OCTMNIST using cross-validation. ViT-2SPN-T achieves a mean AUC of 0.936, accuracy of 0.80, precision of 0.81, recall of 0.80, and an F1-Score of 0.79, outperforming baseline self-supervised learning-based methods. These results highlight the robustness and clinical potential of ViT-2SPN in retinal OCT classification.</p>

## Image Samples (Class: Normal, Drusen, DME, CNV)
![ViT-2SPN Architecture](https://github.com/mrsaraei/ViT-2SPN/blob/4c5f159f72ea6734c440d6060c65685ede869233/figures/Fig_2.svg)

## ViT-2SPN Architecture
![ViT-2SPN Architecture](https://github.com/mrsaraei/ViT-2SPN/blob/4c5f159f72ea6734c440d6060c65685ede869233/figures/Fig_3.svg)

## Experimental Setup
During the self-supervised pretraining phase, the model utilizes the unlabeled OCTMNIST dataset, which consists of 97k training samples. The training process is conducted with a mini-batch size of 128, a learning rate of 0.0001, and a momentum rate of 0.999, spanning a total of 50 epochs. For this phase, the model employs the ViT architecture, which has been pretrained on the ImageNet dataset. In the fine-tuning phase, the model takes advantage of 5k labeled samples from the OCTMNIST dataset, using a 10-fold cross-validation strategy. This method was chosen to promote a more stable and generalized learning process, maximizing the utility of the limited labeled data. Each fold consists of 4.5k training samples and 0.5k validation samples, with an additional 0.5k samples reserved for testing. The decision to reserve 0.5k samples for testing was made to ensure consistency across folds while keeping the test set independent. This setup allows for a robust yet computationally feasible assessment of the model's generalization performance. The fine-tuning process utilizes a batch size of 16, maintains the same learning rate from the pretraining phase, incorporates a dropout rate of 0.5, and also spans 50 epochs
![Experimental Setup](https://github.com/mrsaraei/ViT-2SPN/blob/4f4d4725f7e7c6a62bebd39932dbc73c1de8adb4/figures/Fig_4.svg)

## Result
<p align="center">
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/aff60c412627e0b71dde9c75dc1a807da78107bf/figures/Fig_5.svg" width="100%"/>
</p>

## Performance and Efficiency Assessment on Imbalanced OCTMNIST Dataset
<p align="center">
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/6f3d1526fcdeade4abfe18a247115725e0180f7c/figures/Fig_6.svg" width="100%"/>
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/888605b17530a8ea677e622c0db5ebaed480256b/figures/Fig_7.svg" width="100%"/>
</p>

## Performance and Efficiency Assessment on Imbalanced OCT2017v2 Dataset
![Performance Improvement](https://github.com/mrsaraei/ViT-2SPN/blob/3a74e581af88804c2c761ee94216173308dd823e/figures/Fig_8.svg)

## Command
- **ssp_vit2spn.py:** Trains the self-supervised model using unlabeled images to extract meaningful features.  
- **finetune_vit2spn.py:** Fine-tunes the pretrained model for classification tasks using labeled data.

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
