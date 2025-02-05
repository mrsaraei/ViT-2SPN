# ViT-2SPN
**Vision Transformer-based Dual-Stream Self-Supervised Pretraining Networks for Retinal OCT Classification**

Mohammadreza Saraei [^1], Dr. Igor Kozak ([Website](https://doctors.bannerhealth.com/provider/igor-kozak/2955460?utm_source=gmb&utm_medium=listing&utm_campaign=doc_onlinescheduling&y_source=1_MTA1NzIwNTgzMy03MTUtbG9jYXRpb24ud2Vic2l0ZQ%3D%3D)), Dr. Eung-Joo Lee ([Website](https://www.brunel.ac.uk/people/sebelan-danishvar](https://ece.engineering.arizona.edu/faculty-staff/faculty/eung-joo-lee)))

**Code [[GitHub](https://github.com/mrsaraei/ViT-2SPN/tree/main)] | Data [[MedMNISTv2](https://github.com/MedMNIST/MedMNIST)] | Preprint [[ArXiv](https://doi.org/10.48550/arXiv.2501.17260)] | Publication [[Under Review in MIDL 2025](https://openreview.net/forum?id=2dJkEBWyOE&noteId=2dJkEBWyOE)]**

![SSP Approach](https://github.com/mrsaraei/ViT-2SPN/blob/536d76cd13d13e683cb47f23da126d67d88b7928/figures/Fig_1-1.svg)

<p align='justify'>Optical Coherence Tomography (OCT) is a non-invasive imaging modality essential for diagnosing various eye diseases. Despite its clinical significance, developing OCT-based diagnostic tools faces challenges, such as limited public datasets, sparse annotations, and privacy concerns. Although deep learning has made progress in automating OCT analysis, these challenges remain unresolved. To address these limitations, we introduce the Vision Transformer-based Dual-Stream Self-Supervised Pretraining Network (ViT-2SPN), a novel framework designed to enhance feature extraction and improve diagnostic accuracy. ViT-2SPN employs a three-stage workflow: Supervised Pretraining, Self-Supervised Pretraining (SSP), and Supervised Fine-Tuning. The pretraining phase leverages the OCTMNIST dataset (97,477 unlabeled images across four disease classes) with data augmentation to create dual-augmented views. A Vision Transformer (ViT-Base) backbone extracts features, while a negative cosine similarity loss aligns feature representations. Pretraining is conducted over 50 epochs with a learning rate of 0.0001 and momentum of 0.999. Fine-tuning is performed on a stratified 5.129\% subset of OCTMNIST using 10-fold cross-validation. ViT-2SPN achieves a mean AUC of 0.93, accuracy of 0.77, precision of 0.81, recall of 0.75, and an F1 score of 0.76, outperforming existing SSP-based methods. These results underscore the robustness and clinical potential of ViT-2SPN in retinal OCT classification.</p>

## Data Samples (Class: Normal, Drusen, DME, CNV)
![ViT-2SPN Architecture](https://github.com/mrsaraei/ViT-2SPN/blob/7f74e36f5e2fe3a57ed47c0647571e0067ab9c40/figures/Fig_2-1.svg)

## ViT-2SPN Architecture
![ViT-2SPN Architecture](https://github.com/mrsaraei/ViT-2SPN/blob/cd9882d20e9bb5da4d6076d288a845cd82c49e32/figures/Fig_3.svg)

## Experimental Setup
During the SSP phase, the model utilizes the unlabeled OCTMNIST dataset, which comprises 97,477 training samples. The training process is conducted with a mini-batch size of 128, a learning rate of 0.0001, and a momentum rate of 0.999, spanning a total of 50 epochs. The ViT-base architecture, pretrained on the ImageNet dataset, is employed as the backbone. In the fine-tuning phase, the model leverages 5.129\% of the labeled OCTMNIST dataset, following a 10-fold cross-validation strategy. Each fold consists of 4,500 training samples and 500 validation samples, with an additional 500 samples reserved for testing. The fine-tuning process is carried out using a batch size of 16, the same learning rate from the pretraining phase, a dropout rate of 0.5, and 50 epochs
![Experimental Setup](https://github.com/mrsaraei/ViT-2SPN/blob/071fe697c89e938d2c8f74d5dd2399893c5098ca/figures/Fig_11-1.svg)

## Result
<p align="center">
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/bbf5c596a8babe774bb6332bccf64d9011a79f00/figures/Fig_4.svg" alt="Performance Comparison" width="100%"/>
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/dd8e708ce14316ae11113fe5657a37c633a48bca/figures/Fig_5-1.svg" alt="AUC Curve" width="100%"/>
  <img src="https://github.com/mrsaraei/ViT-2SPN/blob/bbf5c596a8babe774bb6332bccf64d9011a79f00/figures/Fig_6.svg" alt="Confusion Matrix" width="100%"/>
</p>

## Performance Improvement
![Performance Improvement](https://github.com/mrsaraei/ViT-2SPN/blob/2dd568836e7b9c49969ce03d7bef83bc20cb0e8b/figures/Fig_7.svg)

## Command
- **ssp_vit2spn.py:** Trains the self-supervised model using unlabeled images to extract meaningful features.  
- **finetune_vit2spn.py:** Fine-tunes the pretrained model for classification tasks using labeled data.
## Usage
Update the paths in the scripts to reflect your own, and to execute any of the scripts, you can run them as follows:

```bash
python ssp_vit2spn.py
python finetune_vit2spn.py
```

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
