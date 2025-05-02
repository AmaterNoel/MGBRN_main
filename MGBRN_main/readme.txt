This repository contains the source code for training and evaluating our proposed method. The key files and their functionalities are summarized below:

TST_main.py: Supports standalone training of the fine-grained temporal module (STGE).

RT_main.py: Supports standalone training of the coarse-grained temporal module (SL-TGE).

T2BRN_main.py: Trains the backbone network using STGE in both branches.

R2BRN_main.py: Trains the backbone network using SL-TGE in both branches.

MBBRN_main.py: Trains the full method proposed in our paper (MBBRN).

MBBRN_eval.py: Performs standalone evaluation of the proposed method.

MBBRN_example.py: Generates the recognition results of our method.

model/: Contains pre-trained models used in the experiments.