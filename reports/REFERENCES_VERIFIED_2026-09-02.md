# Reference verification — 2026-09-02

Every entry below was checked against a primary registry (Crossref API, arXiv API, PhysioNet,
publisher page) on 2026-09-02. **V** = verified at the URL given; **U** = could not be resolved
(recorded as unresolved, never asserted as absent). This file supersedes the bibliographic
`[TODO]` and `[unverified]` markers that `reports/PREPRINT_DRAFT.md` carried through v0.4.

## Corrections this pass forced

1. **The bioRxiv preprint of the parent paper has a different title.** It is *"Fail closed trust
   gated synthetic augmentation governs tail risk under subject shift in EEG"*, not the npj title.
   Neither Crossref nor bioRxiv records a preprint-to-article relation; the identity rests on the
   identical author list. Cite it as a separate item and do not assert a formal link.
2. **SzCORE framework paper year.** Epilepsia 66(S3):14–24 is the September **2025** issue; the
   article went online 18 Sep 2024. Do not write "2024;66(S3)".
3. **arXiv:2210.15809 is Zheng, Liu, Lai & Prakash, "Coverage-centric coreset selection for high
   pruning rates" (ICLR 2023)**, not Sorscher et al. It is a defensible citation for "random
   selection is a strong baseline at high pruning rates", which is the sense the draft uses; the
   Sorscher et al. paper (arXiv:2206.14486, NeurIPS 2022) is the alternative if the intended claim
   is about data-pruning scaling.
4. **PhysioNet now asks for Guttag (2010, dataset DOI 10.13026/C2K01R) plus Pollard et al. 2026
   (*Nature Health*)**; Goldberger 2000 remains correct as the historical platform citation.
5. **The parent paper uses an article number (634), not pages.**
6. **LCT identity caveat.** The only well-known "lightweight convolution transformer" for
   cross-patient seizure detection is Rukhsar & Tiwari 2023. Whether `models/lct.py` implements
   that architecture has to be confirmed from the source before the citation is used as
   "the LCT of [18]"; otherwise cite it as the inspiration and describe ours.

## Per-reference status

| # | reference | status | notes |
|---|---|---|---|
| 1 | Choi, Yip, Choi, Park 2026, npj Digit Med 9(1):634 | V | Crossref + nature.com; open access, CC BY-NC-ND 4.0; published 25 May 2026 |
| 2 | Choi et al. bioRxiv 10.64898/2026.01.26.701638 | V (title differs) | posted 28 Jan 2026; CC BY 4.0; no registry link to [1] |
| 3 | Moutonnet, Corneck, Tobar, Mandic, arXiv:2601.21752 | V | v1 only, 29 Jan 2026, stat.ME; no journal ref |
| 4 | Shidani, Farghly, Sun, Ganjgahi, Deligiannidis, arXiv:2510.08095 | V | v1 9 Oct 2025, v2 31 Mar 2026; title case normalised |
| 5 | Wolfrath, Wolfrath, Hu, Banerjee, Kothari, arXiv:2409.12116 | V / U venue | 18 Sep 2024; no published venue resolved |
| 6a | Dan, Shahbazinia, Kechris, Atienza, arXiv:2505.18191 | V / U proceedings | v1 19 May 2025 (challenge report title); v2 18 May 2026 (retitled); listed on EPFL's ICML 2026 page, proceedings entry unresolved |
| 6b | Dan et al., SzCORE framework, Epilepsia 66(S3):14–24 | V | doi 10.1111/epi.18113; 12 authors; online 18 Sep 2024 |
| 7 | DeMasi, Kording, Recht, PLoS ONE 12(9):e0184604 | V | doi 10.1371/journal.pone.0184604 |
| 8 | Zheng, Liu, Lai, Prakash, ICLR 2023, arXiv:2210.15809 | V (identity corrected) | see correction 3 |
| 9 | Shoeb 2009 PhD thesis, MIT | V | hdl.handle.net/1721.1/54669 |
| 9b | Guttag 2010, CHB-MIT Scalp EEG Database v1.0.0 | V | doi 10.13026/C2K01R; RRID:SCR_007345 |
| 9c | Goldberger et al. 2000, Circulation 101(23):e215 | V / U end page | Crossref gives e215 only |
| 9d | Pollard et al. 2026, Nat Health 1(8):792–795 | V | doi 10.1038/s44360-026-00096-z |
| 10 | Nadeau & Bengio 2003, Mach Learn 52(3):239–281 | V | doi 10.1023/A:1024068626366 |
| 11 | Lawhern et al. 2018, J Neural Eng 15(5):056013 | V | six authors; doi 10.1088/1741-2552/aace8c |
| 12 | Gulrajani et al. 2017, NIPS 30:5767–5777 | V | arXiv:1704.00028 |
| 13 | Bai, Kolter, Koltun 2018, arXiv:1803.01271 (TCN) | V | no journal ref |
| 14 | Rukhsar & Tiwari 2023, CMPB 242:107856 (LCT) | V / identity caveat | doi 10.1016/j.cmpb.2023.107856 |
| 15 | Lin et al. 2017, ICCV, focal loss | V | doi 10.1109/ICCV.2017.324 |
| 16 | timescoring (ESL-EPFL) | V (repo) / U version | github.com/esl-epfl/timescoring |
| 17 | Bouckaert & Frank 2004, PAKDD, LNCS 3056:3–12 | V | doi 10.1007/978-3-540-24775-3_3 |
| 18 | Dietterich 1998, Neural Comput 10(7):1895–1923 | V | doi 10.1162/089976698300017197 |

## Reference list as used in the manuscript

1. Choi D, Yip C, Choi A, Park J. Trust-gated synthetic EEG augmentation reduces performance drops when generalizing to new patients. *npj Digit Med*. 2026;9(1):634. doi:10.1038/s41746-026-02778-0
2. Choi D, Yip C, Choi A, Park J. Fail closed trust gated synthetic augmentation governs tail risk under subject shift in EEG. *bioRxiv*. Posted 28 Jan 2026. doi:10.64898/2026.01.26.701638
3. Moutonnet N, Corneck J, Tobar F, Mandic D. Synthesizing epileptic seizures: Gaussian processes for EEG generation. arXiv:2601.21752. 2026.
4. Shidani A, Farghly T, Sun Y, Ganjgahi H, Deligiannidis G. Beyond real data: synthetic data through the lens of regularization. arXiv:2510.08095. 2025.
5. Wolfrath N, Wolfrath J, Hu H, Banerjee A, Kothari AN. Stronger baseline models: a key requirement for aligning machine learning research with clinical utility. arXiv:2409.12116. 2024.
6. Dan J, Shahbazinia A, Kechris C, Atienza D. Quantifying the generalization gap in seizure detection: a large-scale empirical benchmark via the SzCORE challenge. arXiv:2505.18191 (v2, 2026; v1 2025 as "SzCORE as a benchmark: report from the seizure detection challenge at the 2025 AI in Epilepsy and Neurological Disorders Conference").
7. Dan J, Pale U, Amirshahi A, Cappelletti W, Ingolfsson TM, Wang X, Cossettini A, Bernini A, Benini L, Beniczky S, Atienza D, Ryvlin P. SzCORE: Seizure Community Open-Source Research Evaluation framework for the validation of electroencephalography-based automated seizure detection algorithms. *Epilepsia*. 2025;66(S3):14–24 (online 18 Sep 2024). doi:10.1111/epi.18113
8. DeMasi O, Kording K, Recht B. Meaningless comparisons lead to false optimism in medical machine learning. *PLoS ONE*. 2017;12(9):e0184604. doi:10.1371/journal.pone.0184604
9. Zheng H, Liu R, Lai F, Prakash A. Coverage-centric coreset selection for high pruning rates. In: *Proc ICLR 2023*. arXiv:2210.15809
10. Shoeb AH. Application of machine learning to epileptic seizure onset detection and treatment [PhD thesis]. Harvard–MIT Division of Health Sciences and Technology, MIT; 2009. http://hdl.handle.net/1721.1/54669
11. Guttag J. CHB-MIT Scalp EEG Database (version 1.0.0). PhysioNet. 2010. doi:10.13026/C2K01R
12. Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*. 2000;101(23):e215. doi:10.1161/01.CIR.101.23.e215
13. Pollard T, Moody BE, Lehman L-wH, Gow BJ, Fernandes C, Xie C, Johnson A, Mark RG, Heldt T. PhysioNet as a global platform for biomedical research. *Nat Health*. 2026;1(8):792–795. doi:10.1038/s44360-026-00096-z
14. Nadeau C, Bengio Y. Inference for the generalization error. *Mach Learn*. 2003;52(3):239–281. doi:10.1023/A:1024068626366
15. Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *J Neural Eng*. 2018;15(5):056013. doi:10.1088/1741-2552/aace8c
16. Gulrajani I, Ahmed F, Arjovsky M, Dumoulin V, Courville A. Improved training of Wasserstein GANs. In: *Advances in Neural Information Processing Systems 30*. 2017:5767–5777. arXiv:1704.00028
17. Bai S, Kolter JZ, Koltun V. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271. 2018.
18. Rukhsar S, Tiwari AK. Lightweight convolution transformer for cross-patient seizure detection in multi-channel EEG signals. *Comput Methods Programs Biomed*. 2023;242:107856. doi:10.1016/j.cmpb.2023.107856
19. Lin T-Y, Goyal P, Girshick R, He K, Dollár P. Focal loss for dense object detection. In: *Proc IEEE ICCV*. 2017:2999–3007. doi:10.1109/ICCV.2017.324
20. Embedded Systems Laboratory, EPFL. timescoring: event- and sample-based scoring for time-series annotations [software]. https://github.com/esl-epfl/timescoring
21. Bouckaert RR, Frank E. Evaluating the replicability of significance tests for comparing learning algorithms. In: *Advances in Knowledge Discovery and Data Mining (PAKDD 2004)*, LNCS 3056. Springer; 2004:3–12. doi:10.1007/978-3-540-24775-3_3
22. Dietterich TG. Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Comput*. 1998;10(7):1895–1923. doi:10.1162/089976698300017197
