# Kalbos garso fonemu klasifikavimo igyvendinimo rezultatai

**MATLAB:** 2026b  
**Protokolo versija:** phoneme-revised-v2.1

## Duomenys ir nutekejimo patikra

Pradiniu irasu: 5404. Po vienodu pozymiu vektoriu deduplikavimo: 5395; pasalinta: 9. Train-test identisku vektoriu po skaidymo: 0.

## Train CV parinktos konfigūracijos

| Modelio seima | Konfiguracija | CV BA | CV F1 |
|---|---|---:|---:|
| LR | LR lambda=0.0001 | 0.6757 | 0.5353 |
| kNN | kNN k=3 weight=inverse | 0.8419 | 0.7824 |
| SVM | SVM C=10 gamma=1 prior=uniform | 0.8657 | 0.7950 |
| CS_SVM | CS-SVM C=10 gamma=1 FNcost=3 | 0.8645 | 0.7901 |
| MLP | MLP layers=[32 16] lambda=0.0001 iter=300 | 0.8358 | 0.7696 |

## Galutiniai rezultatai atskiroje testavimo imtyje

| Modelis | BA | F1 | Recall1 | Accuracy | Brier | AUC |
|---|---:|---:|---:|---:|---:|---:|
| LR | 0.6517 | 0.4991 | 0.4540 | 0.7340 | 0.1673 | 0.7894 |
| kNN | 0.8186 | 0.7468 | 0.7302 | 0.8554 | 0.1045 | 0.9012 |
| RBF_SVM | 0.8423 | 0.7609 | 0.8286 | 0.8480 | 0.1016 | 0.9149 |
| CS_RBF_SVM | 0.8489 | 0.7677 | 0.8444 | 0.8508 | 0.1025 | 0.9138 |
| MLP_ensemble3 | 0.8226 | 0.7496 | 0.7460 | 0.8545 | 0.1051 | 0.9186 |

## Hipotezes patikra

Stipresnis atskaitos metodas pagal train CV: **kNN**. CS-RBF-SVM minus atskaitos metodas: DeltaBA=0.0303; porinio bootstrap 95% intervalas [0.0079; 0.0546], DeltaRecall1=0.1143. Is anksto uzrasyta H1 patvirtinta.

## Tikimybiu kalibravimo diagnostika

| Variant | Slenkstis | BA | F1 |
|---|---:|---:|---:|
| RBF_SVM_calibrated_t0.5 | 0.500 | 0.8254 | 0.7520 |
| RBF_SVM_calibrated_tCal | 0.280 | 0.8504 | 0.7675 |
| CS_RBF_SVM_calibrated_t0.5 | 0.500 | 0.8228 | 0.7473 |
| CS_RBF_SVM_calibrated_tCal | 0.280 | 0.8458 | 0.7599 |

## Atkuriamumas

Bendra trukme 108.9 s, CV 98.4 s. Fiksuotos seed reiksmes saugomos experiment_config.mat. Pagrindini eksperimenta galima pakartoti komanda `run_all`.
