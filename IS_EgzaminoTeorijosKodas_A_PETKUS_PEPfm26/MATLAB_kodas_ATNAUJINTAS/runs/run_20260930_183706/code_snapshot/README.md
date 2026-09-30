# Kalbos garso fonemų klasifikavimas - atnaujintas MATLAB eksperimentas

## Viena komanda rezultatams atkartoti

MATLAB `Current Folder` nustatykite į šį aplanką ir paleiskite:

```matlab
run_all
```

Kiekvienas paleidimas sukuria naują `runs/run_YYYYMMDD_HHMMSS/` aplanką ir jo ZIP archyvą. Ankstesni rezultatai neperrašomi. Fiksuotos atsitiktinės sėklos įrašomos į `experiment_config.mat`, o `code_snapshot/` saugo tikslų paleidimo metu naudotą kodą.

Norint palyginti du pakartotinius paleidimus:

```matlab
compare_runs('runs/run_...','runs/run_...')
```

## Reikalavimai

- MATLAB R2023b (parengta ir dokumentuota šiai versijai)
- Statistics and Machine Learning Toolbox
- interneto ryšys tik jei `phoneme.arff` nėra šalia kodo

Duomenys: OpenML **Phoneme**, ID 1489. Jei vietinio ARFF nėra, `main.m` bando atsisiųsti originalų OpenML failą.

## Praktinė problema

Penki jau apskaičiuoti akustiniai požymiai naudojami atskirti nosinius ir burninius kalbos garsus. Tai nėra visa kalbos atpažinimo sistema ir kodas neskaito mikrofono signalo. Jis sprendžia dvejetainio klasifikavimo uždavinį, kuriame svarbu ne tik bendras teisingų atsakymų procentas, bet ir tai, kad mažesnė burninių garsų klasė nebūtų sistemingai ignoruojama.

## Kas pakeista po dėstytojo pastabų

1. Tapatūs penkių požymių vektoriai konservatyviai deduplikuojami **prieš** train/cal/test skaidymą. Jei tam pačiam vektoriui būtų priskirtos skirtingos klasės, programa sustoja.
2. Sukuriamas `leakage_audit.csv`; `assert` reikalauja nulio tapačių vektorių sutapimų tarp train/cal/test.
3. Hiperparametrai parenkami tik 5 dalių kryžmine patikra mokymo imtyje. Standartizavimo statistikos kiekvienoje dalyje skaičiuojamos tik iš tos dalies mokymo poaibio.
4. MLP derinama ne tik architektūra ir reguliarizacija, bet ir mokymo iteracijų limitas (300 arba 1000).
5. Kalibravimo imtis nenaudojama modelių hiperparametrams derinti; galutinis testas atveriamas tik po `PROTOCOL_LOCKED.txt` sukūrimo.
6. Galutinis visų modelių palyginimas skaičiuojamas tame pačiame atidėtame testavimo rinkinyje.
7. Įtrauktas 2024 m. literatūra pagrįstas **klaidų kainoms jautrus RBF-SVM (CS-RBF-SVM)**. Jis nėra vadinamas pilnu MSHR-FCSSVM, nes šiame kode neįgyvendintas straipsnio MSHR balansavimas ir RIME optimizavimas.
8. SVM klasifikavimo BA/F1 skaičiuojami pagal paties SVM klasės sprendimą. Atskirai kalibruotos tikimybės naudojamos Brier rodikliui, AUC ir kalibravimo kreivei. Taip tikimybių kalibravimas nebepakeičia pagrindinės klasifikavimo taisyklės nepastebimai.
9. Atliekama abliacija, triukšmo atsparumo analizė, klaidų analizė, tikimybės ir faktinės klasės grafikas bei porinė bootstrap patikra.

## Svarbiausi išvesties failai

- `selected_parameters.csv` - tik train CV parinkti hiperparametrai;
- `test_classification_metrics.csv` - galutinis modelių palyginimas teste;
- `test_probability_metrics.csv` - Brier ir AUC;
- `hypothesis_test.csv` - porinė bootstrap hipotezės patikra;
- `leakage_audit.csv` - duomenų nutekėjimo auditas;
- `deduplication_report.csv` - prieš skaidymą sutvarkyti tapatūs požymių vektoriai;
- `ablation_train_cv.csv` - komponentų pašalinimo / pakeitimo bandymai tik train CV;
- `noise_summary.csv` - atsparumo sintetiniam požymių triukšmui santrauka;
- `error_examples.csv` - klaidingi ir ribiniai testiniai pavyzdžiai;
- `test_model_comparison.png`, `confusion_primary.png`, `calibration_primary.png`, `probability_vs_truth.png`, `noise_primary.png` - grafikai;
- `ATASKAITA_AUTOMATINE.md` ir `RUN_SUMMARY.txt` - realaus paleidimo skaitinės santraukos;
- `model_primary.mat` - pagrindinio CS-RBF-SVM modelio paketas.

## Literatūros pagrindas

**Zhu, B., Jing, X., Qiu, L., Li, R. (2024).** *An Imbalanced Data Classification Method Based on Hybrid Resampling and Fine Cost Sensitive Support Vector Machine*. Computers, Materials & Continua, 79(3), 3977-3999. DOI: 10.32604/cmc.2024.048062.

Straipsnyje naudojamas tas pats `Phoneme` rinkinys ir nagrinėjama klaidų kainoms jautraus SVM idėja. Šiame projekte įgyvendinama sąmoningai paprastesnė ir lengvai audituojama modifikacija: klaidingam mažesnės 1 klasės priskyrimui 0 klasei suteikiama didesnė kaina, kuri parenkama tik mokymo kryžmine patikra.

Papildomai literatūros paieškoje rastas 2022 m. darbas apie kontrafaktinį duomenų papildymą klasių disbalansui, kuriame `Phoneme` taip pat naudojamas kaip etaloninis rinkinys (DOI: 10.1016/j.mlwa.2022.100375). Šio darbo metodas nepasirinktas, nes reikalautų atskiro sintetinių duomenų generavimo algoritmo ir apsunkintų gyvą gynimą bei pakartojamumo auditą.

## Svarbus ribotumas

Deduplikavimas apsaugo tik nuo **tapačių požymių vektorių** nutekėjimo. Duomenų rinkinyje nėra kalbėtojo ar pradinio garso įrašo grupės identifikatorių, todėl negalima įrodyti, kad skirtingi, bet iš to paties šaltinio kilę fragmentai nepateko į skirtingas imtis. Ši grėsmė turi būti nurodyta išvadose.
