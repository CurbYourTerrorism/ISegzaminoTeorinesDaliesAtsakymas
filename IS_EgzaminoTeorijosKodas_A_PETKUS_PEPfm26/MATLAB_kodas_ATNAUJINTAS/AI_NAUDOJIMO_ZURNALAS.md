# AI naudojimo žurnalas

Žurnalas dokumentuoja ne tik AI indėlį, bet ir studento / eksperimento kontrolės veiksmus. AI pasiūlymas nelaikomas įrodymu, kol jis nepatikrintas šaltiniu, MATLAB dokumentacija arba realaus paleidimo išvestimi.

## Darbas etapais

1. **Problemos ir terminų patikra.** AI paprašyta perrašyti praktinės problemos paaiškinimą paprasta lietuvių kalba ir suvienodinti terminus: *atskaitos metodas*, *kryžminė patikra*, *subalansuotas tikslumas*, *tikimybių kalibravimas*, *duomenų nutekėjimas*, *klaidų kainoms jautrus klasifikatorius*.
2. **Literatūros patikra.** AI paprašyta ieškoti 2022-2026 m. publikacijų, kuriose naudojamas būtent Phoneme / KEEL Phoneme rinkinys, o ne vien bendrų SVM ar MLP straipsnių.
3. **Eksperimento auditas.** Atskirai tikrintas skaidymas, išankstinis apdorojimas, hiperparametrų parinkimas, kalibravimas ir galutinio testo panaudojimas.
4. **Kodo taisymas.** Tik po audito keistas MATLAB kodas: pridėta deduplikacija, nutekėjimo `assert`, CS-RBF-SVM, MLP mokymo parametro derinimas, protokolo užraktas ir atskirtos klasifikavimo bei tikimybių metrikos.
5. **Atkuriamumas.** Parengtas `run_all`, naujo paleidimo aplankas, kodo kopija, fiksuotos sėklos ir `compare_runs`.
6. **Rezultatų analizė.** Galutiniai naujo eksperimento skaičiai turi būti priimami tik iš MATLAB sugeneruotų CSV / TXT failų. Jei skaičiai silpni, tai nėra priežastis keisti testą ar hipotezę po fakto.

## Priimti AI pasiūlymai

- Įtraukti Zhu ir kt. (2024) darbą, nes jame tiesiogiai naudojamas `Phoneme` rinkinys ir siūlomas MSHR-FCSSVM su klaidų kainoms jautriu SVM komponentu.
- Įgyvendinti **supaprastintą CS-RBF-SVM**, o ne apsimesti, kad atkartota visa MSHR-FCSSVM metodika.
- Tapatų penkių požymių vektorių pasikartojimą pašalinti prieš skaidymą ir programiškai reikalauti nulio tapačių vektorių sutapimų tarp train/cal/test.
- SVM klasifikavimo sprendimą atskirti nuo tikimybių kalibravimo.
- Modelių hiperparametrus ir MLP iteracijų limitą parinkti tik mokymo kryžmine patikra.
- Visų modelių pagrindines metrikas skaičiuoti viename atidėtame teste.

## Atmesti arba apriboti AI pasiūlymai

- **Nevadinti** įgyvendinto metodo `MSHR-FCSSVM`, nes nėra MSHR hibridinio imties balansavimo ir RIME optimizatoriaus.
- Nenaudoti galutinio testo hiperparametrams, tikimybės slenksčiui ar modelio šeimai parinkti.
- Neatrinkti „geriausios“ MLP sėklos pagal testą; trijų iš anksto nustatytų sėklų tikimybės vidurkinamos.
- Nelaikyti sintetinio požymių triukšmo realaus mikrofono triukšmo eksperimentu.
- Neperkelti 2024 m. straipsnio skaičių į šį darbą kaip savo rezultatų: jų protokolas ir metodas skiriasi.

## Aptiktos AI klaidos / nepatikrintos prielaidos ir jų patikra

### 1. Per senas ir formalus literatūros pagrindimas
Ankstesniame plane SVM ir MLP pasirinkimas daugiausia buvo grindžiamas 1995 m. ir 1986 m. algoritmų straipsniais. Jie įrodo, kad metodai egzistuoja ir paaiškina jų principą, bet neparodo, kad jie tinka būtent Phoneme problemai. Patikra: atlikta atskira 2022-2026 m. paieška. Rastas Zhu ir kt. (2024) straipsnis, kuriame Phoneme tiesiogiai naudojamas ir nagrinėjamas klaidų kainoms jautrus SVM.

### 2. Tapatūs požymių vektoriai tarp mokymo ir testo
Ankstesnio MATLAB paleidimo ataskaitoje buvo užfiksuoti **3 tapatūs požymių vektoriai tarp train ir test**. Tai gali šiek tiek optimistiškai paveikti testą. Patikra: naujame kode tapatūs vektoriai deduplikuojami prieš skaidymą, o `leakage_audit.csv` ir `assert` reikalauja nulio sutapimų.

### 3. Tikimybių kalibravimas buvo supainiotas su pagrindiniu klasifikavimo sprendimu
Ankstesniame paleidime RBF-SVM BA pagal natūralų SVM sprendimą buvo **0,8728**, o kalibruotą tikimybę perskyrus per 0,5 slenkstį BA sumažėjo iki **0,8365**. Tai parodė, kad tikimybių kalibravimas ir klasifikavimo riba yra skirtingi klausimai. Pataisa: naujame kode BA/F1 SVM modeliams skaičiuojami pagal `predict` klasę; kalibruota tikimybė naudojama Brier, AUC ir kalibravimo grafikai. Kalibravimo imtyje parinktas slenkstis rodomas tik kaip papildoma diagnostika.

### 4. Atkuriamumas anksčiau reikalavo rankiniu būdu tvarkyti `results` aplanką
Senas `main.m` saugojo viename `results` aplanke ir stabdėsi, jei jis jau egzistavo. Pataisa: `run_all` kiekvieną kartą sukuria naują laiko žyma pažymėtą aplanką, išsaugo tikslaus kodo kopiją ir ZIP archyvą. `compare_runs` leidžia patikrinti du pakartojimus.

## Ką studentas turi patikrinti prieš pateikimą

1. MATLAB R2023b paleisti `run_all` bent kartą, pageidautina du kartus.
2. Patikrinti, kad `leakage_audit.csv` sutapimai yra 0.
3. Peržiūrėti `run_log.txt` dėl įspėjimų ar konvergavimo problemų.
4. Patikrinti `selected_parameters.csv` ir įsitikinti, kad hiperparametrai parinkti iš train CV, ne iš testo.
5. Galutinius skaičius ataskaitoje perkelti iš `test_classification_metrics.csv`, `test_probability_metrics.csv` ir `hypothesis_test.csv`.
6. Paleidus du kartus, naudoti `compare_runs` ir dokumentuoti, ar skirtumai neviršija tolerancijos.
