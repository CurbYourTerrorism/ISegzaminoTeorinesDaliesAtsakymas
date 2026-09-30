# MATLAB kodo gynimo atmintinė

## Viena komanda

```matlab
run_all
```

Kiekvieną kartą sukuriamas naujas `runs/run_...` aplankas. Nieko rankiniu būdu trinti nereikia.

## Kur ką parodyti `main.m`

1. **1 sekcija** - fiksuotos sėklos, hipotezė, triukšmo lygiai ir eksperimento versija.
2. **2-5 sekcijos** - duomenų nuskaitymas, tapačių vektorių deduplikacija, train/cal/test skaidymas ir `leakage_audit.csv`.
3. **6-8 sekcijos** - tik mokymo imties 5 dalių kryžminė patikra ir hiperparametrų atranka.
4. **9 sekcija** - abliaciniai bandymai, kol testas dar neatvertas.
5. **10-13 sekcijos** - galutinis preprocessing, modelių mokymas, tikimybių kalibravimas ir `PROTOCOL_LOCKED.txt`.
6. **14 sekcija** - vienintelis pagrindinis visų modelių palyginimas galutinėje testavimo imtyje.
7. **15 sekcija** - kalibravimo slenksčio diagnostika; ji nėra pagrindinė modelio atranka.
8. **16 sekcija** - H1 porinė bootstrap patikra: CS-RBF-SVM prieš stipresnį LR/kNN atskaitos metodą.
9. **17-18 sekcijos** - triukšmas, klaidų analizė ir grafikai.
10. **19-20 sekcijos** - diegiamas modelis, prognozavimo laikas ir išsaugomos testinės prognozės.

## Kaip paaiškinti svarbiausią pataisymą

Ankstesniame kode kalibruotos SVM tikimybės `p>=0.5` buvo naudojamos ir kaip pagrindinė klasifikavimo taisyklė. Senajame teste tai sumažino BA nuo 0.8728 iki 0.8365. Naujame kode:

- klasė BA/F1 skaičiavimui = `predict(rawModel, X)`;
- kalibruota tikimybė = `predict(calibratedModel, X)` score;
- Brier/AUC/kalibravimo grafikas naudoja tikimybę;
- cal dalyje parinktas slenkstis rodomas tik kaip papildoma diagnostika.

## Kaip paaiškinti CS-RBF-SVM

Standartinis RBF-SVM baudžia klaidas simetriškai. `CS_SVM` atveju:

```matlab
cost = [0 1; fnCost 0];
```

`ClassNames=[0;1]`, todėl `cost(2,1)` yra tikros 1 klasės priskyrimo 0 klasei (FN) kaina. `fnCost` parenkamas tik train CV. Tai supaprastinta 2024 m. Zhu ir kt. Phoneme tyrimo cost-sensitive SVM idėja, ne pilnas MSHR-FCSSVM.

## Jei paprašo pakeisti kodą gyvai

Saugūs pavyzdžiai, kuriems NEREIKIA liesti testo:

- pakeisti kNN tinklelį `k=[21 11 5 3]`, pvz. pridėti `7`;
- pakeisti CS-SVM `fnCost` tinklelį, pvz. pridėti `2.5`;
- pakeisti MLP iteracijų tinklelį `[300 1000]`;
- pridėti triukšmo lygį `0.35`;
- pakeisti bootstrap pakartojimų skaičių diagnostiniam greitam bandymui.

Po tokio pakeitimo turi būti kuriamas **naujas** `run_all` paleidimas. Negalima pažiūrėti į seno testo rezultatą ir pagal jį nuspręsti, kurį hiperparametrą įtraukti kaip „geriausią“.

## Ką reiškia metrikos vienu sakiniu

- **BA** - dviejų klasių jautrumo vidurkis; pagrindinis rodiklis dėl disbalanso.
- **Recall1** - kiek tikrų burninių garsų atpažinta.
- **Precision1** - kiek prognozuotų burninių iš tiesų buvo burniniai.
- **F1** - Recall1 ir Precision1 kompromisas.
- **Accuracy** - bendra teisingų etikečių dalis, bet dėl disbalanso nėra pakankama viena.
- **Brier** - tikimybės kvadratinė paklaida; mažiau geriau.
- **AUC** - kaip gerai modelis surikiuoja 1 klasę aukščiau už 0 per visus slenksčius.

## Duomenų nutekėjimo įrodymas

Parodyti `leakage_audit.csv`. Train-test, train-cal ir cal-test tapačių penkių požymių vektorių skaičius turi būti **0**. Taip pat kode matyti, kad CV preprocessing mokomas tik `idxTrain`, o galutinis `Ztest` apskaičiuojamas tik po protokolo užrakinimo.

Likusi riba: nėra kalbėtojo / originalaus įrašo ID, todėl negalima patikrinti grupinio nutekėjimo tarp skirtingų, bet galimai susijusių fragmentų.
