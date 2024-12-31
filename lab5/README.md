# Ćwiczenie nr 5

## Treść polecenia

1. Zaimplementować sieć neuronową - perceptron wielowarstwowy, oraz mechanizm jej uczenia algorytmem propagacji
wstecznej gradientu.
2. Sprawdzić jakość działania algorytmu dla zadania regresji na zbiorze danych Wine Quality
[https://archive.ics.uci.edu/dataset/186/wine+quality](https://archive.ics.uci.edu/dataset/186/wine+quality).
3. Spróbować znaleźć konfigurację sieci, która pozwala osiągać dobre rezultaty. 

## Uwagi 

- Zbiór danych należy podzielić losowo na podzbiory uczący, i testowy (75-25). Podział musi być stały dla wszystkich eksperymentów,
 tj. należy ustawiać to samo ziarno generatora liczb losowych. 
- Przed rozpoczęciem uczenia należy przyjrzeć się danym i odpowiednio je przygotować.
W szczególności zwrócić uwagę na typ i zakres wartości atrybutów wejściowych oraz na brakujące wartości w zbiorze.
- Do wszystkich powyższych operacji, poza impementacją modelu, można używać gotowych funkcji
bibliotecznych (polecam głównie `scikit-learn`), ale należy rozumieć funkcje, z których się korzysta.

