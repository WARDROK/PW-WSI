# Ćwiczenie nr 4

## Treść polecenia

1. Zaimplementować algorytm regresji logistycznej.  
2. Sprawdzić jakość działania algorytmu dla klasyfikacji na zbiorze danych Census Income
[https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
3. Policzyć wynik dla przynajmniej 3 różnych sposobów przygotowania danych, na przykład usuwając niektóre kolumny, dodając normalizację wartości.

## Uwagi 

- Zbiór danych należy podzielić losowo na podzbiory uczący, i testowy (75-25). Podział musi być stały dla wszystkich eksperymentów,
 tj. należy ustawiać to samo ziarno generatora liczb losowych.
- Jako metryk jakości użyć celności (accuracy), F1 oraz AUROC. 
- Przed rozpoczęciem uczenia należy przyjrzeć się danym i odpowiednio je przygotować.
W szczególności zwrócić uwagę na typ i zakres wartości atrybutów wejściowych oraz na brakujące wartości w zbiorze.
- Do wszystkich powyższych operacji, poza samym algorytmem regresji logistycznej, można używać gotowych implementacji
bibliotecznych (polecam głównie `scikit-learn`), ale należy rozumieć funkcje, z których się korzysta.
