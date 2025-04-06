# Ćwiczenie nr 7

## Treść polecenia

1. Dla zbioru danych o zabójstwach w USA z lat 1980-2014 [https://www.kaggle.com/datasets/mrayushagrawal/us-crime-dataset](https://www.kaggle.com/datasets/mrayushagrawal/us-crime-dataset)
wybrać następujące cechy {Victim Sex, Victim Age, Victim Race, Perpetrator Sex, Perpetrator Age, Perpetrator Race, Relationship, Weapon}
2. Przy pomocy jednej z bibliotek  [pgmpy](https://github.com/pgmpy/pgmpy/tree/dev), [pomegranate](https://github.com/jmschrei/pomegranate), [bnlearn](https://github.com/erdogant/bnlearn) wygenerować sieć 
Bayesowską modelującą zależności pomiędzy tymi cechami. Podpowiedź: należy znaleźć strukturę sieci  (structure learning), 
następnie estymować prawdopodobieństwa warunkowe pomiędzy zmiennymi losowymi (parameter learning).
3. Zwizualizować i przeanalizować nauczoną sieć - jakie są rozkłady prawdopodobieństw pojedynczych cech, 
jakie zależności pomiędzy cechami można zauważyć?
4. Zaimplementować losowy generator danych, który działa zgodnie z rozkładem reprezentowanym przez wygenerowaną sieć.
5. Użyć generatora do wygenerowania kilku losowych morderstw, podając jako argumenty różne obserwacje.


## Uwagi 

- Proszę spróbować stworzyć jedną sieć dla danych globalnych tj. ze wszystkich lat, we wszystkich dostępnych miastach. 
Gdyby występowały problemy wydajnościowe (sieć się uczy za długo, brakuje pamięci), proszę ograniczyć się do jednej/kilku lokalizacji,
ewentualnie także zmniejszyć przedział czasowy.
- Generator danych powinien działać w następujący sposób:
  - Jako argument przyjmuje od użytkownika niepełną obserwację (może być pusta) np. {?, 20, ?, male, ?, asian, friend, strangulation}
  - Zwraca losowo wygenerowaną pełną krotkę {Victim Sex, Victim Age, Victim Race, Perpetrator Sex, Perpetrator Age, Perpetrator Race, Relationship, Weapon},
  przy czym generuje ją zgodnie z rozkładami prawdopodobieństw sieci Bayesowskiej.
- Do uczenia sieci Bayesowskiej powinniśmy używać pewnych danych, należy więc odfiltrować krotki zawierające wartość *Unknown* 
