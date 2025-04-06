# Ćwiczenie nr 2

## Treść polecenia

Zaimplementować algorytm ewolucyjny z mutacją, selekcją ruletkową, krzyżowaniem oraz sukcesją generacyjną.

Użyć zaimplementowany algorytm do wyznaczenia najbliższej trasy w
problemie komiwojażera na zbiorze polskich miast (`data/cities.csv`).

Znaleźć ustawienie hiperparametrów algorytmu, które zwraca przyzwoite wyniki.

Zbadać wpływ następującego hiperparametru na proces optymalizacji (jak dobre rozwiązanie zwraca algorytm,
jak szybko algorytm dochodzi do dobrego rozwiązania):

 - Osoby o numerze indeksu podzielnym przez 3: wielkość populacji   
 - Osoby o numerze indeksu niepodzielnym przez 3, z resztą z dzielenia równą 1: prawdopodobieństwo mutacji
 - Osoby o numerze indeksu niepodzielnym przez 3, z resztą z dzielenia równą 2: prawdopodobieństwo krzyżowania
 
Wypisać najlepszą znalezioną przez algorytm trasę. 
W sprawozdaniu zawrzeć zrzut ekranu z wizualizacji trasy na mapie Polski używając np. Google Maps



## Uwagi
 - Rozwiązanie (osobnik z populacji) jest reprezentowane listą indeksów miast kolejno odwiedzanych np.
    [0,2,1,3] dla danych
    
    |   | A | B | C | D | 
    |---|---|---|---|---|
    | A | 0 | x | x | x | 
    | B | x | 0 | x | x | 
    | C | x | x | 0 | x | 
    | D | x | x | x | 0 | 
     oznacza następującą trasę: [A, C, B, D] 
    
 - Każdy osobnik musi być poprawny tj. każde miasto musi być odwiedzone dokładnie jeden raz,
   miasta początkowe i końcowe są ustalone dla wszystkich osobników.
 - Zostały przygotowane fragmenty kodu odpowiedzialne za wczytanie, sprawdzenie poprawności,
    ewaluację, generowanie losowego rozwiązania, reprezentację rozwiązania w postaci trasy. Należy z nich korzystać,
    ewentualnie wprowadzać konieczne modyfikacje.
 - Każdą z operacji genetycznych (mutacja, krzyżowanie...) można zaimplementować w jednej, wybranej przez siebie wersji.
    Podczas implementacji operacji genetycznych pamiętać o warunkach poprawności osobników
 - Należy pamiętać, żeby przy eksperymentach powtarzać je na wystarczająco licznej próbie losowej,
a wynik podawać jako średnią i odchylenie standardowe
 - Ustalić jedną parę miast (początek, koniec) na wszystkie eksperymenty, zmieniając przykładową parę (Łomża, Częstochowa)
 
