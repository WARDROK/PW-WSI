# Ćwiczenie nr 1

## Treść polecenia

Znana jest funkcja celu

 1. $f(x) = Ax + Bsin(x)$ 
    
    $x \in{(-4\pi, 4\pi)}$
 2. $g(x, y) = \frac{Cxy}{e^{x^2 + y^2}}$
    
    $x \in{(-2, 2)}$
    
    $y \in{(-2, 2)}$

Gdzie A, B, C to kolejne dodatnie cyfry numeru indeksu poczynając od jego końca. Np. dla 321037 A=7, B=3, C=1

Zaimplementować metodę gradientu prostego opisaną na wykładzie.

Użyć zaimplementowany algorytm do wyznaczenia ekstremów funkcji.

Zbadać wpływ następujących parametrów na proces optymalizacji:
 - długość kroku uczącego
 - limit maksymalnej liczby kroków algorytmu
 - rozmieszczenie punktu startowego

Zinterpretować wyniki w kontekście kształtu badanej funkcji. 

## Uwagi

 - Punkty startowe algorytmu należy losować z zadanego przedziału.
 - Implementacja algorytmu powinna być jedna dla dowolnej zadanej funkcji i nazywać się `grad_descent`.
 - Do wizualizacji funkcji można użyć dowolnego narzędzia. Pisanie wizualizatora funkcji nie jest celem tego ćwiczenia.
