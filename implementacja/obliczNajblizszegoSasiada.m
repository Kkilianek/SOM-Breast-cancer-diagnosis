function otoczenie = obliczNajblizszegoSasiada(liczbaWierszySiatki,liczbaKolumnSiatki, ...
    wygranyRzad, wygranaKolumna, wariancjaSzerokosci)
% Ta funkcja oblicza odległość poprzeczną między neuronami i oraz
% wygrywających neuronów

    % Zainicjuj macierz do przechowywania odległości euklidesowej między każdym neuronem
    % i wygrywający neuron do obliczenia funkcji sąsiedztwa
    otoczenie = zeros(liczbaWierszySiatki, liczbaKolumnSiatki);
    
    for r = 1:liczbaWierszySiatki
       for c = 1:liczbaKolumnSiatki
           if (r == wygranyRzad) && (c == wygranaKolumna)
               % funkcja sąsiedztwa dla wygrywającego neuron
               otoczenie(r,c) = 1;
           else
               % funkcja sąsiedztwa dla innych neuronów (Spróbowałem
               % zmienić na wzor z prezentacji (zapalanie neuronow w
               % funkcji Gaussa)
               odleglosc = (wygranyRzad - r)^2+(wygranaKolumna - c)^2;
               otoczenie(r,c) = exp(-odleglosc/(2*wariancjaSzerokosci^2));
           end    
       end
    end
end