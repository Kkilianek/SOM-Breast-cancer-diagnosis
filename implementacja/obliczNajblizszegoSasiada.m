function otoczenie = obliczNajblizszegoSasiada(liczbaWierszySiatki,liczbaKolumnSiatki, ...
    wygranyRzad, wygranaKolumna, wariancjaSzerokosci)
% Opis funkcji...

    otoczenie = zeros(liczbaWierszySiatki, liczbaKolumnSiatki);
    
    for r = 1:liczbaWierszySiatki
       for c = 1:liczbaKolumnSiatki
           if (r == wygranyRzad) && (c == wygranaKolumna)
               otoczenie(r,c) = 1;
           else
               odleglosc = norm([r c] - [wygranyRzad wygranaKolumna],2);
               otoczenie(r,c) = exp(-odleglosc^2/(2*wariancjaSzerokosci^2));
           end    
       end
    end
end