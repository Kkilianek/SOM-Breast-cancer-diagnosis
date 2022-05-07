function AktualizacjWagWektorow = aktualizacjaWag(zbiorTreningowy, somMap, liczbaWierszySiatki, ... 
                        liczbaKolumnSiatki, wymiarDanych, indeks, wspolczynnikUczenia, otoczenie)
% Opis funkcji...

    AktualizacjWagWektorow = zeros(liczbaWierszySiatki, liczbaKolumnSiatki, wymiarDanych);
    
    for r = 1: liczbaWierszySiatki
       for c = 1:liczbaKolumnSiatki
           % Przekształć wymiar aktualnego wektora wagi
           aktualnyWektorWag = reshape(somMap(r,c,:),1,wymiarDanych);
           
           % Zaktualizuj wektor wag dla każdego neuronu
           AktualizacjWagWektorow(r,c,:) = aktualnyWektorWag + wspolczynnikUczenia*otoczenie(r,c)*(zbiorTreningowy(indeks,:)-aktualnyWektorWag);

       end
    end
end