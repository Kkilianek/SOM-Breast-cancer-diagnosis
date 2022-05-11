function aktualizacjWagWektorow = aktualizacjaWag(zbiorTreningowy, somMap, liczbaWierszySiatki, ... 
                        liczbaKolumnSiatki, wymiarDanych, indeks, wspolczynnikUczenia, otoczenie)
% Opis funkcji...

    aktualizacjWagWektorow = zeros(liczbaWierszySiatki, liczbaKolumnSiatki, wymiarDanych);
    
    for r = 1: liczbaWierszySiatki
        for c = 1:liczbaKolumnSiatki
           aktualnyWektorWag = reshape(somMap(r,c,:),1,wymiarDanych);
           aktualizacjWagWektorow(r,c,:) = aktualnyWektorWag + wspolczynnikUczenia*otoczenie(r,c)*(zbiorTreningowy(indeks,:)-aktualnyWektorWag);
        end
    end
end