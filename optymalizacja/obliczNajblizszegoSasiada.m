function otoczenie = obliczNajblizszegoSasiada(liczbaWierszySiatki,liczbaKolumnSiatki, ...
    wygranyRzad, wygranaKolumna, wariancjaSzerokosci)
% Funkcja odpowiedzialna za liczenie otoczenia na podstawie funkcji Gaussa

    otoczenie = zeros(liczbaWierszySiatki, liczbaKolumnSiatki); % prealokacja
    
    for r = 1:liczbaWierszySiatki
        for c = 1:liczbaKolumnSiatki
            if (r == wygranyRzad) && (c == wygranaKolumna)
               otoczenie(r,c) = 1; % jeżeli wybrany neuron jest wygrany zwróć otoczenie równe 1
            else % w przeciwnym przypadku oblicz otoczenie na podstawie poniższego wzoru
               odleglosc = norm([r c] - [wygranyRzad wygranaKolumna],2);
               otoczenie(r,c) = exp(-odleglosc^2/(2*wariancjaSzerokosci^2));
            end    
        end
    end
end