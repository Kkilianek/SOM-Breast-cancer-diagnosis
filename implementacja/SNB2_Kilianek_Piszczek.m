%% Kacper Kilianek (305375), Adam Piszczek (303803) [zespół nr. 22]
% Sieci neuronowe w zastosowaniach biomedycznych (SNB) – Projekt
% Projekt nr. 36: Diagnostyka raka piersi w badaniach mammograficznych za pomocą sieci SOM (katalog: Mammographic Mass_MLR)

%% ========= Przygotowanie środowiska =========

clear;
clc;
clf;
close all;

%% ========= Wczytanie danych =========

% Dane wczytane są w kolejnych kolumnach BI-RADS,Age,Shape,Margin,Density,Severity
try
    M = readtable('mammographic_masses.data.txt'); % przekonwertowanie plików na txt
catch 
    fprintf("Nie udało się otworzyć pliku mammographic_masses.data.txt")
end

%% ========= Preprocessing danych (etap 1) =========

size1 = size(M,1); % zapisanie ilości wektorów cech przed preprocessingiem
M = table2array(M); % zamiana na dane numeryczne

% Proponowany sposób kodowania danych numerycznych -> single
% (single-precision number), ponieważ mamy do czynienia z liczbami 
% całkowitymi typu integer
single(M); 

% usunięcie złych danych pierwszej cechy (BI-RADS) z poza zdefiniowanego zakresu 1-5
% ustawienie warunków pilnujących podane zakresy cech:

% Pierwszy wektor cech (BI-RADS):
Cond1 = M(:,1) > 5;
Cond2 = M(:,1) < 1; 
Condition1 = Cond1 | Cond2; % połaczenie warunków

% Trzeci wektor cech (Kształt):
Cond1 = M(:,3) > 4;
Cond2 = M(:,3) < 1; 
Condition2 = Cond1 | Cond2; 

% Czwarty wektor cech (Margines):
Cond1 = M(:,4) > 5;
Cond2 = M(:,4) < 1; 
Condition3 = Cond1 | Cond2; 

% Trzeci wektor cech (Gęstość):
Cond1 = M(:,5) > 4;
Cond2 = M(:,5) < 1; 
Condition4 = Cond1 | Cond2; 

Conditions = Condition1 | Condition2 | Condition3 | Condition4;
M(Conditions,:) = []; 
size2 = size(M,1); % wielkość macierzy po redukcji błędnych wektorów
numOfDeletedRows = size1 - size2; % ilość danych, które zostały usunięte

%% ========= Preprocessing danych (etap 2) =========

% Podział danych na złośliwe i łagodne, tak aby przedstawić histogramy cech
% w klasach, które są klasyfikowane
M = sortrows(M,6);
malignant = M(:,6) == 1;
benign = M(:,6) == 0;
Malignant = M(malignant,:); % zbiór cech dla przypadku nowotworu złośliwego
Benign = M(benign,:); % zbiór cech dla przypadku nowotworu łagodnego

%% ========= Preprocessing danych (etap 3) =========

% Przed podaniem danych wejściowych do sieci neuronowej musimy jeszcze
% przeprowadzić operację normalizacji, tak aby każda z cech miała
% identyczny wpływ na proces uczenia się sieci. Jeżeli nie dokonalibyśmy
% takich operacji, to jedna z cech (w naszym przypadku dotycząca wieku),
% miałaby największy wpływ na rozkład danych, przez to, że jej rozpiętość
% jest największa wynosząca od 18 do 96 lat. Dlatego też poniżej
% dokonaliśmy przekształceń tak aby każda cecha przyjmowała wartości w
% zakresie od 0 do 1.

% Zgodnie ze wzorem źródło: http://lh3.ggpht.com/_MrdHIr826C4/Sl7eVpJOYfI/AAAAAAAAB34/MCFvz1r_CZQ/s800/7.JPG
M(:,1) = (M(:,1)-min(M(:,1)))/(max(M(:,1)-min(M(:,1)))) * (1-0) + 0;
M(:,2) = (M(:,2)-min(M(:,2)))/(max(M(:,2)-min(M(:,2)))) * (1-0) + 0;
M(:,3) = (M(:,3)-min(M(:,3)))/(max(M(:,3)-min(M(:,3)))) * (1-0) + 0;
M(:,4) = (M(:,4)-min(M(:,4)))/(max(M(:,4)-min(M(:,4)))) * (1-0) + 0;
M(:,5) = (M(:,5)-min(M(:,5)))/(max(M(:,5)-min(M(:,5)))) * (1-0) + 0;

%% ========= Podział danych na zbiór uczący i testowy =========

Test = [Malignant(1:uint64(size(Malignant,1)/2),:) ; Benign(1:uint64(size(Benign,1)/2),:)]; % dane testowe
Train = [Malignant(uint64(size(Malignant,1)/2)+1:size(Malignant,1),:) ; Benign(uint64(size(Benign,1)/2)+1:size(Benign,1),:)]; % dane uczące

%% ========= Implementacja sieci SOM =========

somRow = 100;
somCol = 100;

% Number of iteration for convergence
Iteration = 100;

%%=========== Parameter Setting For SOM ===================================
% Initial size of topological neighbourhood of the winning neuron
width_Initial = 5;

% Time constant for initial topological neighbourhood size
t_width = Iteration/log(width_Initial);

% Initial time-varying learning rate
learningRate_Initial = 1;

% Time constant for the time-varying learning rate
t_learningRate = Iteration;


somMap = inicjalizacjaWag(somRow,somCol,size(Train,2));

train_data = Train;
[dataRow, dataCol] = size(train_data);

for t = 1:Iteration
    width = width_Initial*exp(-t/t_width);
    width_Variance = width^2;
    learningRate = learningRate_Initial*exp(-t/t_learningRate);
    if learningRate <0.025
            learningRate = 0.1;
    end

    [euclideanDist, index] = euklidesowydystans(Train, somMap, somRow, ...
                                            somCol,size(Train,1), size(Train,2));

    [minM,ind] = min(euclideanDist(:));
    [win_Row,win_Col] = ind2sub(size(euclideanDist),ind);

    neighborhood = obliczsasiada( somRow, somCol, win_Row, ...
                                            win_Col, width_Variance);
    somMap = aktualizacjawag( train_data, somMap, somRow, somCol, ...
                                dataCol, index, learningRate, neighborhood);
     % Weight vector of neuron
    dot = zeros(somRow*somCol, dataCol);
    % Matrix for SOM plot grid
    matrix = zeros(somRow*somCol,1);
    % Matrix for SOM plot grid for deletion
    matrix_old = zeros(somRow*somCol,1);
    ind = 1;  
    hold on;
    f1 = figure(1);
    set(f1,'name',strcat('Iteration #',num2str(t)),'numbertitle','off');

    % Retrieve the weight vector of neuron
    for r = 1:somRow
        for c = 1:somCol      
            dot(ind,:)=reshape(somMap(r,c,:),1,dataCol);
            ind = ind + 1;
        end
    end

    % Plot SOM
    for r = 1:somRow
        Row_1 = 1+somRow*(r-1);
        Row_2 = r*somRow;
        Col_1 = somRow*somCol;

        matrix(2*r-1,1) = plot(dot(Row_1:Row_2,1),dot(Row_1:Row_2,2),'--ro','LineWidth',2,'MarkerEdgeColor','g','MarkerFaceColor','g','MarkerSize',4);
        matrix(2*r,1) = plot(dot(r:somCol:Col_1,1),dot(r:somCol:Col_1,2),'--ro','LineWidth',2,'MarkerEdgeColor','g','MarkerFaceColor','g','MarkerSize',4);

        matrix_old(2*r-1,1) = matrix(2*r-1,1);
        matrix_old(2*r,1) = matrix(2*r,1);

    end

    % Delete the SOM plot from previous iteration
    if t~=Iteration  
        for r = 1:somRow
            delete(matrix_old(2*r-1,1));
            delete(matrix_old(2*r,1));
            drawnow;
        end
    end
end


