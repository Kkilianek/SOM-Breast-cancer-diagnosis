function rysujDane(zbiorTreningowy, y)
% Opis funkcji...

    figure;
    title('Wizualizacja danych treningowych')
    hold on;
    cluster1 = find(y==0);
    cluster2 = find(y==0);
    plot(zbiorTreningowy(cluster1,1),zbiorTreningowy(cluster1,2),'yo','LineWidth',2,'MarkerSize',12);
    plot(zbiorTreningowy(cluster2,1),zbiorTreningowy(cluster2,2),'m+','LineWidth',2,'MarkerSize',12);
    
    set(gcf,'un','n','pos',[0,0,1,1])
    figure(gcf)
    hold off

end
