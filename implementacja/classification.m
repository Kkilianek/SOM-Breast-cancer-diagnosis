function result=classification(x,y,wspf,wspp) %zwraca 0 je�li jest fizjo, a 1 je�li pato
for i=1:size(wspf,1)
    if wspf(i,:)==[x,y]
        result=0;
    end
end
for i=1:size(wspp,1)
    if wspp(i,:)==[x,y]
        result=1;
    end
end
end